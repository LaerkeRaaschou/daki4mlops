import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import matplotlib
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from data.dataloader import get_test_loader
from model.resnet18 import ResNet18


BENCHMARK_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_dmon_gpu_id(device):
    """Return the physical GPU id that nvidia-smi should monitor."""
    if device.type != "cuda":
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        first_visible_device = visible_devices.split(",")[0].strip()
        if first_visible_device:
            return first_visible_device

    return str(torch.cuda.current_device())


def initialize_model(num_classes, weights_path, device):
    if weights_path is None:
        raise ValueError("No weights provided. Please provide model weights for inference.")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    model = ResNet18(num_classes)
    weights = torch.load(weights_path, map_location=device)

    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in weights):
        weights = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in weights.items()
        }

    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model


# Image transform used during testing
def build_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class DmonMonitor:

    def __init__(self, output_path, gpu_id=None):
        self.output_path = Path(output_path)
        self.gpu_id = gpu_id
        self.process = None
        self.output = ""
        self.warning = ""

    def __enter__(self):
        if shutil.which("nvidia-smi") is None:
            self.warning = "nvidia-smi not found; GPU utilization fields will be zero."
            return self

        command = ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"]
        if self.gpu_id is not None:
            command = [
                "nvidia-smi",
                "dmon",
                "-i",
                str(self.gpu_id),
                "-s",
                "pucm",
                "-d",
                "1",
            ]

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            self.warning = f"Could not start nvidia-smi dmon: {error}"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is not None:
            self.process.terminate()
            try:
                self.output, _ = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.output, _ = self.process.communicate()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self.output, encoding="utf-8")
        return False


def parse_dmon_output(output):
    sm_values = []
    mem_values = []

    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) < 6:
            continue

        try:
            sm_values.append(float(fields[4]))
            mem_values.append(float(fields[5]))
        except ValueError:
            continue

    return {
        "gpu_sm_avg_pct": sum(sm_values) / len(sm_values) if sm_values else 0.0,
        "gpu_mem_controller_avg_pct": sum(mem_values) / len(mem_values) if mem_values else 0.0,
    }



def dmon_output_path(output_csv, batch_size):
    csv_path = Path(output_csv)
    return csv_path.with_name(f"{csv_path.stem}_dmon_batch{batch_size}.txt")


def benchmark_one_batch_size(cfg, model, batch_size, device, dmon_path, dataset):
    num_workers = int(OmegaConf.select(cfg, "benchmark.num_workers", default=4))

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if len(test_loader.dataset) == 0:
        raise ValueError(f"No samples found in {cfg.inference.data_path}.")

    # Warm up once so cold start does not affect timing
    with torch.inference_mode():
        warmup_images, _ = next(iter(test_loader))
        warmup_images = warmup_images.to(device, non_blocking=True)
        model(warmup_images)
        sync_if_cuda(device)

    processed_samples = 0
    processed_batches = 0
    dmon_gpu_id = get_dmon_gpu_id(device)

    with DmonMonitor(dmon_path, gpu_id=dmon_gpu_id) as dmon:
        if dmon.warning:
            print(dmon.warning)

        sync_if_cuda(device)
        start_time = time.perf_counter()

        with torch.inference_mode():
            for images, _ in test_loader:
                images = images.to(device, non_blocking=True)
                model(images)
                processed_samples += images.size(0)
                processed_batches += 1

        sync_if_cuda(device)
        total_time_s = time.perf_counter() - start_time

    gpu_metrics = parse_dmon_output(dmon.output)

    throughput = processed_samples / total_time_s if total_time_s else 0.0
    latency_ms_per_batch = (
        (total_time_s / processed_batches) * 1000 if processed_batches else 0.0
    )
    latency_ms_per_image = (
        (total_time_s / processed_samples) * 1000 if processed_samples else 0.0
    )

    return {
        "batch_size": batch_size,
        "throughput_images_per_s": throughput,
        "speedup_vs_batch1": 0.0,  # filled in after all batch sizes are measured
        "latency_ms_per_batch": latency_ms_per_batch,
        "latency_ms_per_image": latency_ms_per_image,
        "gpu_sm_avg_pct": gpu_metrics["gpu_sm_avg_pct"],
        "gpu_mem_controller_avg_pct": gpu_metrics["gpu_mem_controller_avg_pct"],
    }


def write_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "batch_size",
        "throughput_images_per_s",
        "speedup_vs_batch1",
        "latency_ms_per_batch",
        "latency_ms_per_image",
        "gpu_sm_avg_pct",
        "gpu_mem_controller_avg_pct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    batch_sizes = [row["batch_size"] for row in rows]
    throughput = [row["throughput_images_per_s"] for row in rows]
    batch_latency = [row["latency_ms_per_batch"] for row in rows]
    image_latency = [row["latency_ms_per_image"] for row in rows]

    x_positions = list(range(len(batch_sizes)))
    x_labels = [str(size) for size in batch_sizes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(x_positions, throughput, marker="o")
    ax1.set_ylabel("Throughput (images/s)")
    ax1.set_title("Batch Inference Benchmark")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_positions, batch_latency, marker="o", label="Latency per batch")
    ax2.plot(x_positions, image_latency, marker="s", label="Latency per image")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_speedup(rows):
    if not rows:
        return

    baseline = rows[0]["throughput_images_per_s"]
    for row in rows:
        row["speedup_vs_batch1"] = (
            row["throughput_images_per_s"] / baseline if baseline else 0.0
        )


def print_short_summary(rows):
    if not rows:
        return

    best = max(rows, key=lambda row: row["throughput_images_per_s"])
    max_throughput = best["throughput_images_per_s"]

    # First batch size within 95% of the maximum observed throughput.
    saturation = next(
        row for row in rows if row["throughput_images_per_s"] >= 0.95 * max_throughput
    )

    print("\nBenchmark summary")
    print(f"Best batch size: {best['batch_size']}")
    print(f"Best throughput: {best['throughput_images_per_s']:.2f} images/s")
    print(f"Best speedup vs batch size 1: {best['speedup_vs_batch1']:.2f}x")
    print(f"Throughput saturates around batch size: {saturation['batch_size']}")


# Batch inference benchmark entrypoint
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    output_csv = OmegaConf.select(
        cfg,
        "benchmark.output_csv",
        default="results/batch_inference_benchmark.csv",
    )
    output_plot = OmegaConf.select(
        cfg,
        "benchmark.output_plot",
        default="results/batch_inference_benchmark.png",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = initialize_model(
        num_classes=cfg.inference.num_classes,
        weights_path=cfg.inference.weights_path,
        device=device,
    )

    initial_loader = get_test_loader(
        mapping_path=cfg.inference.mapping_path,
        test_dir=cfg.inference.data_path,
        transform_test=build_test_transform(),
        test_annotations=cfg.inference.annotations,
        batch_size=1,
        shuffle=False,
    )
    dataset = initial_loader.dataset

    rows = []
    for batch_size in BENCHMARK_BATCH_SIZES:
        print(f"Benchmarking batch_size={batch_size}")
        rows.append(
            benchmark_one_batch_size(
                cfg=cfg,
                model=model,
                batch_size=batch_size,
                device=device,
                dmon_path=dmon_output_path(output_csv, batch_size),
                dataset=dataset,
            )
        )

    add_speedup(rows)
    write_csv(rows, output_csv)
    write_plot(rows, output_plot)
    print_short_summary(rows)

    print(f"\nCSV saved to: {os.path.abspath(output_csv)}")
    print(f"Plot saved to: {os.path.abspath(output_plot)}")


if __name__ == "__main__":
    main()
