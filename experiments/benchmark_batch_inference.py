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

# Load a regular model from saved weights
def initialize_model(num_classes, weights_path, device):
    if weights_path is None:
        raise ValueError(
            "No weights provided. Please provide model weights for inference."
        )
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
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.process = None
        self.output = ""
        self.warning = ""

    def __enter__(self):
        if shutil.which("nvidia-smi") is None:
            self.warning = "nvidia-smi not found; GPU utilization fields will be zero."
            return self

        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
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
        "gpu_mem_controller_avg_pct": (
            sum(mem_values) / len(mem_values) if mem_values else 0.0
        ),
    }


def dmon_output_path(output_csv, batch_size):
    csv_path = Path(output_csv)
    return csv_path.with_name(f"{csv_path.stem}_dmon_batch{batch_size}.txt")


# Measure throughput and latency for one batch size
def benchmark_one_batch_size(cfg, model, batch_size, device, dmon_path, dataset):
    num_workers = int(
        OmegaConf.select(cfg, "benchmark.num_workers", default=4)
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_samples = len(test_loader.dataset)
    if total_samples == 0:
        raise ValueError(f"No samples found in {cfg.inference.data_path}.")

    # Warm up model before timing, ensuring cold start doesn't affect results
    with torch.inference_mode():
        warmup_images, _ = next(iter(test_loader))
        warmup_images = warmup_images.to(device, non_blocking=True)
        model(warmup_images)
        torch.cuda.synchronize()

    processed_samples = 0
    data_loading_time_s = 0.0
    host_to_device_time_s = 0.0
    forward_time_s = 0.0
    loader_iterator = iter(test_loader)

    with DmonMonitor(dmon_path) as dmon:
        if dmon.warning:
            print(dmon.warning)

        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        with torch.inference_mode():
            while True:
                load_start = time.perf_counter()
                try:
                    images, _ = next(loader_iterator)
                except StopIteration:
                    break
                data_loading_time_s += time.perf_counter() - load_start

                transfer_start = time.perf_counter()
                images = images.to(device, non_blocking=True)
                torch.cuda.synchronize()
                host_to_device_time_s += time.perf_counter() - transfer_start

                forward_start = time.perf_counter()
                model(images)
                torch.cuda.synchronize()
                forward_time_s += time.perf_counter() - forward_start
                processed_samples += images.size(0)

        torch.cuda.synchronize()
        total_time_s = time.perf_counter() - start_wall

    gpu_metrics = parse_dmon_output(dmon.output)

    latency_ms_per_image = (
        (total_time_s / processed_samples) * 1000 if processed_samples else 0.0
    )
    throughput_images_per_s = processed_samples / total_time_s if total_time_s else 0.0

    return {
        "batch_size": batch_size,
        "throughput_images_per_s": throughput_images_per_s,
        "latency_ms_per_image": latency_ms_per_image,
        "forward_time_s": forward_time_s,
        "data_loading_time_s": data_loading_time_s,
        "host_to_device_time_s": host_to_device_time_s,
        "gpu_sm_avg_pct": gpu_metrics["gpu_sm_avg_pct"],
        "gpu_mem_controller_avg_pct": gpu_metrics["gpu_mem_controller_avg_pct"],
    }

# Save benchmark rows for report evidence
def write_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "throughput_images_per_s",
        "latency_ms_per_image",
        "forward_time_s",
        "data_loading_time_s",
        "host_to_device_time_s",
        "gpu_sm_avg_pct",
        "gpu_mem_controller_avg_pct",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Plot throughput vs latency trade-off across batch sizes
def write_plot(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    batch_sizes = [row["batch_size"] for row in rows]
    throughput = [row["throughput_images_per_s"] for row in rows]
    latency = [row["latency_ms_per_image"] for row in rows]

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        latency,
        throughput,
        c=batch_sizes,
        cmap="viridis",
        s=80,
    )

    for batch_size, x, y in zip(batch_sizes, latency, throughput):
        ax.annotate(
            str(batch_size),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Latency (ms/image)")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title("Throughput–Latency Trade-off by Batch Size")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Batch size")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    device = torch.device("cuda")
    model = initialize_model(
        num_classes=cfg.inference.num_classes,
        weights_path=cfg.inference.weights_path,
        device=device,
    )
    model.eval()

    rows = []
    # Create the dataset once to avoid cold start for each batch size
    initial_loader = get_test_loader(
        mapping_path=cfg.inference.mapping_path,
        test_dir=cfg.inference.data_path,
        transform_test=build_test_transform(),
        test_annotations=cfg.inference.annotations,
        batch_size=1,
        shuffle=False,
    )
    dataset = initial_loader.dataset

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

    write_csv(rows, output_csv)
    write_plot(rows, output_plot)
    print(f"CSV saved to: {os.path.abspath(output_csv)}")
    print(f"Plot saved to: {os.path.abspath(output_plot)}")


if __name__ == "__main__":
    main()
