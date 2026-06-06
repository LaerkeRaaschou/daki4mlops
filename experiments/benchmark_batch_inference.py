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
import psutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from data.dataloader import get_test_loader
from inference import initialize_model


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


# Read batch sizes from config or use defaults
def parse_batch_sizes(cfg):
    configured = OmegaConf.select(cfg, "benchmark.batch_sizes")
    if configured is None:
        return DEFAULT_BATCH_SIZES

    batch_sizes = [int(batch_size) for batch_size in configured]
    if not batch_sizes:
        raise ValueError("benchmark.batch_sizes must contain at least one batch size.")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("All benchmark batch sizes must be positive.")
    return batch_sizes


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
        "gpu_sm_max_pct": max(sm_values) if sm_values else 0.0,
        "gpu_mem_controller_avg_pct": (
            sum(mem_values) / len(mem_values) if mem_values else 0.0
        ),
        "gpu_mem_controller_max_pct": max(mem_values) if mem_values else 0.0,
    }


def dmon_output_path(output_csv, batch_size):
    csv_path = Path(output_csv)
    return csv_path.with_name(f"{csv_path.stem}_dmon_batch{batch_size}.txt")


# Measure throughput and latency for one batch size
def benchmark_one_batch_size(cfg, model, batch_size, device, dmon_path):
    process = psutil.Process()
    test_loader = get_test_loader(
        mapping_path=cfg.inference.mapping_path,
        test_dir=cfg.inference.data_path,
        transform_test=build_test_transform(),
        test_annotations=cfg.inference.annotations,
        batch_size=batch_size,
        shuffle=False,
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
    processed_batches = 0
    data_loading_time_s = 0.0
    host_to_device_time_s = 0.0
    forward_time_s = 0.0
    peak_memory_rss_mb = process.memory_info().rss / 1024**2
    loader_iterator = iter(test_loader)

    with DmonMonitor(dmon_path) as dmon:
        if dmon.warning:
            print(dmon.warning)

        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

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
                processed_batches += 1
                peak_memory_rss_mb = max(
                    peak_memory_rss_mb,
                    process.memory_info().rss / 1024**2,
                )

        torch.cuda.synchronize()
        cpu_time_s = time.process_time() - start_cpu
        total_time_s = time.perf_counter() - start_wall

    gpu_metrics = parse_dmon_output(dmon.output)

    latency_ms_per_image = (
        (total_time_s / processed_samples) * 1000 if processed_samples else 0.0
    )
    throughput_images_per_s = processed_samples / total_time_s if total_time_s else 0.0
    forward_throughput_images_per_s = (
        processed_samples / forward_time_s if forward_time_s else 0.0
    )
    cpu_utilization_pct = (
        (cpu_time_s / total_time_s) * 100 if total_time_s else 0.0
    )

    return {
        "batch_size": batch_size,
        "total_samples": processed_samples,
        "total_batches": processed_batches,
        "total_time_s": total_time_s,
        "data_loading_time_s": data_loading_time_s,
        "host_to_device_time_s": host_to_device_time_s,
        "forward_time_s": forward_time_s,
        "latency_ms_per_image": latency_ms_per_image,
        "throughput_images_per_s": throughput_images_per_s,
        "forward_only_throughput_images_per_s": forward_throughput_images_per_s,
        "process_cpu_time_s": cpu_time_s,
        "estimated_cpu_utilization_pct": cpu_utilization_pct,
        "peak_memory_rss_mb": peak_memory_rss_mb,
        "gpu_sm_avg_pct": gpu_metrics["gpu_sm_avg_pct"],
        "gpu_sm_max_pct": gpu_metrics["gpu_sm_max_pct"],
        "gpu_mem_controller_avg_pct": gpu_metrics["gpu_mem_controller_avg_pct"],
        "gpu_mem_controller_max_pct": gpu_metrics["gpu_mem_controller_max_pct"],
    }


def add_derived_metrics(rows, saturation_gain_threshold):
    if not rows:
        return rows

    baseline = rows[0]
    baseline_throughput = baseline["throughput_images_per_s"]
    baseline_latency = baseline["latency_ms_per_image"]
    previous_throughput = None

    for row in rows:
        throughput = row["throughput_images_per_s"]
        latency = row["latency_ms_per_image"]
        row["speedup_vs_batch1"] = (
            throughput / baseline_throughput if baseline_throughput else 0.0
        )
        row["latency_change_vs_batch1_pct"] = (
            ((latency - baseline_latency) / baseline_latency) * 100
            if baseline_latency
            else 0.0
        )

        if previous_throughput is None or previous_throughput == 0:
            gain_pct = 0.0
            saturated = False
        else:
            gain_pct = ((throughput - previous_throughput) / previous_throughput) * 100
            saturated = gain_pct < saturation_gain_threshold * 100

        row["throughput_gain_vs_previous_pct"] = gain_pct
        row["saturated_by_5pct_rule"] = saturated
        previous_throughput = throughput

    return rows


# Save benchmark rows for report evidence
def write_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "total_samples",
        "total_batches",
        "total_time_s",
        "data_loading_time_s",
        "host_to_device_time_s",
        "forward_time_s",
        "latency_ms_per_image",
        "throughput_images_per_s",
        "forward_only_throughput_images_per_s",
        "process_cpu_time_s",
        "estimated_cpu_utilization_pct",
        "peak_memory_rss_mb",
        "gpu_sm_avg_pct",
        "gpu_sm_max_pct",
        "gpu_mem_controller_avg_pct",
        "gpu_mem_controller_max_pct",
        "speedup_vs_batch1",
        "throughput_gain_vs_previous_pct",
        "latency_change_vs_batch1_pct",
        "saturated_by_5pct_rule",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Plot throughput and latency across batch sizes
def write_plot(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    batch_sizes = [row["batch_size"] for row in rows]
    throughput = [row["throughput_images_per_s"] for row in rows]
    latency = [row["latency_ms_per_image"] for row in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(batch_sizes, throughput, marker="o", color="tab:blue", label="Throughput")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (images/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, latency, marker="s", color="tab:red", label="Latency")
    ax2.set_ylabel("Latency (ms/image)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    saturated_rows = [row for row in rows if row["saturated_by_5pct_rule"]]
    if saturated_rows:
        ax1.axvline(
            saturated_rows[0]["batch_size"],
            linestyle="--",
            color="tab:gray",
            label="First <5% gain",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Print the best throughput result
def print_summary(rows):
    best = max(rows, key=lambda row: row["throughput_images_per_s"])
    first_saturated = next(
        (row for row in rows if row["saturated_by_5pct_rule"]),
        None,
    )
    print("Batch inference benchmark complete.")
    print(
        f"Best throughput: batch_size={best['batch_size']} "
        f"({best['throughput_images_per_s']:.2f} images/s, "
        f"{best['speedup_vs_batch1']:.2f}x vs batch_size=1)"
    )
    if first_saturated:
        print(
            f"First saturation candidate: batch_size={first_saturated['batch_size']} "
            f"({first_saturated['throughput_gain_vs_previous_pct']:.2f}% gain vs previous)"
        )
    else:
        print("First saturation candidate: none found by the <5% gain rule.")
    print(
        f"GPU utilization at best throughput: avg sm={best['gpu_sm_avg_pct']:.1f}%, "
        f"avg mem={best['gpu_mem_controller_avg_pct']:.1f}%)."
    )


# Batch inference benchmark entrypoint
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    if cfg.device != "cuda":
        raise ValueError("GPU batch inference benchmark requires device=cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

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
    saturation_gain_threshold = float(
        OmegaConf.select(
            cfg,
            "benchmark.saturation_gain_threshold",
            default=0.05,
        )
    )

    device = torch.device("cuda")
    model = initialize_model(
        num_classes=cfg.inference.num_classes,
        weights_path=cfg.inference.weights_path,
        device=device,
    )
    model.eval()

    rows = []
    for batch_size in parse_batch_sizes(cfg):
        print(f"Benchmarking batch_size={batch_size}")
        rows.append(
            benchmark_one_batch_size(
                cfg=cfg,
                model=model,
                batch_size=batch_size,
                device=device,
                dmon_path=dmon_output_path(output_csv, batch_size),
            )
        )

    rows = add_derived_metrics(rows, saturation_gain_threshold)
    write_csv(rows, output_csv)
    write_plot(rows, output_plot)
    print_summary(rows)
    print(f"CSV saved to: {os.path.abspath(output_csv)}")
    print(f"Plot saved to: {os.path.abspath(output_plot)}")


if __name__ == "__main__":
    main()
