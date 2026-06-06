import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from data.dataloader import get_test_loader
from experiments.quantization import load_quantized_model


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


# Measure throughput and latency for one batch size
def benchmark_one_batch_size(cfg, model, batch_size):
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
    with torch.no_grad():
        warmup_images, _ = next(iter(test_loader))
        model(warmup_images)

    start = time.perf_counter()
    processed_samples = 0
    with torch.no_grad():
        for images, _ in test_loader:
            model(images)
            processed_samples += images.size(0)
    total_time_s = time.perf_counter() - start

    latency_ms_per_image = (
        (total_time_s / processed_samples) * 1000 if processed_samples else 0.0
    )
    throughput_images_per_s = processed_samples / total_time_s if total_time_s else 0.0

    return {
        "batch_size": batch_size,
        "total_samples": processed_samples,
        "total_time_s": total_time_s,
        "latency_ms_per_image": latency_ms_per_image,
        "throughput_images_per_s": throughput_images_per_s,
    }


# Save benchmark rows for report evidence
def write_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "total_samples",
        "total_time_s",
        "latency_ms_per_image",
        "throughput_images_per_s",
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Print the best throughput result
def print_summary(rows):
    best = max(rows, key=lambda row: row["throughput_images_per_s"])
    print("Batch inference benchmark complete.")
    print(
        f"Best throughput: batch_size={best['batch_size']} "
        f"({best['throughput_images_per_s']:.2f} images/s)"
    )
    print(
        "Use the CSV/plot to identify the saturation point and discuss latency tradeoffs."
    )


# Batch inference benchmark entrypoint
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    if cfg.device == "cuda":
        print("Using CPU for quantized PyTorch batch benchmark.")

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

    model = load_quantized_model(cfg.inference.weights_path)
    model.eval()

    rows = []
    for batch_size in parse_batch_sizes(cfg):
        print(f"Benchmarking batch_size={batch_size}")
        rows.append(benchmark_one_batch_size(cfg, model, batch_size))

    write_csv(rows, output_csv)
    write_plot(rows, output_plot)
    print_summary(rows)
    print(f"CSV saved to: {os.path.abspath(output_csv)}")
    print(f"Plot saved to: {os.path.abspath(output_plot)}")


if __name__ == "__main__":
    main()
