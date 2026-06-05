import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

import drift_detection as dd


DEFAULT_INPUT_DIR = "data/tiny-imagenet-200/val/images"
DEFAULT_OUTPUT_DIR = "results/simulated_drift"


# Apply artificial image shifts used to validate drift detection
def apply_simulated_shift(image, shift_name, rng):
    shift_name = shift_name.lower()

    if shift_name == "severe":
        shifted = ImageEnhance.Brightness(image).enhance(1.8)
        shifted = ImageEnhance.Contrast(shifted).enhance(2.0)
        values = np.asarray(shifted).astype(np.float32)
        values[:, :, 0] = np.clip(values[:, :, 0] + 25.0, 0, 255)
        values[:, :, 2] = np.clip(values[:, :, 2] - 20.0, 0, 255)
        noise = rng.normal(loc=0.0, scale=35.0, size=values.shape)
        return Image.fromarray(np.clip(values + noise, 0, 255).astype(np.uint8))

    if shift_name == "noise":
        values = np.asarray(image).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=45.0, size=values.shape)
        return Image.fromarray(np.clip(values + noise, 0, 255).astype(np.uint8))

    if shift_name == "dark":
        return ImageEnhance.Brightness(image).enhance(0.35)

    if shift_name == "contrast":
        return ImageEnhance.Contrast(image).enhance(2.4)

    if shift_name == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")

    raise ValueError(
        f"Unsupported shift '{shift_name}'. "
        "Use one of: severe, noise, dark, contrast, grayscale."
    )


# Build clean reference and shifted current dataframes
def build_simulated_drift_frames(
    image_paths,
    shift_name,
    seed,
):
    rng = np.random.default_rng(seed)
    reference_rows = []
    current_rows = []

    for image_path in image_paths:
        with Image.open(image_path) as source:
            clean_image = source.convert("RGB")

        shifted_image = apply_simulated_shift(clean_image, shift_name, rng)
        reference_rows.append(dd.extract_image_features(clean_image, image_path))
        current_rows.append(dd.extract_image_features(shifted_image, image_path))

    return pd.DataFrame(reference_rows), pd.DataFrame(current_rows)


# CLI arguments for simulated drift evidence
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate harsh synthetic image drift and run Evidently."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shift",
        default="severe",
        choices=["severe", "noise", "dark", "contrast", "grayscale"],
    )
    return parser.parse_args()


# Simulated drift evidence entrypoint
def main():
    args = parse_args()

    image_paths = dd.sample_paths(
        dd.list_image_files(args.input_dir),
        sample_limit=args.sample_limit,
        seed=args.seed,
    )
    if not image_paths:
        raise ValueError(f"No images found in: {args.input_dir}")

    reference_data, current_data = build_simulated_drift_frames(
        image_paths=image_paths,
        shift_name=args.shift,
        seed=args.seed,
    )

    summary = dd.run_evidently_report(
        reference_data=reference_data,
        current_data=current_data,
        output_dir=args.output_dir,
    )
    summary.update(
        {
            "sampled_images": len(image_paths),
            "shift": args.shift,
            "input_dir": str(Path(args.input_dir)),
        }
    )
    summary_path = Path(args.output_dir) / "drift_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Simulated Drift Summary")
    print(f"Sampled images: {summary['sampled_images']}")
    print(f"Shift: {summary['shift']}")
    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Drifted columns: {summary['drifted_columns']}")
    print(f"Retraining signal: {summary['retraining_signal']}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
