import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from omegaconf import DictConfig, OmegaConf
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Look for images in the given folder
def list_image_files(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# Sample image paths deterministically for reproducible reports
def sample_paths(paths, sample_limit, seed):
    if sample_limit is None or len(paths) <= sample_limit:
        return list(paths)

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(paths), size=sample_limit, replace=False)
    return [paths[index] for index in sorted(selected.tolist())]


# Convert one image into tabular monitoring features
def extract_image_features(image, image_path):
    rgb = image.convert("RGB")
    values = np.asarray(rgb).astype(np.float32) / 255.0
    channel_mean = values.mean(axis=(0, 1))
    channel_std = values.std(axis=(0, 1))

    return {
        "filename": Path(image_path).name,
        "mean_r": float(channel_mean[0]),
        "mean_g": float(channel_mean[1]),
        "mean_b": float(channel_mean[2]),
        "std_r": float(channel_std[0]),
        "std_g": float(channel_std[1]),
        "std_b": float(channel_std[2]),
        "brightness": float(values.mean()),
        "contrast": float(values.std()),
    }


# Build the tabular dataframe Evidently compares
def build_feature_dataframe(image_paths):
    rows = []

    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        rows.append(extract_image_features(image, image_path))

    return pd.DataFrame(rows)


# Extract simple drift status from Evidently output
def extract_drift_summary(snapshot_dict):
    for metric in snapshot_dict.get("metrics", []):
        if not isinstance(metric, dict):
            continue

        metric_name = str(metric.get("metric_name", ""))
        metric_config = metric.get("config", {})
        metric_value = metric.get("value", {})
        metric_type = str(metric_config.get("type", ""))

        is_drifted_columns_count = (
            metric_name.startswith("DriftedColumnsCount")
            or metric_type.endswith("DriftedColumnsCount")
        )
        if not is_drifted_columns_count:
            continue
        if not isinstance(metric_value, dict):
            raise ValueError("DriftedColumnsCount metric value must be an object.")

        count = metric_value.get("count")
        share = metric_value.get("share")
        if count is None or share is None:
            raise ValueError("DriftedColumnsCount metric must include count and share.")

        return {
            "drift_detected": float(share) >= 0.5,
            "drifted_columns": int(count),
        }

    raise ValueError("Evidently snapshot does not contain DriftedColumnsCount metric.")


# Remove text metadata before passing data to Evidently
def prepare_evidently_dataframe(data):
    object_columns = data.select_dtypes(include=["object"]).columns
    return data.drop(columns=list(object_columns))


# Run Evidently and write report artifacts
def run_evidently_report(reference_data, current_data, output_dir, report_factory=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "drift_summary.json"

    if report_factory is None:
        report_factory = lambda: Report(
            [DataDriftPreset(method="ks", drift_share=0.5, threshold=0.05)],
            include_tests=True,
        )

    report = report_factory()
    snapshot = report.run(
        current_data=prepare_evidently_dataframe(current_data),
        reference_data=prepare_evidently_dataframe(reference_data),
    )

    snapshot_json = snapshot.json()
    if isinstance(snapshot_json, str):
        snapshot_dict = json.loads(snapshot_json)
    else:
        snapshot_dict = snapshot_json

    summary = extract_drift_summary(snapshot_dict)
    summary.update({"retraining_signal": bool(summary["drift_detected"])})
    summary.update({"summary_report": str(summary_path)})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# Drift detection pipeline
def run_drift_detection(cfg):
    if not cfg.drift.enabled:
        print("Drift detection disabled by configuration.")
        return {"drift_detected": False, "drifted_columns": 0}

    reference_paths = sample_paths(
        list_image_files(cfg.drift.reference_path),
        int(cfg.drift.sample_limit),
        int(cfg.drift.seed),
    )
    current_paths = sample_paths(
        list_image_files(cfg.drift.current_path),
        int(cfg.drift.sample_limit),
        int(cfg.drift.seed),
    )

    if not reference_paths:
        raise ValueError(f"No reference images found in: {cfg.drift.reference_path}")
    if not current_paths:
        raise ValueError(f"No current images found in: {cfg.drift.current_path}")

    reference_data = build_feature_dataframe(reference_paths)
    current_data = build_feature_dataframe(current_paths)

    summary = run_evidently_report(
        reference_data=reference_data,
        current_data=current_data,
        output_dir=cfg.drift.output_dir,
    )
    summary.update(
        {
            "reference_samples": int(len(reference_data)),
            "current_samples": int(len(current_data)),
        }
    )

    summary_path = Path(cfg.drift.output_dir) / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Drift Detection Summary")
    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Drifted columns: {summary['drifted_columns']}")
    print(f"Retraining signal: {summary['retraining_signal']}")
    print(f"Summary report: {summary_path}")

    if cfg.drift.fail_on_drift and summary["drift_detected"]:
        raise SystemExit(1)

    return summary


# Hydra entrypoint
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    run_drift_detection(cfg)


if __name__ == "__main__":
    main()
