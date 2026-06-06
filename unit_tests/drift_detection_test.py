import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import drift_detection


def _write_image(path, color):
    image = Image.new("RGB", (4, 4), color=color)
    image.save(path)
    return path


def test_extract_image_features_for_rgb_image(tmp_path):
    image_path = _write_image(tmp_path / "sample.png", (255, 128, 0))

    with Image.open(image_path) as image:
        features = drift_detection.extract_image_features(image, image_path)

    assert features["filename"] == "sample.png"
    assert "width" not in features
    assert "height" not in features
    assert features["mean_r"] == 1.0
    assert np.isclose(features["mean_g"], 128 / 255)
    assert features["mean_b"] == 0.0
    assert features["brightness"] > 0.0
    assert features["contrast"] > 0.0


def test_build_feature_dataframe_is_deterministic_with_sample_limit(tmp_path):
    paths = [
        _write_image(tmp_path / f"{index}.png", (index * 20, 20, 20))
        for index in range(5)
    ]

    sampled_a = drift_detection.sample_paths(paths, sample_limit=3, seed=123)
    sampled_b = drift_detection.sample_paths(paths, sample_limit=3, seed=123)
    frame_a = drift_detection.build_feature_dataframe(sampled_a)
    frame_b = drift_detection.build_feature_dataframe(sampled_b)

    assert [path.name for path in sampled_a] == [path.name for path in sampled_b]
    assert frame_a["filename"].tolist() == frame_b["filename"].tolist()
    assert len(frame_a) == 3


def test_prepare_evidently_dataframe_drops_text_metadata(tmp_path):
    image_path = _write_image(tmp_path / "sample.png", (100, 100, 100))
    frame = drift_detection.build_feature_dataframe([image_path])

    prepared = drift_detection.prepare_evidently_dataframe(frame)

    assert "filename" not in prepared.columns
    assert "brightness" in prepared.columns


def test_extract_drift_summary_reads_evidently_snapshot_shape():
    snapshot = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5,method=ks)",
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": 6.0, "share": 0.75},
            }
        ],
    }

    summary = drift_detection.extract_drift_summary(snapshot)

    assert summary["drift_detected"] is True
    assert summary["drifted_columns"] == 6


def test_extract_drift_summary_uses_aggregate_share_not_column_failures():
    snapshot = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5,method=ks)",
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": 3.0, "share": 0.375},
            },
            {
                "metric_name": "ValueDrift(column=mean_r,method=ks,threshold=0.05)",
                "value": 0.001,
            },
        ],
        "tests": [{"status": "FAIL"}],
    }

    summary = drift_detection.extract_drift_summary(snapshot)

    assert summary["drift_detected"] is False
    assert summary["drifted_columns"] == 3


def test_extract_drift_summary_rejects_missing_aggregate_metric():
    snapshot = {
        "metrics": [
            {
                "metric_name": "ValueDrift(column=mean_r,method=ks,threshold=0.05)",
                "value": 0.001,
            }
        ],
        "tests": [{"status": "FAIL"}],
    }

    with pytest.raises(ValueError, match="DriftedColumnsCount"):
        drift_detection.extract_drift_summary(snapshot)


class _FakeSnapshot:
    def save_html(self, path):
        Path(path).write_text("<html>drift</html>", encoding="utf-8")

    def json(self):
        return json.dumps(
            {
                "metrics": [
                    {
                        "metric_name": "DriftedColumnsCount(drift_share=0.5,method=ks)",
                        "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                        "value": {"count": 6.0, "share": 0.75},
                    }
                ]
            }
        )


class _FakeReport:
    def run(self, current_data, reference_data):
        assert len(current_data) == len(reference_data)
        return _FakeSnapshot()


def test_run_evidently_report_writes_outputs(tmp_path):
    image_path = _write_image(tmp_path / "sample.png", (100, 100, 100))
    frame = drift_detection.build_feature_dataframe([image_path])

    summary = drift_detection.run_evidently_report(
        reference_data=frame,
        current_data=frame,
        output_dir=tmp_path / "drift",
        report_factory=lambda: _FakeReport(),
    )

    assert summary["drift_detected"] is True
    assert summary["drifted_columns"] == 6
    assert summary["retraining_signal"] is True
    assert Path(summary["summary_report"]).is_file()
