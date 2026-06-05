import json

import pytest

import publish_monitoring_metrics


def test_load_latest_jsonl_record_returns_last_record(tmp_path):
    metrics_path = tmp_path / "runtime.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"image_count": 1}),
                json.dumps({"image_count": 2, "retraining_signal": True}),
            ]
        ),
        encoding="utf-8",
    )

    record = publish_monitoring_metrics.load_latest_jsonl_record(metrics_path)

    assert record == {"image_count": 2, "retraining_signal": True}


def test_publish_metrics_posts_runtime_and_drift_payloads(tmp_path):
    runtime_path = tmp_path / "runtime.jsonl"
    drift_path = tmp_path / "drift_summary.json"
    runtime_path.write_text(
        json.dumps({"image_count": 3, "retraining_signal": True}),
        encoding="utf-8",
    )
    drift_path.write_text(
        json.dumps({"drift_detected": True, "drifted_columns": 2}),
        encoding="utf-8",
    )
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return payload

    published = publish_monitoring_metrics.publish_metrics(
        monitoring_url="http://monitoring:8000/",
        runtime_metrics_path=runtime_path,
        drift_summary_path=drift_path,
        post_fn=fake_post,
    )

    assert calls == [
        (
            "http://monitoring:8000/runtime-metrics",
            {"image_count": 3, "retraining_signal": True},
        ),
        (
            "http://monitoring:8000/drift-summary",
            {"drift_detected": True, "drifted_columns": 2},
        ),
    ]
    assert published["runtime"]["image_count"] == 3
    assert published["drift"]["drift_detected"] is True


def test_publish_metrics_requires_source():
    with pytest.raises(ValueError):
        publish_monitoring_metrics.publish_metrics("http://monitoring:8000")
