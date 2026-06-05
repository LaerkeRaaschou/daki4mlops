import json

import runtime_metrics


def test_build_inference_metrics_sets_retraining_signal():
    predictions = [
        {"confidence": 0.2},
        {"confidence": 0.4},
        {"confidence": 0.9},
    ]

    metrics = runtime_metrics.build_inference_metrics(
        predictions=predictions,
        batch_size=2,
        device="cpu",
        quantized=True,
        low_confidence_threshold=0.5,
        signal_threshold=0.5,
    )

    assert metrics["event_type"] == "inference_batch_summary"
    assert metrics["image_count"] == 3
    assert metrics["batch_count"] == 2
    assert metrics["low_confidence_count"] == 2
    assert metrics["retraining_signal"] is True


def test_append_jsonl_creates_parent_directory(tmp_path):
    output_path = tmp_path / "metrics" / "inference.jsonl"

    path = runtime_metrics.append_jsonl({"value": 1}, output_path)

    assert path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"value": 1}
