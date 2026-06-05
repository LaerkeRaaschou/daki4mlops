import json
import math
from datetime import datetime, timezone
from pathlib import Path


# Build one runtime summary record for a completed inference run
def build_inference_metrics(
    predictions,
    batch_size,
    device,
    quantized,
    low_confidence_threshold=0.5,
    signal_threshold=0.5,
):
    image_count = len(predictions)
    confidences = [float(prediction["confidence"]) for prediction in predictions]
    low_confidence_count = sum(
        confidence < low_confidence_threshold for confidence in confidences
    )
    low_confidence_share = low_confidence_count / image_count if image_count else 0.0

    return {
        "event_type": "inference_batch_summary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": image_count,
        "batch_size": int(batch_size),
        "batch_count": math.ceil(image_count / batch_size) if image_count else 0,
        "device": device,
        "quantized": bool(quantized),
        "average_confidence": sum(confidences) / image_count if image_count else 0.0,
        "min_confidence": min(confidences) if image_count else 0.0,
        "max_confidence": max(confidences) if image_count else 0.0,
        "low_confidence_threshold": float(low_confidence_threshold),
        "low_confidence_count": int(low_confidence_count),
        "low_confidence_share": float(low_confidence_share),
        "retraining_signal": low_confidence_share >= signal_threshold,
        "signal_threshold": float(signal_threshold),
    }


# Append runtime monitoring records for later publishing
def append_jsonl(record, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")

    return path
