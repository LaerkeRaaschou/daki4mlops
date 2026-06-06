from copy import deepcopy

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Gauge, generate_latest


app = FastAPI(title="daki4mlops monitoring")

_DEFAULT_STATE = {
    "runtime": {
        "image_count": 0,
        "batch_count": 0,
        "average_confidence": 0.0,
        "low_confidence_share": 0.0,
        "retraining_signal": False,
    },
    "drift": {
        "drift_detected": False,
        "drifted_columns": 0,
        "retraining_signal": False,
    },
}
_STATE = deepcopy(_DEFAULT_STATE)


# Convert bool values into Prometheus gauge values
def _bool_metric(value):
    return 1 if bool(value) else 0


# Register one gauge in the request-local Prometheus registry
def _set_gauge(registry, name, description, value):
    gauge = Gauge(name, description, registry=registry)
    gauge.set(value)


# Reset in-memory monitoring state for tests
def reset_monitoring_state():
    _STATE.clear()
    _STATE.update(deepcopy(_DEFAULT_STATE))


# Update latest runtime inference metrics
def update_runtime_metrics(record):
    runtime = _STATE["runtime"]
    runtime["image_count"] = int(record.get("image_count", runtime["image_count"]))
    runtime["batch_count"] = int(record.get("batch_count", runtime["batch_count"]))
    runtime["average_confidence"] = float(
        record.get("average_confidence", runtime["average_confidence"])
    )
    runtime["low_confidence_share"] = float(
        record.get("low_confidence_share", runtime["low_confidence_share"])
    )
    runtime["retraining_signal"] = bool(
        record.get("retraining_signal", runtime["retraining_signal"])
    )
    return deepcopy(runtime)


# Update latest drift summary and infer retraining signal when needed
def update_drift_summary(summary):
    drift = _STATE["drift"]
    drift["drift_detected"] = bool(
        summary.get("drift_detected", drift["drift_detected"])
    )
    drift["drifted_columns"] = int(
        summary.get("drifted_columns", drift["drifted_columns"])
    )
    if "retraining_signal" in summary:
        drift["retraining_signal"] = bool(summary["retraining_signal"])
    elif "drift_detected" in summary:
        drift["retraining_signal"] = drift["drift_detected"]
    return deepcopy(drift)


# Build Prometheus text format
def build_prometheus_metrics():
    runtime = _STATE["runtime"]
    drift = _STATE["drift"]
    registry = CollectorRegistry()

    _set_gauge(
        registry,
        "daki_inference_images_total",
        "Images processed in the latest inference batch.",
        runtime["image_count"],
    )
    _set_gauge(
        registry,
        "daki_inference_batches_total",
        "Batches processed in the latest inference run.",
        runtime["batch_count"],
    )
    _set_gauge(
        registry,
        "daki_inference_average_confidence",
        "Average confidence from the latest inference run.",
        runtime["average_confidence"],
    )
    _set_gauge(
        registry,
        "daki_inference_low_confidence_share",
        "Share of low-confidence predictions.",
        runtime["low_confidence_share"],
    )
    _set_gauge(
        registry,
        "daki_runtime_retraining_signal",
        "Runtime confidence-based retraining signal.",
        _bool_metric(runtime["retraining_signal"]),
    )
    _set_gauge(
        registry,
        "daki_drift_detected",
        "Whether drift was detected in the latest drift report.",
        _bool_metric(drift["drift_detected"]),
    )
    _set_gauge(
        registry,
        "daki_drifted_columns",
        "Number of drifted columns in the latest drift report.",
        drift["drifted_columns"],
    )
    _set_gauge(
        registry,
        "daki_drift_retraining_signal",
        "Drift-based retraining signal.",
        _bool_metric(drift["retraining_signal"]),
    )

    return generate_latest(registry).decode("utf-8")


# Monitoring API endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runtime-metrics")
def post_runtime_metrics(record: dict):
    return update_runtime_metrics(record)


@app.post("/drift-summary")
def post_drift_summary(summary: dict):
    return update_drift_summary(summary)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        build_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
