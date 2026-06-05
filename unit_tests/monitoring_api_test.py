from fastapi.testclient import TestClient

import monitoring_api


def setup_function():
    monitoring_api.reset_monitoring_state()


def test_build_prometheus_metrics_contains_runtime_and_drift_signals():
    monitoring_api.update_runtime_metrics(
        {
            "image_count": 3,
            "batch_count": 2,
            "average_confidence": 0.7,
            "low_confidence_share": 0.25,
            "retraining_signal": True,
        }
    )
    monitoring_api.update_drift_summary(
        {
            "drift_detected": True,
            "drifted_columns": 4,
            "retraining_signal": True,
        }
    )

    metrics = monitoring_api.build_prometheus_metrics()

    assert "daki_inference_images_total 3" in metrics
    assert "daki_runtime_retraining_signal 1" in metrics
    assert "daki_drift_detected 1" in metrics
    assert "daki_drifted_columns 4" in metrics
    assert "daki_drift_retraining_signal 1" in metrics


def test_monitoring_api_metrics_endpoint_returns_prometheus_text():
    client = TestClient(monitoring_api.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "daki_inference_average_confidence" in response.text


def test_monitoring_api_update_endpoints():
    client = TestClient(monitoring_api.app)

    runtime_response = client.post(
        "/runtime-metrics",
        json={"image_count": 5, "retraining_signal": True},
    )
    drift_response = client.post(
        "/drift-summary",
        json={"drift_detected": True, "drifted_columns": 2},
    )

    assert runtime_response.status_code == 200
    assert runtime_response.json()["image_count"] == 5
    assert runtime_response.json()["retraining_signal"] is True
    assert drift_response.status_code == 200
    assert drift_response.json()["drift_detected"] is True
    assert drift_response.json()["drifted_columns"] == 2
    assert drift_response.json()["retraining_signal"] is True


def test_drift_summary_infers_retraining_signal_when_omitted():
    drift = monitoring_api.update_drift_summary(
        {"drift_detected": True, "drifted_columns": 1}
    )

    assert drift["retraining_signal"] is True


def test_drift_summary_respects_explicit_retraining_signal():
    drift = monitoring_api.update_drift_summary(
        {
            "drift_detected": True,
            "drifted_columns": 1,
            "retraining_signal": False,
        }
    )

    assert drift["drift_detected"] is True
    assert drift["retraining_signal"] is False
