import argparse
import json
from pathlib import Path
from urllib import request


DEFAULT_MONITORING_URL = "http://localhost:8000"


# Read the latest runtime metrics record
def load_latest_jsonl_record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Runtime metrics file not found: {path}")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Runtime metrics file contains no records: {path}")
    return records[-1]


# Read drift summary JSON
def load_json_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Post one metrics payload to the monitoring API
def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=10) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


# Publish runtime and drift artifacts to the monitoring API
def publish_metrics(
    monitoring_url,
    runtime_metrics_path=None,
    drift_summary_path=None,
    post_fn=post_json,
):
    monitoring_url = monitoring_url.rstrip("/")
    published = {}

    if runtime_metrics_path:
        runtime_record = load_latest_jsonl_record(runtime_metrics_path)
        published["runtime"] = post_fn(
            f"{monitoring_url}/runtime-metrics",
            runtime_record,
        )

    if drift_summary_path:
        drift_summary = load_json_file(drift_summary_path)
        published["drift"] = post_fn(
            f"{monitoring_url}/drift-summary",
            drift_summary,
        )

    if not published:
        raise ValueError("Provide at least one metrics source to publish.")

    return published


# CLI arguments for local monitoring evidence
def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish deployment metrics artifacts to the monitoring API."
    )
    parser.add_argument("--monitoring-url", default=DEFAULT_MONITORING_URL)
    parser.add_argument("--runtime-metrics-path", default="")
    parser.add_argument("--drift-summary-path", default="")
    return parser.parse_args()


# Monitoring publisher entrypoint
def main():
    args = parse_args()
    published = publish_metrics(
        monitoring_url=args.monitoring_url,
        runtime_metrics_path=args.runtime_metrics_path or None,
        drift_summary_path=args.drift_summary_path or None,
    )
    print(json.dumps(published, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
