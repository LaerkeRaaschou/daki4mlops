import mlflow
import os
from datetime import datetime

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
GIT_COMMIT = os.environ["GIT_COMMIT"]


def log_deployment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'")

    run_id = runs[0].info.run_id

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("deployment_status", "deployed")
        mlflow.set_tag("deployment_time", datetime.utcnow().isoformat())
        mlflow.set_tag("deployed_image", f"ainger24/daki4mlops:{GIT_COMMIT}")
        mlflow.set_tag("deployed_branch", "main")

    print(f"Deployment logged to MLflow run {run_id}")


if __name__ == "__main__":
    log_deployment()
