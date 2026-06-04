import mlflow
import os

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
MODEL_NAME = "resnet18-tinyimagenet"


def register_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'")

    run = runs[0]
    run_id = run.info.run_id
    print(f"Registering model from run ID: {run_id}")

    model_uri = f"runs:/{run_id}/final_model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Model registered as '{MODEL_NAME}' version {result.version}")


if __name__ == "__main__":
    register_model()