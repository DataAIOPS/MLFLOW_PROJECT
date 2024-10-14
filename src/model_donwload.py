import mlflow
from mlflow.tracking import MlflowClient
import os
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/DataAIOPS/MLFLOW_PROJECT.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'DataAIOPS'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '9775fb6f18d752814f34587ba0e1e3d988a189e4'

def download_best_model(experiment_name="Default", download_path=r"..\artifacts\model_eval\\"):
    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Initialize MLflow client
    client = MlflowClient()

    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    experiment_id = experiment.experiment_id

    # Search for the runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string="tags.mlflow.runName = 'Model Evaluation'")
    
    best_run = None
    best_mse = float("inf")  # Initialize to infinity to find the lowest MSE
    best_r_square = float("-inf")  # Initialize to negative infinity to find the highest R²

    for run in runs:
        metrics = run.data.metrics
        mse = metrics.get("MSE", float("inf"))
        r_square = metrics.get("R_square", float("-inf"))

        # Select run with lowest MSE and highest R²
        if mse < best_mse and r_square > best_r_square:
            best_mse = mse
            best_r_square = r_square
            best_run = run

    if best_run:
        # Get the best model's artifact URI and download the model
        best_run_id = best_run.info.run_id
        print(f"Best run ID: {best_run_id} with MSE: {best_mse} and R²: {best_r_square}")

        # Download the model artifact (assuming it's named 'best_model')
        model_uri = f"runs:/{best_run_id}/best_model"
        local_model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=download_path)
        
        print(f"Model downloaded to: {local_model_path}")
    else:
        print("No suitable run found.")

# Example usage:
download_best_model(experiment_name="Default", download_path=r"..\artifacts\best_model\\")
