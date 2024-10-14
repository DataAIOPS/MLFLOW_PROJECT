import mlflow
import os

# Set the environment variables if not already set in your OS
os.environ['MLFLOW_TRACKING_URI'] = 'https://github.com/DataAIOPS/MLFLOW_PROJECT.git'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'DataAIOPS'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '9775fb6f18d752814f34587ba0e1e3d988a189e4'

# Start a test run
with mlflow.start_run():
    # Log a simple metric (e.g., accuracy)
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.89)
    
    # Create and log a simple artifact (e.g., a text file)
    with open("artifact.txt", "w") as f:
        f.write("Hello, this is a test artifact for DagsHub MLflow connection.")
    mlflow.log_artifact("artifact.txt")

print("Test run completed. Check your DagsHub repository to see if the experiment was logged.")
