import mlflow
import json
import dagshub
from mlflow.tracking import MlflowClient

# Initialize DagsHub
# dagshub.init(repo_owner='Sudip-8345', repo_name='CI_MLOPS', mlflow=True)
# mlflow.set_tracking_uri('https://dagshub.com/Sudip-8345/CI_MLOPS.mlflow')

import os
dagshub_token = os.getenv('DAGSHUB_TOKEN')
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable not set. Please set it to your DagsHub token.")
os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/Sudip-8345/CI_MLOPS.mlflow')
# Set the experiment name in MLflow
mlflow.set_experiment("Final_Model")

def main():
    try:
        # Load run info
        with open("reports/run_info.json", "r") as f:
            run_info = json.load(f)

        run_id = run_info["run_id"]
        model_name = run_info["model_name"]
        model_uri = f"runs:/{run_id}/model.pkl"

        client = MlflowClient()
        
        # Check if model exists
        try:
            client.get_registered_model(model_name)
            print(f"Model '{model_name}' already exists. Creating new version...")
        except:
            print(f"Creating new model '{model_name}'...")
            client.create_registered_model(model_name)

        # Create new version
        result = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        
        print(f"\n✅ Successfully created version {result.version} of model '{model_name}'")
        print(f"Model URI: {model_uri}")

    except Exception as e:
        print(f"\n❌ Registration failed: {e}")
        raise

if __name__ == "__main__":
    main()