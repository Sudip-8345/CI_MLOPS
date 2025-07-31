import mlflow
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import pickle

app = FastAPI(
    title="Water Potability Prediction API",
    description="This API predicts water potability using a machine learning model.",
)

mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/CI_MLOPS.mlflow")

def load_model():
    # client = mlflow.tracking.MlflowClient()
    # version = client.get_latest_versions("Best Model")
    # if not version:
    #     raise ValueError("No model version found in Production stage.")
    # run_id = version[0].run_id
    # model_uri = "models:/Best Model/2"
    # model = mlflow.pyfunc.load_model(model_uri)
    with open("outputs/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

class WaterQualityData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}

@app.post("/predict")
def predict(data: WaterQualityData):
    sample = pd.DataFrame({
        "ph": [data.ph],
        "Hardness": [data.Hardness],
        "Solids": [data.Solids],
        "Chloramines": [data.Chloramines],
        "Sulfate": [data.Sulfate],
        "Conductivity": [data.Conductivity],
        "Organic_carbon": [data.Organic_carbon],
        "Trihalomethanes": [data.Trihalomethanes],
        "Turbidity": [data.Turbidity]
    })

    prediction = model.predict_proba(sample)

    if prediction[0][1] > 0.5:
        return {"prediction": "Potable", "probability": prediction[0][1]}
    else:
        return {"prediction": "Not Potable", "probability": prediction[0][0]}
    

