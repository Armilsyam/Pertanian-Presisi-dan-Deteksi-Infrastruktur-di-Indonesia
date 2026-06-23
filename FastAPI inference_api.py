# inference_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Inference API")

class PredictRequest(BaseModel):
    features: list

MODEL_PATH = "model_store/model.joblib"
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
scaler = model_bundle["scaler"]
features = model_bundle["features"]
task = model_bundle.get("task", "classification")

@app.get("/")
def root():
    return {"status":"ok","task":task,"n_features":len(features)}

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != len(features):
        raise HTTPException(status_code=400, detail=f"Expected {len(features)} features")
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled).tolist()
    result = {"prediction": pred}
    if task == "classification":
        try:
            proba = model.predict_proba(X_scaled).tolist()
            result["probabilities"] = proba
        except Exception:
            pass
    return result
