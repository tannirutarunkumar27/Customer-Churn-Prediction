"""
api/main.py
-----------
FastAPI REST API to serve the customer churn prediction model.

Run with: uvicorn api.main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn using a trained ML model.",
    version="1.0.0",
)

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL_PATH = "models"
model = None

@app.on_event("startup")
def load_model():
    global model
    model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".joblib")]
    if not model_files:
        print("[WARNING] No model found in models/ directory.")
        return
    model_file = sorted(model_files)[-1]   # Load most recently saved
    model = joblib.load(os.path.join(MODEL_PATH, model_file))
    print(f"[✓] Loaded model: {model_file}")


# ── Schemas ───────────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    features: List[float] = Field(
        ...,
        example=[1, 12, 0, 1, 45.5, 546.0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        description="Numerical feature vector for a single customer"
    )


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    label: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Customer Churn Prediction API is running 🚀"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        label = "Churn" if prediction == 1 else "No Churn"
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            label=label,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(data: List[CustomerFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for item in data:
        features = np.array(item.features).reshape(1, -1)
        pred  = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0][1])
        results.append({
            "churn_prediction": pred,
            "churn_probability": round(proba, 4),
            "label": "Churn" if pred == 1 else "No Churn",
        })
    return {"predictions": results, "count": len(results)}
