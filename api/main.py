"""
api/main.py
-----------
Production-grade FastAPI for Customer Churn Prediction.

Endpoints:
  GET  /health            - health check
  POST /predict           - single customer prediction with risk score
  POST /batch_predict     - bulk CSV-style prediction
  GET  /model/info        - model metadata
  GET  /recommendations   - business recommendations table

Run: uvicorn api.main:app --reload --port 8000
"""

import os, time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production ML API for predicting customer churn probability, "
        "risk tier, financial impact, and recommended interventions."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "final_HistGB.joblib")
FEAT_PATH   = os.path.join("reports", "shap_feature_importance.csv")

RISK_BANDS = {
    "Critical": (0.80, 1.01),
    "High":     (0.60, 0.80),
    "Medium":   (0.40, 0.60),
    "Low":      (0.00, 0.40),
}

BUSINESS_RECOMMENDATIONS = [
    {
        "finding":        "Month-to-month contract",
        "risk_signal":    "contract == 0",
        "action":         "Offer 20% discount to switch to annual contract",
        "priority":       "Critical",
    },
    {
        "finding":        "High risk score (complaints + late payments)",
        "risk_signal":    "risk_score > p75",
        "action":         "Assign dedicated account manager; escalate to retention team",
        "priority":       "Critical",
    },
    {
        "finding":        "Low satisfaction / high complaint ratio",
        "risk_signal":    "satisfaction_per_complaint < median",
        "action":         "Trigger customer success outreach within 48 hours",
        "priority":       "High",
    },
    {
        "finding":        "No tech support subscription",
        "risk_signal":    "has_tech_support == 0",
        "action":         "Offer 3-month free tech support trial",
        "priority":       "High",
    },
    {
        "finding":        "High monthly charges relative to income",
        "risk_signal":    "charge_to_income_pct > p75",
        "action":         "Offer a downgraded plan or loyalty discount",
        "priority":       "Medium",
    },
    {
        "finding":        "Low customer satisfaction score",
        "risk_signal":    "customer_satisfaction < 4",
        "action":         "Send personalised satisfaction survey + offer resolution incentive",
        "priority":       "High",
    },
    {
        "finding":        "Short tenure (< 6 months)",
        "risk_signal":    "tenure < 6",
        "action":         "Onboarding follow-up call; assign customer success buddy",
        "priority":       "Medium",
    },
    {
        "finding":        "Multiple late payments",
        "risk_signal":    "late_payments >= 2",
        "action":         "Offer flexible payment plan or auto-pay incentive",
        "priority":       "Medium",
    },
]

# ── Startup: load model & feature list ───────────────────────────────────────
model         = None
feature_names = None
model_meta    = {}

@app.on_event("startup")
def load_model():
    global model, feature_names, model_meta
    if not os.path.exists(MODEL_PATH):
        zip_path = os.path.join("models", "models.zip")
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall("models")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_meta = {
            "model_type":   type(model).__name__,
            "model_file":   MODEL_PATH,
            "loaded_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"[OK] Model loaded: {MODEL_PATH}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")

    if os.path.exists(FEAT_PATH):
        feature_names = pd.read_csv(FEAT_PATH)["feature"].tolist()
        print(f"[OK] Feature list loaded: {len(feature_names)} features")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_BANDS.items():
        if lo <= prob < hi:
            return tier
    return "Low"

def financial_impact(prob: float, monthly_revenue: float = 85.0) -> Dict[str, Any]:
    """Estimate financial impact for a single customer."""
    retention_rate      = 1 - prob
    revenue_at_risk     = round(monthly_revenue * 12 * prob, 2)
    expected_retained   = round(monthly_revenue * 12 * retention_rate, 2)
    campaign_cost       = 50.0    # average cost per retention campaign touchpoint
    net_benefit         = round(expected_retained - campaign_cost, 2)
    roi_pct             = round((net_benefit / campaign_cost) * 100, 1) if campaign_cost > 0 else 0
    return {
        "monthly_revenue":      monthly_revenue,
        "annual_revenue_at_risk": revenue_at_risk,
        "expected_annual_retained": expected_retained,
        "campaign_cost":        campaign_cost,
        "net_benefit":          net_benefit,
        "roi_pct":              roi_pct,
    }

# ── Schemas ───────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="Ordered feature vector matching the model's training features.",
        example=[0.0] * 39,
    )
    monthly_revenue: Optional[float] = Field(
        85.0, description="Customer's average monthly revenue (for ROI calc)."
    )

class PredictionResponse(BaseModel):
    churn_probability:  float
    risk_tier:          str
    churn_prediction:   int
    label:              str
    financial_impact:   Dict[str, Any]
    recommendations:    List[Dict[str, str]]

class BatchCustomer(BaseModel):
    customer_id:    Optional[str] = None
    features:       List[float]
    monthly_revenue: Optional[float] = 85.0

class BatchResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    summary:     Dict[str, Any]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

@app.get("/", tags=["Health"])
def root():
    return {
        "api":          "Customer Churn Prediction API",
        "version":      "2.0.0",
        "docs":         "/docs",
        "health":       "/health",
    }

@app.get("/model/info", tags=["Model"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        **model_meta,
        "feature_count": len(feature_names) if feature_names else "unknown",
        "risk_bands":    RISK_BANDS,
    }

@app.get("/recommendations", tags=["Business"])
def get_recommendations():
    return {
        "count":           len(BUSINESS_RECOMMENDATIONS),
        "recommendations": BUSINESS_RECOMMENDATIONS,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        arr   = np.array(data.features).reshape(1, -1)
        prob  = float(model.predict_proba(arr)[0][1])
        pred  = int(prob >= 0.50)
        tier  = get_risk_tier(prob)
        label = "Churn" if pred == 1 else "No Churn"

        # Filter recommendations to high-priority ones for at-risk customers
        recs = [r for r in BUSINESS_RECOMMENDATIONS if r["priority"] in ("Critical", "High")] \
               if prob >= 0.40 else []

        return PredictionResponse(
            churn_probability = round(prob, 4),
            risk_tier         = tier,
            churn_prediction  = pred,
            label             = label,
            financial_impact  = financial_impact(prob, data.monthly_revenue),
            recommendations   = recs,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=BatchResponse, tags=["Prediction"])
def batch_predict(customers: List[BatchCustomer]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(customers) > 10_000:
        raise HTTPException(status_code=400, detail="Max batch size is 10,000.")
    try:
        predictions = []
        tier_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        total_revenue_at_risk = 0.0

        for i, cust in enumerate(customers):
            arr  = np.array(cust.features).reshape(1, -1)
            prob = float(model.predict_proba(arr)[0][1])
            tier = get_risk_tier(prob)
            tier_counts[tier] += 1
            fi = financial_impact(prob, cust.monthly_revenue)
            total_revenue_at_risk += fi["annual_revenue_at_risk"]

            predictions.append({
                "customer_id":        cust.customer_id or f"customer_{i+1}",
                "churn_probability":  round(prob, 4),
                "risk_tier":          tier,
                "churn_prediction":   int(prob >= 0.50),
                "label":              "Churn" if prob >= 0.50 else "No Churn",
                "annual_revenue_at_risk": fi["annual_revenue_at_risk"],
            })

        return BatchResponse(
            predictions = predictions,
            summary = {
                "total_customers":      len(customers),
                "risk_breakdown":       tier_counts,
                "predicted_churners":   sum(1 for p in predictions if p["churn_prediction"] == 1),
                "total_revenue_at_risk": round(total_revenue_at_risk, 2),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
