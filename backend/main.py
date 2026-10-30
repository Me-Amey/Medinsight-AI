"""
MedInsight AI - FastAPI Backend
Serves ML inference + SHAP explanations for Diabetes, Heart Disease, and CKD.
Run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.model_loader import load_all_models, get_registry, DiseaseModel
from backend.explainer import explain_prediction, format_factors_for_llm
from backend.llm_advisor import generate_clinical_report
from backend.schemas import (
    DiabetesInput, HeartDiseaseInput, CKDInput,
    PredictionResponse, BatchPredictionResponse, HealthResponse
)

VERSION = "1.0.0"


# ── Lifespan: load models once at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[MedInsight AI] Loading models...")
    load_all_models()
    print("[MedInsight AI] Ready.\n")
    yield
    print("[MedInsight AI] Shutting down.")


app = FastAPI(
    title="MedInsight AI",
    description=(
        "Multi-disease predictive analytics API. "
        "Predicts risk for Diabetes, Heart Disease, and Chronic Kidney Disease "
        "with Explainable AI (SHAP) feature contributions."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    return "High"


def _run_inference(
    disease_key: str,
    raw_input: np.ndarray,
) -> PredictionResponse:
    registry = get_registry()
    if disease_key not in registry:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{disease_key}' is not loaded. Run train_all.py first."
        )
    dm: DiseaseModel = registry[disease_key]

    # Scale input
    X_scaled = dm.scaler.transform(raw_input.reshape(1, -1))

    # Predict
    prob = float(dm.model.predict_proba(X_scaled)[0][1])

    # SHAP explanation
    xai = explain_prediction(
        model=dm.model,
        scaler=dm.scaler,
        raw_input=raw_input,
        feature_names=dm.feature_names,
        model_type=dm.model_type,
        top_n=5,
    )

    return PredictionResponse(
        disease=dm.name,
        risk_probability=round(prob, 4),
        risk_percent=round(prob * 100, 2),
        risk_level=_risk_level(prob),
        top_factors=xai["top_factors"],
        model_type=dm.model_type,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API health status and list of loaded models."""
    registry = get_registry()
    return HealthResponse(
        status="ok",
        models_loaded=list(registry.keys()),
        version=VERSION,
    )


@app.post("/predict/diabetes", response_model=PredictionResponse, tags=["Prediction"])
def predict_diabetes(data: DiabetesInput):
    """
    Predict Diabetes risk from Pima Indians feature set.
    Returns probability, risk level, and top SHAP contributing features.
    """
    return _run_inference("diabetes", data.to_array())


@app.post("/predict/heart", response_model=PredictionResponse, tags=["Prediction"])
def predict_heart(data: HeartDiseaseInput):
    """
    Predict Heart Disease risk from UCI Cleveland feature set.
    Returns probability, risk level, and top SHAP contributing features.
    """
    return _run_inference("heart_disease", data.to_array())


@app.post("/predict/ckd", response_model=PredictionResponse, tags=["Prediction"])
def predict_ckd(data: CKDInput):
    """
    Predict Chronic Kidney Disease risk from UCI CKD feature set.
    Returns probability, risk level, and top SHAP contributing features.
    """
    return _run_inference("ckd", data.to_array())


@app.post("/predict/all", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_all(
    diabetes_data:      DiabetesInput,
    heart_data:         HeartDiseaseInput,
    ckd_data:           CKDInput,
):
    """
    Run all three disease predictions in a single request.
    Any individual model failure is captured in the `errors` field
    without blocking the other results.
    """
    results = BatchPredictionResponse()
    errors: Dict[str, str] = {}

    try:
        results.diabetes = _run_inference("diabetes", diabetes_data.to_array())
    except Exception as e:
        errors["diabetes"] = str(e)

    try:
        results.heart_disease = _run_inference("heart_disease", heart_data.to_array())
    except Exception as e:
        errors["heart_disease"] = str(e)

    try:
        results.ckd = _run_inference("ckd", ckd_data.to_array())
    except Exception as e:
        errors["ckd"] = str(e)

    results.errors = errors
    return results


@app.post("/predict/diabetes/report", response_model=PredictionResponse, tags=["Advisory Report"])
def predict_diabetes_with_report(data: DiabetesInput):
    """Predict Diabetes risk AND generate a full LLM clinical advisory report."""
    result = _run_inference("diabetes", data.to_array())
    report_data = generate_clinical_report(
        disease=result.disease,
        prob_percent=result.risk_percent,
        risk_level=result.risk_level,
        top_factors=[f.model_dump() for f in result.top_factors],
    )
    result.clinical_report = report_data["report"]
    result.report_available = report_data["available"]
    return result


@app.post("/predict/heart/report", response_model=PredictionResponse, tags=["Advisory Report"])
def predict_heart_with_report(data: HeartDiseaseInput):
    """Predict Heart Disease risk AND generate a full LLM clinical advisory report."""
    result = _run_inference("heart_disease", data.to_array())
    report_data = generate_clinical_report(
        disease=result.disease,
        prob_percent=result.risk_percent,
        risk_level=result.risk_level,
        top_factors=[f.model_dump() for f in result.top_factors],
    )
    result.clinical_report = report_data["report"]
    result.report_available = report_data["available"]
    return result


@app.post("/predict/ckd/report", response_model=PredictionResponse, tags=["Advisory Report"])
def predict_ckd_with_report(data: CKDInput):
    """Predict CKD risk AND generate a full LLM clinical advisory report."""
    result = _run_inference("ckd", data.to_array())
    report_data = generate_clinical_report(
        disease=result.disease,
        prob_percent=result.risk_percent,
        risk_level=result.risk_level,
        top_factors=[f.model_dump() for f in result.top_factors],
    )
    result.clinical_report = report_data["report"]
    result.report_available = report_data["available"]
    return result


@app.get("/models/info", tags=["System"])
def model_info():
    """Returns metadata for all loaded models (type, features, validation metrics)."""
    registry = get_registry()
    info = {}
    for key, dm in registry.items():
        info[key] = {
            "disease":       dm.name,
            "model_type":    dm.model_type,
            "feature_names": dm.feature_names,
            "metrics":       dm.metrics,
        }
    return info


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
