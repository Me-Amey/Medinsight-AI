"""
API Client - MedInsight AI Frontend
Handles all HTTP calls to the FastAPI backend.
Can also run inference directly (no backend needed) for standalone mode.
"""

import os
import sys
import numpy as np
import requests
from typing import Dict, Optional

# Allow direct imports from project root when running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT     = 30  # seconds


# ── HTTP client (backend mode) ────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> Dict:
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": f"Cannot connect to backend at {BACKEND_URL}. "
                                       "Is uvicorn running?"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request timed out."}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"ok": False, "error": detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _get(endpoint: str) -> Dict:
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Standalone inference (no backend required) ────────────────────────────────

def _standalone_predict(disease_key: str, feature_array: np.ndarray) -> Dict:
    """
    Run inference directly using loaded models.
    Used when the FastAPI backend is not running.
    """
    try:
        from backend.model_loader import load_all_models, get_registry
        from backend.explainer import explain_prediction
        from backend.llm_advisor import generate_clinical_report

        registry = get_registry() if _registry_loaded() else load_all_models()
        if disease_key not in registry:
            return {"ok": False, "error": f"Model '{disease_key}' not loaded."}

        dm = registry[disease_key]
        X_scaled = dm.scaler.transform(feature_array.reshape(1, -1))
        prob     = float(dm.model.predict_proba(X_scaled)[0][1])

        xai = explain_prediction(
            model=dm.model, scaler=dm.scaler,
            raw_input=feature_array, feature_names=dm.feature_names,
            model_type=dm.model_type, top_n=5,
        )

        risk_level = "Low" if prob < 0.35 else ("Moderate" if prob < 0.65 else "High")

        report_data = generate_clinical_report(
            disease=dm.name, prob_percent=prob * 100,
            risk_level=risk_level, top_factors=xai["top_factors"],
        )

        return {
            "ok": True,
            "data": {
                "disease":          dm.name,
                "risk_probability": round(prob, 4),
                "risk_percent":     round(prob * 100, 2),
                "risk_level":       risk_level,
                "top_factors":      xai["top_factors"],
                "model_type":       dm.model_type,
                "clinical_report":  report_data["report"],
                "report_available": report_data["available"],
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _registry_loaded() -> bool:
    try:
        from backend.model_loader import get_registry
        get_registry()
        return True
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def check_backend() -> bool:
    result = _get("/health")
    return result["ok"]


def predict_diabetes(inputs: dict, with_report: bool = True) -> Dict:
    endpoint = "/predict/diabetes/report" if with_report else "/predict/diabetes"
    result   = _post(endpoint, inputs)
    if not result["ok"]:
        # Fallback to standalone
        arr = np.array(list(inputs.values()), dtype=float)
        return _standalone_predict("diabetes", arr)
    return result


def predict_heart(inputs: dict, with_report: bool = True) -> Dict:
    endpoint = "/predict/heart/report" if with_report else "/predict/heart"
    result   = _post(endpoint, inputs)
    if not result["ok"]:
        arr = np.array(list(inputs.values()), dtype=float)
        return _standalone_predict("heart_disease", arr)
    return result


def predict_ckd(inputs: dict, with_report: bool = True) -> Dict:
    endpoint = "/predict/ckd/report" if with_report else "/predict/ckd"
    result   = _post(endpoint, inputs)
    if not result["ok"]:
        arr = np.array(list(inputs.values()), dtype=float)
        return _standalone_predict("ckd", arr)
    return result


def get_model_info() -> Dict:
    return _get("/models/info")
