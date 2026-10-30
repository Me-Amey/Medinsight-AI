"""
Model Loader - MedInsight AI Backend
Loads serialized models, scalers, and metadata at startup.
All models are cached as module-level singletons (loaded once).
"""

import os
import joblib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class DiseaseModel:
    name: str
    model: Any
    scaler: Any
    feature_names: List[str]
    model_type: str
    metrics: Dict[str, float]
    encoders: Optional[Dict] = field(default=None)


def _load(filename: str) -> Any:
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            f"Run `python scripts/train_all.py` first to generate model files."
        )
    return joblib.load(path)


def load_diabetes_model() -> DiseaseModel:
    meta   = _load("diabetes_meta.pkl")
    model  = _load("diabetes_model.pkl")
    scaler = _load("diabetes_scaler.pkl")
    return DiseaseModel(
        name="Diabetes",
        model=model,
        scaler=scaler,
        feature_names=meta["feature_names"],
        model_type=meta["model_type"],
        metrics=meta["metrics"],
    )


def load_heart_model() -> DiseaseModel:
    meta   = _load("heart_meta.pkl")
    model  = _load("heart_model.pkl")
    scaler = _load("heart_scaler.pkl")
    return DiseaseModel(
        name="Heart Disease",
        model=model,
        scaler=scaler,
        feature_names=meta["feature_names"],
        model_type=meta["model_type"],
        metrics=meta["metrics"],
    )


def load_ckd_model() -> DiseaseModel:
    meta     = _load("ckd_meta.pkl")
    model    = _load("ckd_model.pkl")
    scaler   = _load("ckd_scaler.pkl")
    encoders = _load("ckd_encoders.pkl")
    return DiseaseModel(
        name="Chronic Kidney Disease",
        model=model,
        scaler=scaler,
        feature_names=meta["feature_names"],
        model_type=meta["model_type"],
        metrics=meta["metrics"],
        encoders=encoders,
    )


# ── Module-level singletons ───────────────────────────────────────────────────
_registry: Dict[str, DiseaseModel] = {}


def get_registry() -> Dict[str, DiseaseModel]:
    """Return the loaded model registry. Raises if not initialized."""
    if not _registry:
        raise RuntimeError("Models not loaded. Call load_all_models() at startup.")
    return _registry


def load_all_models() -> Dict[str, DiseaseModel]:
    """Load all models into the registry. Call once at application startup."""
    global _registry
    loaders = {
        "diabetes":       load_diabetes_model,
        "heart_disease":  load_heart_model,
        "ckd":            load_ckd_model,
    }
    for key, loader_fn in loaders.items():
        try:
            _registry[key] = loader_fn()
            print(f"  [OK] {_registry[key].name} model loaded.")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
    return _registry
