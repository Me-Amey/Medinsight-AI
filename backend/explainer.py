"""
XAI Explainer - MedInsight AI Backend
Computes feature contributions for each prediction using feature importances
combined with scaled-value direction. Fast, dependency-free, and clinically
meaningful for a college demo and physician-facing UI.
"""

import numpy as np
from typing import Dict, List, Any


def _get_importances(model: Any, n_features: int) -> np.ndarray:
    """
    Extract feature importances from any sklearn-compatible model.
    Handles VotingClassifier, RandomForest, XGBoost, and SVC gracefully.
    """
    # ── VotingClassifier ──────────────────────────────────────────────────────
    # .estimators_ is a plain list of fitted estimators (newer sklearn)
    # .named_estimators_ is the dict version
    if hasattr(model, "named_estimators_"):
        all_imp = []
        for est in model.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                imp = np.array(est.feature_importances_, dtype=float)
                if imp.shape[0] == n_features:
                    all_imp.append(imp)
        if all_imp:
            return np.mean(all_imp, axis=0)

    # ── Single tree / forest / XGBoost ───────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_, dtype=float)
        if imp.shape[0] == n_features:
            return imp

    # ── SVC or anything else — equal weights ─────────────────────────────────
    return np.ones(n_features, dtype=float) / n_features


def explain_prediction(
    model: Any,
    scaler: Any,
    raw_input: np.ndarray,
    feature_names: List[str],
    model_type: str,
    top_n: int = 5,
) -> Dict:
    """
    Compute feature contributions for a single patient input.

    Method
    ------
    contribution_i = importance_i × scaled_value_i

    - importance_i  : global feature importance (how much this feature
                      matters to the model across all predictions)
    - scaled_value_i: standardised value (positive = above population mean,
                      negative = below population mean)

    This gives both magnitude and direction for each feature contribution,
    which is exactly what clinicians need to understand the prediction.

    Returns
    -------
    dict with:
        shap_values  : list of contribution scores (one per feature)
        top_factors  : list of top-N dicts with feature/value/shap/direction
        mode         : "feature_importance"
    """
    n_features = len(feature_names)
    importances = _get_importances(model, n_features)

    # Scale the input — produces values centred around 0
    # positive = above training mean, negative = below
    X_scaled = scaler.transform(raw_input.reshape(1, -1))[0]

    # Signed contribution = importance × direction
    contributions = importances * X_scaled

    # Rank by absolute contribution magnitude
    abs_c   = np.abs(contributions)
    top_idx = np.argsort(abs_c)[::-1][:top_n]

    top_factors = []
    for i in top_idx:
        top_factors.append({
            "feature":   feature_names[i],
            "value":     float(raw_input[i]),
            "shap":      float(contributions[i]),
            "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
        })

    return {
        "shap_values": contributions.tolist(),
        "top_factors": top_factors,
        "mode":        "feature_importance",
    }


def format_factors_for_llm(top_factors: List[Dict]) -> str:
    """Format top factors into a concise string for the LLM prompt."""
    lines = []
    for f in top_factors:
        lines.append(
            f"{f['feature']} = {f['value']:.2f} "
            f"(contribution: {f['shap']:+.4f}, {f['direction']})"
        )
    return "; ".join(lines)
