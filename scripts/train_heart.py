"""
Model Training Script - Heart Disease (Support Vector Classifier)
Target metric: Recall >= 0.90
Run: python scripts/train_heart.py
"""

import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, classification_report,
    confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_splits() -> dict:
    path = os.path.join(PROC_DIR, "heart_processed.pkl")
    if not os.path.exists(path):
        print("[!] Processed data not found. Running preprocessor first...")
        sys.path.insert(0, os.path.dirname(__file__))
        import preprocess_heart as prep
        df = prep.load_data()
        prep.preprocess(df)
    return joblib.load(path)


def train(splits: dict) -> dict:
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    # ── SVC with probability calibration ─────────────────────────────────────
    # Grid search over key SVM hyperparameters, optimising for Recall
    svc = SVC(
        probability=True,
        class_weight="balanced",  # penalise false negatives more
        random_state=42,
    )

    param_grid = {
        "C":      [0.1, 1, 10, 50],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Running GridSearchCV (optimising Recall)...")
    grid = GridSearchCV(
        svc, param_grid, cv=cv,
        scoring="recall",
        n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV Recall: {grid.best_score_:.4f}")

    # Full CV AUC with best model
    cv_aucs = cross_val_score(best_model, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_aucs.mean():.4f} (+/- {cv_aucs.std():.4f})")

    # ── Test set evaluation ───────────────────────────────────────────────────
    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_pred_prob),
        "accuracy":  accuracy_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }

    print("\n  ── Test Set Metrics ──────────────────────────────")
    for k, v in metrics.items():
        flag = " ✓" if k == "recall" and v >= 0.90 else ""
        print(f"  {k:<12}: {v:.4f}{flag}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["No Heart Disease", "Heart Disease"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Serialize ─────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "heart_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[OK] Model saved -> {model_path}")

    meta = {
        "model_path":    model_path,
        "scaler_path":   os.path.join(MODEL_DIR, "heart_scaler.pkl"),
        "feature_names": splits["feature_names"],
        "metrics":       metrics,
        "best_params":   grid.best_params_,
        "model_type":    "SVC (RBF/Linear, calibrated)",
        "disease":       "Heart Disease",
    }
    meta_path = os.path.join(MODEL_DIR, "heart_meta.pkl")
    joblib.dump(meta, meta_path)
    print(f"[OK] Metadata   -> {meta_path}")

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI - Train Heart Disease Model")
    print("=" * 55)
    splits  = load_splits()
    metrics = train(splits)
    target  = 0.90
    status  = "PASSED" if metrics["recall"] >= target else "BELOW TARGET"
    print(f"\n[{status}] Recall {metrics['recall']:.4f} (target >= {target})")
    print("[DONE]\n")
