"""
Model Training Script - Diabetes (XGBoost + Random Forest ensemble)
Target metric: ROC-AUC >= 0.88
Run: python scripts/train_diabetes.py
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_splits() -> dict:
    path = os.path.join(PROC_DIR, "diabetes_processed.pkl")
    if not os.path.exists(path):
        print("[!] Processed data not found. Running preprocessor first...")
        sys.path.insert(0, os.path.dirname(__file__))
        import preprocess_diabetes as prep
        df = prep.load_data()
        prep.preprocess(df)
    return joblib.load(path)


def train(splits: dict) -> dict:
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # ── Soft-voting ensemble ──────────────────────────────────────────────────
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_model), ("rf", rf_model)],
        voting="soft",
        n_jobs=-1,
    )

    print("  Training XGBoost + Random Forest ensemble...")
    ensemble.fit(X_train, y_train)

    # ── 5-fold cross-validation on training set ───────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(ensemble, X_train, y_train, cv=cv,
                              scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_aucs.mean():.4f} (+/- {cv_aucs.std():.4f})")

    # ── Test set evaluation ───────────────────────────────────────────────────
    y_pred      = ensemble.predict(X_test)
    y_pred_prob = ensemble.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_pred_prob),
        "accuracy":  accuracy_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }

    print("\n  ── Test Set Metrics ──────────────────────────────")
    for k, v in metrics.items():
        flag = " ✓" if k == "roc_auc" and v >= 0.88 else ""
        print(f"  {k:<12}: {v:.4f}{flag}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["No Diabetes", "Diabetes"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Serialize ─────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "diabetes_model.pkl")
    joblib.dump(ensemble, model_path)
    print(f"\n[OK] Model saved -> {model_path}")

    meta = {
        "model_path":    model_path,
        "scaler_path":   os.path.join(MODEL_DIR, "diabetes_scaler.pkl"),
        "feature_names": splits["feature_names"],
        "metrics":       metrics,
        "model_type":    "VotingClassifier(XGBoost + RandomForest)",
        "disease":       "Diabetes",
    }
    meta_path = os.path.join(MODEL_DIR, "diabetes_meta.pkl")
    joblib.dump(meta, meta_path)
    print(f"[OK] Metadata   -> {meta_path}")

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI - Train Diabetes Model")
    print("=" * 55)
    splits  = load_splits()
    metrics = train(splits)
    target  = 0.88
    status  = "PASSED" if metrics["roc_auc"] >= target else "BELOW TARGET"
    print(f"\n[{status}] ROC-AUC {metrics['roc_auc']:.4f} (target >= {target})")
    print("[DONE]\n")
