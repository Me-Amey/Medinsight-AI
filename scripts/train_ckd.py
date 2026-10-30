"""
Model Training Script - Chronic Kidney Disease (Random Forest)
Target metric: Accuracy >= 0.92
Run: python scripts/train_ckd.py
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
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
    path = os.path.join(PROC_DIR, "ckd_processed.pkl")
    if not os.path.exists(path):
        print("[!] Processed data not found. Running preprocessor first...")
        sys.path.insert(0, os.path.dirname(__file__))
        import preprocess_ckd as prep
        df = prep.load_data()
        prep.preprocess(df)
    return joblib.load(path)


def train(splits: dict) -> dict:
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    # ── Random Forest with randomised hyperparameter search ──────────────────
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators":      [200, 300, 400, 500],
        "max_depth":         [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Running RandomizedSearchCV (optimising Accuracy)...")
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=30,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV Accuracy: {search.best_score_:.4f}")

    # Full CV AUC
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
        flag = " ✓" if k == "accuracy" and v >= 0.92 else ""
        print(f"  {k:<12}: {v:.4f}{flag}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["No CKD", "CKD"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature importances ───────────────────────────────────────────────────
    importances = best_model.feature_importances_
    feat_names  = splits["feature_names"]
    top_idx     = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 Feature Importances:")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2}. {feat_names[i]:<20} {importances[i]:.4f}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "ckd_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[OK] Model saved -> {model_path}")

    meta = {
        "model_path":    model_path,
        "scaler_path":   os.path.join(MODEL_DIR, "ckd_scaler.pkl"),
        "encoder_path":  os.path.join(MODEL_DIR, "ckd_encoders.pkl"),
        "feature_names": splits["feature_names"],
        "metrics":       metrics,
        "best_params":   search.best_params_,
        "model_type":    "RandomForestClassifier",
        "disease":       "Chronic Kidney Disease",
    }
    meta_path = os.path.join(MODEL_DIR, "ckd_meta.pkl")
    joblib.dump(meta, meta_path)
    print(f"[OK] Metadata   -> {meta_path}")

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI - Train CKD Model")
    print("=" * 55)
    splits  = load_splits()
    metrics = train(splits)
    target  = 0.92
    status  = "PASSED" if metrics["accuracy"] >= target else "BELOW TARGET"
    print(f"\n[{status}] Accuracy {metrics['accuracy']:.4f} (target >= {target})")
    print("[DONE]\n")
