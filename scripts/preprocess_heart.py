"""
Preprocessing & EDA Script - Heart Disease Dataset
UCI Cleveland Heart Disease Dataset (303 records)
Run: python scripts/preprocess_heart.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore")

RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]


def load_data() -> pd.DataFrame:
    raw_path = os.path.join(RAW_DIR, "heart.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        if df.shape[1] == len(COLUMNS):
            df.columns = COLUMNS
        df["target"] = (df["target"] > 0).astype(int)
        print(f"[OK] Loaded heart.csv  ({df.shape[0]} rows)")
    else:
        print("[!]  heart.csv not found - generating synthetic data.")
        df = _generate_synthetic(raw_path)
    return df


def _generate_synthetic(save_path: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n   = 303

    age      = rng.integers(29, 78, n)
    sex      = rng.integers(0, 2, n)
    cp       = rng.integers(0, 4, n)
    trestbps = rng.normal(131, 17, n).clip(94, 200).astype(int)
    chol     = rng.normal(246, 52, n).clip(126, 564).astype(int)
    fbs      = (rng.random(n) < 0.15).astype(int)
    restecg  = rng.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02])
    thalach  = rng.normal(150, 23, n).clip(71, 202).astype(int)
    exang    = (rng.random(n) < 0.33).astype(int)
    oldpeak  = rng.exponential(1.0, n).clip(0, 6.2).round(1)
    slope    = rng.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca       = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal     = rng.choice([1, 2, 3], n, p=[0.05, 0.55, 0.40])

    logit  = -3 + 0.04*age - 0.5*sex + 0.3*cp - 0.01*thalach + 0.3*exang + 0.4*ca
    target = (rng.random(n) < 1/(1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal, "target": target
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Synthetic heart dataset saved -> {save_path}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
            print(f"  [{col}] missing values imputed")
    return df


def run_eda(df: pd.DataFrame):
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eda_plots")
    os.makedirs(plot_dir, exist_ok=True)
    features = [c for c in df.columns if c != "target"]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for ax, feat in zip(axes.flatten(), features):
        ax.hist(df[feat], bins=25, color="coral", edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=10)
    for ax in axes.flatten()[len(features):]:
        ax.axis("off")
    plt.suptitle("Heart Disease - Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "heart_distributions.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5)
    plt.title("Heart Disease - Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "heart_correlation.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    plt.bar(["No Disease", "Heart Disease"], counts.values,
            color=["steelblue", "salmon"], edgecolor="white")
    plt.title("Class Distribution - Heart Disease", fontweight="bold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "heart_class_balance.png"), dpi=150)
    plt.close()
    print(f"[OK] EDA plots saved -> {plot_dir}")


def preprocess(df: pd.DataFrame) -> dict:
    df = handle_missing(df)
    X, y = df.drop("target", axis=1), df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    splits = {
        "X_train": X_train_s, "X_test": X_test_s,
        "y_train": y_train.values, "y_test": y_test.values,
        "feature_names": list(X.columns),
    }
    joblib.dump(splits, os.path.join(PROC_DIR, "heart_processed.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "heart_scaler.pkl"))
    print(f"[OK] Processed splits -> data/processed/heart_processed.pkl")
    print(f"[OK] Scaler           -> models/heart_scaler.pkl")
    print(f"     Train: {X_train_s.shape[0]}  |  Test: {X_test_s.shape[0]}")
    return splits


if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI - Heart Disease Preprocessing & EDA")
    print("=" * 55)
    df = load_data()
    print("\n-- Summary Statistics --")
    print(df.describe().round(2))
    print("\n-- Missing Values --")
    print(df.isnull().sum())
    print("\n-- Running EDA --")
    run_eda(df)
    print("\n-- Preprocessing --")
    preprocess(df)
    print("\n[DONE] Heart disease preprocessing complete.\n")
