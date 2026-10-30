"""
Preprocessing & EDA Script — Diabetes Dataset
UCI Pima Indians Diabetes Dataset (768 records)
Run: python scripts/preprocess_diabetes.py
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

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
ZERO_NOT_ALLOWED = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ── 1. Load or generate dataset ───────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    raw_path = os.path.join(RAW_DIR, "diabetes.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        df.columns = COLUMNS
        print(f"[OK] Loaded diabetes.csv  ({df.shape[0]} rows)")
    else:
        print("[!]  diabetes.csv not found — generating synthetic data for development.")
        df = _generate_synthetic(raw_path)
    return df


def _generate_synthetic(save_path: str) -> pd.DataFrame:
    """Mirrors the statistical distribution of the UCI Pima dataset."""
    rng = np.random.default_rng(42)
    n   = 768
    glucose  = rng.normal(121, 32, n).clip(44, 199).astype(int)
    bmi      = rng.normal(32, 7.9, n).clip(18, 67).round(1)
    age      = rng.integers(21, 82, n)
    logit    = -6 + 0.04 * glucose + 0.07 * bmi + 0.01 * age
    outcome  = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "Pregnancies":              rng.integers(0, 18, n),
        "Glucose":                  glucose,
        "BloodPressure":            rng.normal(69, 19, n).clip(24, 122).astype(int),
        "SkinThickness":            rng.normal(20, 16, n).clip(0, 99).astype(int),
        "Insulin":                  rng.normal(80, 115, n).clip(0, 846).astype(int),
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": rng.exponential(0.47, n).clip(0.07, 2.42).round(3),
        "Age":                      age,
        "Outcome":                  outcome,
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Synthetic dataset saved → {save_path}")
    return df


# ── 2. Handle biologically impossible zeros ───────────────────────────────────
def handle_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_NOT_ALLOWED:
        n_zeros = (df[col] == 0).sum()
        if n_zeros:
            median = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median)
            print(f"  [{col}]  {n_zeros} zeros → replaced with median {median:.2f}")
    return df


# ── 3. EDA plots ──────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eda_plots")
    os.makedirs(plot_dir, exist_ok=True)
    features = [c for c in df.columns if c != "Outcome"]

    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feat in zip(axes.flatten(), features):
        ax.hist(df[feat], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=11)
    axes.flatten()[-1].axis("off")
    plt.suptitle("Diabetes — Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "diabetes_distributions.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Diabetes — Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "diabetes_correlation.png"), dpi=150)
    plt.close()

    # Class balance
    plt.figure(figsize=(5, 4))
    counts = df["Outcome"].value_counts().sort_index()
    plt.bar(["No Diabetes", "Diabetes"], counts.values, color=["steelblue", "salmon"],
            edgecolor="white")
    plt.title("Class Distribution — Diabetes", fontweight="bold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "diabetes_class_balance.png"), dpi=150)
    plt.close()
    print(f"[OK] EDA plots saved → {plot_dir}")


# ── 4. Preprocess & save splits ───────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> dict:
    df = handle_zeros(df)
    X, y = df.drop("Outcome", axis=1), df["Outcome"]

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
    joblib.dump(splits, os.path.join(PROC_DIR, "diabetes_processed.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
    print(f"[OK] Processed splits → data/processed/diabetes_processed.pkl")
    print(f"[OK] Scaler           → models/diabetes_scaler.pkl")
    print(f"     Train: {X_train_s.shape[0]}  |  Test: {X_test_s.shape[0]}")
    return splits


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI — Diabetes Preprocessing & EDA")
    print("=" * 55)
    df = load_data()
    print("\n── Summary Statistics ──")
    print(df.describe().round(2))
    print("\n── Missing Values ──")
    print(df.isnull().sum())
    print("\n── Running EDA ──")
    run_eda(df)
    print("\n── Preprocessing ──")
    preprocess(df)
    print("\n[DONE] Diabetes preprocessing complete.\n")
