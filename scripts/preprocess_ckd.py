"""
Preprocessing & EDA Script - Chronic Kidney Disease Dataset
UCI CKD Dataset (400 records)
Run: python scripts/preprocess_ckd.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore")

RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# UCI CKD column names (arff format headers)
COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "class"
]

# Numeric and categorical feature groups
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc"
]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]


def load_data() -> pd.DataFrame:
    raw_path = os.path.join(RAW_DIR, "ckd.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, na_values=["?", "\t?", " "])
        if df.shape[1] == len(COLUMNS):
            df.columns = COLUMNS
        print(f"[OK] Loaded ckd.csv  ({df.shape[0]} rows)")
    else:
        print("[!]  ckd.csv not found - generating synthetic data.")
        df = _generate_synthetic(raw_path)
    return df


def _generate_synthetic(save_path: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n   = 400

    # CKD patients (n_ckd) vs healthy
    n_ckd = 250

    def col(ckd_mean, ckd_std, healthy_mean, healthy_std, low, high):
        ckd_vals     = rng.normal(ckd_mean, ckd_std, n_ckd).clip(low, high)
        healthy_vals = rng.normal(healthy_mean, healthy_std, n - n_ckd).clip(low, high)
        return np.concatenate([ckd_vals, healthy_vals])

    age  = col(55, 17, 40, 14, 2, 90).round(0)
    bp   = col(80, 16, 71, 12, 50, 180).round(0)
    sg   = np.concatenate([
        rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_ckd, p=[0.30,0.30,0.20,0.15,0.05]),
        rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], n - n_ckd, p=[0.05,0.10,0.20,0.30,0.35])
    ])
    al   = col(2.5, 1.5, 0.2, 0.5, 0, 5).round(0).clip(0, 5)
    su   = col(1.0, 1.5, 0.1, 0.3, 0, 5).round(0).clip(0, 5)
    bgr  = col(150, 70, 100, 20, 70, 490).round(0)
    bu   = col(60, 50, 30, 10, 1.5, 391).round(1)
    sc   = col(4.0, 3.5, 0.9, 0.2, 0.4, 76).round(1)
    sod  = col(132, 12, 140, 4, 111, 163).round(0)
    pot  = col(4.5, 1.5, 4.0, 0.5, 2.5, 47).round(1)
    hemo = col(10.0, 3.0, 14.5, 1.5, 3.1, 17.8).round(1)
    pcv  = col(30, 10, 44, 5, 9, 54).round(0)
    wc   = col(9000, 4000, 7500, 2000, 2200, 26400).round(0)
    rc   = col(3.5, 1.2, 5.0, 0.6, 1.8, 8).round(1)

    def cat_col(p_ckd, p_healthy, vals):
        return np.concatenate([
            rng.choice(vals, n_ckd, p=p_ckd),
            rng.choice(vals, n - n_ckd, p=p_healthy)
        ])

    rbc   = cat_col([0.65, 0.35], [0.10, 0.90], ["abnormal", "normal"])
    pc    = cat_col([0.70, 0.30], [0.15, 0.85], ["abnormal", "normal"])
    pcc   = cat_col([0.55, 0.45], [0.05, 0.95], ["present", "notpresent"])
    ba    = cat_col([0.40, 0.60], [0.05, 0.95], ["present", "notpresent"])
    htn   = cat_col([0.75, 0.25], [0.20, 0.80], ["yes", "no"])
    dm    = cat_col([0.55, 0.45], [0.15, 0.85], ["yes", "no"])
    cad   = cat_col([0.25, 0.75], [0.05, 0.95], ["yes", "no"])
    appet = cat_col([0.60, 0.40], [0.10, 0.90], ["poor", "good"])
    pe    = cat_col([0.50, 0.50], [0.05, 0.95], ["yes", "no"])
    ane   = cat_col([0.55, 0.45], [0.10, 0.90], ["yes", "no"])

    cls = np.array(["ckd"] * n_ckd + ["notckd"] * (n - n_ckd))

    # Shuffle
    idx = rng.permutation(n)
    df  = pd.DataFrame({
        "age": age[idx], "bp": bp[idx], "sg": sg[idx], "al": al[idx],
        "su": su[idx], "rbc": rbc[idx], "pc": pc[idx], "pcc": pcc[idx],
        "ba": ba[idx], "bgr": bgr[idx], "bu": bu[idx], "sc": sc[idx],
        "sod": sod[idx], "pot": pot[idx], "hemo": hemo[idx], "pcv": pcv[idx],
        "wc": wc[idx], "rc": rc[idx], "htn": htn[idx], "dm": dm[idx],
        "cad": cad[idx], "appet": appet[idx], "pe": pe[idx], "ane": ane[idx],
        "class": cls[idx]
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Synthetic CKD dataset saved -> {save_path}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_miss = df[col].isnull().sum()
            if n_miss:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  [{col}] {n_miss} missing -> median imputed")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n_miss = df[col].isnull().sum()
            if n_miss:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  [{col}] {n_miss} missing -> mode imputed")

    # Encode target
    df["class"] = df["class"].str.strip().map({"ckd": 1, "notckd": 0})
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def run_eda(df: pd.DataFrame):
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eda_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Key numeric features
    key_feats = ["age", "bp", "sc", "hemo", "bgr", "bu", "pcv"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, feat in zip(axes.flatten(), key_feats):
        ax.hist(df[feat].dropna(), bins=25, color="mediumseagreen",
                edgecolor="white", alpha=0.85)
        ax.set_title(feat.upper(), fontsize=11)
    axes.flatten()[-1].axis("off")
    plt.suptitle("CKD - Key Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ckd_distributions.png"), dpi=150)
    plt.close()

    # Serum Creatinine by class
    plt.figure(figsize=(7, 5))
    df.boxplot(column="sc", by="class", ax=plt.gca(),
               boxprops=dict(color="teal"), medianprops=dict(color="red"))
    plt.title("Serum Creatinine by CKD Status")
    plt.suptitle("")
    plt.xlabel("CKD (1=Yes, 0=No)")
    plt.ylabel("Serum Creatinine (mg/dL)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ckd_creatinine_boxplot.png"), dpi=150)
    plt.close()

    # Class balance
    plt.figure(figsize=(5, 4))
    counts = df["class"].value_counts().sort_index()
    plt.bar(["No CKD", "CKD"], counts.values,
            color=["steelblue", "salmon"], edgecolor="white")
    plt.title("Class Distribution - CKD", fontweight="bold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ckd_class_balance.png"), dpi=150)
    plt.close()
    print(f"[OK] EDA plots saved -> {plot_dir}")


def preprocess(df: pd.DataFrame) -> dict:
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)

    X = df.drop("class", axis=1)
    y = df["class"]

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
        "encoders": encoders,
    }
    joblib.dump(splits, os.path.join(PROC_DIR, "ckd_processed.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "ckd_scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "ckd_encoders.pkl"))
    print(f"[OK] Processed splits -> data/processed/ckd_processed.pkl")
    print(f"[OK] Scaler           -> models/ckd_scaler.pkl")
    print(f"[OK] Encoders         -> models/ckd_encoders.pkl")
    print(f"     Train: {X_train_s.shape[0]}  |  Test: {X_test_s.shape[0]}")
    return splits


if __name__ == "__main__":
    print("=" * 55)
    print("  MedInsight AI - CKD Preprocessing & EDA")
    print("=" * 55)
    df = load_data()
    print("\n-- Shape --")
    print(df.shape)
    print("\n-- Missing Values (top 10) --")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\n-- Running EDA --")
    run_eda(df)
    print("\n-- Preprocessing --")
    preprocess(df)
    print("\n[DONE] CKD preprocessing complete.\n")
