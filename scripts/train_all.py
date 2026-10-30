"""
Master Training Runner - MedInsight AI
Runs preprocessing + training for all three disease models in sequence.
Run: python scripts/train_all.py
"""

import os
import sys
import time

# Allow importing sibling scripts
sys.path.insert(0, os.path.dirname(__file__))


def separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_all():
    results = {}
    start_total = time.time()

    # ── 1. Diabetes ───────────────────────────────────────────────────────────
    separator("STEP 1/3  |  Diabetes (XGBoost + Random Forest)")
    t0 = time.time()
    import preprocess_diabetes as pd_prep
    import train_diabetes as td

    df = pd_prep.load_data()
    splits = pd_prep.preprocess(df)
    metrics = td.train(splits)
    results["Diabetes"] = {"metrics": metrics, "time": time.time() - t0}

    # ── 2. Heart Disease ──────────────────────────────────────────────────────
    separator("STEP 2/3  |  Heart Disease (SVC)")
    t0 = time.time()
    import preprocess_heart as ph_prep
    import train_heart as th

    df = ph_prep.load_data()
    splits = ph_prep.preprocess(df)
    metrics = th.train(splits)
    results["Heart Disease"] = {"metrics": metrics, "time": time.time() - t0}

    # ── 3. CKD ────────────────────────────────────────────────────────────────
    separator("STEP 3/3  |  Chronic Kidney Disease (Random Forest)")
    t0 = time.time()
    import preprocess_ckd as pc_prep
    import train_ckd as tc

    df = pc_prep.load_data()
    splits = pc_prep.preprocess(df)
    metrics = tc.train(splits)
    results["CKD"] = {"metrics": metrics, "time": time.time() - t0}

    # ── Summary report ────────────────────────────────────────────────────────
    separator("TRAINING SUMMARY")
    targets = {
        "Diabetes":      ("roc_auc",  0.88),
        "Heart Disease": ("recall",   0.90),
        "CKD":           ("accuracy", 0.92),
    }
    all_passed = True
    for disease, info in results.items():
        m       = info["metrics"]
        metric, threshold = targets[disease]
        val     = m[metric]
        passed  = val >= threshold
        status  = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(
            f"  {disease:<25} | {metric}={val:.4f} (>={threshold})  "
            f"[{status}]  | {info['time']:.1f}s"
        )

    total = time.time() - start_total
    print(f"\n  Total training time: {total:.1f}s")
    print(f"\n  Overall: {'ALL TARGETS MET' if all_passed else 'SOME TARGETS MISSED'}")
    print("\n  Serialized models in: models/")
    print("    diabetes_model.pkl  |  diabetes_scaler.pkl  |  diabetes_meta.pkl")
    print("    heart_model.pkl     |  heart_scaler.pkl     |  heart_meta.pkl")
    print("    ckd_model.pkl       |  ckd_scaler.pkl       |  ckd_encoders.pkl  |  ckd_meta.pkl")


if __name__ == "__main__":
    run_all()
