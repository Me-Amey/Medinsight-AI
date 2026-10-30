"""Quick test for the explainer fix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from backend.model_loader import load_all_models, get_registry
from backend.explainer import explain_prediction
import numpy as np

load_all_models()
registry = get_registry()

# Test all three models
tests = {
    "diabetes": np.array([2.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]),
    "heart_disease": np.array([54.0, 1.0, 2.0, 130.0, 250.0, 0.0, 1.0, 150.0, 0.0, 1.0, 1.0, 0.0, 2.0]),
    "ckd": np.array([48.0, 80.0, 1.015, 1.0, 0.0, 120.0, 40.0, 1.2, 137.0, 4.5,
                     13.5, 41.0, 8000.0, 4.8, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

all_ok = True
for key, sample in tests.items():
    dm = registry[key]
    try:
        result = explain_prediction(dm.model, dm.scaler, sample, dm.feature_names, dm.model_type)
        print(f"\n[OK] {dm.name}")
        for f in result["top_factors"]:
            arrow = "+" if f["shap"] > 0 else "-"
            print(f"     {arrow} {f['feature']:<28} val={f['value']:.2f}  contrib={f['shap']:+.4f}")
    except Exception as e:
        print(f"\n[FAIL] {dm.name}: {e}")
        all_ok = False

print("\n[DONE]" if all_ok else "\n[SOME TESTS FAILED]")
