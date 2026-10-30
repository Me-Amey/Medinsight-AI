"""
LLM Clinical Advisory Module - MedInsight AI
Generates structured 3-part clinical reports using an OpenAI-compatible API.
Supports: NVIDIA NIM, Groq, OpenAI, or any OpenAI-spec provider.
"""

import os
import time
import textwrap
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Client factory ────────────────────────────────────────────────────────────

def _build_client() -> Optional[OpenAI]:
    """
    Build an OpenAI-compatible client from environment variables.
    Priority: NVIDIA NIM → Groq → OpenAI → disabled (returns None).
    """
    # 1. NVIDIA NIM
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_key and nvidia_key != "YOUR_NVIDIA_API_KEY":
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key,
        )

    # 2. Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "YOUR_GROQ_API_KEY":
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
        )

    # 3. Standard OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "YOUR_OPENAI_API_KEY":
        return OpenAI(api_key=openai_key)

    return None


# ── Model selection per provider ──────────────────────────────────────────────
_MODEL_MAP = {
    "nvidia":  os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct"),
    "groq":    os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
    "openai":  os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "default": os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct"),
}

def _resolve_model() -> str:
    if os.getenv("NVIDIA_API_KEY", "").startswith("nvapi"):
        return _MODEL_MAP["nvidia"]
    if os.getenv("GROQ_API_KEY"):
        return _MODEL_MAP["groq"]
    if os.getenv("OPENAI_API_KEY"):
        return _MODEL_MAP["openai"]
    return _MODEL_MAP["default"]


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are MedInsight AI, an AI-powered Clinical Decision Support System.
    Your role is to translate machine learning risk scores into clear,
    structured, empathetic clinical advisory reports for healthcare providers.

    Guidelines:
    - Be factual, concise, and clinically accurate.
    - Never make a definitive diagnosis — frame findings as screening results
      that require physician validation.
    - Use empathetic, non-alarming language.
    - Format your response with the exact three section headers provided.
    - Each section should be 2-4 sentences or bullet points.
""").strip()


def _build_prompt(
    disease: str,
    prob_percent: float,
    risk_level: str,
    top_factors: List[Dict],
    patient_context: Optional[Dict] = None,
) -> str:
    """Build the LLM user prompt from prediction results."""

    # Format top contributing features
    factor_lines = []
    for f in top_factors:
        direction_icon = "↑" if f["direction"] == "increases risk" else "↓"
        factor_lines.append(
            f"  - {f['feature']}: {f['value']:.2f} "
            f"({direction_icon} {f['direction']}, SHAP={f['shap']:+.4f})"
        )
    factors_str = "\n".join(factor_lines) if factor_lines else "  - No factors available"

    # Optional patient context
    context_str = ""
    if patient_context:
        context_str = "\nAdditional Patient Context:\n"
        for k, v in patient_context.items():
            context_str += f"  - {k}: {v}\n"

    prompt = textwrap.dedent(f"""
        Patient Screening Results — {disease}
        ─────────────────────────────────────────
        Calculated Risk Probability: {prob_percent:.1f}%
        Risk Severity Level: {risk_level}

        Key Contributing Parameters (SHAP Analysis):
{factors_str}
{context_str}
        ─────────────────────────────────────────
        Generate a structured clinical advisory report with EXACTLY these three sections:

        ## 1. Risk Severity Assessment
        Summarise the risk level and what the key parameters indicate clinically.

        ## 2. Lifestyle & Dietary Recommendations
        Provide 3-5 specific, actionable recommendations targeting the top risk factors.

        ## 3. Suggested Follow-up Diagnostic Tests
        List 3-5 specific lab tests or clinical investigations the attending physician
        should consider ordering to confirm or rule out the condition.
    """).strip()

    return prompt


# ── Main generation function ──────────────────────────────────────────────────

def generate_clinical_report(
    disease: str,
    prob_percent: float,
    risk_level: str,
    top_factors: List[Dict],
    patient_context: Optional[Dict] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    retries: int = 2,
) -> Dict:
    """
    Generate a structured 3-part clinical advisory report via LLM.

    Parameters
    ----------
    disease         : disease name (e.g. "Diabetes")
    prob_percent    : risk probability as percentage (0-100)
    risk_level      : "Low" | "Moderate" | "High"
    top_factors     : list of SHAP factor dicts from explainer.py
    patient_context : optional extra patient info dict
    temperature     : LLM temperature (lower = more deterministic)
    max_tokens      : max response length
    retries         : number of retry attempts on API failure

    Returns
    -------
    dict with keys:
        report      : full report text (str)
        model_used  : LLM model identifier (str)
        provider    : API provider used (str)
        available   : bool (False if no API key configured)
        error       : error message if failed (str | None)
    """
    client = _build_client()

    if client is None:
        return {
            "report":    _fallback_report(disease, prob_percent, risk_level, top_factors),
            "model_used": "rule-based-fallback",
            "provider":   "local",
            "available":  False,
            "error":      "No LLM API key configured. Using rule-based fallback report.",
        }

    model     = _resolve_model()
    prompt    = _build_prompt(disease, prob_percent, risk_level, top_factors, patient_context)
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            report_text = response.choices[0].message.content.strip()
            provider = _detect_provider()
            return {
                "report":    report_text,
                "model_used": model,
                "provider":   provider,
                "available":  True,
                "error":      None,
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))  # exponential backoff

    # All retries failed — return fallback
    return {
        "report":    _fallback_report(disease, prob_percent, risk_level, top_factors),
        "model_used": "rule-based-fallback",
        "provider":   "local",
        "available":  False,
        "error":      f"LLM API failed after {retries + 1} attempts: {last_error}",
    }


def _detect_provider() -> str:
    if os.getenv("NVIDIA_API_KEY", "").startswith("nvapi"):
        return "NVIDIA NIM"
    if os.getenv("GROQ_API_KEY"):
        return "Groq"
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI"
    return "unknown"


# ── Rule-based fallback (no API key required) ─────────────────────────────────

_FOLLOWUP_TESTS = {
    "Diabetes": [
        "Fasting Plasma Glucose (FPG)",
        "HbA1c (Glycated Haemoglobin)",
        "Oral Glucose Tolerance Test (OGTT)",
        "Fasting Lipid Profile",
        "Urine Microalbumin (early nephropathy screening)",
    ],
    "Heart Disease": [
        "12-lead Electrocardiogram (ECG)",
        "Echocardiogram (Echo)",
        "Stress Test (Treadmill / Nuclear)",
        "Troponin I/T (cardiac biomarkers)",
        "Coronary CT Angiography (if indicated)",
    ],
    "Chronic Kidney Disease": [
        "Serum Creatinine & eGFR",
        "Urine Albumin-to-Creatinine Ratio (ACR)",
        "Kidney Ultrasound",
        "Complete Blood Count (CBC)",
        "Electrolyte Panel (Na, K, Bicarbonate)",
    ],
}

_LIFESTYLE_ADVICE = {
    "Low": (
        "Maintain current healthy habits. "
        "Continue regular physical activity (≥150 min/week moderate intensity). "
        "Annual screening is recommended."
    ),
    "Moderate": (
        "Adopt a low-glycaemic, heart-healthy diet (DASH or Mediterranean pattern). "
        "Reduce sodium and saturated fats. "
        "Target 30 minutes of aerobic exercise 5 days/week. "
        "Avoid smoking and limit alcohol."
    ),
    "High": (
        "Immediate lifestyle intervention is recommended. "
        "Consult a registered dietitian for a personalised meal plan. "
        "Structured exercise programme under physician guidance. "
        "Monitor vitals (BP, glucose, weight) weekly. "
        "Strict medication adherence if prescribed."
    ),
}


def _fallback_report(
    disease: str,
    prob_percent: float,
    risk_level: str,
    top_factors: List[Dict],
) -> str:
    """
    Deterministic rule-based report when no LLM API is available.
    Ensures the app remains functional during development and demos.
    """
    top_factor_names = [f["feature"] for f in top_factors[:3]]
    factors_str = ", ".join(top_factor_names) if top_factor_names else "key clinical parameters"

    tests = _FOLLOWUP_TESTS.get(disease, ["Comprehensive metabolic panel", "CBC"])
    tests_str = "\n".join(f"  - {t}" for t in tests)

    lifestyle = _LIFESTYLE_ADVICE.get(risk_level, _LIFESTYLE_ADVICE["Moderate"])

    report = textwrap.dedent(f"""
        ## 1. Risk Severity Assessment
        The screening model calculates a **{risk_level} risk** of {disease}
        with a probability of **{prob_percent:.1f}%**.
        The most significant contributing clinical parameters identified are:
        {factors_str}.
        This is a preliminary screening result — physician review is essential
        before any clinical decision is made.

        ## 2. Lifestyle & Dietary Recommendations
        {lifestyle}

        ## 3. Suggested Follow-up Diagnostic Tests
        The attending physician should consider ordering the following investigations:
{tests_str}

        ─────────────────────────────────────────────────────────────
        Note: This report was generated by the rule-based fallback engine.
        Configure an LLM API key (NVIDIA NIM / Groq / OpenAI) in .env
        for AI-generated personalised advisory reports.
    """).strip()

    return report


# ── Batch report generation ───────────────────────────────────────────────────

def generate_batch_reports(predictions: List[Dict]) -> List[Dict]:
    """
    Generate clinical reports for multiple disease predictions.
    Each item in predictions must have: disease, risk_percent, risk_level, top_factors.
    """
    results = []
    for pred in predictions:
        report_data = generate_clinical_report(
            disease=pred["disease"],
            prob_percent=pred["risk_percent"],
            risk_level=pred["risk_level"],
            top_factors=pred["top_factors"],
        )
        results.append({**pred, **report_data})
    return results
