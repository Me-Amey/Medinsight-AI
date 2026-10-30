"""
MedInsight AI - Streamlit Dashboard  (Premium UI)
Multi-Disease Predictive Analytics & AI Clinical Advisory System
Run: streamlit run frontend/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from frontend.api_client import (
    predict_diabetes, predict_heart, predict_ckd,
    check_backend, get_model_info,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedInsight AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Animated gradient hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0c4a6e 70%, #0f172a 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(14,165,233,0.25);
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-banner::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(14,165,233,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; color: #ffffff;
    letter-spacing: -0.5px; margin: 0; line-height: 1.15;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; font-weight: 400;
}
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #e2e8f0; font-size: 0.75rem; font-weight: 500;
    padding: 0.25rem 0.75rem; border-radius: 999px;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    color: #64748b; text-transform: uppercase; margin-bottom: 0.4rem;
}

/* ── Glass input card ── */
.input-card {
    background: rgba(30,41,59,0.55);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 1.5rem 1.5rem 0.5rem 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}
.input-card-title {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em;
    color: #38bdf8; text-transform: uppercase; margin-bottom: 0.8rem;
}

/* ── Result risk banner ── */
.risk-banner {
    border-radius: 14px; padding: 1.2rem 1.6rem;
    margin: 1.2rem 0; display: flex; align-items: center; gap: 1rem;
}
.risk-low  { background: linear-gradient(90deg, rgba(22,163,74,0.18), rgba(22,163,74,0.05)); border-left: 5px solid #16a34a; }
.risk-mod  { background: linear-gradient(90deg, rgba(234,179,8,0.18),  rgba(234,179,8,0.05));  border-left: 5px solid #eab308; }
.risk-high { background: linear-gradient(90deg, rgba(239,68,68,0.18),  rgba(239,68,68,0.05));  border-left: 5px solid #ef4444; }
.risk-label { font-size: 1.7rem; font-weight: 800; }
.risk-low  .risk-label { color: #4ade80; }
.risk-mod  .risk-label { color: #facc15; }
.risk-high .risk-label { color: #f87171; }
.risk-disease { font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Metric pill ── */
.metric-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 0.35rem 0.9rem;
    font-size: 0.78rem; font-weight: 600; color: #e2e8f0;
}
.metric-pill span { color: #38bdf8; }

/* ── Report box ── */
.report-card {
    background: rgba(14,165,233,0.07);
    border: 1px solid rgba(14,165,233,0.22);
    border-radius: 14px; padding: 1.5rem 1.8rem;
    line-height: 1.85; font-size: 0.92rem;
}

/* ── Analyse button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #0369a1, #0ea5e9, #38bdf8) !important;
    color: #ffffff !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
    transition: all 0.2s !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 6px 28px rgba(14,165,233,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(148,163,184,0.1);
}
.sidebar-logo {
    font-size: 1.6rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px; margin-bottom: 0.1rem;
}
.sidebar-logo span { color: #38bdf8; }
.sidebar-tagline { font-size: 0.75rem; color: #64748b; margin-bottom: 1.5rem; }

/* ── Nav item ── */
div[data-testid="stRadio"] label {
    padding: 0.5rem 0.75rem !important; border-radius: 8px !important;
    transition: background 0.15s !important;
}
div[data-testid="stRadio"] label:hover { background: rgba(255,255,255,0.06) !important; }

/* ── Tab divider ── */
hr { border-color: rgba(148,163,184,0.12) !important; margin: 1.2rem 0 !important; }

/* ── Feature factor card ── */
.factor-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.factor-name  { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
.factor-value { font-size: 0.78rem; color: #94a3b8; }
.factor-up    { color: #f87171; font-weight: 700; font-size: 0.8rem; }
.factor-down  { color: #4ade80; font-weight: 700; font-size: 0.8rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLOR = {"Low": "#4ade80", "Moderate": "#facc15", "High": "#f87171"}
RISK_CSS   = {"Low": "risk-low", "Moderate": "risk-mod", "High": "risk-high"}
RISK_ICON  = {"Low": "✅", "Moderate": "⚠️", "High": "🚨"}


def gauge_chart(prob: float, risk_level: str):
    color = RISK_COLOR.get(risk_level, "#94a3b8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)"},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "rgba(74,222,128,0.12)"},
                {"range": [35, 65], "color": "rgba(250,204,21,0.12)"},
                {"range": [65,100], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 5},
                "thickness": 0.82, "value": prob,
            },
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_chart(top_factors: list, title: str):
    if not top_factors:
        return None
    features = [f["feature"] for f in top_factors]
    vals     = [f["shap"]    for f in top_factors]
    colors   = ["#f87171" if v > 0 else "#4ade80" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> — Key Drivers", font=dict(size=13)),
        xaxis_title="Feature Contribution Score",
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        height=270,
        margin=dict(t=45, b=15, l=5, r=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    )
    return fig


def render_result(data: dict, label: str):
    prob       = data["risk_percent"]
    risk_level = data["risk_level"]
    top_factors= data.get("top_factors", [])
    report     = data.get("clinical_report", "")

    # ── Risk Banner ──
    icon = RISK_ICON.get(risk_level, "")
    st.markdown(f"""
    <div class="risk-banner {RISK_CSS.get(risk_level,'risk-mod')}">
        <div style="font-size:2.8rem">{icon}</div>
        <div>
            <div class="risk-label">{risk_level} Risk &nbsp; {prob:.1f}%</div>
            <div class="risk-disease">{label} Screening Result</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ──
    col_gauge, col_shap = st.columns([1, 1.4])
    with col_gauge:
        st.plotly_chart(gauge_chart(prob, risk_level),
                        use_container_width=True, key=f"g_{label}")

    with col_shap:
        fig = shap_chart(top_factors, label)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"s_{label}")

    # ── Top factors detail ──
    if top_factors:
        st.markdown("**🔬 Top Contributing Parameters**")
        cols = st.columns(min(len(top_factors), 5))
        for i, f in enumerate(top_factors):
            direction_html = (
                f'<span class="factor-up">↑ increases risk</span>'
                if f["direction"] == "increases risk"
                else f'<span class="factor-down">↓ decreases risk</span>'
            )
            with cols[i]:
                st.markdown(f"""
                <div class="factor-card">
                    <div>
                        <div class="factor-name">{f['feature']}</div>
                        <div class="factor-value">Value: {f['value']:.2f}</div>
                        {direction_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Report ──
    if report:
        with st.expander("📋 AI Clinical Advisory Report", expanded=True):
            st.markdown("""
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                        color:#38bdf8; text-transform:uppercase; margin-bottom:0.8rem;">
                Generated by Groq · openai/gpt-oss-120b
            </div>
            """, unsafe_allow_html=True)
            st.markdown(report)

    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem 0;">
            <div class="sidebar-logo">Med<span>Insight</span> AI</div>
            <div class="sidebar-tagline">Clinical Decision Support System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🩸  Diabetes Screening",
             "❤️  Heart Disease Screening",
             "🫘  Kidney Disease Screening",
             "📊  Model Information"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        backend_ok = check_backend()
        if backend_ok:
            st.markdown("""
            <div style="background:rgba(22,163,74,0.15); border:1px solid rgba(22,163,74,0.35);
                        border-radius:8px; padding:0.5rem 0.85rem; font-size:0.82rem; color:#4ade80;">
                🟢 &nbsp; Backend connected
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(234,179,8,0.12); border:1px solid rgba(234,179,8,0.3);
                        border-radius:8px; padding:0.5rem 0.85rem; font-size:0.82rem; color:#facc15;">
                🟡 &nbsp; Standalone mode
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7rem; color:#475569; line-height:1.6;">
            ⚠️ <b>Disclaimer:</b> For research &amp; educational use only.<br>
            Not a substitute for professional medical advice.
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner (shared)
# ─────────────────────────────────────────────────────────────────────────────

def hero(icon: str, title: str, subtitle: str, badges: list):
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{icon} &nbsp;{title}</div>
        <div class="hero-sub">{subtitle}</div>
        <div class="hero-badges">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Diabetes
# ─────────────────────────────────────────────────────────────────────────────

def page_diabetes():
    hero("🩸", "Diabetes Risk Screening",
         "Enter patient vitals and biochemical markers below to compute risk probability.",
         ["XGBoost + Random Forest", "UCI Pima Indians Dataset", "Target ROC-AUC ≥ 0.88"])

    with st.form("diabetes_form"):

        # ── Section 1: Demographics ──
        st.markdown('<div class="input-card-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            age  = st.number_input("Age (years)", min_value=1, max_value=120, value=33,
                                   help="Patient's age in years")
        with d2:
            sex  = st.selectbox("Biological Sex", options=[0, 1],
                                format_func=lambda x: "♀ Female" if x == 0 else "♂ Male",
                                help="Biological sex affects hormonal and metabolic risk factors")
        with d3:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1,
                                          help="Number of times pregnant (0 for males)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 2: Metabolic Markers ──
        st.markdown('<div class="input-card-title">🧪 Metabolic Markers</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, 110,
                                      help="Plasma glucose concentration (2-hour OGTT)")
        with m2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80,
                                      help="2-hour serum insulin")
        with m3:
            bmi     = st.number_input("BMI (kg/m²)", 0.0, 80.0, 28.5, step=0.1,
                                      help="Body Mass Index")
        with m4:
            dpf     = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.47, step=0.01,
                                      help="Diabetes pedigree function — family history score")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 3: Clinical Measurements ──
        st.markdown('<div class="input-card-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            bp   = st.slider("Diastolic Blood Pressure (mmHg)", 0, 200, 72,
                             help="Diastolic blood pressure measurement")
        with c2:
            skin = st.slider("Skin Thickness (mm)", 0, 100, 20,
                             help="Triceps skin fold thickness")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Analyse Diabetes Risk", use_container_width=True)

    if submitted:
        payload = {
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": bp, "skin_thickness": skin,
            "insulin": insulin, "bmi": bmi,
            "diabetes_pedigree_function": dpf, "age": age,
        }
        with st.spinner("Running ML inference + feature attribution..."):
            result = predict_diabetes(payload)
        if not result["ok"]:
            st.error(f"❌ {result['error']}")
        else:
            render_result(result["data"], "Diabetes")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Heart Disease
# ─────────────────────────────────────────────────────────────────────────────

def page_heart():
    hero("❤️", "Heart Disease Risk Screening",
         "Enter cardiovascular parameters to assess coronary artery disease risk.",
         ["Support Vector Classifier", "UCI Cleveland Dataset", "Target Recall ≥ 0.90"])

    with st.form("heart_form"):

        st.markdown('<div class="input-card-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        with h1:
            age  = st.number_input("Age (years)", 1, 120, 54)
        with h2:
            sex  = st.selectbox("Sex", [1, 0], format_func=lambda x: "♂ Male" if x == 1 else "♀ Female")
        with h3:
            cp   = st.selectbox("Chest Pain Type", [0,1,2,3],
                                format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                       2:"Non-Anginal Pain",3:"Asymptomatic"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="input-card-title">📊 Cardiovascular Parameters</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 130)
            chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
            fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x:"Yes" if x else "No")
        with c2:
            restecg  = st.selectbox("Resting ECG", [0,1,2],
                                    format_func=lambda x:{0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
            thalach  = st.number_input("Max Heart Rate", 60, 220, 150)
            exang    = st.selectbox("Exercise Angina", [0,1], format_func=lambda x:"Yes" if x else "No")
        with c3:
            oldpeak  = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)
            slope    = st.selectbox("ST Slope", [0,1,2],
                                    format_func=lambda x:{0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
            ca       = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
            thal     = st.selectbox("Thalassemia", [1,2,3],
                                    format_func=lambda x:{1:"Normal",2:"Fixed Defect",3:"Reversible"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Analyse Heart Disease Risk", use_container_width=True)

    if submitted:
        payload = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                       fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                       oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        with st.spinner("Running ML inference + feature attribution..."):
            result = predict_heart(payload)
        if not result["ok"]:
            st.error(f"❌ {result['error']}")
        else:
            render_result(result["data"], "Heart Disease")


# ─────────────────────────────────────────────────────────────────────────────
# Page: CKD
# ─────────────────────────────────────────────────────────────────────────────

def page_ckd():
    hero("🫘", "Kidney Disease Risk Screening",
         "Enter renal panel and clinical findings to assess Chronic Kidney Disease risk.",
         ["Random Forest Classifier", "UCI CKD Dataset", "Target Accuracy ≥ 0.92"])

    with st.form("ckd_form"):

        st.markdown('<div class="input-card-title">🔬 Blood Chemistry Panel</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age  = st.number_input("Age", 1, 100, 48)
            bp   = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
            sg   = st.number_input("Specific Gravity", 1.000, 1.030, 1.015, step=0.001, format="%.3f")
            al   = st.number_input("Albumin (0–5)", 0, 5, 1)
        with c2:
            su   = st.number_input("Sugar (0–5)", 0, 5, 0)
            bgr  = st.number_input("Blood Glucose (mg/dL)", 70, 500, 120)
            bu   = st.number_input("Blood Urea (mg/dL)", 1, 400, 40)
            sc   = st.number_input("Serum Creatinine (mg/dL)", 0.4, 80.0, 1.2, step=0.1)
        with c3:
            sod  = st.number_input("Sodium (mEq/L)", 100, 170, 137)
            pot  = st.number_input("Potassium (mEq/L)", 2.0, 50.0, 4.5, step=0.1)
            hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, 13.5, step=0.1)
            pcv  = st.number_input("Packed Cell Vol. (%)", 9, 55, 41)
        with c4:
            wc   = st.number_input("WBC (cells/cumm)", 2000, 30000, 8000, step=100)
            rc   = st.number_input("RBC (millions/cmm)", 1.0, 9.0, 4.8, step=0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="input-card-title">🩺 Clinical Findings</div>', unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            rbc  = st.selectbox("Red Blood Cells",  [1,0], format_func=lambda x:"Normal" if x else "Abnormal")
            pc   = st.selectbox("Pus Cells",         [1,0], format_func=lambda x:"Normal" if x else "Abnormal")
            pcc  = st.selectbox("Pus Cell Clumps",  [0,1], format_func=lambda x:"Present" if x else "Not Present")
        with c6:
            ba   = st.selectbox("Bacteria",          [0,1], format_func=lambda x:"Present" if x else "Not Present")
            htn  = st.selectbox("Hypertension",      [0,1], format_func=lambda x:"Yes" if x else "No")
            dm   = st.selectbox("Diabetes Mellitus", [0,1], format_func=lambda x:"Yes" if x else "No")
        with c7:
            cad  = st.selectbox("Coronary Artery Disease", [0,1], format_func=lambda x:"Yes" if x else "No")
            appet= st.selectbox("Appetite",          [0,1], format_func=lambda x:"Poor" if x else "Good")
        with c8:
            pe   = st.selectbox("Pedal Edema",       [0,1], format_func=lambda x:"Yes" if x else "No")
            ane  = st.selectbox("Anemia",             [0,1], format_func=lambda x:"Yes" if x else "No")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Analyse Kidney Disease Risk", use_container_width=True)

    if submitted:
        payload = dict(age=age, bp=bp, sg=sg, al=al, su=su, bgr=bgr, bu=bu, sc=sc,
                       sod=sod, pot=pot, hemo=hemo, pcv=pcv, wc=wc, rc=rc,
                       rbc=rbc, pc=pc, pcc=pcc, ba=ba, htn=htn, dm=dm,
                       cad=cad, appet=appet, pe=pe, ane=ane)
        with st.spinner("Running ML inference + feature attribution..."):
            result = predict_ckd(payload)
        if not result["ok"]:
            st.error(f"❌ {result['error']}")
        else:
            render_result(result["data"], "Chronic Kidney Disease")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Info
# ─────────────────────────────────────────────────────────────────────────────

def page_model_info():
    hero("📊", "Model Information",
         "Validation metrics, feature sets, and explainability methodology for each classifier.",
         ["Scikit-Learn", "XGBoost", "SHAP Explainability"])

    info_result = get_model_info()
    if not info_result["ok"]:
        info_data = {
            "diabetes":      {"disease": "Diabetes",               "model_type": "VotingClassifier(XGBoost + RandomForest)"},
            "heart_disease": {"disease": "Heart Disease",          "model_type": "SVC (RBF/Linear, calibrated)"},
            "ckd":           {"disease": "Chronic Kidney Disease", "model_type": "RandomForestClassifier"},
        }
    else:
        info_data = info_result["data"]

    targets = {
        "diabetes":      ("ROC-AUC",  "≥ 0.88", "🩸"),
        "heart_disease": ("Recall",   "≥ 0.90", "❤️"),
        "ckd":           ("Accuracy", "≥ 0.92", "🫘"),
    }

    for key, info in info_data.items():
        icon, tname, tval = targets.get(key, ("🔬","",""))[-1], targets.get(key, ("","",""))[0], targets.get(key, ("","",""))[1]
        emoji = targets[key][2] if key in targets else "🔬"

        with st.expander(f"{emoji}  {info['disease']}", expanded=True):
            left, right = st.columns([1, 1.8])
            with left:
                st.markdown(f"**Algorithm:** `{info.get('model_type','—')}`")
                tname_val = targets.get(key, ("—","—"))
                st.markdown(f"**Primary Metric Target:** {tname_val[0]} {tname_val[1]}")
                metrics = info.get("metrics", {})
                if metrics:
                    st.markdown("**Validation Results:**")
                    for k, v in metrics.items():
                        bar_pct = int(v * 100)
                        color   = "#4ade80" if v >= 0.85 else "#facc15" if v >= 0.70 else "#f87171"
                        st.markdown(f"""
                        <div style="margin-bottom:0.4rem;">
                            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                                <span>{k.upper()}</span>
                                <span style="color:{color};font-weight:700;">{v:.4f}</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;margin-top:3px;">
                                <div style="background:{color};width:{bar_pct}%;height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            with right:
                feats = info.get("feature_names", [])
                if feats:
                    st.markdown("**Input Feature Set:**")
                    cols = st.columns(2)
                    for i, f in enumerate(feats):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                                        border-radius:6px;padding:0.3rem 0.6rem;margin-bottom:0.3rem;
                                        font-size:0.78rem;color:#94a3b8;">
                                <span style="color:#38bdf8;font-weight:600;">{i+1}.</span> {f}
                            </div>
                            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧮 Explainability — Shapley Values")
    col_eq, col_desc = st.columns([1, 1.2])
    with col_eq:
        st.latex(r"""
            \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
            \frac{|S|!(|N| - |S| - 1)!}{|N|!}
            \left[ v(S \cup \{i\}) - v(S) \right]
        """)
    with col_desc:
        st.markdown("""
        **How it works:**
        - Each feature receives a contribution score φᵢ
        - Scores represent the feature's average marginal effect across all prediction scenarios
        - 🔴 **Red bars** → feature pushes risk **higher**
        - 🟢 **Green bars** → feature pushes risk **lower**
        - Magnitude = how strongly the feature influenced this prediction
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()
    if "Diabetes"  in page: page_diabetes()
    elif "Heart"   in page: page_heart()
    elif "Kidney"  in page: page_ckd()
    elif "Model"   in page: page_model_info()

if __name__ == "__main__":
    main()
