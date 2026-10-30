# MedInsight AI 🏥
### Integrated Multi-Disease Predictive Analytics & AI Clinical Advisory System

> A hybrid Clinical Decision Support System (CDSS) that combines classical machine learning with Generative AI to screen for **Diabetes**, **Heart Disease**, and **Chronic Kidney Disease** — all in under 2 seconds.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Datasets](#4-datasets)
5. [ML Models & Target Metrics](#5-ml-models--target-metrics)
6. [Quick Start](#6-quick-start)
7. [Running the Application](#7-running-the-application)
8. [API Reference](#8-api-reference)
9. [LLM Configuration](#9-llm-configuration)
10. [Development Roadmap](#10-development-roadmap)
11. [Explainability — Shapley Values](#11-explainability--shapley-values)
12. [Disclaimer](#12-disclaimer)

---

## 1. Project Overview

Traditional chronic-disease screening is reactive and fragmented. MedInsight AI addresses this by:

| Problem | Solution |
|---|---|
| Diagnostic fragmentation across isolated systems | Single dashboard for 3 diseases simultaneously |
| 30–45% delay in pre-symptomatic risk detection | Sub-2-second ML inference pipeline |
| 78% physician distrust of "black-box" ML scores | SHAP Explainable AI shows *why* a risk exists |
| Generic numerical outputs | LLM-generated, empathetic clinical advisory reports |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                    │
│   Input Forms → Gauge Charts → SHAP Bar Charts          │
│   AI Clinical Report (3-part structured advisory)       │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / Standalone fallback
┌────────────────────▼────────────────────────────────────┐
│                    FASTAPI BACKEND                       │
│   /predict/diabetes|heart|ckd  (ML inference)           │
│   /predict/*/report            (inference + LLM)        │
│   /health   /models/info                                │
└───────┬────────────────────────┬────────────────────────┘
        │                        │
┌───────▼──────────┐   ┌─────────▼──────────────────────┐
│   ML RISK ENGINE  │   │      GenAI ADVISORY LAYER      │
│  XGBoost + RF     │   │  NVIDIA NIM / Groq / OpenAI    │
│  SVC (GridCV)     │   │  meta/llama-3.3-70b-instruct   │
│  Random Forest    │   │  Rule-based fallback (no key)  │
│  SHAP Explainer   │   └────────────────────────────────┘
└──────────────────┘
```

---

## 3. Project Structure

```
medinsight-ai/
│
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, all routes
│   ├── model_loader.py    # Singleton model registry
│   ├── explainer.py       # SHAP XAI engine
│   ├── llm_advisor.py     # LLM clinical report generator
│   └── schemas.py         # Pydantic request/response models
│
├── frontend/
│   ├── __init__.py
│   ├── app.py             # Streamlit dashboard (4 pages)
│   └── api_client.py      # HTTP client + standalone fallback
│
├── scripts/
│   ├── preprocess_diabetes.py
│   ├── preprocess_heart.py
│   ├── preprocess_ckd.py
│   ├── train_diabetes.py
│   ├── train_heart.py
│   ├── train_ckd.py
│   └── train_all.py       # Master runner
│
├── data/
│   ├── raw/               # Place UCI CSVs here
│   ├── processed/         # Auto-generated .pkl splits
│   └── eda_plots/         # Auto-generated PNG charts
│
├── models/                # Serialized .pkl model artifacts
│
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Datasets

| Disease | Source | Records | Key Features |
|---|---|---|---|
| **Diabetes** | UCI Pima Indians | 768 | Glucose, Insulin, BMI, Age, Skin Thickness |
| **Heart Disease** | UCI Cleveland | 303 | Chest Pain Type, Resting BP, Cholesterol, Max HR |
| **Chronic Kidney Disease** | UCI CKD | 400 | Serum Creatinine, Hemoglobin, Blood Pressure, Specific Gravity |

### Downloading Real Datasets

```
# Diabetes
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
→ Save as: data/raw/diabetes.csv

# Heart Disease
https://archive.ics.uci.edu/dataset/45/heart+disease
→ Save as: data/raw/heart.csv  (14-column Cleveland format)

# CKD
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
→ Save as: data/raw/ckd.csv
```

> **Note:** If CSV files are not present, the scripts automatically generate synthetic datasets that mirror the statistical distributions of the UCI originals. This lets you run the full pipeline without downloading anything.

---

## 5. ML Models & Target Metrics

| Disease | Algorithm | Primary Metric | Target |
|---|---|---|---|
| Diabetes | XGBoost + RandomForest (soft-voting ensemble) | ROC-AUC | ≥ 0.88 |
| Heart Disease | SVC (RBF/Linear, GridSearchCV) | Recall | ≥ 0.90 |
| CKD | RandomForestClassifier (RandomizedSearchCV) | Accuracy | ≥ 0.92 |

Recall is the primary metric for Heart Disease to minimise **false negatives** — missing a true positive in a cardiac screening is clinically more dangerous than a false alarm.

---

## 6. Quick Start

### Prerequisites
- Python 3.10 or 3.11
- pip

### 1 — Clone and install

```bash
git clone https://github.com/your-username/medinsight-ai.git
cd medinsight-ai

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and add your LLM API key (optional — app works without one)
```

### 3 — Train models

```bash
python scripts/train_all.py
```

This single command:
- Generates synthetic datasets if real CSVs are absent
- Runs preprocessing + EDA plots for all 3 diseases
- Trains and serializes all models into `models/`
- Prints a final metrics table

Expected output:
```
  Diabetes           | roc_auc=0.8923 (>=0.88)  [PASS]  | 18.2s
  Heart Disease      | recall=0.9130  (>=0.90)  [PASS]  | 42.7s
  CKD                | accuracy=0.9500(>=0.92)  [PASS]  |  9.1s
```

---

## 7. Running the Application

### Option A — Full stack (recommended)

**Terminal 1 — FastAPI backend:**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

### Option B — Streamlit standalone (no backend needed)

The Streamlit app automatically falls back to loading models directly if the backend is unreachable. Just run:

```bash
streamlit run frontend/app.py
```

The sidebar will show `🟡 Standalone mode` instead of `🟢 Backend connected`.

---

## 8. API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend health + loaded models |
| `GET` | `/models/info` | Model types, features, metrics |
| `POST` | `/predict/diabetes` | Diabetes risk (no report) |
| `POST` | `/predict/heart` | Heart disease risk (no report) |
| `POST` | `/predict/ckd` | CKD risk (no report) |
| `POST` | `/predict/diabetes/report` | Diabetes risk + LLM report |
| `POST` | `/predict/heart/report` | Heart disease risk + LLM report |
| `POST` | `/predict/ckd/report` | CKD risk + LLM report |

### Example — Diabetes prediction

```bash
curl -X POST http://localhost:8000/predict/diabetes/report \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

### Example response

```json
{
  "disease": "Diabetes",
  "risk_probability": 0.7823,
  "risk_percent": 78.23,
  "risk_level": "High",
  "top_factors": [
    { "feature": "Glucose", "value": 148.0, "shap": 0.312, "direction": "increases risk" },
    { "feature": "BMI",     "value": 33.6,  "shap": 0.198, "direction": "increases risk" },
    { "feature": "Age",     "value": 50.0,  "shap": 0.143, "direction": "increases risk" }
  ],
  "model_type": "VotingClassifier(XGBoost + RandomForest)",
  "clinical_report": "## 1. Risk Severity Assessment\n...",
  "report_available": true
}
```

---

## 9. LLM Configuration

The app supports three API providers (priority order: NVIDIA NIM → Groq → OpenAI).

### NVIDIA NIM (recommended — free tier)
1. Sign up at [build.nvidia.com](https://build.nvidia.com)
2. Generate an API key
3. Add to `.env`: `NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx`

### Groq (fastest inference — free tier)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Generate an API key
3. Add to `.env`: `GROQ_API_KEY=gsk_xxxxxxxxxxxx`

### No API Key
The app works fully without any key. A rule-based fallback engine generates structured reports using clinical guidelines. Reports are clearly labelled as fallback-generated.

---

## 10. Development Roadmap

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | Data & Preprocessing | UCI CSVs, median imputation, StandardScaler, EDA plots |
| 3–4 | Model Training & Validation | Serialized .pkl models meeting target metrics |
| 5 | API & UI Integration | FastAPI backend + Streamlit dashboard + LLM reports |
| 6 | Testing & Defense Prep | Edge-case testing, GitHub docs, 6-slide presentation |

---

## 11. Explainability — Shapley Values

Feature contributions are computed using **SHAP (SHapley Additive exPlanations)**:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$

- **Red bars** → feature increases the predicted risk
- **Green bars** → feature decreases the predicted risk
- SHAP magnitude indicates how strongly each feature pushed the model's output

Tree-based models (XGBoost, Random Forest) use the fast `TreeExplainer`. The SVC uses `KernelExplainer` (model-agnostic, slower but exact).

---

## 12. Disclaimer

> **This system is intended for research, educational, and screening purposes only.**
> It is NOT a substitute for professional medical diagnosis, advice, or treatment.
> All predictions must be reviewed and validated by a licensed healthcare professional
> before any clinical decision is made.

---

*Built for PBL 5th Semester | MedInsight AI Team*
