"""
Pydantic request/response schemas - MedInsight AI Backend
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any


# ── Request schemas ───────────────────────────────────────────────────────────

class DiabetesInput(BaseModel):
    pregnancies:               float = Field(..., ge=0, le=20,  description="Number of pregnancies")
    glucose:                   float = Field(..., ge=0, le=300, description="Plasma glucose (mg/dL)")
    blood_pressure:            float = Field(..., ge=0, le=200, description="Diastolic BP (mmHg)")
    skin_thickness:            float = Field(..., ge=0, le=100, description="Triceps skinfold (mm)")
    insulin:                   float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi:                       float = Field(..., ge=0, le=80,  description="Body Mass Index")
    diabetes_pedigree_function:float = Field(..., ge=0, le=3,   description="Diabetes pedigree function")
    age:                       float = Field(..., ge=1, le=120, description="Age in years")

    def to_array(self):
        import numpy as np
        return np.array([
            self.pregnancies, self.glucose, self.blood_pressure,
            self.skin_thickness, self.insulin, self.bmi,
            self.diabetes_pedigree_function, self.age
        ], dtype=float)


class HeartDiseaseInput(BaseModel):
    age:      float = Field(..., ge=1,  le=120, description="Age in years")
    sex:      int   = Field(..., ge=0,  le=1,   description="Sex (1=Male, 0=Female)")
    cp:       int   = Field(..., ge=0,  le=3,   description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80, le=220, description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=100,le=600, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,  le=1,   description="Fasting blood sugar > 120 mg/dl (1=True)")
    restecg:  int   = Field(..., ge=0,  le=2,   description="Resting ECG results (0-2)")
    thalach:  float = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang:    int   = Field(..., ge=0,  le=1,   description="Exercise induced angina (1=Yes)")
    oldpeak:  float = Field(..., ge=0,  le=7,   description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,  le=2,   description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,  le=4,   description="Number of major vessels (0-3)")
    thal:     int   = Field(..., ge=0,  le=3,   description="Thalassemia (1=Normal, 2=Fixed, 3=Reversible)")

    def to_array(self):
        import numpy as np
        return np.array([
            self.age, self.sex, self.cp, self.trestbps, self.chol,
            self.fbs, self.restecg, self.thalach, self.exang,
            self.oldpeak, self.slope, self.ca, self.thal
        ], dtype=float)


class CKDInput(BaseModel):
    age:  float = Field(..., ge=1,   le=100,  description="Age in years")
    bp:   float = Field(..., ge=50,  le=200,  description="Blood pressure (mmHg)")
    sg:   float = Field(..., ge=1.0, le=1.03, description="Specific gravity")
    al:   float = Field(..., ge=0,   le=5,    description="Albumin level (0-5)")
    su:   float = Field(..., ge=0,   le=5,    description="Sugar level (0-5)")
    bgr:  float = Field(..., ge=70,  le=500,  description="Blood glucose random (mg/dL)")
    bu:   float = Field(..., ge=1,   le=400,  description="Blood urea (mg/dL)")
    sc:   float = Field(..., ge=0.4, le=80,   description="Serum creatinine (mg/dL)")
    sod:  float = Field(..., ge=100, le=170,  description="Sodium (mEq/L)")
    pot:  float = Field(..., ge=2,   le=50,   description="Potassium (mEq/L)")
    hemo: float = Field(..., ge=3,   le=18,   description="Hemoglobin (g/dL)")
    pcv:  float = Field(..., ge=9,   le=55,   description="Packed cell volume (%)")
    wc:   float = Field(..., ge=2000,le=30000,description="White blood cell count (cells/cumm)")
    rc:   float = Field(..., ge=1,   le=9,    description="Red blood cell count (millions/cmm)")
    # Categorical — encoded as int (0/1) after LabelEncoder
    rbc:  int   = Field(..., ge=0,   le=1,    description="Red blood cells (0=abnormal, 1=normal)")
    pc:   int   = Field(..., ge=0,   le=1,    description="Pus cell (0=abnormal, 1=normal)")
    pcc:  int   = Field(..., ge=0,   le=1,    description="Pus cell clumps (0=notpresent, 1=present)")
    ba:   int   = Field(..., ge=0,   le=1,    description="Bacteria (0=notpresent, 1=present)")
    htn:  int   = Field(..., ge=0,   le=1,    description="Hypertension (0=no, 1=yes)")
    dm:   int   = Field(..., ge=0,   le=1,    description="Diabetes mellitus (0=no, 1=yes)")
    cad:  int   = Field(..., ge=0,   le=1,    description="Coronary artery disease (0=no, 1=yes)")
    appet:int   = Field(..., ge=0,   le=1,    description="Appetite (0=good, 1=poor)")
    pe:   int   = Field(..., ge=0,   le=1,    description="Pedal edema (0=no, 1=yes)")
    ane:  int   = Field(..., ge=0,   le=1,    description="Anemia (0=no, 1=yes)")

    def to_array(self):
        import numpy as np
        return np.array([
            self.age, self.bp, self.sg, self.al, self.su,
            self.rbc, self.pc, self.pcc, self.ba,
            self.bgr, self.bu, self.sc, self.sod, self.pot,
            self.hemo, self.pcv, self.wc, self.rc,
            self.htn, self.dm, self.cad, self.appet, self.pe, self.ane
        ], dtype=float)


# ── Response schemas ──────────────────────────────────────────────────────────

class TopFactor(BaseModel):
    feature:   str
    value:     float
    shap:      float
    direction: str


class PredictionResponse(BaseModel):
    disease:           str
    risk_probability:  float          = Field(..., description="Risk probability 0.0–1.0")
    risk_percent:      float          = Field(..., description="Risk probability as percentage")
    risk_level:        str            = Field(..., description="Low / Moderate / High")
    top_factors:       List[TopFactor]
    model_type:        str
    clinical_report:   Optional[str]  = None
    report_available:  bool           = False


class BatchPredictionResponse(BaseModel):
    diabetes:      Optional[PredictionResponse] = None
    heart_disease: Optional[PredictionResponse] = None
    ckd:           Optional[PredictionResponse] = None
    errors:        Dict[str, str] = {}


class HealthResponse(BaseModel):
    status:         str
    models_loaded:  List[str]
    version:        str
