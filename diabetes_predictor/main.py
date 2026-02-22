from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Diabetes Risk Predictor Microservice",
    description="Predicts diabetes risk based on patient features using a pre-trained model",
    version="1.0.0"
)

# ─────────────────────────────────────────
# Load pre-trained model and artifacts
# ─────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "diabetes_features.pkl"))
    print("✅ Diabetes model successfully loaded")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = scaler = features = None

# ─────────────────────────────────────────
# Data mappings & schemas
# ─────────────────────────────────────────
GENDER_MAP = {"male": 1, "female": 0, "other": 2}
SMOKING_MAP = {
    "never": 0,
    "no info": 1,
    "former": 2,
    "current": 3,
    "ever": 4,
    "not current": 5
}

class PatientInput(BaseModel):
    gender: str = Field(..., example="Male")
    age: float = Field(..., example=45.0)
    hypertension: int = Field(..., example=0)
    heart_disease: int = Field(..., example=0)
    smoking_history: str = Field(..., example="never")
    bmi: float = Field(..., example=27.5)
    hba1c_level: float = Field(..., example=5.8)
    blood_glucose_level: float = Field(..., example=120.0)


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def encode_input(data: PatientInput) -> np.ndarray:
    """Convert input data to model-compatible feature vector"""
    gender_enc = GENDER_MAP.get(data.gender.lower(), 2)
    smoking_enc = SMOKING_MAP.get(data.smoking_history.lower(), 1)

    return np.array([[
        gender_enc,
        data.age,
        data.hypertension,
        data.heart_disease,
        smoking_enc,
        data.bmi,
        data.hba1c_level,
        data.blood_glucose_level,
    ]])


def get_risk_level(prob: float) -> str:
    """Determine risk category based on probability"""
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Very High"


def get_risk_factors(data: PatientInput, prob: float) -> list[str]:
    """Identify main contributing risk factors"""
    factors = []

    if data.blood_glucose_level > 200:
        factors.append("High blood glucose")
    if data.hba1c_level > 6.5:
        factors.append("Elevated HbA1c (indicative of diabetes)")
    if data.bmi > 30:
        factors.append("Obesity (BMI > 30)")
    if data.hypertension:
        factors.append("Hypertension")
    if data.heart_disease:
        factors.append("Heart disease")
    if data.age > 60:
        factors.append("Age over 60")
    if data.smoking_history.lower() == "current":
        factors.append("Current smoker")

    return factors


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Diabetes Risk Predictor",
        "status": "operational"
    }


@app.post("/predict")
def predict(data: PatientInput):
    """
    Predict diabetes risk probability and return risk assessment
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    X = encode_input(data)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0][1])  # probability of diabetes (class 1)

    risk_level = get_risk_level(prob)
    risk_factors = get_risk_factors(data, prob)

    recommendation = (
        "Urgent consultation with an endocrinologist is strongly recommended"
        if prob > 0.60
        else "Periodic screening and lifestyle monitoring recommended"
    )

    return {
        "diabetes_probability": round(prob, 4),
        "diabetes_probability_percent": f"{prob * 100:.1f}%",
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "note": "This is a screening tool only — not a medical diagnosis"
    }


@app.get("/health")
def health():
    """Health check endpoint for container orchestration"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR
    }