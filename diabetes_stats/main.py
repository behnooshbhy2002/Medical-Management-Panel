from fastapi import FastAPI
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Diabetes Statistical Analyzer",
    description="Provides statistical insights and aggregations from diabetes patient dataset",
    version="1.0.0"
)

DATA_PATH = os.getenv("DATA_PATH", "/app/data/diabetes_prediction_dataset.csv")

# ─────────────────────────────────────────
# Data loading (cached on first access)
# ─────────────────────────────────────────
_df: pd.DataFrame = None

def get_df() -> pd.DataFrame:
    """
    Load dataset once and cache it in memory.
    Falls back to synthetic random data if real file is not found.
    """
    global _df
    if _df is None:
        if not os.path.exists(DATA_PATH):
            # Fallback: generate synthetic data for development/testing
            np.random.seed(42)
            n = 500
            _df = pd.DataFrame({
                "age": np.random.randint(20, 80, n),
                "bmi": np.random.uniform(18, 45, n),
                "hba1c_level": np.random.uniform(4, 9, n),
                "blood_glucose_level": np.random.uniform(80, 300, n),
                "hypertension": np.random.randint(0, 2, n),
                "heart_disease": np.random.randint(0, 2, n),
                "diabetes": np.random.randint(0, 2, n),
                "gender": np.random.choice(["Male", "Female"], n),
                "smoking_history": np.random.choice(["never", "former", "current"], n),
            })
        else:
            _df = pd.read_csv(DATA_PATH)
            # Normalize column names
            _df.columns = [c.lower().strip() for c in _df.columns]
    
    return _df


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Diabetes Statistical Analyzer",
        "status": "operational"
    }


@app.get("/stats")
def get_stats():
    """
    Comprehensive statistical summary comparing diabetic vs non-diabetic patients
    """
    df = get_df()
    
    diabetic = df[df["diabetes"] == 1]
    healthy = df[df["diabetes"] == 0]
    
    return {
        "overview": {
            "total_patients": len(df),
            "diabetic_count": len(diabetic),
            "healthy_count": len(healthy),
            "diabetes_prevalence_percent": round(len(diabetic) / len(df) * 100, 2),
        },
        "bmi": {
            "diabetic_mean": round(float(diabetic["bmi"].mean()), 2),
            "healthy_mean": round(float(healthy["bmi"].mean()), 2),
            "overall_mean": round(float(df["bmi"].mean()), 2),
            "diabetic_obesity_percent": round(float((diabetic["bmi"] > 30).mean() * 100), 2),
            "diabetic_severe_obesity_percent": round(float((diabetic["bmi"] > 35).mean() * 100), 2),
        },
        "hba1c": {
            "diabetic_mean": round(float(diabetic["hba1c_level"].mean()), 2),
            "healthy_mean": round(float(healthy["hba1c_level"].mean()), 2),
            "diabetic_above_threshold_percent": round(float((diabetic["hba1c_level"] > 6.5).mean() * 100), 2),
        },
        "blood_glucose": {
            "diabetic_mean": round(float(diabetic["blood_glucose_level"].mean()), 2),
            "healthy_mean": round(float(healthy["blood_glucose_level"].mean()), 2),
        },
        "age": {
            "diabetic_mean_age": round(float(diabetic["age"].mean()), 1),
            "healthy_mean_age": round(float(healthy["age"].mean()), 1),
            "age_distribution": {
                "under_30": int((df["age"] < 30).sum()),
                "30_to_49": int(((df["age"] >= 30) & (df["age"] < 50)).sum()),
                "50_to_64": int(((df["age"] >= 50) & (df["age"] < 65)).sum()),
                "65_and_above": int((df["age"] >= 65).sum()),
            }
        },
        "comorbidities": {
            "diabetic_hypertension_percent": round(float(diabetic["hypertension"].mean() * 100), 2),
            "diabetic_heart_disease_percent": round(float(diabetic["heart_disease"].mean() * 100), 2),
        },
        "gender_distribution": {
            "diabetic_by_gender": diabetic["gender"].value_counts().to_dict(),
            "overall_by_gender": df["gender"].value_counts().to_dict(),
        },
        "smoking": {
            "diabetic_smoking_current_percent": round(
                float((diabetic["smoking_history"] == "current").mean() * 100), 2
            ),
        }
    }


@app.get("/stats/age-groups")
def age_group_stats():
    """
    Diabetes prevalence broken down by age groups
    """
    df = get_df()
    
    bins = [0, 30, 45, 60, 75, 120]
    labels = ["<30", "30-44", "45-59", "60-74", "75+"]
    
    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )
    
    result = (
        df.groupby("age_group", observed=True)["diabetes"]
        .agg(["mean", "count"])
        .reset_index()
    )
    
    result["mean"] = (result["mean"] * 100).round(2)
    result = result.rename(columns={"mean": "diabetes_prevalence_percent"})
    
    return result.to_dict(orient="records")


@app.get("/health")
def health():
    """Simple health check endpoint"""
    df = get_df()
    return {
        "status": "ok",
        "data_loaded": _df is not None,
        "row_count": len(df) if df is not None else 0,
        "data_source": "synthetic" if not os.path.exists(DATA_PATH) else "real"
    }