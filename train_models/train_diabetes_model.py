import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df


# ─────────────────────────────────────────
# 2. Preprocess Data
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()
    
    # Encode gender
    le_gender = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"])  # Female=0, Male=1, Other=2
    
    # Encode smoking_history
    smoking_map = {
        "never": 0,
        "No Info": 1,
        "former": 2,
        "current": 3,
        "ever": 4,
        "not current": 5,
    }
    df["smoking_history"] = df["smoking_history"].map(smoking_map).fillna(1)
    
    # Features and target
    feature_cols = [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
    ]
    
    # Make sure column names are lowercase
    df.columns = [c.lower() for c in df.columns]
    feature_cols = [c.lower() for c in feature_cols]
    
    X = df[feature_cols]
    y = df["diabetes"]
    
    print(f"\n📊 Target distribution:\n{y.value_counts()}")
    print(f"\nDiabetes positive rate: {y.mean()*100:.2f}%")
    
    return X, y, feature_cols


# ─────────────────────────────────────────
# 3. Train Model
# ─────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",       # Because the dataset is imbalanced
        random_state=42,
        C=1.0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📈 Model Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler


# ─────────────────────────────────────────
# 4. Save Model & Artifacts
# ─────────────────────────────────────────
def save_model(model, scaler, feature_cols, output_dir: str = "../models"):
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model,        os.path.join(output_dir, "diabetes_model.pkl"))
    joblib.dump(scaler,       os.path.join(output_dir, "diabetes_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "diabetes_features.pkl"))
    
    print(f"\n✅ Model artifacts saved to: {output_dir}/")
    print("   - diabetes_model.pkl")
    print("   - diabetes_scaler.pkl")
    print("   - diabetes_features.pkl")


# ─────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\University-Master\\Term-3\\Cloud Computing\\PRO\\Medical-Management-Panel\\data\\diabetes_prediction_dataset.csv"
    
    df = load_data(csv_path)
    X, y, feature_cols = preprocess(df)
    model, scaler = train(X, y)
    save_model(model, scaler, feature_cols)