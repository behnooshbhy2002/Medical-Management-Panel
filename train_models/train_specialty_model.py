import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
import re

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded: {df.shape[0]} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMedical specialties found ({df['medical_specialty'].nunique()} unique):")
    print(df["medical_specialty"].value_counts().head(15))
    return df


# ─────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)         # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)     # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text


def preprocess(df: pd.DataFrame, min_samples: int = 50):
    df = df.copy()
    
    # Drop rows with missing target text or label
    df = df.dropna(subset=["transcription", "medical_specialty"])
    
    # Clean transcription text
    df["transcription"] = df["transcription"].apply(clean_text)
    
    # Remove very short texts
    df = df[df["transcription"].str.len() > 50]
    
    # Keep only specialties with enough samples
    specialty_counts = df["medical_specialty"].value_counts()
    valid_specialties = specialty_counts[specialty_counts >= min_samples].index
    df = df[df["medical_specialty"].isin(valid_specialties)]
    
    print(f"\n📊 After filtering: {df.shape[0]} rows remain")
    print(f"Number of specialties: {df['medical_specialty'].nunique()}")
    
    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["medical_specialty"])
    
    X = df["transcription"]
    y = df["label"]
    
    return X, y, le


# ─────────────────────────────────────────
# 3. Train Model using Pipeline
# ─────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
   
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="saga",
            random_state=42,
            # n_jobs=-1  # بهتر است حذف شود
        ))
    ])
   
    print("\n⏳ Training model...")
    pipeline.fit(X_train, y_train)
   
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
   
    print(f"\n📈 Overall accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
   
    return pipeline


# ─────────────────────────────────────────
# 4. Save Model & Artifacts
# ─────────────────────────────────────────
def save_model(pipeline, le, output_dir: str = "../models"):
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(output_dir, "specialty_pipeline.pkl"))
    joblib.dump(le,       os.path.join(output_dir, "specialty_label_encoder.pkl"))
    
    # Save class names for later use (inference / API)
    classes = list(le.classes_)
    joblib.dump(classes, os.path.join(output_dir, "specialty_classes.pkl"))
    
    print(f"\n✅ Model artifacts saved to: {output_dir}/")
    print("   - specialty_pipeline.pkl")
    print("   - specialty_label_encoder.pkl")
    print("   - specialty_classes.pkl")
    
    print(f"\nDetectable specialties ({len(classes)}):")
    for i, c in enumerate(classes):
        print(f" {i:2d}: {c}")


# ─────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    csv_path = "D:\\University-Master\\Term-3\\Cloud Computing\\PRO\\Medical-Management-Panel\\data\\mtsamples.csv"
    
    df = load_data(csv_path)
    X, y, le = preprocess(df)
    pipeline = train(X, y)
    save_model(pipeline, le)