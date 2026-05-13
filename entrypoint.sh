#!/bin/bash
set -e
echo "============================================"
echo "  MedBI -- Medical Intelligence Platform"
echo "============================================"

# ── Default LLM provider to Groq for cloud deployment ──────────────────────
# Override by setting LLM_PROVIDER env var in HF Spaces secrets
export LLM_PROVIDER="${LLM_PROVIDER:-groq}"
echo "  LLM Provider : $LLM_PROVIDER"

# Warn early if Groq key is missing
if [ "$LLM_PROVIDER" = "groq" ] && [ -z "$GROQ_API_KEY" ]; then
  echo "  WARNING: GROQ_API_KEY is not set!"
  echo "  Set it in HF Spaces → Settings → Variables and secrets"
  echo "  Chatbot will return error messages until key is added."
fi

# Train LoS model if not present
if [ ! -f "models/Hospital_LoS_Model.joblib" ]; then
  echo "[1/3] Training LoS prediction model..."
  python3 - << 'PYEOF'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

try:
    df_adm = pd.read_csv("data/Preprocessed_Data-at-Admission.csv")
    df_los = pd.read_csv("data/Preprocessed_Hospital-LoS.csv")
    df = df_adm.merge(df_los, left_on="id", right_on="parent_id", how="inner")

    df["sex_enc"] = LabelEncoder().fit_transform(df["sex"].astype(str))
    df["icu_enc"] = (df["admission_disposition"].str.upper() == "ICU").astype(int)
    df["intubated_enc"] = (df["intubated"].astype(str).str.lower() == "yes").astype(int)

    features = ["age","sex_enc","systolic_blood_pressure","diastolic_blood_pressure",
                "heart_rate","respiratory_rate","oxygen_saturation","temperature",
                "icu_enc","intubated_enc"]

    df_clean = df[features + ["hospital_length_of_stay"]].copy()
    for col in features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean["hospital_length_of_stay"] = pd.to_numeric(
        df_clean["hospital_length_of_stay"], errors="coerce")
    df_clean = df_clean.dropna()

    X = df_clean[features].values
    y = df_clean["hospital_length_of_stay"].values

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, "models/Hospital_LoS_Model.joblib")
    print(f"  Model trained on {len(X)} samples. Saved to models/Hospital_LoS_Model.joblib")
except Exception as e:
    print(f"  Warning: Could not train model: {e}")
    print("  LoS predictor will use fallback estimates.")
PYEOF
else
  echo "[1/3] LoS model found, skipping training."
fi

# Retrain intent classifier with current sklearn version (avoids version mismatch warnings)
echo "  Retraining intent classifier for current sklearn version..."
python3 - << 'PYEOF'
import joblib, os
os.makedirs("models", exist_ok=True)
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    SAMPLES = [
        ("how long will patient stay","predict_los"),
        ("predict length of stay","predict_los"),
        ("days in hospital","predict_los"),
        ("expected discharge","predict_los"),
        ("los estimate","predict_los"),
        ("what medication","medication_query"),
        ("drug prescribed","medication_query"),
        ("dosage for patient","medication_query"),
        ("medicine given","medication_query"),
        ("which drugs","medication_query"),
        ("average age","analytics_query"),
        ("how many icu","analytics_query"),
        ("mortality rate","analytics_query"),
        ("statistics for","analytics_query"),
        ("percentage of patients","analytics_query"),
        ("what happened to patient","general_medical"),
        ("tell me about","general_medical"),
        ("symptoms of","general_medical"),
        ("diagnosis","general_medical"),
        ("comorbidities","general_medical"),
    ]
    X, y = zip(*SAMPLES)
    clf = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
    clf.fit(X, y)
    joblib.dump(clf, "models/intent_classifier.joblib")
    print("  Intent classifier retrained OK.")
except Exception as e:
    print(f"  Warning: Could not retrain intent classifier: {e}")
PYEOF

# Build vector store if not present
if [ ! -d "vector_store" ] || [ -z "$(ls -A vector_store 2>/dev/null)" ]; then
  echo "[2/3] Building RAG vector store (first run — takes ~3 minutes)..."
  python3 -m rag.ingest
else
  echo "[2/3] Vector store found, skipping ingestion."
fi

echo "[3/3] Starting FastAPI on port 7860..."
exec uvicorn deployment.api:app --host 0.0.0.0 --port 7860 --workers 1
