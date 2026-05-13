#!/bin/bash
set -e
echo "============================================"
echo "  MedBI -- Medical Intelligence Platform"
echo "============================================"

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

# Build vector store if not present
if [ ! -d "vector_store" ] || [ -z "$(ls -A vector_store 2>/dev/null)" ]; then
  echo "[2/3] Building RAG vector store (first run — takes ~3 minutes)..."
  python3 -m rag.ingest
else
  echo "[2/3] Vector store found, skipping ingestion."
fi

echo "[3/3] Starting FastAPI on port 7860..."
exec uvicorn deployment.api:app --host 0.0.0.0 --port 7860 --workers 1
