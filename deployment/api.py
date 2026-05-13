"""deployment/api.py — FastAPI backend for the Medical Healthcare Chatbot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import settings

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Healthcare Chatbot API",
    description="RAG-powered medical assistant for COVID-19 inpatient data",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import get_pipeline as _gp
        _pipeline = _gp()
    return _pipeline

# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list
    source_count: int
    timestamp: str

class LoSPredictRequest(BaseModel):
    age: float
    sex: int
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    heart_rate: float
    respiratory_rate: float
    oxygen_saturation: float
    temperature: float
    icu_admission: int
    intubated: int

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def serve_index():
    """Serve the main dashboard UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "MedBI API is running. Visit /docs for API reference."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    pipeline = get_pipeline()
    result = pipeline.chat(req.message)
    return ChatResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=result["sources"],
        source_count=result["source_count"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    pipeline = get_pipeline()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            result = pipeline.chat(payload.get("message", ""))
            for word in result["answer"].split():
                await websocket.send_json({"token": word + " "})
            await websocket.send_json({
                "done": True,
                "intent": result["intent"],
                "sources": result["sources"],
                "source_count": result["source_count"],
            })
    except WebSocketDisconnect:
        pass

@app.post("/api/predict/los")
def predict_los(req: LoSPredictRequest):
    pipeline = get_pipeline()
    return pipeline.predict_los(req.model_dump())

@app.get("/api/analytics/summary")
def analytics_summary():
    try:
        df_adm = pd.read_csv(settings.ADMISSION_CSV)
        df_los = pd.read_csv(settings.HOSPITAL_LOS_CSV)
        total = len(df_adm)
        icu = int((df_adm["admission_disposition"].str.upper() == "ICU").sum()) if "admission_disposition" in df_adm.columns else 0
        ward = int((df_adm["admission_disposition"].str.upper() == "WARD").sum()) if "admission_disposition" in df_adm.columns else 0
        avg_los = round(float(pd.to_numeric(df_los["hospital_length_of_stay"], errors="coerce").mean()), 1) if "hospital_length_of_stay" in df_los.columns else None
        avg_icu = round(float(pd.to_numeric(df_los["icu_length_of_stay"], errors="coerce").mean()), 1) if "icu_length_of_stay" in df_los.columns else None
        mortality = None
        if "did_the_patient_expire_in_hospital" in df_los.columns:
            expired = (df_los["did_the_patient_expire_in_hospital"].astype(str).str.lower() == "yes").sum()
            mortality = round(float(expired) / total * 100, 1)
        return {
            "total_patients": total,
            "icu_admissions": icu,
            "ward_admissions": ward,
            "icu_pct": round(icu / total * 100, 1) if total else 0,
            "avg_hospital_los_days": avg_los,
            "avg_icu_los_days": avg_icu,
            "mortality_pct": mortality,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients")
def list_patients(
    disposition: Optional[str] = Query(None),
    sex: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        df = pd.read_csv(settings.ADMISSION_CSV)
        if disposition:
            df = df[df["admission_disposition"].str.upper() == disposition.upper()]
        if sex:
            df = df[df["sex"].str.capitalize() == sex.capitalize()]
        total = len(df)
        start = (page - 1) * page_size
        cols = [c for c in ["id","age","sex","reason_for_admission","comorbidities","admission_disposition"] if c in df.columns]
        records = df.iloc[start:start + page_size][cols].fillna("").to_dict("records")
        return {"total": total, "page": page, "page_size": page_size, "patients": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int):
    try:
        df_adm = pd.read_csv(settings.ADMISSION_CSV)
        df_los = pd.read_csv(settings.HOSPITAL_LOS_CSV)
        adm_row = df_adm[df_adm["id"] == patient_id]
        if adm_row.empty:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        los_col = "parent_id" if "parent_id" in df_los.columns else "id"
        los_row = df_los[df_los[los_col] == patient_id]
        return {
            "patient_id": patient_id,
            "admission": adm_row.iloc[0].fillna("").to_dict(),
            "outcome": los_row.iloc[0].fillna("").to_dict() if not los_row.empty else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/medications/search")
def search_medications(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    pipeline = get_pipeline()
    results = pipeline.search_medications(q, k=k)
    return {"query": q, "results": results}
