"""config.py — Central configuration for the Medical Healthcare Chatbot."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"

class Settings(BaseSettings):
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL_PATH: str = str(MODELS_DIR / "llama-2-7b-chat.ggmlv3.q4_0.bin")
    OLLAMA_MODEL: str = "llama3"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = str(VECTOR_STORE_DIR)
    CHROMA_COLLECTION_PATIENTS: str = "patient_records"
    CHROMA_COLLECTION_MEDS: str = "medications"
    RAG_TOP_K: int = 5
    RAW_DATA_PATH: str = str(DATA_DIR / "Canada_Hosp1_COVID_InpatientData.xlsx")
    ADMISSION_CSV: str = str(DATA_DIR / "Preprocessed_Data-at-Admission.csv")
    DAYS_CSV: str = str(DATA_DIR / "Preprocessed_Days-Breakdown.csv")
    HOSPITAL_LOS_CSV: str = str(DATA_DIR / "Preprocessed_Hospital-LoS.csv")
    MEDICATIONS_CSV: str = str(DATA_DIR / "Preprocessed_Medications.csv")
    LOS_MODEL_PATH: str = str(MODELS_DIR / "Hospital_LoS_Model.joblib")
    INTENT_MODEL_PATH: str = str(MODELS_DIR / "intent_classifier.joblib")
    NER_MODEL_PATH: str = str(MODELS_DIR / "ner_model.joblib")
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
