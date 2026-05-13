"""
rag/ingest.py — Build the ChromaDB vector store from your medical datasets.
What it does:
  1. Loads Preprocessed_Data-at-Admission.csv  -> patient summary documents
  2. Loads Preprocessed_Hospital-LoS.csv       -> outcome documents
  3. Loads Preprocessed_Medications.csv        -> drug knowledge base
  4. Embeds all into ChromaDB for RAG retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Use the new langchain-huggingface package (fixes deprecated import)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings


def make_patient_document(row_admission: dict, row_los: dict) -> Document:
    """Convert a patient admission row + LoS row into a LangChain Document."""
    pid = row_admission.get("id", "unknown")

    content = (
        f"Patient ID: {pid}\n"
        f"Age: {row_admission.get('age', 'N/A')} | Sex: {row_admission.get('sex', 'N/A')}\n"
        f"Reason for Admission: {row_admission.get('reason_for_admission', 'N/A')}\n"
        f"Comorbidities: {row_admission.get('comorbidities', 'N/A')}\n"
        f"Admission Disposition: {row_admission.get('admission_disposition', 'N/A')}\n"
        f"Medications: {row_admission.get('medications', 'N/A')}\n"
        f"Vitals - BP: {row_admission.get('systolic_blood_pressure','N/A')}/"
        f"{row_admission.get('diastolic_blood_pressure','N/A')} mmHg\n"
        f"Vitals - HR: {row_admission.get('heart_rate','N/A')} bpm | "
        f"RR: {row_admission.get('respiratory_rate','N/A')} | "
        f"SpO2: {row_admission.get('oxygen_saturation','N/A')}%\n"
        f"Vitals - Temp: {row_admission.get('temperature','N/A')} C | "
        f"Intubated: {row_admission.get('intubated','N/A')}\n"
        f"COVID Vaccinated: {row_admission.get('received_covid_vaccine','N/A')}"
    )

    if row_los:
        content += (
            f"\nHospital Length of Stay: {row_los.get('hospital_length_of_stay','N/A')} days\n"
            f"ICU Length of Stay: {row_los.get('icu_length_of_stay','N/A')} days\n"
            f"Mechanical Ventilation Duration: {row_los.get('time_on_mechanical_ventilation','N/A')} days\n"
            f"Patient Expired: {row_los.get('did_the_patient_expire_in_hospital','N/A')}"
        )

    metadata = {
        "patient_id": str(pid),
        "age": str(row_admission.get("age", "")),
        "sex": str(row_admission.get("sex", "")),
        "admission_disposition": str(row_admission.get("admission_disposition", "")),
        "comorbidities": str(row_admission.get("comorbidities", ""))[:200],
        "doc_type": "patient_record",
    }
    return Document(page_content=content, metadata=metadata)


def make_medication_document(row: dict) -> Document:
    """Convert a medication row into a LangChain Document."""
    name = str(row.get("name", "unknown"))
    content = f"Medication: {name}"
    metadata = {"doc_type": "medication", "name": name[:200]}
    return Document(page_content=content, metadata=metadata)


def build_vector_store():
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Loading patient admission data...")
    df_admission = pd.read_csv(settings.ADMISSION_CSV)
    df_los       = pd.read_csv(settings.HOSPITAL_LOS_CSV)
    df_meds      = pd.read_csv(settings.MEDICATIONS_CSV)

    # Index LoS by patient_id for fast lookup
    if "parent_id" in df_los.columns:
        los_index = df_los.set_index("parent_id").to_dict("index")
    elif "id" in df_los.columns:
        los_index = df_los.set_index("id").to_dict("index")
    else:
        los_index = {}

    print(f"Building {len(df_admission)} patient documents...")
    patient_docs = []
    for _, row in df_admission.iterrows():
        pid     = row.get("id")
        los_row = los_index.get(pid)
        doc     = make_patient_document(row.to_dict(), los_row)
        patient_docs.append(doc)

    print("Building medication documents (sampling up to 5000)...")
    df_meds_sample = df_meds.drop_duplicates(subset=["name"]).head(5000) if "name" in df_meds.columns else df_meds.head(5000)
    med_docs = [make_medication_document(row.to_dict()) for _, row in df_meds_sample.iterrows()]

    print(f"Embedding {len(patient_docs)} patient records into ChromaDB...")
    Chroma.from_documents(
        documents=patient_docs,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION_PATIENTS,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    print(f"Embedding {len(med_docs)} medications into ChromaDB...")
    Chroma.from_documents(
        documents=med_docs,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION_MEDS,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    print(f"\nVector store built successfully!")
    print(f"  Patient records : {len(patient_docs)} docs -> {settings.CHROMA_PERSIST_DIR}")
    print(f"  Medications     : {len(med_docs)} docs  -> {settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    build_vector_store()
