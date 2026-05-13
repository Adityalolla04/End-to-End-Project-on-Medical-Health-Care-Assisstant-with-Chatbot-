"""
rag/pipeline.py — Core RAG pipeline for the Medical Healthcare Chatbot.
Compatible with langchain 1.x / langchain-community 0.4.x

Retrieval-Augmented Generation flow:
  User Query
    → Intent classification
    → Embed query with HuggingFace
    → ChromaDB similarity search (top-k patient records)
    → Augment prompt with retrieved context
    → LLM generates grounded, cited response
    → (Optional) Tool call: LoS Predictor if prediction intent detected
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

# ── Medical System Prompt ────────────────────────────────────────────────────
MEDICAL_SYSTEM_PROMPT = """You are a hospital AI assistant for a COVID-19 inpatient database.

STRICT RULES:
1. Answer ONLY from the retrieved patient context provided below.
2. If the context doesn't contain enough information, say "I don't have enough data on this."
3. NEVER hallucinate diagnoses, medications, or statistics.
4. Always cite the Patient ID(s) you are drawing from.
5. Recommend consulting a qualified clinician for medical decisions.

Retrieved Patient Context:
{context}

Question: {question}

Answer (with citations):"""

MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=MEDICAL_SYSTEM_PROMPT,
)


class MedicalRAGPipeline:
    """
    End-to-end RAG pipeline for the Medical Healthcare Chatbot.
    Compatible with langchain 1.x / langchain-community 0.4.x
    """

    def __init__(self):
        print("🔧 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("🗃  Connecting to ChromaDB vector store...")
        self.patient_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_PATIENTS,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        self.med_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_MEDS,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

        print("🤖 Loading LLM...")
        self.llm = self._load_llm()

        print("📋 Loading LoS predictor model...")
        try:
            self.los_model = joblib.load(settings.LOS_MODEL_PATH)
        except Exception as e:
            self.los_model = None
            print(f"⚠️  LoS model not loaded: {e}")

        print("🎯 Loading intent classifier...")
        try:
            self.intent_classifier = joblib.load(settings.INTENT_MODEL_PATH)
        except Exception:
            self.intent_classifier = None

        # Conversation history (simple list of tuples)
        self.chat_history: list[tuple[str, str]] = []

        print("✅ RAG Pipeline ready!")

    def _load_llm(self):
        """Load LLM based on LLM_PROVIDER config setting."""
        provider = settings.LLM_PROVIDER.lower()

        if provider == "local":
            from langchain_community.llms import LlamaCpp
            return LlamaCpp(
                model_path=settings.LLM_MODEL_PATH,
                n_ctx=4096,
                n_threads=4,
                temperature=0.1,
                max_tokens=512,
                verbose=False,
            )
        elif provider == "ollama":
            from langchain_community.llms import Ollama
            return Ollama(model=settings.OLLAMA_MODEL, temperature=0.1)

        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.1,
                max_tokens=1024,
            )

        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1,
                max_tokens=1024,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
            )
        elif provider == "mock":
            # Used in CI/testing only
            from unittest.mock import MagicMock
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Mock response for testing."
            return mock_llm
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider}'. "
                "Choose: local | ollama | groq | claude | openai | mock"
            )

    def classify_intent(self, query: str) -> str:
        """Classify user intent — trained classifier or rule-based fallback."""
        if self.intent_classifier is not None:
            try:
                return self.intent_classifier.predict([query])[0]
            except Exception:
                pass
        # Rule-based fallback
        q = query.lower()
        if any(w in q for w in ["predict", "how long", "stay", "los", "days in hospital"]):
            return "predict_los"
        if any(w in q for w in ["medication", "drug", "medicine", "dose", "dosage"]):
            return "medication_query"
        if any(w in q for w in ["average", "statistics", "cohort", "how many", "percent"]):
            return "analytics_query"
        return "general_medical"

    def _retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """Retrieve top-k patient docs from ChromaDB and format as context."""
        docs = self.patient_store.similarity_search(query, k=settings.RAG_TOP_K)
        context = "\n\n---\n".join(d.page_content for d in docs)
        sources = [
            {
                "patient_id": d.metadata.get("patient_id", "N/A"),
                "doc_type":   d.metadata.get("doc_type", "N/A"),
                "snippet":    d.page_content[:150] + "...",
            }
            for d in docs
        ]
        return context, sources

    def predict_los(self, features: dict) -> dict:
        """Run the LoS prediction model (R²=0.9915)."""
        if self.los_model is None:
            return {"error": "LoS prediction model not loaded"}
        try:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            prediction = self.los_model.predict(feature_vector)[0]
            return {
                "predicted_los_days": round(float(prediction), 1),
                "confidence": "High (R²=0.9915)",
                "model": "Hospital_LoS_Model.joblib",
            }
        except Exception as e:
            return {"error": str(e)}

    def search_medications(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search over the medication vector store."""
        docs = self.med_store.similarity_search(query, k=k)
        return [
            {"name": d.metadata.get("name", ""), "content": d.page_content}
            for d in docs
        ]

    def chat(self, user_query: str) -> dict:
        """
        Main RAG chat method.
        Returns: answer, intent, sources, source_count
        """
        intent = self.classify_intent(user_query)

        # 1. Retrieve relevant patient context
        context, sources = self._retrieve_context(user_query)

        # 2. Build augmented prompt
        prompt = MEDICAL_PROMPT.format(context=context, question=user_query)

        # 3. Generate response via LLM
        try:
            result = self.llm.invoke(prompt)
            # Handle both str and AIMessage return types
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            answer = f"LLM error: {e}. Please check your LLM_PROVIDER setting in .env"

        # 4. Store in simple history
        self.chat_history.append((user_query, answer))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return {
            "answer":       answer,
            "intent":       intent,
            "sources":      sources,
            "source_count": len(sources),
        }


# ── Singleton loader (lazy) ──────────────────────────────────────────────────
_pipeline: Optional[MedicalRAGPipeline] = None


def get_pipeline() -> MedicalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalRAGPipeline()
    return _pipeline
