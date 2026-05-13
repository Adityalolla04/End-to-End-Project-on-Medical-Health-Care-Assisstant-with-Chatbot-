---
title: Medical RAG Chatbot
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
---

# 🏥 MedBI — Medical RAG Intelligence Platform

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built on **508 real COVID-19 inpatient records** from Canada Hospital 1 (2020–21).

## 🚀 Live Features

- **AI Chat** — Ask natural language questions about patient outcomes, medications, and demographics. Powered by RAG (ChromaDB + HuggingFace embeddings).
- **Power BI-Style Dashboard** — 8 interactive Chart.js charts with real patient data (KPIs, age distribution, mortality, LoS, SpO₂).
- **Patient Explorer** — Sortable, filterable table of all 508 patients with CSV export.
- **Clinical Outcomes** — Scatter plots, histograms, and mortality comparisons.
- **LoS Predictor** — Random Forest model (R²=0.99) predicting hospital length-of-stay from vitals.

## 📊 Dataset

| Metric | Value |
|--------|-------|
| Total patients | 508 |
| ICU admissions | 43 (8.5%) |
| Ward admissions | 465 (91.5%) |
| Mortality rate | 17.7% |
| Avg hospital LoS | 12.4 days |
| Male / Female | 296 / 212 |

## 🛠️ Tech Stack

- **Backend:** FastAPI + Python 3.12 + Pydantic v2
- **RAG:** LangChain + ChromaDB + HuggingFace all-MiniLM-L6-v2
- **Frontend:** Bootstrap 5 + Chart.js (dark theme)
- **ML:** Scikit-learn Random Forest LoS predictor
- **Deployment:** Docker + Hugging Face Spaces

## 🔗 API Endpoints

```
GET  /health                    → API status
POST /api/chat                  → RAG chatbot
GET  /api/analytics/summary     → Dataset statistics
GET  /api/patients              → Paginated patient list
GET  /api/patients/{id}         → Patient detail
POST /api/predict/los           → LoS prediction
GET  /api/medications/search    → Semantic drug search
WS   /ws/chat                   → Streaming WebSocket chat
```

## 💻 Run Locally

```bash
git clone https://github.com/Adityalolla04/End-to-End-Project-on-Medical-Health-Care-Assisstant-with-Chatbot-.git
cd End-to-End-Project-on-Medical-Health-Care-Assisstant-with-Chatbot-
pip install -r requirements_hf.txt
python -m rag.ingest
uvicorn deployment.api:app --port 8000
```

Open `http://localhost:8000`

## 🤖 With Ollama (local LLM)

```bash
ollama pull llama3.2:3b
ollama serve
```

The dashboard auto-detects Ollama and enables streaming responses.

---

*Built by [Aditya Srivatsav](https://www.linkedin.com/in/adityasrivatsav)*
