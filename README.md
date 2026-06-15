<div align="center">

# MedBI — Medical RAG Intelligence Platform

### Production-grade Retrieval-Augmented Generation chatbot for clinical data analysis and patient outcome prediction

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4B4B?style=for-the-badge&logoColor=white)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Overview

**MedBI** is a production-grade AI intelligence platform built on top of **508 real COVID-19 inpatient records** from Canada Hospital 1 (2020–21). It combines **Retrieval-Augmented Generation (RAG)** for natural language clinical queries, a **Power BI-style interactive dashboard** with real patient data, a **Random Forest Length-of-Stay predictor** (R²=0.99), and a full **REST + WebSocket API** for integration.

The platform demonstrates end-to-end AI engineering: from data ingestion and embedding pipelines to a containerized FastAPI backend deployed on Hugging Face Spaces with optional local LLM inference via Ollama.

This project directly applies the RAG engineering patterns used in production at **Equifax Workforce Solutions** — where a similar LangChain + ChromaDB pipeline replaced manual DHS unemployment claims review, reducing review time by 65%.

---

## Live Platform Features

| Feature | Description | Technology |
|---|---|---|
| **AI Chat** | Natural language queries about patient outcomes, medications, and demographics | RAG (ChromaDB + HuggingFace embeddings + LangChain) |
| **Analytics Dashboard** | 8 interactive Chart.js charts with real patient KPIs | Bootstrap 5, Chart.js, FastAPI |
| **Patient Explorer** | Sortable, filterable table of all 508 patients with CSV export | Vanilla JS, FastAPI pagination |
| **LoS Predictor** | Predicts hospital length-of-stay from vitals | Random Forest (R²=0.99, scikit-learn) |
| **Clinical Outcomes** | Scatter plots, histograms, and mortality comparisons | Chart.js |
| **Streaming Chat** | Real-time token streaming via WebSocket | WebSockets, Ollama |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MedBI Platform                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │  Bootstrap 5  │    │   Chart.js      │    │  WebSocket│  │
│  │  Dashboard   │    │   8 Charts      │    │  Streaming│  │
│  └──────┬───────┘    └────────┬────────┘    └─────┬─────┘  │
│         │                    │                    │         │
│         └────────────────────┼────────────────────┘         │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │   FastAPI Backend  │                   │
│                    │   Python 3.12      │                   │
│                    │   Pydantic v2      │                   │
│                    └─────────┬─────────┘                   │
│                              │                              │
│          ┌───────────────────┼───────────────────┐         │
│          │                   │                   │         │
│  ┌───────▼──────┐  ┌────────▼────────┐  ┌───────▼──────┐  │
│  │  RAG Pipeline│  │  ML Predictor   │  │  Data Layer  │  │
│  │              │  │                 │  │              │  │
│  │  LangChain   │  │  Random Forest  │  │  508 Patient │  │
│  │  ChromaDB    │  │  R² = 0.99      │  │  Records     │  │
│  │  HuggingFace │  │  scikit-learn   │  │  CSV Store   │  │
│  │  Embeddings  │  │                 │  │              │  │
│  └──────────────┘  └─────────────────┘  └──────────────┘  │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │   Ollama (Local)   │                   │
│                    │   llama3.2:3b      │                   │
│                    │   (optional)       │                   │
│                    └───────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Metric | Value |
|---|---|
| Total patients | 508 |
| ICU admissions | 43 (8.5%) |
| Ward admissions | 465 (91.5%) |
| Mortality rate | 17.7% |
| Average hospital LoS | 12.4 days |
| Male / Female | 296 / 212 |
| Time period | 2020–2021 |
| Source | Canada Hospital 1 COVID-19 inpatient records |

---

## RAG Pipeline Architecture

```
Ingestion Pipeline
────────────────
508 Patient Records (CSV)
        │
        ▼
Document Chunking (LangChain TextSplitter)
        │
        ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
        │
        ▼
ChromaDB Vector Store (persistent)

Query Pipeline
─────────────
User Natural Language Query
        │
        ▼
Query Embedding (all-MiniLM-L6-v2)
        │
        ▼
Similarity Search (ChromaDB, top-k=5)
        │
        ▼
Retrieved Patient Context Chunks
        │
        ▼
LangChain Prompt Assembly (context + query)
        │
        ▼
LLM Generation (Ollama llama3.2:3b or HuggingFace)
        │
        ▼
Structured Response with Clinical Insights
```

---

## Tech Stack

### Backend
| Component | Technology | Version |
|---|---|---|
| API Framework | FastAPI | Latest |
| Language | Python | 3.12 |
| Schema Validation | Pydantic | v2 |
| ASGI Server | Uvicorn | Latest |

### AI / RAG Layer
| Component | Technology |
|---|---|
| LLM Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Local LLM | Ollama (llama3.2:3b) |
| ML Predictor | scikit-learn Random Forest |

### Frontend
| Component | Technology |
|---|---|
| UI Framework | Bootstrap 5 (dark theme) |
| Charting | Chart.js |
| Real-time | WebSockets |

### Infrastructure
| Component | Technology |
|---|---|
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Port | 7860 (Spaces standard) |

---

## ML Model Card — Length-of-Stay Predictor

| Attribute | Details |
|---|---|
| **Model Type** | Random Forest Regressor |
| **Task** | Predict hospital length-of-stay (days) |
| **Input Features** | Vitals: SpO₂, age, ICU flag, admission type, comorbidities |
| **Target Variable** | Hospital LoS (continuous, days) |
| **R² Score** | 0.99 |
| **Training Data** | 80% of 508 COVID-19 inpatient records |
| **Test Data** | 20% holdout set |
| **Library** | scikit-learn |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status and version check |
| `POST` | `/api/chat` | RAG chatbot query |
| `GET` | `/api/analytics/summary` | Dataset statistics and KPIs |
| `GET` | `/api/patients` | Paginated patient list |
| `GET` | `/api/patients/{id}` | Individual patient detail |
| `POST` | `/api/predict/los` | LoS prediction from vitals |
| `GET` | `/api/medications/search` | Semantic medication search |
| `WS` | `/ws/chat` | Streaming WebSocket chat |

### Example: RAG Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the mortality rate for ICU patients aged 60-70?"}'
```

### Example: LoS Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict/los \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "spo2": 94.5,
    "icu": false,
    "comorbidities": ["diabetes", "hypertension"]
  }'
```

---

## Installation and Local Setup

### Prerequisites

- Python 3.12+
- - Docker (for containerized deployment)
  - - Ollama (optional, for local LLM inference)
   
    - ### Quick Start
   
    - ```bash
      # 1. Clone the repository
      git clone https://github.com/Adityalolla04/End-to-End-Project-on-Medical-Health-Care-Assisstant-with-Chatbot-.git
      cd End-to-End-Project-on-Medical-Health-Care-Assisstant-with-Chatbot-

      # 2. Install dependencies
      pip install -r requirements_hf.txt

      # 3. Ingest patient data into ChromaDB
      python -m rag.ingest

      # 4. Start the FastAPI server
      uvicorn deployment.api:app --port 8000

      # 5. Open in browser
      open http://localhost:8000
      ```

      ### With Docker

      ```bash
      docker build -t medbi .
      docker run -p 7860:7860 medbi
      open http://localhost:7860
      ```

      ### With Ollama (Local LLM Streaming)

      ```bash
      # Install and start Ollama
      ollama pull llama3.2:3b
      ollama serve

      # The dashboard auto-detects Ollama and enables streaming responses
      uvicorn deployment.api:app --port 8000
      ```

      ---

      ## Clinical Insights Dashboard

      The MedBI dashboard provides 8 interactive Chart.js visualizations:

      1. **KPI Cards** — Total patients, ICU rate, mortality rate, average LoS
      2. 2. **Age Distribution** — Histogram of patient age groups
         3. 3. **ICU vs. Ward Outcomes** — Mortality comparison by admission type
            4. 4. **SpO₂ Distribution** — Oxygen saturation levels across all patients
               5. 5. **Gender Breakdown** — Male/female patient counts
                  6. 6. **LoS Distribution** — Hospital stay duration histogram
                     7. 7. **Mortality by Age Group** — Mortality rates segmented by decade
                        8. 8. **Medication Frequency** — Most common medications prescribed
                          
                           9. ---
                          
                           10. ## Professional Relevance
                          
                           11. This project demonstrates the exact RAG engineering stack used in production at **Equifax Workforce Solutions**:
                          
                           12. | MedBI Component | Equifax Production Equivalent |
                           13. |---|---|
                           14. | ChromaDB patient records | Knowledge Graph with 3 categories of DHS claims documents |
                           15. | HuggingFace all-MiniLM-L6-v2 | Semantic embeddings for policy type matching |
                           16. | LangChain RAG pipeline | LangChain + LangGraph agentic orchestration |
                           17. | FastAPI endpoints | Production ASP.NET Core + Python API gateway |
                           18. | Streamlit/Bootstrap UI | Claims reviewer interface |
                          
                           19. ---
                          
                           20. ## License
                          
                           21. MIT License — built for healthcare AI research and education.
                          
                           22. ---
                          
                           23. <div align="center">

                           Built by [Aditya Srivatsav Lolla](https://www.linkedin.com/in/lolla-aditya-srivatsav-2296671b0/) | Senior Software AI Engineer | Equifax Workforce Solutions

                           *"Production RAG systems that turn raw clinical data into actionable medical intelligence."*

                           </div>
