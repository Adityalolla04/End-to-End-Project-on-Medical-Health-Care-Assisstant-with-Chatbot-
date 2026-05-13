# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps (needed for llama-cpp-python, spaCy, chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# Download spaCy model
RUN /install/bin/python -m spacy download en_core_web_sm


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy entire project
COPY . .

# Create vector store directory
RUN mkdir -p /app/vector_store

# Environment variables (override in docker-compose / deployment)
ENV LLM_PROVIDER=ollama \
    OLLAMA_MODEL=llama3 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000 8501

# Default: start FastAPI backend
# Override CMD in docker-compose for Streamlit
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
