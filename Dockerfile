FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy project (exclude venv and large cache dirs)
COPY . .

# Pre-build the vector store at container startup
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Build vector store on first run, then start API
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 7860

CMD ["./entrypoint.sh"]
