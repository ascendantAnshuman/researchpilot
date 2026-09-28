# ResearchPilot backend: FastAPI + LangGraph agents + ChromaDB + local MiniLM embeddings.
#
#   docker build -t researchpilot-backend ./backend
#   docker run -p 8000:8000 --env-file .env -v researchpilot-data:/data researchpilot-backend
#
# Runs ONE uvicorn worker on purpose: research runs and their live event streams
# live in this process's memory (see app/services/run_manager.py).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False \
    DATA_DIR=/data \
    PORT=8000

# DejaVu fonts give the PDF export full Unicode coverage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls ~2 GB of CUDA libraries we do not need),
# then the pinned requirements, which find torch already satisfied.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
    && pip install -r requirements.txt

# Bake the embedding model into the image so the first request does not download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app

RUN useradd --create-home --uid 1000 researchpilot \
    && mkdir -p /data \
    && chown -R researchpilot:researchpilot /data /app /opt/models
USER researchpilot

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)" || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
