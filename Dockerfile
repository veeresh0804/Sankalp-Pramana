# ─── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download ML model weights at BUILD time so runtime never touches the internet
# NOTE: TRANSFORMERS_OFFLINE must NOT be set here — we need to download now
COPY download_models.py .
RUN python download_models.py

# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy pre-downloaded HuggingFace model cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY . .
RUN mkdir -p data

# Non-root user
RUN useradd -m appuser \
    && chown -R appuser /app \
    && mkdir -p /home/appuser/.cache \
    && cp -r /root/.cache/huggingface /home/appuser/.cache/huggingface \
    && chown -R appuser /home/appuser/.cache

USER appuser

# ── Runtime env vars ─────────────────────────────────────────────────────────
# Cache paths
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
# OFFLINE MODE: weights are baked in — never call HuggingFace Hub at runtime
# This prevents the 295s rate-limit delay that caused 503 errors
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
# Cloud Run port
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]