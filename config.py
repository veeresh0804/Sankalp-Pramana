"""
config.py — Central configuration for AI 3D Backend
All environment-sensitive settings live here.
"""

import os

# ──────────────────────────────────────────────
# API metadata
# ──────────────────────────────────────────────
APP_TITLE       = "AI 3D Model Retrieval API"
APP_VERSION     = "1.0.0"
APP_DESCRIPTION = (
    "Semantic 3D model retrieval using vector search, "
    "CLIP validation, ranking, and RAG explanation."
)

# ──────────────────────────────────────────────
# Vector / embedding settings
# ──────────────────────────────────────────────
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH  = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
METADATA_PATH     = os.getenv("METADATA_PATH",    "data/model_metadata.json")

# ──────────────────────────────────────────────
# CLIP settings
# ──────────────────────────────────────────────
CLIP_MODEL_NAME   = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED   = os.getenv("CLIP_PRETRAINED", "openai")
CLIP_THRESHOLD    = float(os.getenv("CLIP_THRESHOLD", "0.25"))

# ──────────────────────────────────────────────
# Retrieval / ranking
# ──────────────────────────────────────────────
TOP_K_CANDIDATES  = int(os.getenv("TOP_K_CANDIDATES", "10"))
MIN_CONFIDENCE    = float(os.getenv("MIN_CONFIDENCE",  "0.3"))
FALLBACK_ENABLED  = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"

# Comma-separated list of sources to query, in priority order.
# Options: objaverse, sketchfab, polyhaven, mock
# Objaverse downloads ~200 MB metadata on first run — enable deliberately.
RETRIEVAL_SOURCES = os.getenv("RETRIEVAL_SOURCES", "sketchfab,polyhaven,mock")

# ──────────────────────────────────────────────
# Sketchfab API (free token from sketchfab.com)
# ──────────────────────────────────────────────
SKETCHFAB_API_TOKEN = os.getenv("SKETCHFAB_API_TOKEN", "")

# ──────────────────────────────────────────────
# Google Cloud Storage
# ──────────────────────────────────────────────
GCS_BUCKET_NAME   = os.getenv("GCS_BUCKET_NAME", "ai-3d-models-bucket")
GCS_PROJECT_ID    = os.getenv("GCS_PROJECT_ID",  "your-gcp-project-id")

# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "3600"))   # 1 hour

# ──────────────────────────────────────────────
# RAG / LLM
# ──────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")        # optional — for GPT-powered RAG
USE_LLM_RAG       = os.getenv("USE_LLM_RAG", "false").lower() == "true"
