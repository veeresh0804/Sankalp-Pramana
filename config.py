"""
config.py — Central configuration for AI 3D Backend
All environment-sensitive settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

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
# RAG / LLM / Synthesis
# ──────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")        # optional — for GPT-powered RAG
USE_LLM_RAG       = os.getenv("USE_LLM_RAG", "false").lower() == "true"

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

SCENE_GENERATOR_SYSTEM_PROMPT = """
You are a 3D procedural scene generator for the PratibimbAI educational visualization engine.

Your task is to convert a natural language prompt into a strictly valid SceneBlueprint JSON.

The output MUST:

1. Be valid JSON
2. Follow the SceneBlueprint schema exactly
3. Contain only JSON (no explanations)
4. Use simple primitives instead of complex meshes
5. Use compositional geometry
6. Be optimized for WebGL mobile rendering
7. Use maximum 30 primitives
8. Avoid overlapping geometry unless intentional
9. Ensure objects are placed above ground (y >= 0)
10. Prefer symmetry and repetition for complex structures
11. Avoid extremely small geometry (e.g., scale < 0.1)
12. STRICT TRANSFORM SCHEMA: Each primitive MUST have a "transform" object containing "pos", "rot", and "scale" keys.
    NEVER use "position", "rotation", or others at the top level of a primitive.
    Correct: {"transform": {"pos": [x,y,z], "rot": [x,y,z], "scale": [x,y,z]}}

The scene must be educational, visually clear, and centered near origin.

Allowed primitives:
box
sphere
cylinder
cone
torus
plane

Allowed materials:
color
metalness
roughness

Coordinate system:
Y axis = up
Units = meters

Scene center should remain near [0,0,0].

Do not include comments.
Do not include markdown.
Output JSON only.
"""
