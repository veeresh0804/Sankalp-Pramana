"""
vector_search/embedding_index.py
Semantic similarity search using SentenceTransformers + FAISS.

The index is built ONCE at app startup via initialize_index().
Requests never trigger index building — eliminating Cloud Run 503 timeouts.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K_CANDIDATES,
)

logger = logging.getLogger(__name__)

# Globals — populated by initialize_index() at startup
_model    = None   # SentenceTransformer
_index    = None   # faiss.IndexFlatIP
_metadata: List[Dict[str, Any]] = []
_ready    = False  # True once initialize_index() completes


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[VectorSearch] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _build_default_index():
    """Build a FAISS index from the mock dataset when no pre-built index exists."""
    import faiss

    from retrieval.search_models import _MOCK_DATASET  # reuse mock catalogue

    sent_model = _load_model()

    texts = [f"{m['name']} {' '.join(m['tags'])}" for m in _MOCK_DATASET]
    embeddings = sent_model.encode(texts, normalize_embeddings=True).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product ≈ cosine on normalised vecs
    index.add(embeddings)

    meta = [{"name": m["name"], "url": m["url"], "format": m["format"],
             "quality": m["quality"]} for m in _MOCK_DATASET]

    return index, meta


def _ensure_index():
    global _index, _metadata

    if _index is not None:
        return

    index_path = Path(FAISS_INDEX_PATH)
    meta_path  = Path(METADATA_PATH)

    if index_path.exists() and meta_path.exists():
        import faiss
        logger.info("[VectorSearch] Loading pre-built FAISS index from disk")
        _index    = faiss.read_index(str(index_path))
        with meta_path.open("r") as f:
            _metadata = json.load(f)
    else:
        logger.info("[VectorSearch] No pre-built index found — building from default dataset")
        _index, _metadata = _build_default_index()

        # Optionally persist for next run
        try:
            import faiss
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(_index, str(index_path))
            with meta_path.open("w") as f:
                json.dump(_metadata, f, indent=2)
            logger.info("[VectorSearch] FAISS index saved to disk")
        except Exception as e:
            logger.warning(f"[VectorSearch] Could not save index: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Public startup initialiser — called ONCE from main.py lifespan
# ──────────────────────────────────────────────────────────────────────────────

def initialize_index() -> None:
    """
    Eagerly load the embedding model and build the FAISS index.
    Call this ONCE at app startup (FastAPI lifespan) — never inside a request.
    """
    global _model, _index, _metadata, _ready

    if _ready:
        logger.info("[VectorSearch] Index already initialised — skipping")
        return

    logger.info("[VectorSearch] ► Starting index initialisation...")
    _ensure_index()
    _ready = True
    logger.info(f"[VectorSearch] ✓ Index ready — {len(_metadata)} entries embedded")

# ──────────────────────────────────────────────────────────────────────────────
# Request-time search (index guaranteed ready by startup)
# ──────────────────────────────────────────────────────────────────────────────

def semantic_search(concept: str, top_k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Return up to `top_k` model candidates ranked by semantic similarity.
    Requires initialize_index() to have been called at app startup.
    Each result: {name, url, format, quality, vector_score}
    """
    if not _ready:
        logger.warning("[VectorSearch] Index not ready — returning empty (startup still in progress?)")
        return []

    try:
        query_vec = _model.encode([concept], normalize_embeddings=True).astype("float32")
        scores, indices = _index.search(query_vec, min(top_k, len(_metadata)))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(_metadata[idx])
            entry["vector_score"] = float(score)
            results.append(entry)

        logger.info(f"[VectorSearch] {len(results)} results for '{concept}'")
        return results

    except Exception as e:
        logger.error(f"[VectorSearch] Error during search: {e}")
        return []
