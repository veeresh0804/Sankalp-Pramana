"""
retrieval/search_models.py
Primary retrieval entry point for the pipeline.

Source priority:
  1. Sketchfab   (5M+ models, free API token required)
  2. Poly Haven  (CC0 free assets, no auth)
  3. Mock        (always-available offline fallback)
  4. Objaverse   (800k+, enabled via ENABLE_OBJAVERSE=true — slow first run)

The source list is controlled by the RETRIEVAL_SOURCES env variable, e.g.:
  RETRIEVAL_SOURCES=sketchfab,polyhaven,mock
  RETRIEVAL_SOURCES=objaverse,sketchfab,polyhaven,mock
"""

from __future__ import annotations
import logging
import os
from typing import List, Dict, Any

from config import TOP_K_CANDIDATES
from retrieval.dataset_loader import fetch_candidates

logger = logging.getLogger(__name__)

# Configurable source list — change via env var without redeploying
_DEFAULT_SOURCES = os.getenv("RETRIEVAL_SOURCES", "sketchfab,polyhaven,mock")
RETRIEVAL_SOURCES: List[str] = [s.strip() for s in _DEFAULT_SOURCES.split(",") if s.strip()]


def search_models(concept: str, top_k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Search for 3D model candidates matching the given concept.

    Returns a list of candidate dicts, each with:
      name, url, format, quality, keyword_score, vector_score, source
    """
    logger.info(
        f"[SearchModels] concept='{concept}'  "
        f"sources={RETRIEVAL_SOURCES}  top_k={top_k}"
    )

    candidates = fetch_candidates(
        query=concept,
        limit=top_k,
        sources=RETRIEVAL_SOURCES,
    )

    if not candidates:
        logger.warning(f"[SearchModels] All sources returned empty — using hardcoded fallback")
        candidates = _hardcoded_fallback(concept)

    logger.info(f"[SearchModels] {len(candidates)} candidates returned")
    return candidates


def _hardcoded_fallback(concept: str) -> List[Dict[str, Any]]:
    """Last-resort fallback so the API never returns empty-handed."""
    return [{
        "name":          f"{concept} (Fallback)",
        "url":           "https://storage.googleapis.com/ai-3d-models-bucket/fallback_cube.glb",
        "format":        "glb",
        "quality":       0.3,
        "keyword_score": 0.0,
        "vector_score":  0.0,
        "source":        "fallback",
    }]


# ── Expose mock catalogue for /models endpoint & FAISS index builder ──────────
from retrieval.dataset_loader import _MOCK_CATALOGUE as _MOCK_DATASET
