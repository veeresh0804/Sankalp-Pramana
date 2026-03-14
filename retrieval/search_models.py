"""
retrieval/search_models.py
Primary retrieval entry point for the pipeline.

Source priority:
  1. Sketchfab   (5M+ models, free API token required)
  2. Poly Haven  (CC0 free assets, no auth)
  3. Mock        (always-available offline fallback)

The source list is controlled by the RETRIEVAL_SOURCES env variable, e.g.:
  RETRIEVAL_SOURCES=sketchfab,polyhaven,mock
  RETRIEVAL_SOURCES=objaverse,sketchfab,polyhaven,mock
"""

from __future__ import annotations
import logging
import os
from typing import List, Dict, Any

from rapidfuzz import fuzz

from config import TOP_K_CANDIDATES
from retrieval.dataset_loader import fetch_candidates

logger = logging.getLogger(__name__)

# Configurable source list — change via env var without redeploying
_DEFAULT_SOURCES = os.getenv("RETRIEVAL_SOURCES", "sketchfab,polyhaven,mock")
RETRIEVAL_SOURCES: List[str] = [s.strip() for s in _DEFAULT_SOURCES.split(",") if s.strip()]


def _compute_similarity(a: str, b: str) -> float:
    """Token-set similarity normalized to [0,1]."""
    return round(fuzz.token_set_ratio(a, b) / 100.0, 3)


def _popularity(candidate: Dict[str, Any]) -> float:
    # Prefer explicit quality; otherwise derive a tiny signal from like/view counts if present
    if "quality" in candidate:
        return float(candidate.get("quality", 0))
    likes = candidate.get("like_count", 0) or candidate.get("likes", 0)
    views = candidate.get("view_count", 0) or candidate.get("views", 0)
    # lightweight normalization
    return min(1.0, (likes / 5000.0) * 0.6 + (views / 50000.0) * 0.4)


def search_models(concept: str, top_k: int = TOP_K_CANDIDATES, sources: List[str] = None) -> List[Dict[str, Any]]:
    """
    Search for 3D model candidates matching the given concept.

    Returns a list of candidate dicts, each with:
      name, url, format, quality, keyword_score, vector_score, source
    """
    selected_sources = sources if sources else RETRIEVAL_SOURCES
    logger.info(
        f"[SearchModels] concept='{concept}'  "
        f"sources={selected_sources}  top_k={top_k}"
    )

    candidates = fetch_candidates(
        query=concept,
        limit=top_k,
        sources=selected_sources,
    )

    # Filter out entries without preview/thumbnail
    filtered: List[Dict[str, Any]] = []
    for c in candidates:
        if not (c.get("thumbnail") or c.get("preview")):
            continue

        name = c.get("name", "")
        sim = _compute_similarity(concept, name)
        pop = _popularity(c)
        score = round(0.7 * sim + 0.3 * pop, 3)

        enriched = dict(c)
        enriched["keyword_score"] = sim
        enriched["clip_score"] = score  # reused downstream by ranking_engine
        enriched["vector_score"] = 0.0
        enriched["final_score"] = score
        enriched["popularity"] = pop
        filtered.append(enriched)

    if not filtered:
        logger.warning(f"[SearchModels] All sources returned empty — using hardcoded fallback")
        filtered = _hardcoded_fallback(concept)

    filtered.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    logger.info(f"[SearchModels] {len(filtered)} candidates returned")
    return filtered


def deep_web_search(concept: str) -> List[Dict[str, Any]]:
    """
    Experimental deep search that tries broader queries and extra sources.
    Triggered when standard search confidence is < 0.80.
    """
    logger.info(f"[DeepSearch] Aggressive retrieval for: '{concept}'")
    
    # Try multiple query variants
    variants = [
        concept,
        f"{concept} 3d model",
        f"{concept} low poly",
        f"realistic {concept}"
    ]
    
    all_deep_candidates = []
    seen = set()
    
    for var in variants:
        # Increase top_k for deep search
        candidates = search_models(var, top_k=20)
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                all_deep_candidates.append(c)
                
    if not all_deep_candidates:
        return []

    # In a production system, we would re-run CLIP and Ranking here.
    # For this implementation, we simulate a 'scored' result set.
    # We sort by quality + keyword_score
    for c in all_deep_candidates:
        c["final_score"] = round((c.get("keyword_score", 0) * 0.4) + (c.get("quality", 0) * 0.6), 3)
    
    all_deep_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return all_deep_candidates


def _hardcoded_fallback(concept: str) -> List[Dict[str, Any]]:
    """Last-resort fallback so the API never returns empty-handed."""
    return [{
        "name":          f"{concept} (Fallback)",
        "url":           "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",
        "format":        "glb",
        "quality":       0.3,
        "keyword_score": 0.0,
        "vector_score":  0.0,
        "source":        "fallback",
    }]


# ── Expose mock catalogue for /models endpoint & FAISS index builder ──────────
from retrieval.dataset_loader import _MOCK_CATALOGUE as _MOCK_DATASET
