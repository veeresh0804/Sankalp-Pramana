"""
ranking/ranking_engine.py
Combines keyword score, vector score, CLIP score, and model quality into
a weighted final confidence score, then returns the single best model.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Weighted blend of individual scores
_WEIGHTS = {
    "clip_score":    0.45,   # visual-semantic alignment
    "vector_score":  0.30,   # semantic embedding similarity
    "keyword_score": 0.15,   # fast keyword match
    "quality":       0.10,   # dataset metadata quality flag
}


def _composite_score(model: Dict[str, Any]) -> float:
    """Compute a weighted composite confidence score."""
    total = 0.0
    for key, weight in _WEIGHTS.items():
        value = model.get(key, 0.0)
        # Clamp to [0, 1]
        value = max(0.0, min(1.0, float(value)))
        total += weight * value
    return round(total, 4)


def rank_models(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rank all validated candidates by composite score.
    Returns the single best model dict with a 'final_score' key added.
    Falls back to a placeholder if the list is empty.
    """
    if not candidates:
        logger.warning("[Ranking] No candidates to rank — returning fallback model")
        return {
            "name":        "Fallback Primitive",
            "url":         "https://storage.googleapis.com/ai-3d-models-bucket/fallback_cube.glb",
            "format":      "glb",
            "final_score": 0.0,
        }

    for model in candidates:
        model["final_score"] = _composite_score(model)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    best = candidates[0]
    logger.info(
        f"[Ranking] Best model: '{best.get('name')}' "
        f"score={best['final_score']:.3f}"
    )
    return best
