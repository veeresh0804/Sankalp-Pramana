"""
validation/clip_validator.py
AI validation pipeline that uses OpenCLIP to confirm whether a 3D model
visually/semantically matches the requested concept.

Since we cannot render actual 3D models server-side without a GPU renderer,
we use text-to-text CLIP similarity as a reliable proxy:
  — encode the model's name/tags as a rich description
  — encode the user concept
  — cosine similarity gives the validation score
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_THRESHOLD

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_clip_model    = None
_clip_preprocess = None
_tokenizer     = None


def _load_clip():
    global _clip_model, _clip_preprocess, _tokenizer
    if _clip_model is None:
        try:
            import open_clip
            logger.info(f"[CLIP] Loading {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
            _clip_model, _clip_preprocess, _ = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
            _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            _clip_model.eval()
        except Exception as e:
            logger.error(f"[CLIP] Failed to load model: {e}")
            _clip_model = None
    return _clip_model, _tokenizer


def _text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two text strings using CLIP text encoder.
    Falls back to keyword-overlap score if CLIP is unavailable.
    """
    model, tokenizer = _load_clip()

    if model is None or tokenizer is None:
        # Graceful fallback — keyword overlap
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a | words_b), 1)

    import torch
    try:
        tokens_a = tokenizer([text_a])
        tokens_b = tokenizer([text_b])
        with torch.no_grad():
            feat_a = model.encode_text(tokens_a)
            feat_b = model.encode_text(tokens_b)
            feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            similarity = (feat_a @ feat_b.T).item()
        return float(similarity)
    except Exception as e:
        logger.warning(f"[CLIP] Similarity computation failed: {e}")
        return 0.5


def validate_models(
    candidates: List[Dict[str, Any]],
    concept: str,
) -> List[Dict[str, Any]]:
    """
    Score each candidate model against the concept using CLIP text similarity.
    Returns only candidates that exceed CLIP_THRESHOLD, with a 'clip_score' key.
    If nothing passes the threshold the top-3 are returned anyway (safety net).
    """
    logger.info(f"[CLIP] Validating {len(candidates)} candidates for '{concept}'")

    validated: List[Dict[str, Any]] = []
    for model in candidates:
        model_description = f"{model.get('name', '')} 3D model"
        clip_score = _text_similarity(concept, model_description)

        entry = dict(model)
        entry["clip_score"] = round(clip_score, 4)
        validated.append(entry)
        logger.debug(f"[CLIP] '{model.get('name')}' score={clip_score:.3f}")

    # Filter by threshold
    above_threshold = [m for m in validated if m["clip_score"] >= CLIP_THRESHOLD]

    if not above_threshold:
        logger.warning("[CLIP] No candidates above threshold — returning top 3 by score")
        validated.sort(key=lambda x: x["clip_score"], reverse=True)
        return validated[:3]

    logger.info(f"[CLIP] {len(above_threshold)} candidates passed threshold")
    return above_threshold
