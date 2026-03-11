"""
cache/cache_manager.py
Simple in-memory TTL cache for API responses.
Prevents repeat AI pipeline calls for the same concept within the TTL window.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional, Tuple

from config import CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

# {concept_key: (result_dict, expiry_timestamp)}
_cache: Dict[str, Tuple[Any, float]] = {}


def _cache_key(concept: str) -> str:
    return concept.strip().lower()


def get(concept: str) -> Optional[Dict[str, Any]]:
    """Return cached result if still within TTL, else None."""
    key = _cache_key(concept)
    entry = _cache.get(key)
    if entry is None:
        return None
    result, expiry = entry
    if time.time() > expiry:
        del _cache[key]
        logger.debug(f"[Cache] Expired entry for '{concept}'")
        return None
    logger.info(f"[Cache] HIT for '{concept}'")
    return result


def set(concept: str, result: Dict[str, Any]) -> None:
    """Store result in cache with TTL from config."""
    key    = _cache_key(concept)
    expiry = time.time() + CACHE_TTL_SECONDS
    _cache[key] = (result, expiry)
    logger.info(f"[Cache] SET for '{concept}' (TTL={CACHE_TTL_SECONDS}s)")


def clear() -> None:
    """Wipe the entire cache (useful for testing / admin endpoints)."""
    _cache.clear()
    logger.info("[Cache] Cleared all entries")


def stats() -> Dict[str, int]:
    now = time.time()
    active = sum(1 for _, (_, exp) in _cache.items() if exp > now)
    return {"total_entries": len(_cache), "active_entries": active}
