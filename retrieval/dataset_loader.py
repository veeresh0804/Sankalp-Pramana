from __future__ import annotations
"""
retrieval/dataset_loader.py
Real dataset connectors for Objaverse and Sketchfab.

Priority order when search_models.py calls fetch_candidates():
  1. Objaverse  — free, 800k+ annotated 3D objects, no API key needed
  2. Sketchfab  — 5M+ models, requires free API token
  3. Poly Haven — free CC0 HDRi + models (stub)
  4. Mock dataset — always-available offline fallback
"""


import logging
import os
import time
from typing import List, Dict, Any, Optional

USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Sketchfab token (set via env var — keep out of source control)
# ─────────────────────────────────────────────────────────────────────────────
SKETCHFAB_API_TOKEN = os.getenv("SKETCHFAB_API_TOKEN", "")
SKETCHFAB_API_BASE  = "https://api.sketchfab.com/v3"

# ─────────────────────────────────────────────────────────────────────────────
# 1. OBJAVERSE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_objaverse(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Objaverse — DISABLED in V2 optimization to reduce container size.
    """
    # logger.info(f"[Objaverse] Search skipped (uninstalled)")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKETCHFAB
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_sketchfab(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Sketchfab Data API v3 — 5M+ models, many free/downloadable.

    Requires a free API token:
      1. Create account at https://sketchfab.com
      2. Go to Settings → Password & API → Copy your API Token
      3. Set env var:  SKETCHFAB_API_TOKEN=your_token_here

    Docs: https://docs.sketchfab.com/data-api/v3/
    """
    if not SKETCHFAB_API_TOKEN:
        logger.warning(
            "[Sketchfab] No API token set. "
            "Set SKETCHFAB_API_TOKEN env var to enable Sketchfab search."
        )
        return []

    try:
        import requests

        logger.info(f"[Sketchfab] Searching for: '{query}'")

        params = {
            "q":            query,
            "type":         "models",
            "downloadable": "true",          # only freely downloadable
            "sort_by":      "-likeCount",    # most liked first
            "count":        limit,
            "archives_flavours": "false",
        }
        headers = {"Authorization": f"Token {SKETCHFAB_API_TOKEN}"}

        resp = requests.get(
            f"{SKETCHFAB_API_BASE}/search",
            params=params,
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, Any]] = []
        for model in data.get("results", []):
            uid        = model.get("uid", "")
            name       = model.get("name", "Unknown")
            like_count = model.get("likeCount", 0)
            view_count = model.get("viewCount", 0)

            # Thumbnail for visual reference
            thumbnails = model.get("thumbnails", {}).get("images", [])
            thumb_url  = thumbnails[0]["url"] if thumbnails else ""

            # Download URL — need a separate API call to get the actual GLB link
            download_url = _get_sketchfab_download_url(uid, headers)

            if not download_url:
                continue

            quality = min(1.0, (like_count / 5000) * 0.6 + (view_count / 50000) * 0.4)

            results.append({
                "name":          name,
                "url":           download_url,
                "format":        "glb",
                "quality":       round(quality, 3),
                "keyword_score": _score_text(query.lower(), name.lower()),
                "vector_score":  0.0,
                "source":        "sketchfab",
                "uid":           uid,
                "thumbnail":     thumb_url,
            })

        logger.info(f"[Sketchfab] Returned {len(results)} results for '{query}'")
        return results

    except ImportError:
        logger.warning("[Sketchfab] requests not installed — run: pip install requests")
        return []
    except Exception as e:
        logger.error(f"[Sketchfab] Error: {e}")
        return []


def _get_sketchfab_download_url(uid: str, headers: dict) -> Optional[str]:
    """Resolve the actual GLB download URL for a Sketchfab model UID."""
    try:
        import requests
        resp = requests.get(
            f"{SKETCHFAB_API_BASE}/models/{uid}/download",
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 200:
            glb = resp.json().get("glb", {})
            return glb.get("url")
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. POLY HAVEN (stub — no auth needed, CC0 assets)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_polyhaven(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch free CC0 3D models from Poly Haven API.
    Docs: https://api.polyhaven.com/
    """
    try:
        import requests
        logger.info(f"[PolyHaven] Searching for: '{query}'")

        resp = requests.get("https://api.polyhaven.com/assets?type=models", timeout=10)
        resp.raise_for_status()
        all_assets: Dict[str, Any] = resp.json()

        query_lower = query.lower()
        results: List[Dict[str, Any]] = []

        for asset_id, meta in all_assets.items():
            name   = meta.get("name", asset_id)
            tags   = " ".join(meta.get("tags", []))
            cats   = " ".join(meta.get("categories", []))
            score  = _score_text(query_lower, f"{name} {tags} {cats}".lower())
            if score > 0:
                results.append({
                    "name":          name,
                    "url":           f"https://dl.polyhaven.org/file/ph-assets/Models/glb/4k/{asset_id}.glb",
                    "format":        "glb",
                    "quality":       0.85,
                    "keyword_score": score,
                    "vector_score":  0.0,
                    "source":        "polyhaven",
                    "uid":           asset_id,
                })

        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        logger.info(f"[PolyHaven] Returned {min(limit, len(results))} results for '{query}'")
        return results[:limit]

    except ImportError:
        logger.warning("[PolyHaven] requests not installed")
        return []
    except Exception as e:
        logger.error(f"[PolyHaven] Error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOCK FALLBACK CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_CATALOGUE: List[Dict[str, Any]] = [
    {"name": "F1 Racing Car",    "tags": ["f1", "car", "race", "formula one"],     "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",       "format": "glb", "quality": 0.95},
    {"name": "Human Heart",      "tags": ["heart", "anatomy", "organ", "medical"], "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",   "format": "glb", "quality": 0.90},
    {"name": "Taj Mahal",        "tags": ["taj mahal", "monument", "india"],       "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",     "format": "glb", "quality": 0.88},
    {"name": "Fighter Jet",      "tags": ["jet", "fighter", "aircraft", "plane"],  "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",   "format": "glb", "quality": 0.87},
    {"name": "Human Brain",      "tags": ["brain", "neuron", "anatomy", "medical"],"url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",   "format": "glb", "quality": 0.91},
    {"name": "Solar System",     "tags": ["solar system", "planet", "space"],      "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",  "format": "glb", "quality": 0.85},
    {"name": "DNA Helix",        "tags": ["dna", "helix", "biology", "molecule"],  "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",     "format": "glb", "quality": 0.89},
    {"name": "Eiffel Tower",     "tags": ["eiffel", "tower", "paris", "france"],   "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",  "format": "glb", "quality": 0.86},
    {"name": "Human Skeleton",   "tags": ["skeleton", "bone", "anatomy"],          "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",      "format": "glb", "quality": 0.88},
    {"name": "Volcano",          "tags": ["volcano", "geology", "earth"],          "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",       "format": "glb", "quality": 0.84},
    {"name": "Space Shuttle",    "tags": ["shuttle", "space", "rocket", "nasa"],   "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb", "format": "glb", "quality": 0.92},
    {"name": "Water Molecule",   "tags": ["water", "molecule", "h2o", "chemistry"],"url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb","format": "glb", "quality": 0.83},
]


def fetch_from_mock(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Always-available offline fallback using the curated mock catalogue."""
    if not USE_MOCK_DATA:
        logger.info("[Mock] USE_MOCK_DATA disabled — skipping mock catalogue")
        return []

    query_lower = query.lower()
    results = []
    for m in _MOCK_CATALOGUE:
        combined = f"{m['name']} {' '.join(m['tags'])}".lower()
        score = _score_text(query_lower, combined)
        results.append({
            "name":          m["name"],
            "url":           m["url"],
            "format":        m["format"],
            "quality":       m["quality"],
            "keyword_score": score,
            "vector_score":  0.0,
            "source":        "mock",
        })
    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point used by search_models.py
# ─────────────────────────────────────────────────────────────────────────────

def fetch_candidates(
    query: str,
    limit: int = 10,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch 3D model candidates from all configured real sources, with mock fallback.

    sources: list of source names to try. Default = ["sketchfab", "polyhaven", "mock"]
             Objaverse disabled by default (slow first-time download).
             Enable with sources=["objaverse", "sketchfab", "polyhaven", "mock"]
    """
    if sources is None:
        # Sketchfab + Poly Haven by default (fast, no large download)
        # Objaverse requires ~200 MB metadata download on first run
        sources = ["sketchfab", "polyhaven"]
        if USE_MOCK_DATA:
            sources.append("mock")

    all_results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    source_fns = {
        "objaverse": fetch_from_objaverse,
        "sketchfab": fetch_from_sketchfab,
        "polyhaven": fetch_from_polyhaven,
    }

    if USE_MOCK_DATA:
        source_fns["mock"] = fetch_from_mock

    for source in sources:
        fn = source_fns.get(source)
        if fn is None:
            logger.warning(f"[Loader] Unknown source: {source}")
            continue

        try:
            items = fn(query, limit=limit)
            added = 0
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    item.setdefault("source", source)
                    all_results.append(item)
                    added += 1
            logger.info(f"[Loader] {source}: +{added} unique candidates")
        except Exception as e:
            logger.error(f"[Loader] {source} failed: {e}")

    # Sort combined pool by keyword_score descending
    all_results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
    logger.info(f"[Loader] Total candidates: {len(all_results)} for '{query}'")
    return all_results[:limit * 2]   # return generous pool for CLIP to filter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_text(query: str, text: str) -> float:
    """Keyword overlap score between query and target text."""
    query = query.lower()
    text = text.lower()
    
    # Substring bonus (e.g., 'tajmahal' inside 'taj mahal' without spaces)
    normalized_text = text.replace(" ", "").replace("-", "")
    if query.replace(" ", "") in normalized_text:
        return 1.0
        
    q_words = set(query.split())
    t_words = set(text.split())
    overlap  = q_words & t_words
    if not q_words:
        return 0.0
    return round(len(overlap) / len(q_words), 3)
