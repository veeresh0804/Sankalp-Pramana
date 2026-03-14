"""
main.py — AI 3D Model Retrieval API
FastAPI entry-point, Cloud Run compatible (port 8080).

Pipeline per request:
  1. Check cache
  2. Semantic search  (vector_search)
  3. Keyword search   (retrieval)
  4. Merge & dedupe candidates
  5. CLIP validation
  6. Weighted ranking
  7. RAG explanation
  8. Store in cache
  9. Return response
"""

from __future__ import annotations
import logging
import time
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import APP_TITLE, APP_DESCRIPTION, APP_VERSION, MIN_CONFIDENCE, CONFIDENCE_THRESHOLD, THRESHOLD, DEEP_SEARCH_ENABLED, FALLBACK_ENABLED
from retrieval.search_models   import search_models
from retrieval.intent_classifier import classify_intent, get_prioritized_sources
from ranking.ranking_engine    import rank_models
from rag.explanation_engine     import generate_explanation
from cache.cache_manager       import get as cache_get, set as cache_set, stats as cache_stats, clear as cache_clear
from synthesis.scene_blueprint_generator import generate_scene_blueprint, validate_scene_blueprint
from validation.semantic_filter import filter_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Global readiness flag — set True when models are fully loaded
# ──────────────────────────────────────────────────────────────────────────────
_models_ready = False


def _load_models_background():
    """
    Lightweight readiness initializer (heavy models removed).
    """
    global _models_ready
    try:
        _models_ready = True
        logger.info("[✓ Startup BG] Server ready — lightweight mode (no FAISS/CLIP)")
    except Exception as e:
        logger.error(f"[✗ Startup BG] FAISS failed: {e}")
        _models_ready = True  # allow requests even if broken (fallback kicks in)


@asynccontextmanager
async def lifespan(app_instance):
    """
    Start model loading in background thread immediately.
    Server responds to /health right away → Cloud Run TCP probe passes.
    """
    logger.info("[► Startup] Server ready — loading AI models in background...")
    thread = threading.Thread(target=_load_models_background, daemon=True)
    thread.start()
    yield
    logger.info("[Shutdown] Cleaning up...")


# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,        # ← startup/shutdown hook
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    concept: str = Field(..., example="F1 car", description="3D concept to search for")
    top_k:   int = Field(5, ge=1, le=20, description="Number of candidates before ranking")


class SearchResponse(BaseModel):
    concept:     str
    model_url:   str
    model_name:  str
    format:      str
    confidence:  float
    description: str
    cached:      bool
    latency_ms:  float


class HealthResponse(BaseModel):
    status:  str
    version: str
    cache:   dict


class VisualizeRequest(BaseModel):
    query:      str = Field(..., example="taj mahal", description="3D concept to visualize")
    style:      str = Field("realistic", description="Visual style")
    complexity: str = Field("medium", description="Scene complexity")
    network:    Optional[str] = Field(None, description="Client-reported network quality: fast | slow")
    dataSaver:  Optional[bool] = Field(False, description="Client data saver flag")


class VisualizeResponse(BaseModel):
    """
    Unified response schema for /visualize endpoint.
    Strictly follows PratibimbAI architecture specification.
    """
    success:        bool
    type:           str  # 'glb' | 'scene'
    data:           dict # SceneBlueprint or { "url": "..." }
    explanation:    str = ""  # Educational explanation
    processingTime: int   # Processing time in milliseconds
    source:         str  # 'search' | 'generated'


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return HealthResponse(
        status="AI 3D Retrieval API is running ",
        version=APP_VERSION,
        cache=cache_stats(),
    )


@app.post("/search_model", response_model=SearchResponse, tags=["AI Pipeline"])
def search_model(request: SearchRequest):
    """Legacy endpoint for 3D model retrieval."""
    concept = request.concept.strip()
    result = _run_search_pipeline(concept, request.top_k)
    return SearchResponse(**result)


@app.post("/visualize", response_model=VisualizeResponse, tags=["AI Pipeline"])
def visualize(request: VisualizeRequest):
    """
    Unified visualization endpoint.
    1. Search for a real GLB model.
    2. If not found (or low confidence), generate a procedural SceneBlueprint.
    3. Return in the VisualizeResponse format.
    """
    t0 = time.perf_counter()
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    # Network-adaptive shortcut: skip retrieval if client reports slow network or data saver
    if request.dataSaver or (request.network and request.network.lower() == "slow"):
        logger.info(f"[Decision] Network/dataSaver shortcut activated (network={request.network}, dataSaver={request.dataSaver}) — generating scene directly.")
        blueprint = generate_scene_blueprint(query, request.style, request.complexity)
        latency = int((time.perf_counter() - t0) * 1000)

        if blueprint:
            scene_explanation = generate_explanation(query)
            return VisualizeResponse(
                success=True,
                type="scene",
                processingTime=latency,
                source="generated",
                data=blueprint,
                explanation=scene_explanation
            )

        logger.warning("[Decision] Scene generation failed in network shortcut; returning minimal fallback scene.")
        minimal_scene = {
            "meta": {"title": "Fallback Scene"},
            "environment": {"background": "#1a1a2e"},
            "primitives": [
                {
                    "type": "box",
                    "material": {"color": "#8ab4f8"},
                    "transform": {"pos": [0, 0.5, 0], "rot": [0, 0, 0], "scale": [1, 1, 1]}
                }
            ],
            "assets": []
        }
        scene_explanation = generate_explanation(query)
        return VisualizeResponse(
            success=True,
            type="scene",
            processingTime=latency,
            source="generated",
            data=minimal_scene,
            explanation=scene_explanation
        )

    # 1. Search Pipeline
    search_result = _run_search_pipeline(query, top_k=5)

    # --- PRATIBIMBAI DECISION ENGINE (THRESHOLD = 0.55) ---
    confidence = search_result["confidence"]
    explanation = search_result.get("description", "")

    # Low-confidence guard: immediately prefer procedural scene generation over Box.glb
    if confidence < THRESHOLD:
        logger.info(f"[Decision] Confidence {confidence} < THRESHOLD {THRESHOLD}. Generating procedural scene.")
        blueprint = generate_scene_blueprint(query, request.style, request.complexity)
        latency = int((time.perf_counter() - t0) * 1000)

        if blueprint:
            scene_explanation = generate_explanation(query)
            return VisualizeResponse(
                success=True,
                type="scene",
                processingTime=latency,
                source="generated",
                data=blueprint,
                explanation=scene_explanation
            )

        # Safety net: minimal primitive scene instead of Box.glb
        logger.warning("[Decision] Scene generation failed at low confidence; returning minimal primitive scene.")
        minimal_scene = {
            "meta": {"title": "Fallback Scene"},
            "environment": {"background": "#1a1a2e"},
            "primitives": [
                {
                    "type": "box",
                    "material": {"color": "#8ab4f8"},
                    "transform": {"pos": [0, 0.5, 0], "rot": [0, 0, 0], "scale": [1, 1, 1]}
                }
            ],
            "assets": []
        }
        scene_explanation = generate_explanation(query)
        return VisualizeResponse(
            success=True,
            type="scene",
            processingTime=latency,
            source="generated",
            data=minimal_scene,
            explanation=scene_explanation
        )
    
    # Decision: Return GLB if score >= THRESHOLD, else generate procedural scene
    # THRESHOLD = 0.55 per PratibimbAI specification
    is_above_threshold = (
        confidence >= THRESHOLD and 
        "Box.glb" not in search_result["model_url"]
    )

    if is_above_threshold:
        logger.info(f"[Decision] Score {confidence} >= THRESHOLD {THRESHOLD}. Returning GLB.")
        latency = int((time.perf_counter() - t0) * 1000)
        return VisualizeResponse(
            success=True,
            type="glb",
            processingTime=latency,
            source="search",
            data={"url": search_result["model_url"]},
            explanation=explanation
        )

    # Tier 2: Search Extension (Score < THRESHOLD, but Deep Search is active)
    if DEEP_SEARCH_ENABLED:
        logger.info(f"[Decision] Score {confidence} < THRESHOLD {THRESHOLD}. Triggering Deep Search for '{query}'...")
        from retrieval.search_models import deep_web_search
        
        deep_result = deep_web_search(query)
        # Check if deep search found a better model above threshold
        if deep_result and deep_result[0].get("final_score", 0) >= THRESHOLD:
            best_deep = deep_result[0]
            logger.info(f"[DeepSearch] Found match above threshold: '{best_deep['name']}' ({best_deep['final_score']})")
            latency = int((time.perf_counter() - t0) * 1000)
            return VisualizeResponse(
                success=True,
                type="glb",
                processingTime=latency,
                source="search",
                data={"url": best_deep["url"]},
                explanation=explanation
            )

    # Tier 3: Procedural Scene Generation (Score < THRESHOLD, Deep Search failed)
    reason = f"Top model scored {confidence}, below THRESHOLD of {THRESHOLD}."
    logger.info(f"[Decision] {reason} Generating procedural scene.")
    blueprint = generate_scene_blueprint(query, request.style, request.complexity)
    
    latency = int((time.perf_counter() - t0) * 1000)

    if blueprint:
        # Generate explanation for the procedural scene
        scene_explanation = generate_explanation(query)
        return VisualizeResponse(
            success=True,
            type="scene",
            processingTime=latency,
            source="generated",
            data=blueprint,
            explanation=scene_explanation
        )
    
    # Absolute Fallback (if synthesis fails) - return the low-confidence GLB
    logger.warning(f"[Fallback] Scene generation failed. Returning fallback GLB.")
    return VisualizeResponse(
        success=True,
        type="glb",
        processingTime=latency,
        source="search",
        data={"url": search_result["model_url"]},
        explanation=explanation
    )


def _run_search_pipeline(concept: str, top_k: int) -> dict:
    """Internal helper to run the full retrieval + validation + ranking pipeline."""
    t0 = time.perf_counter()

    # Readiness guard — models load in background
    if not _models_ready:
        raise HTTPException(
            status_code=503,
            detail="AI models are loading, please retry in ~20 seconds"
        )

    # ── 1. Cache check ──────────────────────────────────────────────────────
    cached = cache_get(concept)
    if cached:
        cached["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        cached["cached"] = True
        return cached

    # ── 2. Intent Classification (NEW) ──────────────────────────────────────
    intent = classify_intent(concept)
    sources = get_prioritized_sources(intent)

    # ── 3. Targeted candidate retrieval ─────────────────────────────────────
    logger.info(f"[Pipeline] Processing: '{concept}' (Intent: {intent}, Sources: {sources})")

    keyword_candidates = search_models(concept, sources=sources)

    # Merge & deduplicate by URL
    seen: set[str] = set()
    all_candidates: List[dict] = []
    for c in keyword_candidates:
        if c["url"] not in seen:
            seen.add(c["url"])
            all_candidates.append(c)

    if not all_candidates:
        if not FALLBACK_ENABLED:
            raise HTTPException(status_code=404, detail=f"No models found for '{concept}'")
        logger.warning(f"[Pipeline] No candidates — using fallback for '{concept}'")
        all_candidates = [{
            "name":          "Generic 3D Object",
            "url":           "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb",
            "format":        "glb",
            "quality":       0.3,
            "keyword_score": 0.0,
            "vector_score":  0.0,
        }]

    # ── 4. Ranking (lightweight candidates) ──────────────────────────────────
    best = rank_models(all_candidates)

    # ── 5. RAG explanation ──────────────────────────────────────────────────
    explanation = generate_explanation(concept)

    # ── 6. Build + cache response ───────────────────────────────────────────
    latency = round((time.perf_counter() - t0) * 1000, 2)
    confidence = round(best.get("final_score", best.get("clip_score", 0.5)), 3)

    if confidence < MIN_CONFIDENCE and FALLBACK_ENABLED:
        logger.warning(f"[Pipeline] Low confidence ({confidence}) — prefer procedural scene (no GLB override)")

    payload = {
        "concept":     concept,
        "model_url":   best["url"],
        "model_name":  best.get("name", "Unknown"),
        "format":      best.get("format", "glb"),
        "confidence":  confidence,
        "description": explanation,
        "cached":      False,
        "latency_ms":  latency,
    }

    cache_set(concept, payload)
    return payload


@app.get("/models", tags=["Utility"])
def list_available_models(
    limit: int = Query(default=20, ge=1, le=100),
):
    """List available 3D models (from GCS when deployed, from mock dataset locally)."""
    from retrieval.search_models import _MOCK_DATASET
    return {
        "total": len(_MOCK_DATASET),
        "models": [
            {"name": m["name"], "url": m["url"], "format": m["format"]}
            for m in _MOCK_DATASET[:limit]
        ]
    }


@app.delete("/cache", tags=["Admin"])
def clear_cache():
    """Clear the in-memory response cache."""
    cache_clear()
    return {"message": "Cache cleared successfully"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": APP_VERSION}