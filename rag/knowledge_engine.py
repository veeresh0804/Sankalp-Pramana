"""
rag/knowledge_engine.py
RAG (Retrieval-Augmented Generation) explanation engine.

Phase 1: curated knowledge base (works offline, no API keys needed).
Phase 2: Wikipedia / LangChain integration (see stub at bottom).
Phase 3: LLM-powered (OpenAI / Vertex AI) — enabled via USE_LLM_RAG in config.
"""

from __future__ import annotations
import logging
from typing import Optional

from config import USE_LLM_RAG, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Curated Knowledge Base
# ──────────────────────────────────────────────────────────────────────────────
_KNOWLEDGE_BASE: dict[str, str] = {
    "f1 car":          "A Formula One (F1) car is a single-seat, open-cockpit racing car used in the FIA Formula One World Championship. These high-performance machines produce enormous downforce, reaching speeds over 360 km/h.",
    "formula one":     "Formula One is the highest class of single-seater auto racing sanctioned by the FIA. Teams from around the world compete using sophisticated aerodynamic vehicles with hybrid power units.",
    "human heart":     "The human heart is a muscular organ that pumps blood through the circulatory system. It has four chambers — right/left atria and right/left ventricles — and beats approximately 100,000 times per day.",
    "heart":           "The heart is the central organ of the cardiovascular system, responsible for circulating oxygenated blood throughout the body via rhythmic contractions.",
    "taj mahal":       "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, India. Built by Mughal emperor Shah Jahan, it is a UNESCO World Heritage Site and one of the Seven Wonders of the World.",
    "fighter jet":     "A fighter jet is a high-speed military aircraft designed for air-to-air combat. Modern fighters use fly-by-wire controls, afterburning turbofan engines, and advanced radar/missile systems.",
    "human brain":     "The human brain is the command center of the nervous system, containing approximately 86 billion neurons. It controls thought, memory, emotion, touch, motor skills, vision, breathing, and every other body process.",
    "brain":           "The brain is the organ that serves as the center of the nervous system in vertebrates. It integrates sensory information and coordinates bodily functions.",
    "solar system":    "The Solar System consists of the Sun and all objects gravitationally bound to it, including eight planets, dwarf planets, moons, asteroids, and comets spanning billions of kilometers.",
    "dna":             "DNA (Deoxyribonucleic acid) is a double-helix molecule that carries genetic information. It encodes the instructions for the development, functioning, growth, and reproduction of all known organisms.",
    "dna helix":       "The DNA double helix is a right-handed spiral formed by two complementary strands of nucleotides. Its structure was first described by Watson and Crick in 1953.",
    "eiffel tower":    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed between 1887–1889 for the World's Fair, it stands 330 metres tall and welcomes millions of visitors annually.",
    "skeleton":        "The human skeleton consists of 206 bones that provide structural support, protect vital organs, and enable movement through muscles and joints.",
    "human skeleton":  "The adult human skeleton has 206 bones arranged into the axial skeleton (skull, spine, ribs) and the appendicular skeleton (limbs, shoulder/hip girdles).",
    "volcano":         "A volcano is an opening in the Earth's crust through which lava, volcanic ash, and gases escape. Volcanoes are found along tectonic plate boundaries and hotspots around the world.",
    "space shuttle":   "The NASA Space Shuttle was a reusable spacecraft system operated from 1981 to 2011. It carried crews and cargo to low Earth orbit and was the first reusable orbital spacecraft.",
    "water molecule":  "A water molecule (H₂O) consists of one oxygen atom covalently bonded to two hydrogen atoms, giving it a bent geometry and a partial negative charge on oxygen — making it a polar molecule.",
    "water":           "Water (H₂O) is an inorganic compound essential for all known life. It covers ~71% of Earth's surface and plays a critical role in climate regulation, metabolism, and chemical reactions.",
}


def _keyword_lookup(concept: str) -> Optional[str]:
    """Try exact and partial keyword matches against the knowledge base."""
    key = concept.strip().lower()
    if key in _KNOWLEDGE_BASE:
        return _KNOWLEDGE_BASE[key]
    # Partial match
    for kb_key, description in _KNOWLEDGE_BASE.items():
        if kb_key in key or key in kb_key:
            return description
    return None


def _wikipedia_lookup(concept: str) -> Optional[str]:
    """Fetch a summary from Wikipedia (requires network + wikipedia-api package)."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='AI-3D-Visualizer/1.0 (research project)'
        )
        page = wiki.page(concept)
        if page.exists():
            return page.summary[:600]  # First ~600 chars
    except ImportError:
        logger.debug("[RAG] wikipediaapi not installed, skipping Wikipedia lookup")
    except Exception as e:
        logger.warning(f"[RAG] Wikipedia lookup failed: {e}")
    return None


def _llm_generate(concept: str) -> Optional[str]:
    """Generate explanation using OpenAI GPT (optional — requires OPENAI_API_KEY)."""
    if not USE_LLM_RAG or not OPENAI_API_KEY:
        return None
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": (
                    f"Provide a concise educational description (2-3 sentences) "
                    f"of '{concept}' suitable for a 3D interactive learning app."
                )
            }],
            max_tokens=150,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"[RAG] LLM generation failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_explanation(concept: str) -> str:
    """
    Generate a natural-language explanation for the given concept.
    Priority chain:
      1. LLM (if enabled + API key set)
      2. Wikipedia (if package installed + network available)
      3. Curated knowledge base
      4. Generic fallback
    """
    logger.info(f"[RAG] Generating explanation for: '{concept}'")

    # 1 — LLM
    result = _llm_generate(concept)
    if result:
        logger.info("[RAG] Explanation from LLM")
        return result

    # 2 — Wikipedia
    result = _wikipedia_lookup(concept)
    if result:
        logger.info("[RAG] Explanation from Wikipedia")
        return result

    # 3 — Curated KB
    result = _keyword_lookup(concept)
    if result:
        logger.info("[RAG] Explanation from curated knowledge base")
        return result

    # 4 — Generic fallback
    logger.info("[RAG] Using generic fallback explanation")
    return (
        f"'{concept}' is an educational 3D model retrieved from the AI knowledge base. "
        "Explore its structure interactively in the 3D viewer."
    )
