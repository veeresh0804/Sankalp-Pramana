"""
retrieval/intent_classifier.py
LLM-powered query intent classification for source prioritization.
Categorizes queries (e.g., monument, science, furniture) to target the best retrieval sources.
"""

import logging
import google.generativeai as genai
from typing import Dict, List

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Intent Categories and their mapping to retrieval sources
INTENT_SOURCE_MAPPING = {
    "monument":   ["sketchfab"],
    "science":    ["sketchfab"],
    "nature":     ["polyhaven", "sketchfab"],
    "architecture": ["sketchfab", "polyhaven"],
    "generic":    ["sketchfab", "polyhaven", "mock"]
}

CLASSIFICATION_PROMPT = """
You are a 3D model search intent classifier.
Categorize the user's query into EXACTLY ONE of these categories:
- monument (buildings, landmarks, statues)
- science (anatomy, molecules, astronomy, technology)
- nature (trees, rocks, animals, terrains)
- architecture (interior design, building components, furniture)
- generic (anything else)

User Query: "{query}"
Category:"""

def classify_intent(query: str) -> str:
    """
    Use Gemini to classify the user's search intent.
    Returns the category name.
    """
    if not GEMINI_API_KEY:
        return "generic"

    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(CLASSIFICATION_PROMPT.format(query=query))
        
        category = response.text.strip().lower()
        
        # Validate category
        for valid_cat in INTENT_SOURCE_MAPPING.keys():
            if valid_cat in category:
                logger.info(f"[Intent] Query '{query}' classified as: {valid_cat}")
                return valid_cat
        
        return "generic"

    except Exception as e:
        logger.error(f"[Intent] Classification failed: {e}")
        return "generic"

def get_prioritized_sources(intent: str) -> List[str]:
    """Return a list of source names prioritized for the given intent."""
    return INTENT_SOURCE_MAPPING.get(intent, INTENT_SOURCE_MAPPING["generic"])
