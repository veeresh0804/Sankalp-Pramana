"""
validation/semantic_filter.py
LLM-powered semantic filtering to remove irrelevant search candidates.
Uses Gemini to judge if a model name/description matches the user's intent.
"""

import json
import logging
import google.generativeai as genai
from typing import List, Dict, Any

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Filter prompt for the LLM
FILTER_SYSTEM_PROMPT = """
You are a 3D model relevance judge. 
Your task is to filter a list of 3D model titles based on a user's query.
You must discard models that are not relevant to the user's educational or visual needs.

Rules:
1. Return ONLY a JSON list of indices (0-based) that are RELEVANT.
2. Be strict. If the user wants a "solar system", an "air conditioner" is NOT relevant.
3. If the user wants a "human heart", a "heart emoji" or "valve part" might be relevant only if it's medical.
4. If a model is a generic primitive (like "Box" or "Sphere") and the query is specific, keep it only if no better options exist.
5. Return format: [0, 2, 5]
"""

def filter_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Gemini to filter out irrelevant candidates.
    Returns a subset of the input list.
    """
    if not GEMINI_API_KEY or not candidates:
        return candidates

    try:
        # Prepare the list for the LLM
        titles = [f"{i}: {c.get('name', 'Unnamed')}" for i, c in enumerate(candidates)]
        prompt = f"User Query: \"{query}\"\n\nCandidates:\n" + "\n".join(titles) + "\n\nRelevant indices:"

        model = genai.GenerativeModel('models/gemini-1.5-flash', system_instruction=FILTER_SYSTEM_PROMPT)
        response = model.generate_content(prompt)
        
        text = response.text.strip()
        # Extract JSON list from response
        if "[" in text and "]" in text:
            relevant_indices = json.loads(text[text.find("["):text.find("]")+1])
            
            filtered = [candidates[i] for i in relevant_indices if 0 <= i < len(candidates)]
            logger.info(f"[SemanticFilter] Filtered {len(candidates)} -> {len(filtered)} candidates")
            return filtered
        else:
            logger.warning(f"[SemanticFilter] Could not parse LLM response: {text}")
            return candidates

    except Exception as e:
        logger.error(f"[SemanticFilter] Error during LLM filtering: {e}")
        return candidates
