"""
rag/explanation_engine.py
Educational explanation generation for PratibimbAI.
Uses Gemini 2.5 Pro to generate concise educational descriptions.
"""

from __future__ import annotations
import logging
from typing import Optional

from ai.llm_client import generate, is_available

logger = logging.getLogger(__name__)

# System prompt for educational explanations
EXPLANATION_SYSTEM_PROMPT = """
You are an educational content generator for a 3D interactive learning app called PratibimbAI.

Your task is to provide concise, educational explanations for 3D visualizations.

Rules:
1. Keep explanations to 2-4 sentences
2. Focus on educational value and key facts
3. Use simple, accessible language
4. Avoid jargon unless necessary
5. Include interesting details when relevant
6. Be accurate and factual
"""


def generate_explanation(query: str) -> str:
    """
    Generate an educational explanation for the given query using Gemini 2.5 Pro.
    
    This function wraps the knowledge_engine's generate_explanation but
    ensures LLM-based explanations are prioritized when available.
    
    Args:
        query: The concept or object to explain
    
    Returns:
        Educational explanation string
    """
    logger.info(f"[ExplanationEngine] Generating explanation for: '{query}'")
    
    # Try LLM first if available
    if is_available():
        try:
            prompt = f"Provide a concise educational explanation for: {query}"
            result = generate(prompt, system_instruction=EXPLANATION_SYSTEM_PROMPT)
            if result:
                logger.info("[ExplanationEngine] Generated explanation via Gemini 2.5 Pro")
                return result
        except Exception as e:
            logger.warning(f"[ExplanationEngine] LLM generation failed: {e}")
    
    # Fallback to knowledge engine
    from rag.knowledge_engine import generate_explanation as _fallback
    logger.info("[ExplanationEngine] Falling back to knowledge engine")
    return _fallback(query)
