"""
ai/llm_client.py
Centralized Gemini 2.5 Pro client for all LLM operations.
All modules must use this client instead of direct model initialization.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Model configuration
GEMINI_MODEL_NAME = "gemini-2.5-pro"

# Global model instance (lazy initialization)
_model: Optional[genai.GenerativeModel] = None
_api_key: Optional[str] = None


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY", "")


def _initialize_model() -> Optional[genai.GenerativeModel]:
    """Initialize the Gemini model if API key is available."""
    global _model, _api_key
    
    api_key = _get_api_key()
    if not api_key:
        logger.warning("[LLMClient] GEMINI_API_KEY not set. LLM features will be disabled.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        _api_key = api_key
        _model = genai.GenerativeModel(f'models/{GEMINI_MODEL_NAME}')
        logger.info(f"[LLMClient] Initialized Gemini model: {GEMINI_MODEL_NAME}")
        return _model
    except Exception as e:
        logger.error(f"[LLMClient] Failed to initialize Gemini: {e}")
        return None


def generate(prompt: str, system_instruction: Optional[str] = None) -> str:
    """
    Generate text using Gemini 2.5 Pro.
    
    Args:
        prompt: The user prompt to send to the model.
        system_instruction: Optional system instruction for the model.
    
    Returns:
        The generated text response, or empty string on failure.
    """
    global _model
    
    # Lazy initialization
    if _model is None:
        _model = _initialize_model()
    
    if _model is None:
        logger.error("[LLMClient] Cannot generate: model not initialized (check GEMINI_API_KEY)")
        return ""
    
    try:
        # If system instruction provided, create a new model instance with it
        if system_instruction:
            model = genai.GenerativeModel(
                f'models/{GEMINI_MODEL_NAME}',
                system_instruction=system_instruction
            )
        else:
            model = _model
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"[LLMClient] Generation failed: {e}")
        return ""


def is_available() -> bool:
    """Check if the LLM client is available (API key configured and model initialized)."""
    global _model
    if _model is None:
        _model = _initialize_model()
    return _model is not None
