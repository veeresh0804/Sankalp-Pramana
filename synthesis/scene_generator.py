"""
synthesis/scene_generator.py
Procedural 3D scene generation using Google Gemini.
Converts natural language prompts into SceneBlueprint JSON.
"""

from __future__ import annotations
import json
import logging
import re
import google.generativeai as genai
from typing import Optional, Dict, Any

from config import GEMINI_API_KEY, SCENE_GENERATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("[Synthesis] GEMINI_API_KEY not found. Procedural generation will be disabled.")

def generate_scene_blueprint(query: str, style: str = "realistic", complexity: str = "medium") -> Optional[Dict[str, Any]]:
    """
    Generate a SceneBlueprint JSON using Gemini.
    """
    if not GEMINI_API_KEY:
        logger.error("[Synthesis] Cannot generate scene: GEMINI_API_KEY missing.")
        return None

    try:
        model = genai.GenerativeModel('models/gemini-flash-latest', system_instruction=SCENE_GENERATOR_SYSTEM_PROMPT)
        
        prompt = f"User request: \"{query}\" (Style: {style}, Complexity: {complexity})\n\nGenerate SceneBlueprint JSON following the schema and rules."
        
        logger.info(f"[Synthesis] Generating scene for: '{query}'")
        response = model.generate_content(prompt)
        
        # Clean up the response (remove markdown if any)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            blueprint = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to fix common LLM JSON errors (like trailing commas)
            logger.warning("[Synthesis] JSON decode failed, attempting recovery...")
            text = re.sub(r',\s*([\]}])', r'\1', text) # Remove trailing commas
            blueprint = json.loads(text)

        # Basic schema validation/fixing
        if "primitives" in blueprint:
            for p in blueprint["primitives"]:
                # Force transform structure if LLM put keys at root
                if "transform" not in p:
                    p["transform"] = {
                        "pos": p.pop("position", p.pop("pos", [0,0,0])),
                        "rot": p.pop("rotation", p.pop("rot", [0,0,0])),
                        "scale": p.pop("scale", [1,1,1])
                    }
                # Ensure transform has all keys
                t = p["transform"]
                if "pos" not in t: t["pos"] = p.pop("position", [0,0,0])
                if "rot" not in t: t["rot"] = p.pop("rotation", [0,0,0])
                if "scale" not in t: t["scale"] = [1,1,1]

        logger.info(f"[Synthesis] Successfully generated scene with {len(blueprint.get('primitives', []))} primitives")
        return blueprint

    except Exception as e:
        logger.error(f"[Synthesis] Scene generation failed: {e}")
        return None
