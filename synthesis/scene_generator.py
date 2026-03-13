"""
synthesis/scene_generator.py
Procedural 3D scene generation using Gemini 2.5 Pro.
Converts natural language prompts into SceneBlueprint JSON.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Optional, Dict, Any

from ai.llm_client import generate, is_available
from config import SCENE_GENERATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def generate_scene_blueprint(query: str, style: str = "realistic", complexity: str = "medium") -> Optional[Dict[str, Any]]:
    """
    Generate a SceneBlueprint JSON using Gemini 2.5 Pro.
    """
    if not is_available():
        logger.error("[Synthesis] Cannot generate scene: LLM client not available.")
        return None

    try:
        prompt = f"User request: \"{query}\" (Style: {style}, Complexity: {complexity})\n\nGenerate SceneBlueprint JSON following the schema and rules."
        
        logger.info(f"[Synthesis] Generating scene for: '{query}'")
        text = generate(prompt, system_instruction=SCENE_GENERATOR_SYSTEM_PROMPT)
        
        if not text:
            logger.error("[Synthesis] Empty response from LLM")
            return None
        
        # Clean up the response (remove markdown if any)
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
