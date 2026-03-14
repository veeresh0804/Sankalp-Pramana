"""
synthesis/scene_generator.py
Procedural 3D scene generation using Gemini 3 flash preview.
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
    Generate a SceneBlueprint JSON using Gemini 3 flash preview.
    """
    if not is_available():
        logger.error("[Synthesis] Cannot generate scene: LLM client not available.")
        return None

    try:
        prompt = (
            "You are a procedural 3D scene generator for an educational visualization system.\n\n"
            "Return ONLY valid JSON following this schema:\n"
            "{ meta:{}, environment:{}, primitives:[], assets:[] }\n\n"
            "Primitive types allowed: box, sphere, cylinder, cone, torus, plane.\n"
            "Every primitive must include transform.pos, transform.rot, transform.scale arrays.\n\n"
            "Scene complexity rules:\n"
            "low complexity → about 6 primitives\n"
            "medium complexity → about 12–15 primitives\n"
            "high complexity → about 18–24 primitives\n\n"
            "Scene planning method:\n"
            "Step 1: determine the major structural components of the concept.\n"
            "Step 2: distribute primitives across those components to build the structure.\n"
            "Step 3: place primitives spatially to form a balanced 3D layout.\n\n"
            "Planning examples:\n"
            "heart → ventricles, atria, vessels\n"
            "solar system → sun, planets, asteroid belt\n"
            "atom → nucleus, electron orbit\n"
            "molecule → atoms and bonds\n\n"
            "Placement rules:\n"
            "avoid overlapping objects\n"
            "spread primitives across space\n"
            "vary primitive sizes to represent hierarchy\n"
            "maintain clear educational structure\n\n"
            f"query: {query}\n"
            f"style: {style}\n"
            f"complexity: {complexity}\n\n"
            "Return JSON only."
        )
        
        logger.info(f"[Synthesis] Generating scene for: '{query}'")
        text = generate(
            prompt,
            system_instruction=SCENE_GENERATOR_SYSTEM_PROMPT,
        )
        
        if not text:
            logger.error("[Synthesis] Empty response from LLM")
            return None
        
        # Clean up the response (remove markdown / preamble)
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()
        start = text.find("{")
        if start != -1:
            text = text[start:]
        
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
