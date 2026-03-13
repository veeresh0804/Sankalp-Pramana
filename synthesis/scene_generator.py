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
        prompt = (
            "You are a procedural 3D scene generator for an educational visualization system.\n\n"
            "Return ONLY valid JSON following the SceneBlueprint schema:\n\n"
            "{\n"
            " \"meta\": {},\n"
            " \"environment\": {},\n"
            " \"primitives\": [],\n"
            " \"assets\": []\n"
            "}\n\n"
            "Scene Generation Rules:\n\n"
            "• Always include the root keys: meta, environment, primitives, assets.\n"
            "• Use multiple primitives arranged spatially (avoid single-shape scenes).\n"
            "• Every primitive must include:\n"
            "  transform.pos\n"
            "  transform.rot\n"
            "  transform.scale\n\n"
            "Allowed primitive types:\n\n"
            "box\n"
            "sphere\n"
            "cylinder\n"
            "torus\n"
            "cone\n"
            "plane\n\n"
            "Do NOT generate any other primitive types.\n\n"
            "Spatial Design Guidelines:\n\n"
            "Solar System:\n"
            "- large central sphere for the sun\n"
            "- smaller spheres for planets\n"
            "- increasing distances from the center\n\n"
            "Atom:\n"
            "- central nucleus sphere\n"
            "- several smaller spheres orbiting as electrons\n\n"
            "Human Heart:\n"
            "- multiple spheres or cylinders forming connected chambers\n\n"
            "Molecule or chemical structure:\n"
            "- spheres connected by cylinders representing bonds\n\n"
            "Design Principles:\n\n"
            "• Spread primitives across the scene\n"
            "• Avoid overlapping objects\n"
            "• Use varied sizes to represent hierarchy\n"
            "• Prefer educational clarity over geometric randomness\n\n"
            "Complexity Guidance:\n\n"
            "low complexity → ~3 primitives\n"
            "medium complexity → ~5–7 primitives\n"
            "high complexity → 8+ primitives\n\n"
            "Example SceneBlueprint:\n\n"
            "{\n"
            " \"meta\": {\"title\": \"Atom\"},\n"
            " \"environment\": {\"background\": \"#000000\"},\n"
            " \"primitives\": [\n"
            "  {\"type\": \"sphere\", \"transform\": {\"pos\": [0,0,0], \"rot\": [0,0,0], \"scale\": [1,1,1]}},\n"
            "  {\"type\": \"sphere\", \"transform\": {\"pos\": [2,0,0], \"rot\": [0,0,0], \"scale\": [0.3,0.3,0.3]}}\n"
            " ],\n"
            " \"assets\": []\n"
            "}\n\n"
            "Return ONLY the JSON object. Do not include explanations, comments, or markdown formatting.\n\n"
            f"User Request:\n\"{query}\"\n\n"
            f"Style: {style}\n"
            f"Complexity: {complexity}\n"
        )
        
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
