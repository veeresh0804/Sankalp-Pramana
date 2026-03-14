"""
synthesis/scene_blueprint_generator.py
SceneBlueprint generation and validation for procedural 3D scenes.
Validates SceneBlueprint compatibility with the Flutter renderer.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Required top-level keys for SceneBlueprint
REQUIRED_KEYS = {"meta", "environment", "primitives", "assets"}

# Allowed primitive types
ALLOWED_PRIMITIVES = {"box", "sphere", "cylinder", "torus", "cone", "plane"}

# Required transform keys
REQUIRED_TRANSFORM_KEYS = {"pos", "rot", "scale"}


def validate_scene_blueprint(scene: Dict[str, Any]) -> bool:
    """
    Validate SceneBlueprint compatibility with the Flutter renderer.
    
    Checks:
    1. Required top-level keys (meta, environment, primitives, assets)
    2. Primitives have valid type and transform structure
    3. Transform has pos, rot, scale arrays
    
    Args:
        scene: The SceneBlueprint dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(scene, dict):
        logger.error("[SceneValidation] Scene is not a dictionary")
        return False
    
    # Check required top-level keys
    missing_keys = REQUIRED_KEYS - set(scene.keys())
    if missing_keys:
        logger.error(f"[SceneValidation] Missing required keys: {missing_keys}")
        return False
    
    # Validate primitives
    primitives = scene.get("primitives", [])
    if not isinstance(primitives, list):
        logger.error("[SceneValidation] Primitives is not a list")
        return False
    
    for i, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            logger.error(f"[SceneValidation] Primitive {i} is not a dictionary")
            return False
        
        # Check primitive type
        prim_type = primitive.get("type", "").lower()
        if prim_type not in ALLOWED_PRIMITIVES:
            logger.error(f"[SceneValidation] Primitive {i} has invalid type: {prim_type}")
            return False
        
        # Check transform structure
        transform = primitive.get("transform", {})
        if not isinstance(transform, dict):
            logger.error(f"[SceneValidation] Primitive {i} transform is not a dictionary")
            return False
        
        # Check required transform keys
        missing_transform = REQUIRED_TRANSFORM_KEYS - set(transform.keys())
        if missing_transform:
            logger.error(f"[SceneValidation] Primitive {i} missing transform keys: {missing_transform}")
            return False
        
        # Validate transform arrays
        for key in REQUIRED_TRANSFORM_KEYS:
            val = transform.get(key)
            if not isinstance(val, list) or len(val) != 3:
                logger.error(f"[SceneValidation] Primitive {i} transform.{key} must be a 3-element array")
                return False
    
    logger.info(f"[SceneValidation] Scene blueprint valid with {len(primitives)} primitives")
    return True


def fix_scene_blueprint(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to fix common issues in a SceneBlueprint.
    
    Fixes:
    - Missing top-level keys (adds defaults)
    - Missing transform structure (creates from position/rotation/scale)
    - Missing transform keys (adds defaults)
    
    Args:
        scene: The SceneBlueprint dictionary to fix
    
    Returns:
        Fixed SceneBlueprint dictionary
    """
    fixed = dict(scene)
    
    # Add missing top-level keys
    if "meta" not in fixed:
        fixed["meta"] = {}
    if "environment" not in fixed:
        fixed["environment"] = {"background": "#1a1a2e"}
    if "primitives" not in fixed:
        fixed["primitives"] = []
    if "assets" not in fixed:
        fixed["assets"] = []
    
    # Fix primitives
    primitives = fixed.get("primitives", [])
    for i, p in enumerate(primitives):
        if not isinstance(p, dict):
            continue
        
        # Ensure valid primitive type
        prim_type = p.get("type", "box").lower()
        if prim_type not in ALLOWED_PRIMITIVES:
            p["type"] = "box"
        
        # Fix transform structure
        if "transform" not in p:
            p["transform"] = {
                "pos": p.pop("position", p.pop("pos", [0, 0, 0])),
                "rot": p.pop("rotation", p.pop("rot", [0, 0, 0])),
                "scale": p.pop("scale", [1, 1, 1])
            }
        
        # Ensure transform has all required keys
        t = p["transform"]
        if "pos" not in t:
            t["pos"] = p.pop("position", [0, 0, 0])
        if "rot" not in t:
            t["rot"] = p.pop("rotation", [0, 0, 0])
        if "scale" not in t:
            t["scale"] = [1, 1, 1]
        
        # Ensure arrays are valid
        for key in ["pos", "rot", "scale"]:
            val = t.get(key)
            if not isinstance(val, list) or len(val) != 3:
                t[key] = [0, 0, 0] if key != "scale" else [1, 1, 1]
    
    return fixed


def generate_scene_blueprint(query: str, style: str = "realistic", complexity: str = "medium") -> Optional[Dict[str, Any]]:
    """
    Generate a SceneBlueprint JSON using Gemini 3 flash preview.
    This is a wrapper that imports and calls the scene_generator module.
    
    Args:
        query: Natural language description of the scene
        style: Visual style (realistic, stylized, etc.)
        complexity: Scene complexity (low, medium, high)
    
    Returns:
        Validated SceneBlueprint dictionary or None on failure
    """
    from synthesis.scene_generator import generate_scene_blueprint as _generate
    
    scene = _generate(query, style, complexity)
    if scene is None:
        return None
    
    # Fix any issues and validate
    fixed = fix_scene_blueprint(scene)
    if validate_scene_blueprint(fixed):
        return fixed
    
    logger.warning("[SceneBlueprint] Generated scene failed validation after fixes")
    return fixed  # Return fixed version anyway, let renderer handle it
