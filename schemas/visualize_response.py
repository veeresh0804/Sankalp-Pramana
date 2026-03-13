"""
schemas/visualize_response.py
Pydantic schema for /visualize endpoint response.
Strictly follows PratibimbAI architecture specification.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VisualizeResponse(BaseModel):
    """
    Unified response schema for the /visualize endpoint.
    
    Fields:
        success: Whether the request was successful
        type: Response type - 'glb' for GLB model URL, 'scene' for SceneBlueprint
        data: Either {"url": "..."} for GLB or SceneBlueprint object for scene
        explanation: Educational explanation of the concept
        processingTime: Processing time in milliseconds
        source: Source of the response - 'search' or 'generated'
    """
    
    success: bool = Field(
        ...,
        description="Whether the visualization request was successful"
    )
    type: Literal["glb", "scene"] = Field(
        ...,
        description="Response type: 'glb' for GLB model URL, 'scene' for SceneBlueprint JSON"
    )
    data: dict = Field(
        ...,
        description="Response data: {'url': '...'} for GLB or SceneBlueprint object for scene"
    )
    explanation: str = Field(
        "",
        description="Educational explanation of the visualized concept"
    )
    processingTime: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    source: Literal["search", "generated"] = Field(
        ...,
        description="Source of the response: 'search' for retrieved GLB, 'generated' for procedural scene"
    )


class SceneBlueprint(BaseModel):
    """
    SceneBlueprint schema for procedural 3D scene generation.
    Compatible with Flutter renderer.
    
    Fields:
        meta: Metadata about the scene
        environment: Environment settings (lighting, background, etc.)
        primitives: List of 3D primitives (box, sphere, cylinder, torus, cone, plane)
        assets: List of external assets (optional)
    """
    
    meta: dict = Field(
        default_factory=dict,
        description="Scene metadata (name, description, etc.)"
    )
    environment: dict = Field(
        default_factory=dict,
        description="Environment settings (lighting, background, fog, etc.)"
    )
    primitives: list = Field(
        default_factory=list,
        description="List of 3D primitives with transforms and materials"
    )
    assets: list = Field(
        default_factory=list,
        description="List of external assets (textures, models, etc.)"
    )


class Primitive(BaseModel):
    """
    Single primitive in a SceneBlueprint.
    """
    type: Literal["box", "sphere", "cylinder", "torus", "cone", "plane"] = Field(
        ...,
        description="Primitive type"
    )
    transform: dict = Field(
        ...,
        description="Transform with pos, rot, scale arrays"
    )
    material: Optional[dict] = Field(
        default=None,
        description="Material properties (color, metalness, roughness)"
    )
