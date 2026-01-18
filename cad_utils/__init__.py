"""
CAD Utils Package for Blender MCP

This package provides utilities for importing and processing CAD files (DXF/DWG)
into Blender 3D models.

Modules:
    cad_parser: DXF file parsing and entity extraction
    geometry_builder: 3D geometry construction from 2D CAD data
    material_mapper: Material assignment based on CAD layer names
"""

__version__ = "1.0.0"

from . import cad_parser
from . import geometry_builder

__all__ = ['cad_parser', 'geometry_builder']
