"""
CAD Utils Package for Blender MCP

This package provides utilities for importing and processing CAD files (DXF/DWG)
into Blender 3D models.

Modules:
    cad_parser: DXF file parsing and entity extraction
    geometry_builder: 3D geometry construction from 2D CAD data
    material_mapper: Material assignment based on CAD layer names
    view_detector: Automatic detection of architectural views (floor plans, elevations)
    elevation_parser: Parse elevation views to extract opening information
    view_integrator: Integrate multiple views into complete 3D building data
"""

__version__ = "1.0.0"

from . import cad_parser
from . import geometry_builder
from . import view_detector
from . import elevation_parser
from . import view_integrator

__all__ = ['cad_parser', 'geometry_builder', 'view_detector', 'elevation_parser', 'view_integrator']
