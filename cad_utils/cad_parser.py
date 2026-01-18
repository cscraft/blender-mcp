"""
CAD Parser Module

Parses DXF files and extracts architectural elements:
- Walls (LWPOLYLINE, LINE, ARC entities)
- Layers with material information
- Floor detection for multi-story buildings

Uses ezdxf library for DXF parsing.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger("CADParser")

# Import ezdxf with detailed error handling
try:
    import ezdxf
    logger.info(f"ezdxf imported successfully from: {ezdxf.__file__}")
    logger.info(f"ezdxf version: {ezdxf.version}")
    logger.info(f"ezdxf dir: {dir(ezdxf)[:20]}")  # First 20 attributes
except ImportError as e:
    logger.error(f"Failed to import ezdxf: {e}")
    raise


def parse_dxf(filepath: str) -> Dict[str, Any]:
    """
    Parse a DXF file and extract architectural elements.

    This function reads a DXF file using ezdxf, identifies walls (closed polylines),
    and layers (material information).

    Args:
        filepath: Absolute path to the DXF file

    Returns:
        dict: Parsed data in the format:
            {
                'filename': str,
                'units': str,
                'layers': dict,
                'walls': list,
                'bounds': tuple
            }

    Raises:
        FileNotFoundError: If filepath doesn't exist
        ValueError: If DXF file is corrupted or invalid

    Example:
        >>> parsed = parse_dxf('/path/to/floor_plan.dxf')
        >>> print(len(parsed['walls']))
        12
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DXF file not found: {filepath}")

    try:
        # Debug: Check if ezdxf has readfile
        if not hasattr(ezdxf, 'readfile'):
            available_attrs = [attr for attr in dir(ezdxf) if not attr.startswith('_')]
            raise AttributeError(f"ezdxf module does not have 'readfile'. Available: {available_attrs[:10]}")

        # Read DXF document
        doc = ezdxf.readfile(str(filepath))
        modelspace = doc.modelspace()

        # Extract units (default to millimeters if not specified)
        units = _get_units(doc)

        # Extract layers
        layers = _extract_layers(doc)

        # Extract walls from polylines and lines
        walls = _extract_walls(modelspace, units)

        # Extract openings (windows and doors)
        openings = _extract_openings(modelspace, units)

        # Calculate bounding box
        bounds = _calculate_bounds(walls)

        result = {
            'filename': filepath.name,
            'units': units,
            'layers': layers,
            'walls': walls,
            'openings': openings,
            'bounds': bounds
        }

        logger.info(f"Parsed DXF: {len(walls)} walls, {len(layers)} layers")
        return result

    except ezdxf.DXFStructureError as e:
        raise ValueError(f"Invalid DXF file structure: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing DXF file: {e}")


def _get_units(doc: Any) -> str:
    """
    Extract units from DXF document.

    Args:
        doc: ezdxf Document object

    Returns:
        str: 'millimeters', 'meters', 'inches', or 'feet'
    """
    # Try to get units from header variables
    try:
        insunits = doc.header.get('$INSUNITS', 0)
        units_map = {
            0: 'unitless',
            1: 'inches',
            2: 'feet',
            4: 'millimeters',
            5: 'centimeters',
            6: 'meters',
        }
        return units_map.get(insunits, 'millimeters')
    except:
        # Default to millimeters (common in architectural drawings)
        return 'millimeters'


def _extract_layers(doc: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract layer information from DXF.

    Args:
        doc: ezdxf Document object

    Returns:
        dict: Layer name -> layer properties
    """
    layers = {}

    for layer in doc.layers:
        layers[layer.dxf.name] = {
            'color': layer.dxf.color if hasattr(layer.dxf, 'color') else 7,
            'linetype': layer.dxf.linetype if hasattr(layer.dxf, 'linetype') else 'CONTINUOUS',
            'on': not layer.is_off() if hasattr(layer, 'is_off') else True
        }

    return layers


def _extract_walls(modelspace, units: str) -> List[Dict[str, Any]]:
    """
    Extract wall entities from modelspace.

    Identifies closed polylines, polylines, and connected lines as potential walls.

    Args:
        modelspace: ezdxf modelspace object
        units: Unit string for conversion

    Returns:
        list: List of wall dictionaries with vertices and metadata
    """
    walls = []
    unit_scale = _get_unit_scale(units)

    # Extract LWPOLYLINE entities (most common for walls)
    for entity in modelspace.query('LWPOLYLINE'):
        try:
            vertices = [(p[0] * unit_scale, p[1] * unit_scale)
                       for p in entity.get_points('xy')]

            if len(vertices) >= 3:  # At least a triangle
                wall = {
                    'vertices': vertices,
                    'closed': entity.closed,
                    'layer': entity.dxf.layer,
                    'type': 'LWPOLYLINE'
                }
                walls.append(wall)

        except Exception as e:
            logger.warning(f"Failed to process LWPOLYLINE: {e}")
            continue

    # Extract POLYLINE entities
    for entity in modelspace.query('POLYLINE'):
        try:
            vertices = [(p.dxf.location[0] * unit_scale, p.dxf.location[1] * unit_scale)
                       for p in entity.vertices]

            if len(vertices) >= 3:
                wall = {
                    'vertices': vertices,
                    'closed': entity.is_closed,
                    'layer': entity.dxf.layer,
                    'type': 'POLYLINE'
                }
                walls.append(wall)

        except Exception as e:
            logger.warning(f"Failed to process POLYLINE: {e}")
            continue

    # Extract LINE entities (will be grouped into walls later if connected)
    lines = []
    for entity in modelspace.query('LINE'):
        try:
            start = (entity.dxf.start[0] * unit_scale, entity.dxf.start[1] * unit_scale)
            end = (entity.dxf.end[0] * unit_scale, entity.dxf.end[1] * unit_scale)

            line = {
                'start': start,
                'end': end,
                'layer': entity.dxf.layer,
                'type': 'LINE'
            }
            lines.append(line)

        except Exception as e:
            logger.warning(f"Failed to process LINE: {e}")
            continue

    # Group connected lines into polylines
    if lines:
        grouped_walls = _group_lines_to_walls(lines, unit_scale)
        walls.extend(grouped_walls)

    logger.info(f"Extracted {len(walls)} wall entities")
    return walls


def _group_lines_to_walls(lines: List[Dict], tolerance: float = 0.001) -> List[Dict[str, Any]]:
    """
    Group connected LINE entities into wall polylines.

    Args:
        lines: List of line dictionaries
        tolerance: Distance tolerance for connecting lines (in meters)

    Returns:
        list: Grouped wall polylines
    """
    walls = []
    used = set()

    for i, line in enumerate(lines):
        if i in used:
            continue

        # Start a new polyline
        vertices = [line['start'], line['end']]
        layer = line['layer']
        used.add(i)

        # Try to extend the polyline
        extended = True
        while extended:
            extended = False
            last_point = vertices[-1]

            # Find a line that connects to the last point
            for j, other_line in enumerate(lines):
                if j in used or other_line['layer'] != layer:
                    continue

                # Check if other_line connects to last_point
                dist_to_start = _distance(last_point, other_line['start'])
                dist_to_end = _distance(last_point, other_line['end'])

                if dist_to_start < tolerance:
                    vertices.append(other_line['end'])
                    used.add(j)
                    extended = True
                    break
                elif dist_to_end < tolerance:
                    vertices.append(other_line['start'])
                    used.add(j)
                    extended = True
                    break

        # Check if polyline is closed
        closed = _distance(vertices[0], vertices[-1]) < tolerance
        if closed:
            vertices.pop()  # Remove duplicate last vertex

        if len(vertices) >= 3:
            wall = {
                'vertices': vertices,
                'closed': closed,
                'layer': layer,
                'type': 'GROUPED_LINES'
            }
            walls.append(wall)

    return walls


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate 2D distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def _get_unit_scale(units: str) -> float:
    """
    Get scale factor to convert from DXF units to meters.

    Args:
        units: Unit string

    Returns:
        float: Scale factor
    """
    scale_map = {
        'millimeters': 0.001,
        'centimeters': 0.01,
        'meters': 1.0,
        'inches': 0.0254,
        'feet': 0.3048,
        'unitless': 0.001  # Assume millimeters
    }
    return scale_map.get(units, 0.001)


def _calculate_bounds(walls: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box of all walls.

    Args:
        walls: List of wall dictionaries

    Returns:
        tuple: (min_x, min_y, max_x, max_y) in meters
    """
    if not walls:
        return (0.0, 0.0, 0.0, 0.0)

    all_x = []
    all_y = []

    for wall in walls:
        for vertex in wall['vertices']:
            all_x.append(vertex[0])
            all_y.append(vertex[1])

    return (min(all_x), min(all_y), max(all_x), max(all_y))


def _extract_openings(modelspace, units: str) -> List[Dict[str, Any]]:
    """
    Extract opening entities (windows and doors) from modelspace.

    Identifies INSERT entities (blocks) with names matching window/door patterns.

    Args:
        modelspace: ezdxf modelspace object
        units: Unit string for conversion

    Returns:
        list: List of opening dictionaries with position, size, rotation, and type
    """
    openings = []
    unit_scale = _get_unit_scale(units)

    # Common block name patterns for windows and doors
    window_patterns = ['WINDOW', 'WIN', 'W_', 'FENSTER', '窓']
    door_patterns = ['DOOR', 'DR', 'D_', 'TUR', 'PORTE', '扉', 'ドア']

    # Extract INSERT entities (blocks)
    for entity in modelspace.query('INSERT'):
        try:
            block_name = entity.dxf.name.upper()

            # Determine opening type
            opening_type = None
            if any(pattern in block_name for pattern in window_patterns):
                opening_type = 'window'
            elif any(pattern in block_name for pattern in door_patterns):
                opening_type = 'door'
            else:
                # Skip blocks that don't match window/door patterns
                continue

            # Extract position
            insert_point = entity.dxf.insert
            position = (
                insert_point[0] * unit_scale,
                insert_point[1] * unit_scale,
                insert_point[2] * unit_scale if len(insert_point) > 2 else 0.0
            )

            # Extract rotation (in degrees)
            rotation = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0.0

            # Extract scale
            xscale = entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1.0
            yscale = entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1.0
            zscale = entity.dxf.zscale if hasattr(entity.dxf, 'zscale') else 1.0

            # Estimate size (will be refined when placing 3D models)
            # Default sizes based on architectural standards
            if opening_type == 'window':
                width = 1.2 * xscale  # 1.2m default window width
                height = 1.2 * zscale  # 1.2m default window height
            else:  # door
                width = 0.9 * xscale  # 0.9m default door width
                height = 2.1 * zscale  # 2.1m default door height

            opening = {
                'type': opening_type,
                'block_name': entity.dxf.name,
                'position': position,
                'rotation': rotation,
                'width': width * unit_scale,
                'height': height * unit_scale,
                'layer': entity.dxf.layer,
                'scale': (xscale, yscale, zscale)
            }
            openings.append(opening)

        except Exception as e:
            logger.warning(f"Failed to process INSERT block: {e}")
            continue

    logger.info(f"Extracted {len(openings)} openings ({sum(1 for o in openings if o['type']=='window')} windows, {sum(1 for o in openings if o['type']=='door')} doors)")
    return openings
