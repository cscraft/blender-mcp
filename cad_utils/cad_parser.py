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


def parse_dxf(filepath: str, user_scale: float = 1.0) -> Dict[str, Any]:
    """
    Parse a DXF file and extract architectural elements.

    This function reads a DXF file using ezdxf, identifies walls (closed polylines),
    and layers (material information).

    Args:
        filepath: Absolute path to the DXF file
        user_scale: User-specified scale factor (default 1.0, use 0.1 to make 10x smaller)

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
        >>> parsed = parse_dxf('/path/to/floor_plan.dxf', user_scale=0.1)
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
        walls = _extract_walls(modelspace, units, user_scale)

        # Extract openings (windows and doors)
        openings = _extract_openings(modelspace, units, user_scale)

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


def _extract_walls(modelspace, units: str, user_scale: float = 1.0) -> List[Dict[str, Any]]:
    """
    Extract wall entities from modelspace.

    Identifies closed polylines, polylines, and connected lines as potential walls.
    Also detects architectural standard double-line walls (2 parallel polylines).

    Args:
        modelspace: ezdxf modelspace object
        units: Unit string for conversion
        user_scale: User-specified scale factor

    Returns:
        list: List of wall dictionaries with vertices and metadata
    """
    walls = []
    unit_scale = _get_unit_scale(units) * user_scale

    # Extract all LWPOLYLINE entities first
    lwpolylines = []
    for entity in modelspace.query('LWPOLYLINE'):
        try:
            vertices = [(p[0] * unit_scale, p[1] * unit_scale)
                       for p in entity.get_points('xy')]

            if len(vertices) >= 3:  # At least a triangle
                lwpolylines.append({
                    'vertices': vertices,
                    'closed': entity.closed,
                    'layer': entity.dxf.layer,
                    'type': 'LWPOLYLINE',
                    'entity': entity
                })

        except Exception as e:
            logger.warning(f"Failed to process LWPOLYLINE: {e}")
            continue

    # Try to detect double-line walls (architectural standard)
    # Group polylines by layer
    layer_groups = {}
    for poly in lwpolylines:
        layer = poly['layer']
        if layer not in layer_groups:
            layer_groups[layer] = []
        layer_groups[layer].append(poly)

    # For each layer, check if there are pairs of parallel polylines
    used_indices = set()
    for layer, polys in layer_groups.items():
        # Check for wall-related layer names
        is_wall_layer = any(keyword in layer.upper() for keyword in ['WALL', 'MUR', 'WAND', 'PARED'])

        if is_wall_layer and len(polys) >= 2:
            # Try to find pairs of parallel polylines with same vertex count
            for i in range(len(polys)):
                if i in used_indices:
                    continue

                poly1 = polys[i]
                if not poly1['closed'] or len(poly1['vertices']) < 4:
                    continue

                for j in range(i + 1, len(polys)):
                    if j in used_indices:
                        continue

                    poly2 = polys[j]
                    if not poly2['closed'] or len(poly2['vertices']) != len(poly1['vertices']):
                        continue

                    # Check if polylines are parallel (similar shape, different size)
                    thickness = _calculate_wall_thickness(poly1['vertices'], poly2['vertices'])

                    if thickness is not None and 0.05 <= thickness <= 0.5:  # Reasonable wall thickness (5cm to 50cm)
                        # This is a double-line wall - use the centerline
                        centerline = _calculate_centerline(poly1['vertices'], poly2['vertices'])

                        wall = {
                            'vertices': centerline,
                            'closed': True,
                            'layer': layer,
                            'type': 'DOUBLE_LINE_WALL',
                            'thickness': thickness,
                            'outer_vertices': poly1['vertices'] if _is_outer_polyline(poly1['vertices'], poly2['vertices']) else poly2['vertices'],
                            'inner_vertices': poly2['vertices'] if _is_outer_polyline(poly1['vertices'], poly2['vertices']) else poly1['vertices']
                        }
                        walls.append(wall)
                        used_indices.add(i)
                        used_indices.add(j)
                        break

    # Add remaining single-line polylines as walls (centerline representation)
    for i, poly in enumerate(lwpolylines):
        if i not in used_indices:
            wall = {
                'vertices': poly['vertices'],
                'closed': poly['closed'],
                'layer': poly['layer'],
                'type': 'LWPOLYLINE'
            }
            walls.append(wall)

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


def _extract_openings(modelspace, units: str, user_scale: float = 1.0) -> List[Dict[str, Any]]:
    """
    Extract opening entities (windows and doors) from modelspace.

    Identifies INSERT entities (blocks) with names matching window/door patterns.

    Args:
        modelspace: ezdxf modelspace object
        units: Unit string for conversion
        user_scale: User-specified scale factor

    Returns:
        list: List of opening dictionaries with position, size, rotation, and type
    """
    openings = []
    unit_scale = _get_unit_scale(units) * user_scale

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
            # Default sizes based on architectural standards (in meters)
            if opening_type == 'window':
                width = 1.2 * xscale  # 1.2m default window width
                height = 1.2 * zscale  # 1.2m default window height
            else:  # door
                width = 0.9 * xscale  # 0.9m default door width
                height = 2.1 * zscale  # 2.1m default door height

            # Note: width and height are already in meters, don't apply unit_scale again
            opening = {
                'type': opening_type,
                'block_name': entity.dxf.name,
                'position': position,
                'rotation': rotation,
                'width': width,  # Already in meters
                'height': height,  # Already in meters
                'layer': entity.dxf.layer,
                'scale': (xscale, yscale, zscale)
            }
            openings.append(opening)

        except Exception as e:
            logger.warning(f"Failed to process INSERT block: {e}")
            continue

    logger.info(f"Extracted {len(openings)} openings ({sum(1 for o in openings if o['type']=='window')} windows, {sum(1 for o in openings if o['type']=='door')} doors)")
    return openings


def _calculate_wall_thickness(vertices1: List[tuple], vertices2: List[tuple]) -> float:
    """
    Calculate the thickness between two parallel polylines.

    Args:
        vertices1: List of (x, y) tuples for first polyline
        vertices2: List of (x, y) tuples for second polyline

    Returns:
        float: Wall thickness in meters, or None if polylines are not parallel
    """
    import math

    if len(vertices1) != len(vertices2):
        return None

    # Calculate distances between corresponding vertices
    distances = []
    for v1, v2 in zip(vertices1, vertices2):
        dist = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
        distances.append(dist)

    # Check if distances are consistent (parallel polylines should have similar distances)
    if not distances:
        return None

    avg_distance = sum(distances) / len(distances)
    max_deviation = max(abs(d - avg_distance) for d in distances)

    # If deviation is too large, polylines are not parallel
    if max_deviation > avg_distance * 0.3:  # 30% tolerance
        return None

    return avg_distance


def _calculate_centerline(vertices1: List[tuple], vertices2: List[tuple]) -> List[tuple]:
    """
    Calculate the centerline between two parallel polylines.

    Args:
        vertices1: List of (x, y) tuples for first polyline
        vertices2: List of (x, y) tuples for second polyline

    Returns:
        list: List of (x, y) tuples representing the centerline
    """
    centerline = []
    for v1, v2 in zip(vertices1, vertices2):
        center_x = (v1[0] + v2[0]) / 2.0
        center_y = (v1[1] + v2[1]) / 2.0
        centerline.append((center_x, center_y))

    return centerline


def _is_outer_polyline(vertices1: List[tuple], vertices2: List[tuple]) -> bool:
    """
    Determine if vertices1 represents the outer polyline compared to vertices2.

    Args:
        vertices1: List of (x, y) tuples for first polyline
        vertices2: List of (x, y) tuples for second polyline

    Returns:
        bool: True if vertices1 is the outer polyline
    """
    # Calculate the area of each polyline (using shoelace formula)
    def polygon_area(vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    area1 = polygon_area(vertices1)
    area2 = polygon_area(vertices2)

    # The outer polyline should have a larger area
    return area1 > area2


def _remove_duplicate_vertices(vertices: List[List[float]], tolerance: float = 0.001) -> List[List[float]]:
    """
    Remove duplicate consecutive vertices within tolerance.

    Args:
        vertices: List of [x, y] coordinates
        tolerance: Distance tolerance in meters (default 0.001m = 1mm)

    Returns:
        list: Deduplicated vertices
    """
    if len(vertices) <= 1:
        return vertices

    deduplicated = [vertices[0]]

    for i in range(1, len(vertices)):
        prev_x, prev_y = deduplicated[-1]
        curr_x, curr_y = vertices[i]

        # Calculate distance
        dist = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5

        # Only add if distance > tolerance
        if dist > tolerance:
            deduplicated.append([curr_x, curr_y])

    return deduplicated


def _validate_and_normalize_elements(elements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize building elements according to data rules (NEW SCHEMA VERSION 2.0).

    Rules:
    1. All dimensions in meters
    2. Perimeter walls must be closed polylines
    3. Openings must be blocks with center point + dimension attributes
    4. Wall thickness: 0.05m <= thickness <= 0.5m
    5. Heights: walls 1.5-10m, windows 0.3-3m, doors 1.8-3m
    6. Remove duplicate vertices (tolerance 0.001m)
    7. Round coordinates to 3 decimal places (mm precision)
    8. Normalize layer names (uppercase, trim whitespace)

    Args:
        elements: Raw building elements dictionary

    Returns:
        dict: Validated and normalized building elements
    """
    validated = elements.copy()

    # UPDATED: Validate floors[] structure (new schema)
    for floor_idx, floor in enumerate(validated.get('floors', [])):
        floor_id = floor.get('id', f'{floor_idx+1}F')

        # Validate walls within this floor
        valid_walls = []
        for wall in floor.get('elements', {}).get('walls', []):
            try:
                # Normalize layer name
                wall['layer'] = wall.get('layer', 'WALL').upper().strip()

                # Validate wall type
                if wall.get('type') == 'DOUBLE_LINE_WALL':
                    outer = wall.get('outer_boundary', [])
                    inner = wall.get('inner_boundary', [])

                    # Check closed polyline (first == last point)
                    if len(outer) >= 4 and outer[0] != outer[-1]:
                        logger.warning(f"{floor_id} Wall {wall['id']}: outer boundary not closed, auto-closing")
                        outer.append(outer[0])
                        wall['outer_boundary'] = outer

                    if len(inner) >= 4 and inner[0] != inner[-1]:
                        logger.warning(f"{floor_id} Wall {wall['id']}: inner boundary not closed, auto-closing")
                        inner.append(inner[0])
                        wall['inner_boundary'] = inner

                    # Validate minimum vertices (3 points + closure = 4)
                    if len(outer) < 4:
                        logger.error(f"{floor_id} Wall {wall['id']}: outer boundary has < 3 vertices, skipping")
                        continue

                    # Round coordinates to 3 decimal places
                    wall['outer_boundary'] = [[round(x, 3), round(y, 3)] for x, y in outer]
                    wall['inner_boundary'] = [[round(x, 3), round(y, 3)] for x, y in inner]

                    # Remove duplicate consecutive vertices
                    wall['outer_boundary'] = _remove_duplicate_vertices(wall['outer_boundary'])
                    wall['inner_boundary'] = _remove_duplicate_vertices(wall['inner_boundary'])

                else:  # CENTERLINE_WALL
                    centerline = wall.get('centerline', [])

                    # Check closed polyline
                    if len(centerline) >= 4 and centerline[0] != centerline[-1]:
                        logger.warning(f"{floor_id} Wall {wall['id']}: centerline not closed, auto-closing")
                        centerline.append(centerline[0])
                        wall['centerline'] = centerline

                    # Validate minimum vertices
                    if len(centerline) < 4:
                        logger.error(f"{floor_id} Wall {wall['id']}: centerline has < 3 vertices, skipping")
                        continue

                    # Round and deduplicate
                    wall['centerline'] = [[round(x, 3), round(y, 3)] for x, y in centerline]
                    wall['centerline'] = _remove_duplicate_vertices(wall['centerline'])

                # Validate wall thickness
                thickness = wall.get('thickness', 0.12)
                if not (0.05 <= thickness <= 0.5):
                    logger.warning(f"{floor_id} Wall {wall['id']}: thickness {thickness}m outside valid range (0.05-0.5m)")

                # Validate wall height
                height = wall.get('height', 2.4)
                if not (1.5 <= height <= 10.0):
                    logger.warning(f"{floor_id} Wall {wall['id']}: height {height}m outside typical range (1.5-10m)")

                # Round dimensions
                wall['thickness'] = round(thickness, 3)
                wall['height'] = round(height, 3)
                wall['z_offset'] = round(wall.get('z_offset', 0.0), 3)

                valid_walls.append(wall)

            except Exception as e:
                logger.error(f"{floor_id} Wall {wall.get('id', 'unknown')} validation failed: {e}")

        floor['elements']['walls'] = valid_walls
        logger.info(f"{floor_id} Wall validation: {len(valid_walls)} passed")

        # Validate openings within this floor
        valid_openings = []
        for opening in floor.get('elements', {}).get('openings', []):
            try:
                # Normalize layer name
                opening['layer'] = opening.get('layer', 'WALL').upper().strip()

                # Validate position (center point)
                position = opening.get('position', [0, 0, 0])
                if len(position) != 3:
                    logger.error(f"{floor_id} Opening {opening['id']}: invalid position, skipping")
                    continue

                # Round position to mm precision
                opening['position'] = [round(x, 3) for x in position]

                # Validate dimensions
                width = opening.get('width', 1.0)
                height = opening.get('height', 1.0)
                opening_type = opening.get('type', 'window')

                if opening_type == 'window':
                    if not (0.3 <= height <= 3.0):
                        logger.warning(f"{floor_id} Opening {opening['id']}: window height {height}m outside typical range (0.3-3m)")
                else:  # door
                    if not (1.8 <= height <= 3.0):
                        logger.warning(f"{floor_id} Opening {opening['id']}: door height {height}m outside typical range (1.8-3m)")

                if not (0.3 <= width <= 5.0):
                    logger.warning(f"{floor_id} Opening {opening['id']}: width {width}m outside typical range (0.3-5m)")

                # Round dimensions
                opening['width'] = round(width, 3)
                opening['height'] = round(height, 3)
                opening['depth'] = round(opening.get('depth', 0.35), 3)
                opening['rotation'] = round(opening.get('rotation', 0), 1)

                # Validate rotation (-360 to 360)
                if not (-360 <= opening['rotation'] <= 360):
                    logger.warning(f"{floor_id} Opening {opening['id']}: rotation {opening['rotation']}° outside range")
                    opening['rotation'] = opening['rotation'] % 360

                valid_openings.append(opening)

            except Exception as e:
                logger.error(f"{floor_id} Opening {opening.get('id', 'unknown')} validation failed: {e}")

        floor['elements']['openings'] = valid_openings
        logger.info(f"{floor_id} Opening validation: {len(valid_openings)} passed")

        # Validate slabs within this floor
        valid_slabs = []
        for slab in floor.get('elements', {}).get('slabs', []):
            try:
                slab['layer'] = slab.get('layer', 'FLOOR').upper().strip()

                boundary = slab.get('boundary', [])
                if len(boundary) >= 4 and boundary[0] != boundary[-1]:
                    logger.warning(f"{floor_id} Slab {slab['id']}: boundary not closed, auto-closing")
                    boundary.append(boundary[0])
                    slab['boundary'] = boundary

                if len(boundary) < 4:
                    logger.error(f"{floor_id} Slab {slab['id']}: boundary has < 3 vertices, skipping")
                    continue

                slab['boundary'] = [[round(x, 3), round(y, 3)] for x, y in boundary]
                slab['boundary'] = _remove_duplicate_vertices(slab['boundary'])

                # Validate thickness_mm (NEW: in mm as per schema)
                thickness_mm = slab.get('thickness_mm', 150)
                if not (50 <= thickness_mm <= 500):  # 50mm to 500mm reasonable range
                    logger.warning(f"{floor_id} Slab {slab['id']}: thickness {thickness_mm}mm outside range")

                slab['thickness_mm'] = int(thickness_mm)

                valid_slabs.append(slab)

            except Exception as e:
                logger.error(f"{floor_id} Slab {slab.get('id', 'unknown')} validation failed: {e}")

        floor['elements']['slabs'] = valid_slabs
        logger.info(f"{floor_id} Slab validation: {len(valid_slabs)} passed")

    return validated


def parse_dxf_to_elements(
    filepath: str,
    user_scale: float = 1.0,
    wall_height: float = 2.4,
    use_floor_plan_only: bool = False,
    use_gemini_integration: bool = True,
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a DXF file and extract architectural elements in structured JSON format.

    This function converts DXF data into a standardized building elements structure
    that can be used for Geometry Nodes-based 3D generation.

    Args:
        filepath: Absolute path to the DXF file
        user_scale: User-specified scale factor (default 1.0)
        wall_height: Default wall height in meters (default 2.4m)
        use_floor_plan_only: If True, use only floor plan (deprecated, use use_gemini_integration instead)
        use_gemini_integration: If True, use Gemini AI to detect and integrate multiple views (default True)
        gemini_api_key: Gemini API key (required if use_gemini_integration=True)

    Returns:
        dict: Building elements in JSON-compatible format:
            {
                'version': '1.0',
                'units': 'meters',
                'scale': float,
                'elements': {
                    'walls': [...],
                    'floors': [...],
                    'openings': [...],
                    'columns': [...],
                    'beams': [...],
                    'roofs': [...]
                },
                'metadata': {...}
            }

    Example:
        >>> elements = parse_dxf_to_elements('/path/to/plan.dxf', user_scale=0.1)
        >>> print(len(elements['elements']['walls']))
        4
    """
    import json
    import tempfile
    from datetime import datetime
    from . import view_detector, view_integrator

    # Multi-view integration with Gemini AI
    detected_views = None
    dxf_bounds = None
    floor_plan_bounds = None
    floor_plan_bounds_padded = None
    image_path = None

    # NEW: Calibration data from Phase 1
    calibration_data = None

    if use_gemini_integration and gemini_api_key:
        try:
            from . import dxf_visualizer, gemini_view_detector, gemini_calibration

            # Step 1: Render DXF to image
            logger.info("Rendering DXF to image for Gemini analysis...")
            temp_image = tempfile.mktemp(suffix='.png')

            image_path, dxf_bounds = dxf_visualizer.render_dxf_to_image(
                filepath,
                temp_image,
                dpi=200
            )

            logger.info(f"DXF rendered to {image_path}")

            # Step 1.5: NEW - Phase 1 Calibration (Grid detection, scale, structure type)
            logger.info("Phase 1: Calibrating drawing (grid detection, scale, structure type)...")
            try:
                calibration_data = gemini_calibration.calibrate_drawing(
                    image_path,
                    gemini_api_key,
                    model_name="gemini-2.0-flash"  # CHANGED: 最新の安定モデル
                )
                logger.info(f"Calibration complete: grid_detected={calibration_data['grid_system']['detected']}, "
                           f"structure={calibration_data['structure_type']}")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}, continuing without grid detection")
                calibration_data = None

            # Step 2: Detect views with Gemini
            logger.info("Detecting views with Gemini Vision API...")
            detected_views = gemini_view_detector.detect_views_with_gemini(
                image_path,
                gemini_api_key,
                model_name="gemini-2.5-flash"
            )

            logger.info(f"Gemini detected {len(detected_views)} views")

            # Step 3: Find floor plan for initial parsing
            floor_plan = gemini_view_detector.find_floor_plan_view(detected_views)

            if floor_plan:
                # Convert to DXF coordinates (tight bounds for wall filtering)
                floor_plan_bounds_tight = gemini_view_detector.convert_image_coords_to_dxf_coords(
                    floor_plan['bbox_pixels'],
                    image_path,
                    dxf_bounds
                )
                logger.info(f"Floor plan bounds (tight): {floor_plan_bounds_tight}")

                # Create padded bounds for opening filtering
                # Gemini often detects bboxes that are too tight for openings on edges
                padding = 0.3  # meters (reduced from 0.5 to avoid including elevations)
                floor_plan_bounds_padded = (
                    floor_plan_bounds_tight[0] - padding,
                    floor_plan_bounds_tight[1] - padding,
                    floor_plan_bounds_tight[2] + padding,
                    floor_plan_bounds_tight[3] + padding
                )

                # Use tight bounds for wall filtering to exclude elevation walls
                floor_plan_bounds = floor_plan_bounds_tight
                logger.info(f"Using tight bounds for walls, padded bounds for openings")

        except Exception as e:
            logger.error(f"Gemini integration failed: {e}, falling back to standard parsing")
            use_gemini_integration = False

    elif use_floor_plan_only:
        try:
            doc = ezdxf.readfile(filepath)
            modelspace = doc.modelspace()

            # Detect all views
            detected_views = view_detector.detect_views(modelspace, gap_threshold=1.5)

            if detected_views:
                # Find primary floor plan
                floor_plan = view_detector.find_primary_floor_plan(detected_views)

                if floor_plan:
                    floor_plan_bounds = floor_plan.bounds
                    logger.info(f"Using floor plan view only: bounds={floor_plan_bounds}")
                    logger.info(f"  Detected {len(detected_views)} total views, using floor plan only")
                else:
                    logger.warning("No floor plan detected, using entire DXF")
            else:
                logger.warning("View detection found no separate views, using entire DXF")

        except Exception as e:
            logger.warning(f"View detection failed: {e}, using entire DXF")

    # Parse DXF using existing function
    parsed = parse_dxf(filepath, user_scale)

    # Initialize building elements structure (NEW SCHEMA - Gemini体系準拠)
    # NEW: Apply calibration data from Phase 1 if available
    structure_type = 'unknown'
    grid_system = {
        'detected': False,
        'x_axes': [],
        'y_axes': [],
        'origin': None
    }

    if calibration_data:
        structure_type = calibration_data.get('structure_type', 'unknown')
        grid_system = calibration_data.get('grid_system', grid_system)
        logger.info(f"Applied calibration: structure={structure_type}, grid_detected={grid_system['detected']}")

    elements = {
        'version': '2.0',  # Updated version for new schema
        'units': 'meters',
        'scale': user_scale,
        'project_info': {
            'name': Path(filepath).stem,
            'unit': 'mm',  # Internal unit in mm as per Gemini_read.md
            'structure_type': structure_type,  # NEW: Set from Gemini calibration
            'source_file': parsed['filename'],
            'import_date': datetime.now().isoformat()
        },
        'grid_system': grid_system,  # NEW: Set from Gemini calibration
        'floors': [
            {
                'id': '1F',
                'elevation_mm': 0,  # Ground level (will be updated from section analysis)
                'floor_height_mm': int(wall_height * 1000),  # Convert to mm (default 2400mm)
                'ceiling_height_mm': int(wall_height * 1000),  # Will be updated from section (e.g., 2700mm)
                'slab_level_mm': 0,  # Will be set from section analysis
                'elements': {
                    'walls': [],
                    'openings': [],
                    'columns': [],
                    'beams': [],
                    'slabs': []
                }
            }
        ],
        'metadata': {
            'original_units': parsed['units'],
            'total_floors': 1,
            'building_height': wall_height,
            'schema_version': '2.0'
        }
    }

    # Add calibration metadata if available
    if calibration_data:
        elements['metadata']['calibration'] = {
            'confidence': calibration_data.get('confidence', 'low'),
            'scale_factor': calibration_data.get('scale_factor'),
            'notes': calibration_data.get('notes', '')
        }

    # Helper function to check if entity is within floor plan bounds
    def is_within_bounds(vertices, bounds):
        """Check if any vertex is within the bounds"""
        if not bounds:
            return True  # No filtering if no bounds detected

        min_x, min_y, max_x, max_y = bounds
        for v in vertices:
            x, y = v[0], v[1]
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False

    # Convert walls to structured format (add to first floor)
    wall_count = 0
    for i, wall in enumerate(parsed['walls']):
        # Filter by floor plan bounds if enabled
        vertices = wall.get('vertices', wall.get('outer_vertices', []))
        if floor_plan_bounds and not is_within_bounds(vertices, floor_plan_bounds):
            logger.debug(f"Skipping wall_{i:03d} (outside floor plan bounds)")
            continue

        wall_element = {
            'id': f"wall_{wall_count:03d}",
            'type': wall.get('type', 'CENTERLINE_WALL'),
            'layer': wall.get('layer', 'WALL'),
            'thickness': wall.get('thickness', 0.12),
            'height': wall_height,
            'z_offset': 0.0,
            'confidence': 'high'  # NEW: confidence tag for manual review
        }

        # Handle different wall types
        if wall.get('type') == 'DOUBLE_LINE_WALL':
            wall_element['outer_boundary'] = wall.get('outer_vertices', [])
            wall_element['inner_boundary'] = wall.get('inner_vertices', [])
            wall_element['material_hint'] = _infer_material_from_layer(wall.get('layer', ''))
        else:
            # Centerline wall
            wall_element['centerline'] = wall.get('vertices', [])
            wall_element['material_hint'] = _infer_material_from_layer(wall.get('layer', ''))

        # Add to first floor's walls (NEW: floors[] structure)
        elements['floors'][0]['elements']['walls'].append(wall_element)
        wall_count += 1

    # Convert openings to structured format (add to first floor)
    opening_count = 0
    for i, opening in enumerate(parsed['openings']):
        # Filter by floor plan bounds if enabled (use padded bounds for openings)
        position = opening.get('position', [0, 0, 0])
        # Use padded bounds for openings if available, otherwise use regular bounds
        bounds_for_openings = floor_plan_bounds_padded if floor_plan_bounds_padded else floor_plan_bounds

        if bounds_for_openings:
            min_x, min_y, max_x, max_y = bounds_for_openings
            if not (min_x <= position[0] <= max_x and min_y <= position[1] <= max_y):
                logger.debug(f"Skipping opening_{i:03d} (outside floor plan bounds)")
                continue

        opening_type = opening.get('type', 'window')
        opening_element = {
            'id': f"{opening_type}_{opening_count:03d}",
            'type': opening_type,
            'block_name': opening.get('block_name', f"{opening_type.upper()}_{opening_count:02d}"),
            'layer': opening.get('layer', 'WALL'),
            'position': list(position),
            'width': opening.get('width', 1.0),
            'height': opening.get('height', 1.0 if opening_type == 'window' else 2.1),
            'rotation': opening.get('rotation', 0),
            'confidence': 'medium'  # NEW: confidence tag
        }

        # Add opening-specific properties
        if opening_type == 'window':
            opening_element['sill_height'] = 0.9
            opening_element['depth'] = 0.35
        else:  # door
            opening_element['depth'] = 0.40

        # Add to first floor's openings (NEW: floors[] structure)
        elements['floors'][0]['elements']['openings'].append(opening_element)
        opening_count += 1

    # Integrate elevation data if Gemini detected multiple views
    if use_gemini_integration and detected_views and len(detected_views) > 1:
        try:
            logger.info("Integrating elevation data with floor plan...")

            floor_plan_data = {
                'walls': elements['floors'][0]['elements']['walls'],  # UPDATED: use floors[] structure
                'openings': elements['floors'][0]['elements']['openings'],  # UPDATED: use floors[] structure
                'floors': []  # Legacy compatibility
            }

            integrated_data = view_integrator.integrate_views(
                floor_plan_data,
                detected_views,
                filepath,
                dxf_bounds,
                image_path,
                gemini_api_key=gemini_api_key  # NEW: Pass API key for section analysis
            )

            # Replace with integrated data (UPDATED: write back to floors[0])
            elements['floors'][0]['elements']['walls'] = integrated_data['walls']
            elements['floors'][0]['elements']['openings'] = integrated_data['openings']

            # Add integration metadata
            if 'metadata' in integrated_data:
                elements['metadata'].update(integrated_data['metadata'])

            # NEW: Integrate section level data if available
            if 'section_data' in integrated_data and integrated_data['section_data']:
                logger.info("Integrating section level data into floors structure...")
                from . import section_analyzer

                elements['floors'] = section_analyzer.integrate_section_data_to_floors(
                    elements['floors'],
                    integrated_data['section_data']
                )

                logger.info(f"Section data integrated: {len(integrated_data['section_data'].get('floor_levels', []))} floors updated")

            logger.info(f"Integration complete: {len(integrated_data['openings'])} openings with elevation data")

        except Exception as e:
            logger.error(f"View integration failed: {e}, using floor plan data only")
            import traceback
            traceback.print_exc()

    # Create default slab based on wall bounds (UPDATED: use slabs in floors[])
    if parsed['bounds']:
        min_x, min_y, max_x, max_y = parsed['bounds']
        floor_boundary = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y]
        ]

        slab_element = {
            'id': 'slab_001',
            'layer': 'FLOOR',
            'boundary': floor_boundary,
            'thickness_mm': 150,  # NEW: thickness in mm (default for office buildings)
            'material': 'Concrete',  # NEW: explicit material
            'confidence': 'low'  # NEW: auto-generated slab has low confidence
        }
        elements['floors'][0]['elements']['slabs'].append(slab_element)

    # NEW: Detect columns (Phase 3)
    logger.info("Phase 3: Detecting columns...")
    try:
        from . import column_detector

        # Use calibration data for structure type and grid system
        structure_type = calibration_data.get('structure_type', 'unknown') if calibration_data else 'unknown'
        grid_system = calibration_data.get('grid_system') if calibration_data else None

        # Detect columns
        columns = column_detector.detect_columns(
            filepath,
            grid_system=grid_system,
            structure_type=structure_type,
            floor_bounds=floor_plan_bounds,
            units_scale=user_scale
        )

        # Add columns to first floor
        elements['floors'][0]['elements']['columns'] = columns

        logger.info(f"Detected {len(columns)} columns")

    except Exception as e:
        logger.warning(f"Column detection failed: {e}, continuing without columns")
        import traceback
        traceback.print_exc()

    # NEW: Detect beams (Phase 4)
    logger.info("Phase 4: Detecting beams...")
    try:
        from . import beam_detector

        columns = elements['floors'][0]['elements']['columns']
        floor_height_mm = elements['floors'][0].get('floor_height_mm', 4000)
        ceiling_height_mm = elements['floors'][0].get('ceiling_height_mm', 2700)
        structure_type = calibration_data.get('structure_type', 'unknown') if calibration_data else 'unknown'
        grid_system = calibration_data.get('grid_system') if calibration_data else None

        beams = beam_detector.detect_beams(
            filepath,
            columns=columns,
            floor_height_mm=floor_height_mm,
            ceiling_height_mm=ceiling_height_mm,
            grid_system=grid_system,
            structure_type=structure_type,
            units_scale=user_scale
        )

        # Add beams to first floor
        elements['floors'][0]['elements']['beams'] = beams

        logger.info(f"Detected {len(beams)} beams")

    except Exception as e:
        logger.warning(f"Beam detection failed: {e}, continuing without beams")
        import traceback
        traceback.print_exc()

    # NEW: Enhance slab thickness calculation (Phase 5)
    logger.info("Phase 5: Calculating slab thickness...")
    try:
        from . import slab_calculator

        structure_type = calibration_data.get('structure_type', 'unknown') if calibration_data else 'unknown'
        floor_info = elements['floors'][0]

        # Enhance each slab with calculated thickness
        slabs = elements['floors'][0]['elements']['slabs']
        for slab in slabs:
            enhanced_slab = slab_calculator.enhance_slab_data(
                slab,
                structure_type=structure_type,
                floor_info=floor_info
            )
            slab.update(enhanced_slab)

        logger.info(f"Enhanced {len(slabs)} slabs with thickness calculation")

        # Log slab statistics
        if slabs:
            thicknesses = [s.get('thickness_mm', 0) for s in slabs]
            avg_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0
            logger.info(f"  Slab thickness range: {min(thicknesses)}mm - {max(thicknesses)}mm (avg: {avg_thickness:.0f}mm)")

    except Exception as e:
        logger.warning(f"Slab thickness calculation failed: {e}, using defaults")
        import traceback
        traceback.print_exc()

    logger.info(f"Converted to building elements: {len(elements['floors'][0]['elements']['walls'])} walls, "
                f"{len(elements['floors'][0]['elements']['openings'])} openings, "
                f"{len(elements['floors'][0]['elements']['columns'])} columns, "
                f"{len(elements['floors'][0]['elements']['beams'])} beams, "
                f"{len(elements['floors'][0]['elements']['slabs'])} slabs")

    # Validate and normalize elements
    validated_elements = _validate_and_normalize_elements(elements)

    return validated_elements


def save_elements_json(elements: Dict[str, Any], output_path: str) -> None:
    """
    Save building elements to a JSON file.

    Args:
        elements: Building elements dictionary from parse_dxf_to_elements()
        output_path: Path to save JSON file

    Example:
        >>> elements = parse_dxf_to_elements('plan.dxf')
        >>> save_elements_json(elements, 'plan_elements.json')
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved building elements to: {output_path}")


def _infer_material_from_layer(layer_name: str) -> str:
    """
    Infer material type from CAD layer name.

    Args:
        layer_name: Layer name from DXF (e.g., 'WALL_CONCRETE', 'FLOOR_WOOD')

    Returns:
        str: Material hint ('concrete', 'brick', 'wood', 'glass', 'tile', etc.)
    """
    layer_upper = layer_name.upper()

    if 'CONCRETE' in layer_upper or 'CONC' in layer_upper:
        return 'concrete'
    elif 'BRICK' in layer_upper:
        return 'brick'
    elif 'WOOD' in layer_upper or 'TIMBER' in layer_upper:
        return 'wood'
    elif 'GLASS' in layer_upper or 'WINDOW' in layer_upper:
        return 'glass'
    elif 'TILE' in layer_upper:
        return 'tile'
    elif 'METAL' in layer_upper or 'STEEL' in layer_upper:
        return 'metal'
    elif 'STONE' in layer_upper:
        return 'stone'
    elif 'PLASTER' in layer_upper or 'GYPSUM' in layer_upper:
        return 'plaster'
    else:
        # Default based on element type
        if 'WALL' in layer_upper:
            return 'concrete'
        elif 'FLOOR' in layer_upper or 'SLAB' in layer_upper:
            return 'concrete'
        elif 'ROOF' in layer_upper:
            return 'concrete'
        else:
            return 'generic'
