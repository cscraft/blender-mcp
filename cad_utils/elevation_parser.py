"""
Elevation Parser Module

Extracts information from elevation views (side views) in architectural DXF files.
Specifically extracts window and door openings with their heights and sizes.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("ElevationParser")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logger.warning("ezdxf not available")


def parse_elevation(
    dxf_path: str,
    elevation_bounds: Tuple[float, float, float, float],
    elevation_direction: str,
    units_scale: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Parse elevation view to extract opening information.

    Args:
        dxf_path: Path to DXF file
        elevation_bounds: (min_x, min_y, max_x, max_y) bounds of elevation view in DXF coords
        elevation_direction: 'north', 'south', 'east', or 'west'
        units_scale: Scale factor for unit conversion

    Returns:
        List of opening dictionaries with structure:
        [
            {
                'x_position': float,  # Horizontal position in elevation
                'sill_height': float,  # Bottom edge height (meters)
                'top_height': float,   # Top edge height (meters)
                'height': float,       # Opening height (meters)
                'width': float,        # Opening width (meters)
                'type': str,           # 'window' or 'door'
                'layer': str           # DXF layer name
            },
            ...
        ]
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for elevation parsing")

    logger.info(f"Parsing {elevation_direction} elevation: bounds={elevation_bounds}")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = elevation_bounds
    openings = []

    # Extract rectangles from LWPOLYLINE entities
    for entity in msp.query('LWPOLYLINE'):
        try:
            points = list(entity.get_points('xy'))

            if len(points) < 4:
                continue

            # Check if all points are within elevation bounds
            if not all(min_x <= p[0] <= max_x and min_y <= p[1] <= max_y for p in points):
                continue

            # Check if it's a rectangle (4 or 5 points for closed shape)
            if len(points) in [4, 5]:
                rect_info = _analyze_rectangle(points, entity.dxf.layer)

                if rect_info and _is_opening(rect_info):
                    # FIXED: Convert absolute DXF Y coordinate to floor-relative height
                    # In elevation views, the floor slab is at min_y of the elevation bounds
                    # So sill_height should be: (opening_min_y - floor_level)
                    floor_level_y = min_y  # Elevation view floor level (bottom of bounds)

                    sill_height_relative = (rect_info['min_y'] - floor_level_y) * units_scale
                    top_height_relative = (rect_info['max_y'] - floor_level_y) * units_scale

                    opening = {
                        'x_position': rect_info['center_x'] * units_scale,
                        'sill_height': sill_height_relative,  # NOW floor-relative
                        'top_height': top_height_relative,    # NOW floor-relative
                        'height': rect_info['height'] * units_scale,
                        'width': rect_info['width'] * units_scale,
                        'type': _classify_opening_type(rect_info),
                        'layer': entity.dxf.layer,
                        'direction': elevation_direction
                    }
                    openings.append(opening)
                    logger.debug(f"Found opening: {opening['type']} at x={opening['x_position']:.2f}m, "
                               f"sill_height={sill_height_relative:.2f}m (floor-relative), "
                               f"height={opening['height']:.2f}m")

        except Exception as e:
            logger.debug(f"Failed to process LWPOLYLINE: {e}")
            continue

    # Extract rectangles from LINE entities (4 lines forming a rectangle)
    lines_in_bounds = []
    for entity in msp.query('LINE'):
        try:
            start = (entity.dxf.start[0], entity.dxf.start[1])
            end = (entity.dxf.end[0], entity.dxf.end[1])

            # Check if line is within bounds
            if (min_x <= start[0] <= max_x and min_y <= start[1] <= max_y and
                min_x <= end[0] <= max_x and min_y <= end[1] <= max_y):
                lines_in_bounds.append({
                    'start': start,
                    'end': end,
                    'layer': entity.dxf.layer
                })

        except Exception as e:
            logger.debug(f"Failed to process LINE: {e}")
            continue

    # Group lines into rectangles
    rect_openings = _find_rectangles_from_lines(lines_in_bounds, units_scale, elevation_direction)
    openings.extend(rect_openings)

    logger.info(f"Extracted {len(openings)} openings from {elevation_direction} elevation")

    return openings


def _analyze_rectangle(points: List[Tuple[float, float]], layer: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a closed polyline to extract rectangle properties.

    Returns:
        dict with keys: min_x, max_x, min_y, max_y, width, height, center_x, center_y, layer
    """
    if len(points) < 4:
        return None

    # Remove duplicate last point if closed
    if points[0] == points[-1]:
        points = points[:-1]

    if len(points) != 4:
        return None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    width = max_x - min_x
    height = max_y - min_y

    # Filter out very small or very large shapes
    if width < 0.1 or height < 0.1:  # Too small
        return None

    if width > 10.0 or height > 10.0:  # Too large (likely a wall)
        return None

    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'width': width,
        'height': height,
        'center_x': (min_x + max_x) / 2,
        'center_y': (min_y + max_y) / 2,
        'layer': layer
    }


def _is_opening(rect_info: Dict[str, Any]) -> bool:
    """
    Determine if a rectangle represents an opening (window/door).

    Criteria:
    - Layer name contains 'WINDOW', 'DOOR', 'WIN', 'OPENING'
    - Size is reasonable for an opening (0.3m - 3m in either dimension)
    """
    layer = rect_info['layer'].upper()

    # Check layer name
    opening_keywords = ['WINDOW', 'WIN', 'DOOR', 'OPENING', 'FENSTER', '窓', '扉']
    layer_indicates_opening = any(kw in layer for kw in opening_keywords)

    # Check size
    width = rect_info['width']
    height = rect_info['height']

    reasonable_size = (0.3 <= width <= 3.0) and (0.3 <= height <= 3.0)

    return layer_indicates_opening or reasonable_size


def _classify_opening_type(rect_info: Dict[str, Any]) -> str:
    """
    Classify opening as 'window' or 'door' based on layer name and dimensions.
    """
    layer = rect_info['layer'].upper()

    # Check layer name first
    if any(kw in layer for kw in ['DOOR', 'TUR', '扉']):
        return 'door'

    if any(kw in layer for kw in ['WINDOW', 'WIN', 'FENSTER', '窓']):
        return 'window'

    # Classify by dimensions
    # Doors are typically taller (height > width) and reach floor (min_y close to 0)
    height = rect_info['height']
    width = rect_info['width']

    if height > 1.8 and rect_info['min_y'] < 0.5:  # Tall and near floor
        return 'door'
    else:
        return 'window'


def _find_rectangles_from_lines(
    lines: List[Dict],
    units_scale: float,
    elevation_direction: str
) -> List[Dict[str, Any]]:
    """
    Group 4 lines into rectangles representing openings.
    """
    # This is a simplified implementation
    # A full implementation would need more sophisticated line grouping
    rectangles = []

    # For now, we skip this and rely on LWPOLYLINE detection
    # TODO: Implement line grouping algorithm if needed

    return rectangles


def match_opening_to_floor_plan(
    elevation_opening: Dict[str, Any],
    floor_plan_openings: List[Dict[str, Any]],
    tolerance: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Match an opening from elevation view to corresponding opening in floor plan.

    Args:
        elevation_opening: Opening from elevation view
        floor_plan_openings: List of openings from floor plan
        tolerance: Maximum distance (meters) to consider a match

    Returns:
        Matched floor plan opening, or None if no match found
    """
    elevation_x = elevation_opening['x_position']

    for fp_opening in floor_plan_openings:
        # Check if opening is on the correct wall direction
        if fp_opening.get('wall_direction') != elevation_opening['direction']:
            continue

        # Check X coordinate proximity
        fp_x = fp_opening['position'][0]  # X coordinate from floor plan

        if abs(fp_x - elevation_x) < tolerance:
            logger.debug(f"Matched elevation opening at x={elevation_x:.2f} "
                        f"to floor plan opening at x={fp_x:.2f}")
            return fp_opening

    return None
