"""
View Integrator Module

Integrates information from multiple architectural views (floor plans, elevations, sections)
to create complete 3D building element data.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import math

logger = logging.getLogger("ViewIntegrator")


def determine_wall_direction(
    opening_position: Tuple[float, float],
    walls: List[Dict[str, Any]],
    building_bounds: Tuple[float, float, float, float]
) -> str:
    """
    Determine which wall direction (north/south/east/west) an opening belongs to.

    Args:
        opening_position: (x, y) position from floor plan
        walls: List of wall dictionaries
        building_bounds: (min_x, min_y, max_x, max_y) of building

    Returns:
        'north', 'south', 'east', or 'west'
    """
    x, y = opening_position
    min_x, min_y, max_x, max_y = building_bounds

    # Tolerance for considering a point "on" an edge
    tolerance = 0.3

    # Check if point is on any of the four edges (north/south/east/west)
    # Priority: Check horizontal edges (north/south) first, then vertical (east/west)

    # Check south edge (min_y)
    if abs(y - min_y) < tolerance:
        logger.debug(f"Opening at ({x:.2f}, {y:.2f}) is on south edge (y≈{min_y:.2f})")
        return 'south'

    # Check north edge (max_y)
    if abs(y - max_y) < tolerance:
        logger.debug(f"Opening at ({x:.2f}, {y:.2f}) is on north edge (y≈{max_y:.2f})")
        return 'north'

    # Check west edge (min_x)
    if abs(x - min_x) < tolerance:
        logger.debug(f"Opening at ({x:.2f}, {y:.2f}) is on west edge (x≈{min_x:.2f})")
        return 'west'

    # Check east edge (max_x)
    if abs(x - max_x) < tolerance:
        logger.debug(f"Opening at ({x:.2f}, {y:.2f}) is on east edge (x≈{max_x:.2f})")
        return 'east'

    # If not on any edge, find the nearest edge
    edges = [
        ('south', abs(y - min_y)),
        ('north', abs(y - max_y)),
        ('west', abs(x - min_x)),
        ('east', abs(x - max_x))
    ]

    nearest_edge = min(edges, key=lambda e: e[1])[0]
    logger.debug(f"Opening at ({x:.2f}, {y:.2f}) assigned to nearest edge: {nearest_edge}")

    return nearest_edge


def integrate_elevation_data(
    floor_plan_openings: List[Dict[str, Any]],
    elevations: Dict[str, List[Dict[str, Any]]],
    building_bounds: Tuple[float, float, float, float],
    match_tolerance: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Integrate elevation view data into floor plan openings.

    Args:
        floor_plan_openings: Openings from floor plan (with x, y positions)
        elevations: Dict mapping direction ('north', 'south', etc.) to elevation openings
        building_bounds: Building bounding box
        match_tolerance: Maximum distance for matching (meters)

    Returns:
        Integrated openings with complete 3D information
    """
    integrated_openings = []

    logger.info(f"Starting integration with {len(floor_plan_openings)} floor plan openings")
    logger.info(f"Available elevations: {list(elevations.keys())}")
    logger.info(f"Building bounds: {building_bounds}")

    for fp_opening in floor_plan_openings:
        # Determine which wall this opening is on
        position = fp_opening.get('position', [0, 0, 0])
        x, y = position[0], position[1]

        wall_direction = determine_wall_direction((x, y), [], building_bounds)

        # Find corresponding elevation
        elevation_openings = elevations.get(wall_direction, [])

        if not elevation_openings:
            logger.warning(f"No elevation data for {wall_direction} wall, using defaults")
            # Use default values
            integrated_opening = fp_opening.copy()
            integrated_opening['wall_direction'] = wall_direction
            integrated_opening['data_source'] = 'floor_plan_only'
            integrated_openings.append(integrated_opening)
            continue

        # Match opening by X coordinate (for south/north) or Y coordinate (for east/west)
        matched = False

        if wall_direction in ['south', 'north']:
            # Match by X coordinate
            logger.debug(f"  Trying to match by X coordinate: fp_x={x:.2f}")
            for elev_opening in elevation_openings:
                distance = abs(elev_opening['x_position'] - x)
                logger.debug(f"    Elev opening at x={elev_opening['x_position']:.2f}, distance={distance:.2f}m")
                if distance < match_tolerance:
                    # Match found! Integrate data
                    integrated_opening = _merge_opening_data(
                        fp_opening,
                        elev_opening,
                        wall_direction
                    )
                    integrated_openings.append(integrated_opening)
                    matched = True
                    logger.info(f"✓ Matched {fp_opening['id']} with elevation data "
                              f"(x={x:.2f}m, height={elev_opening['height']:.2f}m)")
                    break

        else:  # east or west
            # Match by Y coordinate
            for elev_opening in elevation_openings:
                if abs(elev_opening['x_position'] - y) < match_tolerance:
                    integrated_opening = _merge_opening_data(
                        fp_opening,
                        elev_opening,
                        wall_direction
                    )
                    integrated_openings.append(integrated_opening)
                    matched = True
                    logger.info(f"Matched {fp_opening['id']} with elevation data "
                              f"(y={y:.2f}m, height={elev_opening['height']:.2f}m)")
                    break

        if not matched:
            logger.warning(f"No elevation match for {fp_opening['id']} at ({x:.2f}, {y:.2f}), using defaults")
            integrated_opening = fp_opening.copy()
            integrated_opening['wall_direction'] = wall_direction
            integrated_opening['data_source'] = 'floor_plan_only'
            integrated_openings.append(integrated_opening)

    logger.info(f"Integrated {len(integrated_openings)} openings with elevation data")

    return integrated_openings


def _merge_opening_data(
    floor_plan_opening: Dict[str, Any],
    elevation_opening: Dict[str, Any],
    wall_direction: str
) -> Dict[str, Any]:
    """
    Merge floor plan and elevation opening data.

    Floor plan provides: X, Y position, rotation
    Elevation provides: Z position (sill height), width, height
    """
    # Start with floor plan data
    merged = floor_plan_opening.copy()

    # Update position with Z coordinate from elevation
    position = merged.get('position', [0, 0, 0])
    merged['position'] = [
        position[0],
        position[1],
        elevation_opening['sill_height']  # Z from elevation
    ]

    # Update dimensions from elevation
    merged['width'] = elevation_opening['width']
    merged['height'] = elevation_opening['height']
    merged['sill_height'] = elevation_opening['sill_height']
    merged['top_height'] = elevation_opening['top_height']

    # Update type if elevation provides more specific info
    if 'type' in elevation_opening:
        merged['type'] = elevation_opening['type']

    # Add metadata
    merged['wall_direction'] = wall_direction
    merged['data_source'] = 'integrated'
    merged['elevation_layer'] = elevation_opening.get('layer', '')

    return merged


def calculate_building_bounds(walls: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box of the building from walls.

    Only uses walls from structural layers (WALL_CONCRETE, WALL, etc.)
    and filters by Y-range to exclude elevation view walls.

    Strategy:
    1. Find the largest (by area) closed polygon from WALL_CONCRETE layers
    2. Use that as the building bounds
    3. This excludes elevation views which are typically smaller rectangles

    Returns:
        (min_x, min_y, max_x, max_y)
    """
    # Only use walls from structural layers
    structural_layers = ['WALL_CONCRETE', 'WALL', 'WALL_INTERIOR', 'WALL_EXTERIOR']

    # Find all wall polygons with their bounds
    wall_polygons = []

    for wall in walls:
        layer = wall.get('layer', '').upper()

        # Skip non-structural layers
        if layer not in structural_layers and not layer.startswith('WALL'):
            continue

        points = []
        if 'centerline' in wall:
            points = wall['centerline']
        elif 'outer_boundary' in wall:
            points = wall['outer_boundary']

        if len(points) >= 3:  # Valid polygon
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            wall_polygons.append({
                'points': points,
                'bbox': bbox,
                'area': area,
                'layer': wall.get('layer', '')
            })

    if not wall_polygons:
        logger.warning("No structural wall polygons found")
        return (0, 0, 10, 10)

    # Use the largest polygon (by area) as the building bounds
    # This should be the main floor plan, not elevation views
    largest_polygon = max(wall_polygons, key=lambda p: p['area'])
    bounds = largest_polygon['bbox']

    logger.info(f"Calculated building bounds from largest polygon (area={largest_polygon['area']:.2f}m²): {bounds}")
    logger.info(f"  Total polygons found: {len(wall_polygons)}, using largest from layer '{largest_polygon['layer']}'")

    return bounds


def integrate_views(
    floor_plan_data: Dict[str, Any],
    detected_views: List[Dict],
    dxf_path: str,
    dxf_bounds: Tuple[float, float, float, float],
    image_path: str,
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main integration function: combine floor plan with elevations and sections.

    Args:
        floor_plan_data: Parsed floor plan data (walls, openings)
        detected_views: List of detected views from Gemini
        dxf_path: Path to DXF file
        dxf_bounds: Overall DXF bounds
        image_path: Path to rendered DXF image (for coordinate conversion)
        gemini_api_key: Optional API key for section analysis

    Returns:
        Integrated building elements dict
    """
    from . import elevation_parser, gemini_view_detector, section_analyzer

    # Parse elevation views
    elevations = {}
    section_bounds = None

    for view in detected_views:
        view_type = view['view_type']

        if 'ELEVATION' in view_type or 'elevation' in view_type.lower():
            # Determine direction
            if 'SOUTH' in view_type or 'south' in view_type.lower():
                direction = 'south'
            elif 'NORTH' in view_type or 'north' in view_type.lower():
                direction = 'north'
            elif 'EAST' in view_type or 'east' in view_type.lower():
                direction = 'east'
            elif 'WEST' in view_type or 'west' in view_type.lower():
                direction = 'west'
            elif 'FRONT' in view_type:
                direction = 'south'  # Assume front is south
            elif 'SIDE' in view_type:
                direction = 'east'   # Assume side is east
            else:
                direction = 'unknown'

            # Convert image bbox to DXF coords
            dxf_view_bounds = gemini_view_detector.convert_image_coords_to_dxf_coords(
                view['bbox_pixels'],
                image_path,
                dxf_bounds
            )

            # Parse elevation
            try:
                elevation_openings = elevation_parser.parse_elevation(
                    dxf_path,
                    dxf_view_bounds,
                    direction,
                    units_scale=1.0
                )

                elevations[direction] = elevation_openings
                logger.info(f"Parsed {direction} elevation: {len(elevation_openings)} openings")

            except Exception as e:
                logger.error(f"Failed to parse {direction} elevation: {e}")
                elevations[direction] = []

        elif 'SECTION' in view_type or 'section' in view_type.lower():
            # Store section bounds for later analysis
            section_bounds = gemini_view_detector.convert_image_coords_to_dxf_coords(
                view['bbox_pixels'],
                image_path,
                dxf_bounds
            )
            logger.info(f"Detected section view at bounds: {section_bounds}")

    # Calculate building bounds from walls
    building_bounds = calculate_building_bounds(floor_plan_data.get('walls', []))

    # Integrate elevation data into floor plan openings
    integrated_openings = integrate_elevation_data(
        floor_plan_data.get('openings', []),
        elevations,
        building_bounds
    )

    # NEW: Integrate section level data if available
    section_data = None
    if section_bounds and gemini_api_key:
        try:
            logger.info("Analyzing section drawing for level markers...")
            section_data = section_analyzer.parse_section_levels(
                image_path,
                section_bounds,
                gemini_api_key,
                model_name="gemini-2.0-flash"
            )
            logger.info(f"Section analysis complete: {len(section_data.get('floor_levels', []))} floors detected")
        except Exception as e:
            logger.warning(f"Section analysis failed: {e}, continuing without section data")

    # Return integrated data
    metadata = {
        'integration': 'multi_view',
        'elevations_used': list(elevations.keys()),
        'building_bounds': building_bounds
    }

    if section_data:
        metadata['section_analysis'] = {
            'confidence': section_data.get('overall_confidence', 'unknown'),
            'floors_detected': len(section_data.get('floor_levels', []))
        }

    return {
        'walls': floor_plan_data.get('walls', []),
        'openings': integrated_openings,
        'floors': floor_plan_data.get('floors', []),
        'section_data': section_data,  # NEW: Include section data
        'metadata': metadata
    }
