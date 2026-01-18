"""
Column Detector Module - Phase 3 of Gemini Integration

Detects structural columns from architectural floor plans:
- RC (Reinforced Concrete) columns: HATCH patterns, SOLID fills, rectangles/circles
- S (Steel) columns: H-shaped sections, box sections, INSERT blocks

This module analyzes DXF entities to identify column locations, typically at
grid intersections (通り芯交点), and classifies them by structure type.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import math

logger = logging.getLogger("ColumnDetector")

try:
    import ezdxf
    from ezdxf.math import Vec2
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logger.warning("ezdxf not available")


def detect_columns(
    dxf_path: str,
    grid_system: Optional[Dict[str, Any]] = None,
    structure_type: str = "unknown",
    floor_bounds: Optional[Tuple[float, float, float, float]] = None,
    units_scale: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Detect columns from DXF floor plan.

    Args:
        dxf_path: Path to DXF file
        grid_system: Grid system data from calibration (optional)
        structure_type: "RC_structure", "S_structure", or "unknown"
        floor_bounds: (min_x, min_y, max_x, max_y) to filter columns within floor plan
        units_scale: Scale factor for unit conversion

    Returns:
        List of column dictionaries:
        [
            {
                "id": "c_001",
                "position": [x, y],  # meters
                "grid_position": "X1-Y1",  # if grid detected
                "section_type": "RECT_400x400" | "CIRCLE_400" | "H_400x200",
                "width": 0.4,  # meters
                "depth": 0.4,  # meters
                "material": "Concrete" | "Steel",
                "layer": "COLUMN",
                "confidence": "high" | "medium" | "low"
            }
        ]

    Raises:
        RuntimeError: If ezdxf not available
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for column detection")

    logger.info(f"Starting column detection: structure_type={structure_type}")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    columns = []

    # Step 1: Detect HATCH entities (RC columns with hatching patterns)
    hatch_columns = _detect_hatch_columns(msp, floor_bounds, units_scale)
    columns.extend(hatch_columns)
    logger.info(f"Detected {len(hatch_columns)} columns from HATCH entities")

    # Step 2: Detect SOLID entities (filled rectangles/circles)
    solid_columns = _detect_solid_columns(msp, floor_bounds, units_scale)
    columns.extend(solid_columns)
    logger.info(f"Detected {len(solid_columns)} columns from SOLID entities")

    # Step 3: Detect filled polylines (common for RC columns)
    polyline_columns = _detect_polyline_columns(msp, floor_bounds, units_scale)
    columns.extend(polyline_columns)
    logger.info(f"Detected {len(polyline_columns)} columns from LWPOLYLINE entities")

    # Step 4: Detect CIRCLE entities (circular columns)
    circle_columns = _detect_circle_columns(msp, floor_bounds, units_scale)
    columns.extend(circle_columns)
    logger.info(f"Detected {len(circle_columns)} columns from CIRCLE entities")

    # Step 5: Detect INSERT blocks (H-beams, box sections for S structure)
    insert_columns = _detect_insert_columns(msp, floor_bounds, units_scale)
    columns.extend(insert_columns)
    logger.info(f"Detected {len(insert_columns)} columns from INSERT blocks")

    # Remove duplicates (same position within tolerance)
    columns = _remove_duplicate_columns(columns, tolerance=0.1)

    # Assign grid positions if grid system is available
    if grid_system and grid_system.get("detected"):
        columns = _assign_grid_positions(columns, grid_system, units_scale)

    # Infer material from structure type if not already set
    for col in columns:
        if col["material"] == "unknown":
            if structure_type == "RC_structure":
                col["material"] = "Concrete"
            elif structure_type == "S_structure":
                col["material"] = "Steel"

    # Generate IDs
    for i, col in enumerate(columns, start=1):
        col["id"] = f"c_{i:03d}"

    logger.info(f"Total columns detected: {len(columns)}")

    return columns


def _detect_hatch_columns(
    modelspace,
    floor_bounds: Optional[Tuple[float, float, float, float]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect columns from HATCH entities (concrete hatching patterns).
    """
    columns = []

    for entity in modelspace.query('HATCH'):
        try:
            # Get boundary paths
            if not entity.paths:
                continue

            # Get bounding box
            bbox = entity.bounding_box
            if bbox is None:
                continue

            min_x, min_y, _ = bbox[0]
            max_x, max_y, _ = bbox[1]

            # Filter by floor bounds
            if floor_bounds:
                fb_min_x, fb_min_y, fb_max_x, fb_max_y = floor_bounds
                if not (fb_min_x <= min_x <= fb_max_x and fb_min_y <= min_y <= fb_max_y):
                    continue

            width = abs(max_x - min_x) * units_scale
            depth = abs(max_y - min_y) * units_scale

            # Filter by size (columns are typically 0.3m - 1.0m)
            if not (0.3 <= width <= 1.0 and 0.3 <= depth <= 1.0):
                continue

            center_x = ((min_x + max_x) / 2) * units_scale
            center_y = ((min_y + max_y) / 2) * units_scale

            # Check if roughly square or circular
            aspect_ratio = max(width, depth) / min(width, depth)
            if aspect_ratio <= 1.5:  # Roughly square
                section_type = f"RECT_{int(width*1000)}x{int(depth*1000)}"
            else:
                continue  # Skip non-square hatches

            columns.append({
                "position": [center_x, center_y],
                "grid_position": None,
                "section_type": section_type,
                "width": width,
                "depth": depth,
                "material": "Concrete",  # HATCH typically indicates concrete
                "layer": entity.dxf.layer,
                "confidence": "high"
            })

        except Exception as e:
            logger.debug(f"Failed to process HATCH: {e}")
            continue

    return columns


def _detect_solid_columns(
    modelspace,
    floor_bounds: Optional[Tuple[float, float, float, float]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect columns from SOLID entities (filled rectangles).
    """
    columns = []

    for entity in modelspace.query('SOLID'):
        try:
            # SOLID has 3-4 corner points
            points = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2]
            if hasattr(entity.dxf, 'vtx3'):
                points.append(entity.dxf.vtx3)

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Filter by floor bounds
            if floor_bounds:
                fb_min_x, fb_min_y, fb_max_x, fb_max_y = floor_bounds
                if not (fb_min_x <= min_x <= fb_max_x and fb_min_y <= min_y <= fb_max_y):
                    continue

            width = abs(max_x - min_x) * units_scale
            depth = abs(max_y - min_y) * units_scale

            # Filter by size
            if not (0.3 <= width <= 1.0 and 0.3 <= depth <= 1.0):
                continue

            center_x = ((min_x + max_x) / 2) * units_scale
            center_y = ((min_y + max_y) / 2) * units_scale

            section_type = f"RECT_{int(width*1000)}x{int(depth*1000)}"

            columns.append({
                "position": [center_x, center_y],
                "grid_position": None,
                "section_type": section_type,
                "width": width,
                "depth": depth,
                "material": "Concrete",
                "layer": entity.dxf.layer,
                "confidence": "medium"
            })

        except Exception as e:
            logger.debug(f"Failed to process SOLID: {e}")
            continue

    return columns


def _detect_polyline_columns(
    modelspace,
    floor_bounds: Optional[Tuple[float, float, float, float]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect columns from LWPOLYLINE entities (closed, filled rectangles).
    """
    columns = []

    for entity in modelspace.query('LWPOLYLINE'):
        try:
            # Check if closed
            if not entity.closed:
                continue

            points = list(entity.get_points('xy'))
            if len(points) < 4:
                continue

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Filter by floor bounds
            if floor_bounds:
                fb_min_x, fb_min_y, fb_max_x, fb_max_y = floor_bounds
                if not (fb_min_x <= min_x <= fb_max_x and fb_min_y <= min_y <= fb_max_y):
                    continue

            width = abs(max_x - min_x) * units_scale
            depth = abs(max_y - min_y) * units_scale

            # Filter by size (columns are 0.3m - 1.0m)
            if not (0.3 <= width <= 1.0 and 0.3 <= depth <= 1.0):
                continue

            # Check if roughly rectangular (4 or 5 points for closed shape)
            if len(points) not in [4, 5]:
                continue

            center_x = ((min_x + max_x) / 2) * units_scale
            center_y = ((min_y + max_y) / 2) * units_scale

            section_type = f"RECT_{int(width*1000)}x{int(depth*1000)}"

            # Check layer name for hints
            layer = entity.dxf.layer.upper()
            material = "Concrete"
            confidence = "medium"

            if any(kw in layer for kw in ['COLUMN', 'COL', 'PILLAR', '柱']):
                confidence = "high"

            columns.append({
                "position": [center_x, center_y],
                "grid_position": None,
                "section_type": section_type,
                "width": width,
                "depth": depth,
                "material": material,
                "layer": entity.dxf.layer,
                "confidence": confidence
            })

        except Exception as e:
            logger.debug(f"Failed to process LWPOLYLINE: {e}")
            continue

    return columns


def _detect_circle_columns(
    modelspace,
    floor_bounds: Optional[Tuple[float, float, float, float]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect columns from CIRCLE entities (circular columns).
    """
    columns = []

    for entity in modelspace.query('CIRCLE'):
        try:
            center_x = entity.dxf.center[0] * units_scale
            center_y = entity.dxf.center[1] * units_scale
            radius = entity.dxf.radius * units_scale
            diameter = radius * 2

            # Filter by floor bounds
            if floor_bounds:
                fb_min_x, fb_min_y, fb_max_x, fb_max_y = floor_bounds
                if not (fb_min_x <= center_x / units_scale <= fb_max_x and
                        fb_min_y <= center_y / units_scale <= fb_max_y):
                    continue

            # Filter by size (columns are typically 0.3m - 1.0m diameter)
            if not (0.3 <= diameter <= 1.0):
                continue

            section_type = f"CIRCLE_{int(diameter*1000)}"

            # Check layer name
            layer = entity.dxf.layer.upper()
            confidence = "medium"

            if any(kw in layer for kw in ['COLUMN', 'COL', 'PILLAR', '柱']):
                confidence = "high"

            columns.append({
                "position": [center_x, center_y],
                "grid_position": None,
                "section_type": section_type,
                "width": diameter,
                "depth": diameter,
                "material": "Concrete",  # Circular columns typically RC
                "layer": entity.dxf.layer,
                "confidence": confidence
            })

        except Exception as e:
            logger.debug(f"Failed to process CIRCLE: {e}")
            continue

    return columns


def _detect_insert_columns(
    modelspace,
    floor_bounds: Optional[Tuple[float, float, float, float]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect columns from INSERT blocks (H-beams, box sections for steel structure).
    """
    columns = []

    for entity in modelspace.query('INSERT'):
        try:
            # Get block name
            block_name = entity.dxf.name.upper()

            # Check if block name indicates column
            column_keywords = ['COLUMN', 'COL', 'H-', 'BOX', 'STEEL', 'S-', '柱', 'H型', '□']
            if not any(kw in block_name for kw in column_keywords):
                continue

            center_x = entity.dxf.insert[0] * units_scale
            center_y = entity.dxf.insert[1] * units_scale

            # Filter by floor bounds
            if floor_bounds:
                fb_min_x, fb_min_y, fb_max_x, fb_max_y = floor_bounds
                if not (fb_min_x <= center_x / units_scale <= fb_max_x and
                        fb_min_y <= center_y / units_scale <= fb_max_y):
                    continue

            # Try to infer section type from block name
            section_type = "H_400x200"  # Default for steel
            material = "Steel"

            if 'H-' in block_name or 'H型' in block_name:
                section_type = "H_400x200"
            elif 'BOX' in block_name or '□' in block_name:
                section_type = "BOX_400x400"

            # Default dimensions for steel columns
            width = 0.4
            depth = 0.2 if 'H-' in section_type else 0.4

            columns.append({
                "position": [center_x, center_y],
                "grid_position": None,
                "section_type": section_type,
                "width": width,
                "depth": depth,
                "material": material,
                "layer": entity.dxf.layer,
                "confidence": "medium"
            })

        except Exception as e:
            logger.debug(f"Failed to process INSERT: {e}")
            continue

    return columns


def _remove_duplicate_columns(
    columns: List[Dict[str, Any]],
    tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Remove duplicate columns that are at the same position.

    Args:
        columns: List of column dictionaries
        tolerance: Maximum distance (meters) to consider columns as duplicates

    Returns:
        Deduplicated list of columns
    """
    if not columns:
        return []

    unique_columns = []

    for col in columns:
        is_duplicate = False

        for existing_col in unique_columns:
            distance = math.sqrt(
                (col["position"][0] - existing_col["position"][0]) ** 2 +
                (col["position"][1] - existing_col["position"][1]) ** 2
            )

            if distance < tolerance:
                # Duplicate found - keep the one with higher confidence
                is_duplicate = True

                # Replace if new column has higher confidence
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                if confidence_order.get(col["confidence"], 0) > confidence_order.get(existing_col["confidence"], 0):
                    unique_columns.remove(existing_col)
                    unique_columns.append(col)

                break

        if not is_duplicate:
            unique_columns.append(col)

    logger.info(f"Removed {len(columns) - len(unique_columns)} duplicate columns")

    return unique_columns


def _assign_grid_positions(
    columns: List[Dict[str, Any]],
    grid_system: Dict[str, Any],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Assign grid positions to columns based on proximity to grid intersections.

    Args:
        columns: List of column dictionaries
        grid_system: Grid system data from calibration
        units_scale: Scale factor for unit conversion

    Returns:
        Updated columns with grid_position assigned
    """
    x_axes = grid_system.get("x_axes", [])
    y_axes = grid_system.get("y_axes", [])

    if not x_axes or not y_axes:
        return columns

    # Convert grid coordinates to world coordinates (meters)
    grid_intersections = []
    for x_axis in x_axes:
        for y_axis in y_axes:
            grid_intersections.append({
                "label": f"{x_axis['label']}-{y_axis['label']}",
                "x": x_axis["world_x_mm"] / 1000,  # mm to meters
                "y": y_axis.get("world_y_mm", 0) / 1000
            })

    # Assign columns to nearest grid intersection (within 0.5m tolerance)
    tolerance = 0.5

    for col in columns:
        col_x, col_y = col["position"]

        nearest_grid = None
        min_distance = float('inf')

        for grid in grid_intersections:
            distance = math.sqrt(
                (col_x - grid["x"]) ** 2 +
                (col_y - grid["y"]) ** 2
            )

            if distance < min_distance and distance < tolerance:
                min_distance = distance
                nearest_grid = grid

        if nearest_grid:
            col["grid_position"] = nearest_grid["label"]
            logger.debug(f"Assigned column at ({col_x:.2f}, {col_y:.2f}) to grid {nearest_grid['label']}")

    return columns
