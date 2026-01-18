"""
Beam Detection Module - Phase 4 of 3-Phase Chain-of-Thought

Detects beams from architectural floor plans:
- Dashed lines (LINETYPE=DASHED) between columns
- Polylines connecting column centers
- Beam depth estimation based on span

This is Phase 4 of the implementation plan.
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("BeamDetector")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logger.warning("ezdxf not available")


def detect_beams(
    dxf_path: str,
    columns: List[Dict[str, Any]],
    floor_height_mm: float = 4000,
    ceiling_height_mm: float = 2700,
    grid_system: Optional[Dict[str, Any]] = None,
    structure_type: str = "unknown",
    units_scale: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Detect beams from DXF floor plan.

    Args:
        dxf_path: Path to DXF file
        columns: List of detected columns
        floor_height_mm: Floor-to-floor height in mm (default: 4000)
        ceiling_height_mm: Ceiling height in mm (default: 2700)
        grid_system: Optional grid system data
        structure_type: "RC_structure", "S_structure", or "unknown"
        units_scale: DXF units scale factor

    Returns:
        List of beam dictionaries:
        [{
            "id": "b_001",
            "start_position": [x1, y1],
            "end_position": [x2, y2],
            "start_column": "c_001" or None,
            "end_column": "c_002" or None,
            "span_mm": 7200,
            "depth_mm": 600,
            "width_mm": 200,
            "bottom_level_mm": 3400,
            "material": "Concrete" | "Steel",
            "layer": "BEAM",
            "confidence": "high" | "medium" | "low"
        }]

    Detection methods:
    1. Dashed lines (LINETYPE=DASHED) - most reliable
    2. Lines on BEAM layer
    3. Lines connecting column centers
    4. Grid-based beam generation (fallback)
    """
    if not EZDXF_AVAILABLE:
        logger.error("ezdxf is required for beam detection")
        return []

    if not Path(dxf_path).exists():
        logger.error(f"DXF file not found: {dxf_path}")
        return []

    logger.info(f"Starting beam detection from: {dxf_path}")
    logger.info(f"  Input columns: {len(columns)}")
    logger.info(f"  Floor height: {floor_height_mm}mm")
    logger.info(f"  Ceiling height: {ceiling_height_mm}mm")

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception as e:
        logger.error(f"Failed to read DXF: {e}")
        return []

    # Detection pipeline
    beams = []
    beam_id_counter = 1

    # Method 1: Dashed lines
    dashed_beams = _detect_dashed_beams(msp, units_scale)
    logger.info(f"  Method 1 (Dashed lines): {len(dashed_beams)} beams")
    for beam_data in dashed_beams:
        beam_data["id"] = f"b_{beam_id_counter:03d}"
        beam_id_counter += 1
        beams.append(beam_data)

    # Method 2: BEAM layer lines
    layer_beams = _detect_beam_layer_lines(msp, units_scale)
    logger.info(f"  Method 2 (BEAM layer): {len(layer_beams)} beams")
    for beam_data in layer_beams:
        beam_data["id"] = f"b_{beam_id_counter:03d}"
        beam_id_counter += 1
        beams.append(beam_data)

    # Method 3: Lines connecting columns
    if columns:
        column_beams = _detect_column_connections(msp, columns, units_scale)
        logger.info(f"  Method 3 (Column connections): {len(column_beams)} beams")
        for beam_data in column_beams:
            beam_data["id"] = f"b_{beam_id_counter:03d}"
            beam_id_counter += 1
            beams.append(beam_data)

    # Method 4: Grid-based beam generation (fallback)
    if grid_system and grid_system.get("detected") and len(beams) == 0:
        grid_beams = _generate_grid_beams(grid_system, columns, units_scale)
        logger.info(f"  Method 4 (Grid-based): {len(grid_beams)} beams")
        for beam_data in grid_beams:
            beam_data["id"] = f"b_{beam_id_counter:03d}"
            beam_id_counter += 1
            beams.append(beam_data)

    # Remove duplicates
    beams = _remove_duplicate_beams(beams, tolerance=0.1)
    logger.info(f"  After deduplication: {len(beams)} beams")

    # Assign column references
    beams = _assign_column_references(beams, columns, tolerance=0.3)

    # Calculate beam properties
    for beam in beams:
        _calculate_beam_properties(
            beam,
            floor_height_mm,
            ceiling_height_mm,
            structure_type
        )

    logger.info(f"Beam detection complete: {len(beams)} beams detected")
    return beams


def _detect_dashed_beams(
    modelspace,
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect beams from dashed lines (LINETYPE=DASHED).

    Args:
        modelspace: DXF modelspace
        units_scale: Units scale factor

    Returns:
        List of beam data (without IDs)
    """
    beams = []

    for entity in modelspace.query('LINE'):
        # Check linetype
        linetype = entity.dxf.linetype.upper() if hasattr(entity.dxf, 'linetype') else 'CONTINUOUS'

        if 'DASH' in linetype or 'HIDDEN' in linetype:
            start = entity.dxf.start
            end = entity.dxf.end

            # Only horizontal or vertical beams
            dx = abs(end.x - start.x)
            dy = abs(end.y - start.y)

            if dx < 0.01 or dy < 0.01:  # Nearly straight
                beams.append({
                    "start_position": [start.x * units_scale, start.y * units_scale],
                    "end_position": [end.x * units_scale, end.y * units_scale],
                    "layer": entity.dxf.layer,
                    "confidence": "high"
                })

    # Also check LWPOLYLINE with dashed linetype
    for entity in modelspace.query('LWPOLYLINE'):
        linetype = entity.dxf.linetype.upper() if hasattr(entity.dxf, 'linetype') else 'CONTINUOUS'

        if 'DASH' in linetype or 'HIDDEN' in linetype:
            points = list(entity.get_points('xy'))
            if len(points) >= 2:
                # Create beam for each segment
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i + 1]

                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])

                    if dx < 0.01 or dy < 0.01:
                        beams.append({
                            "start_position": [p1[0] * units_scale, p1[1] * units_scale],
                            "end_position": [p2[0] * units_scale, p2[1] * units_scale],
                            "layer": entity.dxf.layer,
                            "confidence": "high"
                        })

    return beams


def _detect_beam_layer_lines(
    modelspace,
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect beams from lines on BEAM layer.

    Args:
        modelspace: DXF modelspace
        units_scale: Units scale factor

    Returns:
        List of beam data
    """
    beams = []

    for entity in modelspace.query('LINE'):
        layer = entity.dxf.layer.upper()

        if 'BEAM' in layer or 'GIRDER' in layer or '梁' in layer:
            start = entity.dxf.start
            end = entity.dxf.end

            # Filter by length (beams are typically > 1m)
            length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
            if length * units_scale < 1.0:
                continue

            beams.append({
                "start_position": [start.x * units_scale, start.y * units_scale],
                "end_position": [end.x * units_scale, end.y * units_scale],
                "layer": entity.dxf.layer,
                "confidence": "medium"
            })

    return beams


def _detect_column_connections(
    modelspace,
    columns: List[Dict[str, Any]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Detect beams by finding lines that connect column centers.

    Args:
        modelspace: DXF modelspace
        columns: List of detected columns
        units_scale: Units scale factor

    Returns:
        List of beam data
    """
    if not columns:
        return []

    beams = []
    column_positions = [tuple(col["position"]) for col in columns]

    # Check all LINE entities
    for entity in modelspace.query('LINE'):
        start = entity.dxf.start
        end = entity.dxf.end

        start_pos = (start.x * units_scale, start.y * units_scale)
        end_pos = (end.x * units_scale, end.y * units_scale)

        # Check if line connects two columns
        tolerance = 0.3  # 300mm tolerance
        start_near_column = any(
            _distance_2d(start_pos, col_pos) < tolerance
            for col_pos in column_positions
        )
        end_near_column = any(
            _distance_2d(end_pos, col_pos) < tolerance
            for col_pos in column_positions
        )

        if start_near_column and end_near_column:
            # This line connects two columns - likely a beam
            beams.append({
                "start_position": [start_pos[0], start_pos[1]],
                "end_position": [end_pos[0], end_pos[1]],
                "layer": entity.dxf.layer,
                "confidence": "medium"
            })

    return beams


def _generate_grid_beams(
    grid_system: Dict[str, Any],
    columns: List[Dict[str, Any]],
    units_scale: float
) -> List[Dict[str, Any]]:
    """
    Generate beams along grid lines (fallback method).

    Args:
        grid_system: Grid system data
        columns: List of columns
        units_scale: Units scale factor

    Returns:
        List of beam data
    """
    beams = []

    x_axes = grid_system.get("x_axes", [])
    y_axes = grid_system.get("y_axes", [])

    if not x_axes or not y_axes:
        return []

    # Generate horizontal beams (along X direction)
    for y_axis in y_axes:
        y_world = y_axis.get("world_y_mm", y_axis.get("world_x_mm", 0)) / 1000.0
        x_min = x_axes[0]["world_x_mm"] / 1000.0
        x_max = x_axes[-1]["world_x_mm"] / 1000.0

        beams.append({
            "start_position": [x_min, y_world],
            "end_position": [x_max, y_world],
            "layer": "GRID_BEAM",
            "confidence": "low"
        })

    # Generate vertical beams (along Y direction)
    for x_axis in x_axes:
        x_world = x_axis["world_x_mm"] / 1000.0
        y_min = y_axes[0].get("world_y_mm", y_axes[0].get("world_x_mm", 0)) / 1000.0
        y_max = y_axes[-1].get("world_y_mm", y_axes[-1].get("world_x_mm", 0)) / 1000.0

        beams.append({
            "start_position": [x_world, y_min],
            "end_position": [x_world, y_max],
            "layer": "GRID_BEAM",
            "confidence": "low"
        })

    return beams


def _remove_duplicate_beams(
    beams: List[Dict[str, Any]],
    tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Remove duplicate beams within tolerance.

    Args:
        beams: List of beam data
        tolerance: Distance tolerance in meters

    Returns:
        Deduplicated beam list
    """
    if not beams:
        return []

    unique_beams = []

    for beam in beams:
        is_duplicate = False

        for existing in unique_beams:
            # Check if start/end positions match (in either direction)
            start_match = (
                _distance_2d(beam["start_position"], existing["start_position"]) < tolerance and
                _distance_2d(beam["end_position"], existing["end_position"]) < tolerance
            )
            reverse_match = (
                _distance_2d(beam["start_position"], existing["end_position"]) < tolerance and
                _distance_2d(beam["end_position"], existing["start_position"]) < tolerance
            )

            if start_match or reverse_match:
                is_duplicate = True
                # Keep higher confidence beam
                if beam.get("confidence") == "high" and existing.get("confidence") != "high":
                    unique_beams.remove(existing)
                    unique_beams.append(beam)
                break

        if not is_duplicate:
            unique_beams.append(beam)

    return unique_beams


def _assign_column_references(
    beams: List[Dict[str, Any]],
    columns: List[Dict[str, Any]],
    tolerance: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Assign start_column and end_column IDs to beams.

    Args:
        beams: List of beam data
        columns: List of column data
        tolerance: Distance tolerance in meters

    Returns:
        Beams with column references
    """
    for beam in beams:
        start_pos = beam["start_position"]
        end_pos = beam["end_position"]

        # Find nearest columns
        start_column = None
        end_column = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')

        for col in columns:
            col_pos = col["position"]
            dist_start = _distance_2d(start_pos, col_pos)
            dist_end = _distance_2d(end_pos, col_pos)

            if dist_start < min_start_dist and dist_start < tolerance:
                min_start_dist = dist_start
                start_column = col["id"]

            if dist_end < min_end_dist and dist_end < tolerance:
                min_end_dist = dist_end
                end_column = col["id"]

        beam["start_column"] = start_column
        beam["end_column"] = end_column

    return beams


def _calculate_beam_properties(
    beam: Dict[str, Any],
    floor_height_mm: float,
    ceiling_height_mm: float,
    structure_type: str
):
    """
    Calculate beam depth, width, and bottom level.

    Args:
        beam: Beam data dict (modified in-place)
        floor_height_mm: Floor-to-floor height
        ceiling_height_mm: Ceiling height
        structure_type: "RC_structure" or "S_structure"
    """
    # Calculate span
    dx = beam["end_position"][0] - beam["start_position"][0]
    dy = beam["end_position"][1] - beam["start_position"][1]
    span_m = math.sqrt(dx**2 + dy**2)
    span_mm = span_m * 1000

    beam["span_mm"] = span_mm

    # Estimate beam depth using span/12 rule (common structural heuristic)
    # For office buildings: beam depth ≈ span / 10 to span / 15
    depth_mm = span_mm / 12

    # Apply min/max constraints
    depth_mm = max(400, min(depth_mm, 1000))  # 400mm - 1000mm range

    beam["depth_mm"] = depth_mm

    # Estimate beam width based on structure type
    if "RC" in structure_type.upper():
        width_mm = max(200, depth_mm * 0.5)  # RC beams: width ≈ depth/2
    elif "S" in structure_type.upper():
        width_mm = max(150, depth_mm * 0.3)  # Steel beams: narrower
    else:
        width_mm = 200  # Default

    beam["width_mm"] = width_mm

    # Calculate bottom level (beam bottom is below ceiling)
    # Beam top is at floor_height, beam bottom is at floor_height - depth
    bottom_level_mm = ceiling_height_mm
    beam["bottom_level_mm"] = bottom_level_mm

    # Material inference
    if "RC" in structure_type.upper():
        beam["material"] = "Concrete"
    elif "S" in structure_type.upper():
        beam["material"] = "Steel"
    else:
        beam["material"] = "Concrete"  # Default to concrete


def _distance_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Calculate 2D Euclidean distance.

    Args:
        p1: Point 1 (x, y)
        p2: Point 2 (x, y)

    Returns:
        Distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)
