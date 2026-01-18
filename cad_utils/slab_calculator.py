"""
Slab Thickness Calculator - Phase 5

Calculates appropriate slab thickness based on span and structure type.
Uses structural engineering heuristics from Gemini_read.md.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("SlabCalculator")


def calculate_slab_thickness(
    boundary: List[Tuple[float, float]],
    structure_type: str = "unknown",
    floor_type: str = "office"
) -> int:
    """
    Calculate appropriate slab thickness based on span.

    Args:
        boundary: Slab boundary polygon [(x, y), ...]
        structure_type: "RC_structure", "S_structure", or "unknown"
        floor_type: "office", "residential", "warehouse", etc.

    Returns:
        Slab thickness in mm

    Heuristics (Gemini_read.md):
    - Span ≤ 6m: 150mm (standard office slab)
    - 6m < Span ≤ 8m: 200mm (medium span)
    - Span > 8m: 250mm (long span)
    - S造 (Steel): Can be thinner (-50mm) due to composite design
    """
    if not boundary or len(boundary) < 3:
        logger.warning("Invalid boundary, using default 150mm thickness")
        return 150

    # Calculate maximum span
    max_span_m = _calculate_max_span(boundary)
    max_span_mm = max_span_m * 1000

    logger.info(f"Calculating slab thickness: max_span={max_span_m:.2f}m, structure={structure_type}")

    # Base thickness from span (RC造基準)
    if max_span_m <= 6.0:
        base_thickness = 150  # Standard office slab
    elif max_span_m <= 8.0:
        base_thickness = 200  # Medium span
    elif max_span_m <= 10.0:
        base_thickness = 250  # Long span
    else:
        base_thickness = 300  # Very long span (post-tensioned concrete)

    # Adjust for structure type
    if "S" in structure_type.upper() or "STEEL" in structure_type.upper():
        # Steel composite slabs can be thinner
        base_thickness = max(100, base_thickness - 50)
        logger.info(f"  S造 adjustment: -{50}mm")

    # Adjust for floor type
    if floor_type == "warehouse" or floor_type == "heavy_load":
        base_thickness += 50
        logger.info(f"  Heavy load adjustment: +50mm")
    elif floor_type == "roof":
        base_thickness = max(120, base_thickness - 30)
        logger.info(f"  Roof adjustment: -30mm")

    # Apply minimum thickness constraint
    thickness = max(100, base_thickness)

    # Round to standard increments (50mm) - but preserve the base values
    # Don't round, use exact values for standard thicknesses
    logger.info(f"  Final slab thickness: {thickness}mm")
    return thickness


def enhance_slab_data(
    slab_data: Dict[str, Any],
    structure_type: str = "unknown",
    floor_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhance slab data with calculated thickness and other properties.

    Args:
        slab_data: Slab dictionary from cad_parser
        structure_type: "RC_structure", "S_structure", or "unknown"
        floor_info: Optional floor metadata (elevation, ceiling_height, etc.)

    Returns:
        Enhanced slab dictionary with thickness_mm, material, z_offset

    Example input:
        {
            "id": "slab_001",
            "layer": "FLOOR",
            "boundary": [[0,0], [12,0], [12,12], [0,12], [0,0]],
            "confidence": "low"
        }

    Example output:
        {
            "id": "slab_001",
            "layer": "FLOOR",
            "boundary": [[0,0], [12,0], [12,12], [0,12], [0,0]],
            "thickness_mm": 200,
            "material": "Concrete",
            "z_offset": 0.0,
            "top_level_mm": 0,
            "bottom_level_mm": -200,
            "confidence": "medium"
        }
    """
    boundary = slab_data.get("boundary", [])

    # Calculate thickness if not provided
    if "thickness_mm" not in slab_data or slab_data["thickness_mm"] is None:
        floor_type = floor_info.get("floor_type", "office") if floor_info else "office"
        thickness_mm = calculate_slab_thickness(boundary, structure_type, floor_type)
        slab_data["thickness_mm"] = thickness_mm

        # Upgrade confidence since we calculated it
        if slab_data.get("confidence") == "low":
            slab_data["confidence"] = "medium"

    # Infer material from structure type
    if "material" not in slab_data or slab_data["material"] is None:
        if "RC" in structure_type.upper():
            slab_data["material"] = "Concrete"
        elif "S" in structure_type.upper() or "STEEL" in structure_type.upper():
            slab_data["material"] = "Steel_Composite"  # Composite deck
        else:
            slab_data["material"] = "Concrete"  # Default

    # Calculate z_offset and levels
    if floor_info:
        slab_level_mm = floor_info.get("slab_level_mm", 0)
        elevation_mm = floor_info.get("elevation_mm", 0)

        # Slab top is at slab_level
        # Slab bottom is at slab_level - thickness
        slab_data["z_offset"] = slab_level_mm / 1000.0  # Convert to meters
        slab_data["top_level_mm"] = slab_level_mm
        slab_data["bottom_level_mm"] = slab_level_mm - slab_data["thickness_mm"]
    else:
        # Default: slab top at z=0
        slab_data["z_offset"] = 0.0
        slab_data["top_level_mm"] = 0
        slab_data["bottom_level_mm"] = -slab_data["thickness_mm"]

    # Convert thickness to meters for Blender
    slab_data["thickness"] = slab_data["thickness_mm"] / 1000.0

    return slab_data


def _calculate_max_span(boundary: List[Tuple[float, float]]) -> float:
    """
    Calculate maximum span from boundary polygon.

    Strategy:
    1. Find bounding box
    2. Max span = max(width, height) of bounding box
    3. This is a conservative estimate

    Args:
        boundary: Polygon boundary [(x, y), ...]

    Returns:
        Maximum span in meters
    """
    if len(boundary) < 3:
        return 0.0

    xs = [pt[0] for pt in boundary]
    ys = [pt[1] for pt in boundary]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    max_span = max(width, height)

    logger.debug(f"Boundary spans: width={width:.2f}m, height={height:.2f}m, max={max_span:.2f}m")

    return max_span


def _round_to_increment(value: int, increment: int = 50) -> int:
    """
    Round value to nearest increment.

    Args:
        value: Value to round
        increment: Rounding increment (default: 50mm)

    Returns:
        Rounded value

    Example:
        >>> _round_to_increment(180, 50)
        200
        >>> _round_to_increment(140, 50)
        150
    """
    return int(round(value / increment) * increment)


def calculate_slab_volume(
    boundary: List[Tuple[float, float]],
    thickness_mm: int
) -> float:
    """
    Calculate slab volume for material estimation.

    Args:
        boundary: Slab boundary polygon [(x, y), ...]
        thickness_mm: Slab thickness in mm

    Returns:
        Volume in cubic meters (m³)
    """
    if len(boundary) < 3:
        return 0.0

    # Calculate area using shoelace formula
    area = 0.0
    n = len(boundary)

    for i in range(n):
        j = (i + 1) % n
        area += boundary[i][0] * boundary[j][1]
        area -= boundary[j][0] * boundary[i][1]

    area = abs(area) / 2.0  # m²

    # Volume = area × thickness
    thickness_m = thickness_mm / 1000.0
    volume = area * thickness_m  # m³

    logger.debug(f"Slab volume: area={area:.2f}m², thickness={thickness_m:.3f}m, volume={volume:.2f}m³")

    return volume


def estimate_concrete_weight(
    volume_m3: float,
    density: float = 2400.0
) -> float:
    """
    Estimate concrete weight for structural analysis.

    Args:
        volume_m3: Concrete volume in m³
        density: Concrete density in kg/m³ (default: 2400 for normal concrete)

    Returns:
        Weight in kilograms (kg)
    """
    weight_kg = volume_m3 * density
    logger.debug(f"Concrete weight: {volume_m3:.2f}m³ × {density}kg/m³ = {weight_kg:.0f}kg")
    return weight_kg
