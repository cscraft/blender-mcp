"""
Geometry Builder Module

Constructs 3D Blender geometry from 2D CAD data:
- Extrudes wall polylines to 3D meshes
- Applies wall thickness
- Creates proper mesh topology

Note: This module is designed to be imported by Blender addon (requires bpy).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("GeometryBuilder")

# Import will fail if not running in Blender, but that's expected
try:
    import bpy
    import bmesh
    from mathutils import Vector
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    logger.warning("bpy not available - geometry_builder requires Blender environment")


def build_walls(
    walls: List[Dict[str, Any]],
    height: float = 2.4,
    thickness: float = 0.12,
    z_offset: float = 0.0
) -> List[Any]:
    """
    Build 3D wall geometry from 2D wall data.

    Args:
        walls: List of wall dictionaries from cad_parser
        height: Wall height in meters (default 2.4m)
        thickness: Wall thickness in meters (default 0.12m = 120mm)
        z_offset: Z-axis offset for multi-floor buildings (default 0.0m)

    Returns:
        list: List of Blender mesh objects

    Example:
        >>> objects = build_walls(parsed_data['walls'], height=2.4, thickness=0.12)
        >>> print(f"Created {len(objects)} wall objects")
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("geometry_builder requires Blender environment (bpy not available)")

    objects = []

    for i, wall in enumerate(walls):
        try:
            obj = _create_wall_object(wall, height, thickness, z_offset, i)
            if obj:
                objects.append(obj)
        except Exception as e:
            logger.warning(f"Failed to create wall {i}: {e}")
            continue

    logger.info(f"Created {len(objects)} wall objects")
    return objects


def _create_wall_object(
    wall: Dict[str, Any],
    height: float,
    thickness: float,
    z_offset: float,
    index: int
) -> Optional[Any]:
    """
    Create a single wall mesh object.

    Args:
        wall: Wall dictionary with vertices and metadata
        height: Wall height
        thickness: Wall thickness
        z_offset: Z-axis offset
        index: Wall index for naming

    Returns:
        Blender mesh object or None if creation fails
    """
    vertices = wall['vertices']
    if len(vertices) < 2:
        logger.warning(f"Wall {index} has less than 2 vertices, skipping")
        return None

    # Create mesh and object
    mesh = bpy.data.meshes.new(name=f"Wall_{index}")
    obj = bpy.data.objects.new(f"Wall_{index}", mesh)

    # Store CAD layer as custom property
    obj['cad_layer'] = wall['layer']

    # Use BMesh for flexible mesh construction
    bm = bmesh.new()

    try:
        # Create wall with thickness using offset polyline method
        if wall['closed'] and len(vertices) >= 3:
            # For closed polylines, create a solid wall with thickness
            _create_solid_wall(bm, vertices, height, thickness, z_offset)
        else:
            # For open polylines, create a simple extruded wall
            _create_simple_wall(bm, vertices, height, thickness, z_offset)

        # Write bmesh to mesh
        bm.to_mesh(mesh)
        mesh.update()

    finally:
        bm.free()

    return obj


def _create_solid_wall(
    bm: 'bmesh.types.BMesh',
    vertices: List[Tuple[float, float]],
    height: float,
    thickness: float,
    z_offset: float
) -> None:
    """
    Create a solid wall with thickness from a closed polyline.

    Creates two offset polylines (inner and outer) and connects them.

    Args:
        bm: BMesh object
        vertices: 2D vertices of the wall centerline
        height: Wall height
        thickness: Wall thickness
        z_offset: Z-axis offset
    """
    half_thickness = thickness / 2.0

    # Calculate offset polylines (inner and outer)
    outer_verts = _offset_polyline(vertices, half_thickness, outward=True)
    inner_verts = _offset_polyline(vertices, half_thickness, outward=False)

    # Create bottom face (outer polyline)
    bottom_outer = [bm.verts.new((v[0], v[1], z_offset)) for v in outer_verts]
    if len(bottom_outer) >= 3:
        bm.faces.new(bottom_outer)

    # Create bottom face (inner polyline, reversed for correct normal)
    bottom_inner = [bm.verts.new((v[0], v[1], z_offset)) for v in reversed(inner_verts)]
    if len(bottom_inner) >= 3:
        bm.faces.new(bottom_inner)

    # Create top faces
    top_outer = [bm.verts.new((v[0], v[1], z_offset + height)) for v in outer_verts]
    top_inner = [bm.verts.new((v[0], v[1], z_offset + height)) for v in reversed(inner_verts)]

    if len(top_outer) >= 3:
        bm.faces.new(reversed(top_outer))
    if len(top_inner) >= 3:
        bm.faces.new(top_inner)

    # Connect outer wall sides
    for i in range(len(outer_verts)):
        next_i = (i + 1) % len(outer_verts)
        quad = [
            bottom_outer[i],
            bottom_outer[next_i],
            top_outer[next_i],
            top_outer[i]
        ]
        bm.faces.new(quad)

    # Connect inner wall sides (reversed order for inner surface)
    inner_verts_forward = list(reversed(bottom_inner))
    top_inner_forward = list(reversed(top_inner))

    for i in range(len(inner_verts)):
        next_i = (i + 1) % len(inner_verts)
        quad = [
            inner_verts_forward[next_i],
            inner_verts_forward[i],
            top_inner_forward[i],
            top_inner_forward[next_i]
        ]
        bm.faces.new(quad)


def _create_simple_wall(
    bm: 'bmesh.types.BMesh',
    vertices: List[Tuple[float, float]],
    height: float,
    thickness: float,
    z_offset: float
) -> None:
    """
    Create a simple extruded wall for open polylines.

    Args:
        bm: BMesh object
        vertices: 2D vertices of the wall centerline
        height: Wall height
        thickness: Wall thickness (applied as offset)
        z_offset: Z-axis offset
    """
    half_thickness = thickness / 2.0

    # Create offset polylines
    outer_verts = _offset_polyline(vertices, half_thickness, outward=True, closed=False)
    inner_verts = _offset_polyline(vertices, half_thickness, outward=False, closed=False)

    # Create bottom vertices
    bottom_outer = [bm.verts.new((v[0], v[1], z_offset)) for v in outer_verts]
    bottom_inner = [bm.verts.new((v[0], v[1], z_offset)) for v in inner_verts]

    # Create top vertices
    top_outer = [bm.verts.new((v[0], v[1], z_offset + height)) for v in outer_verts]
    top_inner = [bm.verts.new((v[0], v[1], z_offset + height)) for v in inner_verts]

    # Connect outer wall
    for i in range(len(outer_verts) - 1):
        quad = [
            bottom_outer[i],
            bottom_outer[i + 1],
            top_outer[i + 1],
            top_outer[i]
        ]
        bm.faces.new(quad)

    # Connect inner wall
    for i in range(len(inner_verts) - 1):
        quad = [
            bottom_inner[i + 1],
            bottom_inner[i],
            top_inner[i],
            top_inner[i + 1]
        ]
        bm.faces.new(quad)

    # End caps
    # Start cap
    cap_start = [bottom_outer[0], bottom_inner[0], top_inner[0], top_outer[0]]
    bm.faces.new(cap_start)

    # End cap
    cap_end = [bottom_inner[-1], bottom_outer[-1], top_outer[-1], top_inner[-1]]
    bm.faces.new(cap_end)


def _offset_polyline(
    vertices: List[Tuple[float, float]],
    offset: float,
    outward: bool = True,
    closed: bool = True
) -> List[Tuple[float, float]]:
    """
    Offset a 2D polyline by a given distance.

    Args:
        vertices: Original polyline vertices
        offset: Offset distance (positive)
        outward: True for outward offset, False for inward
        closed: Whether the polyline is closed

    Returns:
        list: Offset polyline vertices
    """
    if len(vertices) < 2:
        return vertices

    offset_verts = []
    sign = 1.0 if outward else -1.0

    n = len(vertices)

    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            # For open polylines, handle endpoints specially
            if i == 0:
                # First vertex: offset perpendicular to first edge
                v1 = Vector(vertices[i])
                v2 = Vector(vertices[i + 1])
                edge_dir = (v2 - v1).normalized()
                perp = Vector((-edge_dir.y, edge_dir.x)) * offset * sign
                offset_verts.append((v1.x + perp.x, v1.y + perp.y))
            else:
                # Last vertex: offset perpendicular to last edge
                v1 = Vector(vertices[i - 1])
                v2 = Vector(vertices[i])
                edge_dir = (v2 - v1).normalized()
                perp = Vector((-edge_dir.y, edge_dir.x)) * offset * sign
                offset_verts.append((v2.x + perp.x, v2.y + perp.y))
        else:
            # Middle vertices or closed polyline
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            v_prev = Vector(vertices[prev_i])
            v_curr = Vector(vertices[i])
            v_next = Vector(vertices[next_i])

            # Calculate edge directions
            edge1_dir = (v_curr - v_prev).normalized()
            edge2_dir = (v_next - v_curr).normalized()

            # Calculate perpendiculars
            perp1 = Vector((-edge1_dir.y, edge1_dir.x))
            perp2 = Vector((-edge2_dir.y, edge2_dir.x))

            # Average perpendicular (bisector)
            bisector = (perp1 + perp2).normalized()

            # Calculate offset distance (accounting for angle)
            # Using miter joint
            angle_cos = perp1.dot(bisector)
            if abs(angle_cos) > 0.001:
                miter_offset = offset / angle_cos
                # Limit extreme miter lengths
                miter_offset = max(-offset * 10, min(offset * 10, miter_offset))
            else:
                miter_offset = offset

            offset_point = v_curr + bisector * miter_offset * sign
            offset_verts.append((offset_point.x, offset_point.y))

    return offset_verts


def build_openings(
    openings: List[Dict[str, Any]],
    z_offset: float = 0.0,
    use_placeholder: bool = True
) -> List[Any]:
    """
    Build 3D opening geometry (windows and doors) from 2D opening data.

    Args:
        openings: List of opening dictionaries from cad_parser
        z_offset: Z-axis offset for multi-floor buildings (default 0.0m)
        use_placeholder: If True, create simple placeholder boxes; if False, use Sketchfab models

    Returns:
        list: List of Blender mesh objects

    Example:
        >>> objects = build_openings(parsed_data['openings'], z_offset=0.0)
        >>> print(f"Created {len(objects)} opening objects")
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("geometry_builder requires Blender environment (bpy not available)")

    objects = []

    for i, opening in enumerate(openings):
        try:
            if use_placeholder:
                obj = _create_placeholder_opening(opening, z_offset, i)
            else:
                # Sketchfab integration will be handled in addon.py
                obj = _create_placeholder_opening(opening, z_offset, i)

            if obj:
                objects.append(obj)
        except Exception as e:
            logger.warning(f"Failed to create opening {i}: {e}")
            continue

    logger.info(f"Created {len(objects)} opening objects")
    return objects


def _create_placeholder_opening(
    opening: Dict[str, Any],
    z_offset: float,
    index: int
) -> Optional[Any]:
    """
    Create a simple placeholder box for an opening.

    Args:
        opening: Opening dictionary with position, size, and rotation
        z_offset: Z-axis offset
        index: Opening index for naming

    Returns:
        Blender mesh object or None if creation fails
    """
    import math

    opening_type = opening['type']
    position = opening['position']
    width = opening['width']
    height = opening['height']
    rotation = opening['rotation']

    # Create a simple box mesh
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = f"{opening_type.capitalize()}_{index}"

    # Store CAD metadata
    obj['cad_layer'] = opening['layer']
    obj['cad_type'] = opening_type
    obj['cad_block_name'] = opening['block_name']

    # Calculate Z position based on opening type
    if opening_type == 'window':
        # Windows typically start at 0.9m from floor
        z_position = z_offset + 0.9 + height / 2.0
        depth = 0.1  # Thin placeholder
    else:  # door
        # Doors start at floor level
        z_position = z_offset + height / 2.0
        depth = 0.2  # Slightly thicker for doors

    # Scale the box to match opening size
    obj.scale = (width, depth, height)

    # Position the opening
    obj.location = (
        position[0],
        position[1],
        z_position
    )

    # Rotate around Z-axis
    obj.rotation_euler = (0, 0, math.radians(rotation))

    # Apply transforms
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    return obj


def create_opening_in_wall(
    wall_obj: Any,
    opening_obj: Any
) -> None:
    """
    Cut an opening into a wall using Boolean modifier.

    Args:
        wall_obj: Blender wall mesh object
        opening_obj: Blender opening mesh object (placeholder box)
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("geometry_builder requires Blender environment")

    # Add Boolean modifier to wall
    bool_modifier = wall_obj.modifiers.new(name=f"Cut_{opening_obj.name}", type='BOOLEAN')
    bool_modifier.operation = 'DIFFERENCE'
    bool_modifier.object = opening_obj
    bool_modifier.use_self = True

    # Hide the opening cutter object (it's just for boolean operation)
    opening_obj.hide_viewport = True
    opening_obj.hide_render = True

    logger.info(f"Created boolean cut for {opening_obj.name} in {wall_obj.name}")
