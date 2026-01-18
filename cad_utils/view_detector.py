"""
View Detector Module

Automatically detects and separates different architectural views (floor plans, elevations, sections)
within a single DXF file.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("ViewDetector")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logger.warning("ezdxf not available - view detection requires ezdxf")


class ViewType(Enum):
    """Types of architectural views"""
    FLOOR_PLAN = "floor_plan"
    ELEVATION_NORTH = "elevation_north"
    ELEVATION_SOUTH = "elevation_south"
    ELEVATION_EAST = "elevation_east"
    ELEVATION_WEST = "elevation_west"
    SECTION = "section"
    SITE_PLAN = "site_plan"
    UNKNOWN = "unknown"


@dataclass
class DetectedView:
    """Represents a detected view in the DXF file"""
    view_type: ViewType
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    confidence: float  # 0.0-1.0
    title_text: Optional[str] = None
    scale_text: Optional[str] = None
    entity_count: int = 0


def detect_views_from_text(modelspace) -> List[DetectedView]:
    """
    Detect views by analyzing TEXT and MTEXT entities.

    Args:
        modelspace: ezdxf modelspace object

    Returns:
        List of detected views with confidence scores
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for view detection")

    detected = []

    # Keywords for view type detection
    keywords = {
        ViewType.FLOOR_PLAN: ['FLOOR PLAN', 'PLAN', '平面図', '平面'],
        ViewType.ELEVATION_SOUTH: ['SOUTH ELEVATION', 'ELEVATION SOUTH', '南立面', '南側立面'],
        ViewType.ELEVATION_NORTH: ['NORTH ELEVATION', 'ELEVATION NORTH', '北立面', '北側立面'],
        ViewType.ELEVATION_EAST: ['EAST ELEVATION', 'ELEVATION EAST', '東立面', '東側立面'],
        ViewType.ELEVATION_WEST: ['WEST ELEVATION', 'ELEVATION WEST', '西立面', '西側立面'],
        ViewType.SECTION: ['SECTION', '断面図', '断面'],
        ViewType.SITE_PLAN: ['SITE PLAN', 'LOCATION', '配置図', '敷地'],
    }

    # Extract all text entities
    text_entities = []
    for entity_type in ['TEXT', 'MTEXT']:
        for entity in modelspace.query(entity_type):
            try:
                if entity_type == 'TEXT':
                    text = entity.dxf.text.upper()
                    position = (entity.dxf.insert[0], entity.dxf.insert[1])
                else:  # MTEXT
                    text = entity.text.upper()
                    position = (entity.dxf.insert[0], entity.dxf.insert[1])

                text_entities.append({
                    'text': text,
                    'position': position,
                    'original': entity.dxf.text if entity_type == 'TEXT' else entity.text
                })
            except Exception as e:
                logger.debug(f"Failed to extract text: {e}")
                continue

    # Match keywords
    for text_info in text_entities:
        text = text_info['text']
        for view_type, kw_list in keywords.items():
            for keyword in kw_list:
                if keyword in text:
                    # Estimate view bounds around the text position
                    # This is a rough estimate - will be refined by spatial clustering
                    pos = text_info['position']
                    detected.append(DetectedView(
                        view_type=view_type,
                        bounds=(pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5),
                        confidence=0.9,  # High confidence from text detection
                        title_text=text_info['original']
                    ))
                    logger.info(f"Detected {view_type.value} from text: '{keyword}'")
                    break

    return detected


def cluster_entities_spatially(modelspace, gap_threshold: float = 1.5) -> List[Tuple[float, float, float, float]]:
    """
    Cluster entities spatially to identify separate views.

    Args:
        modelspace: ezdxf modelspace object
        gap_threshold: Minimum gap (in meters) to consider as separate views

    Returns:
        List of bounding boxes for each cluster
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for spatial clustering")

    # Collect all entity bounding boxes
    entity_bounds = []

    for entity in modelspace:
        try:
            # Get bounding box for different entity types
            if hasattr(entity, 'bounding_box'):
                bbox = entity.bounding_box
                if bbox:
                    min_point, max_point = bbox
                    entity_bounds.append((min_point[0], min_point[1], max_point[0], max_point[1]))
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                min_x = min(start[0], end[0])
                max_x = max(start[0], end[0])
                min_y = min(start[1], end[1])
                max_y = max(start[1], end[1])
                entity_bounds.append((min_x, min_y, max_x, max_y))
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points('xy'))
                if points:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    entity_bounds.append((min(x_coords), min(y_coords), max(x_coords), max(y_coords)))
            elif entity.dxftype() == 'INSERT':
                pos = entity.dxf.insert
                # Rough estimate - blocks can vary greatly
                entity_bounds.append((pos[0] - 0.5, pos[1] - 0.5, pos[0] + 0.5, pos[1] + 0.5))

        except Exception as e:
            logger.debug(f"Failed to get bounds for {entity.dxftype()}: {e}")
            continue

    if not entity_bounds:
        logger.warning("No entity bounds found for clustering")
        return []

    # Simple clustering algorithm: group entities with gaps < threshold
    clusters = []
    used = [False] * len(entity_bounds)

    def merge_bounds(b1, b2):
        return (
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3])
        )

    def bounds_distance(b1, b2):
        """Calculate minimum distance between two bounding boxes"""
        # Check if overlapping or touching
        if (b1[2] >= b2[0] and b1[0] <= b2[2] and
            b1[3] >= b2[1] and b1[1] <= b2[3]):
            return 0.0

        # Calculate minimum distance
        dx = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
        dy = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
        return (dx**2 + dy**2)**0.5

    for i in range(len(entity_bounds)):
        if used[i]:
            continue

        # Start new cluster
        cluster = entity_bounds[i]
        used[i] = True
        cluster_changed = True

        # Iteratively add nearby entities
        while cluster_changed:
            cluster_changed = False
            for j in range(len(entity_bounds)):
                if used[j]:
                    continue

                dist = bounds_distance(cluster, entity_bounds[j])
                if dist < gap_threshold:
                    cluster = merge_bounds(cluster, entity_bounds[j])
                    used[j] = True
                    cluster_changed = True

        clusters.append(cluster)

    logger.info(f"Spatial clustering found {len(clusters)} separate groups (gap_threshold={gap_threshold}m)")
    return clusters


def detect_views(modelspace, gap_threshold: float = 1.5) -> List[DetectedView]:
    """
    Automatically detect architectural views in a DXF file.

    Combines text-based detection and spatial clustering.

    Args:
        modelspace: ezdxf modelspace object
        gap_threshold: Minimum gap (in meters) between views

    Returns:
        List of detected views, sorted by confidence (highest first)
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for view detection")

    # Step 1: Detect views from text annotations
    text_views = detect_views_from_text(modelspace)

    # Step 2: Cluster entities spatially
    spatial_clusters = cluster_entities_spatially(modelspace, gap_threshold)

    # Step 3: Merge text-based and spatial information
    final_views = []

    for cluster_bounds in spatial_clusters:
        # Check if this cluster matches any text-detected view
        matched = False
        for text_view in text_views:
            # Check if text position is within this cluster
            text_center = (
                (text_view.bounds[0] + text_view.bounds[2]) / 2,
                (text_view.bounds[1] + text_view.bounds[3]) / 2
            )

            if (cluster_bounds[0] <= text_center[0] <= cluster_bounds[2] and
                cluster_bounds[1] <= text_center[1] <= cluster_bounds[3]):
                # Update text view with accurate bounds
                final_views.append(DetectedView(
                    view_type=text_view.view_type,
                    bounds=cluster_bounds,
                    confidence=text_view.confidence,
                    title_text=text_view.title_text
                ))
                matched = True
                break

        if not matched:
            # Unknown view type - add as generic
            final_views.append(DetectedView(
                view_type=ViewType.UNKNOWN,
                bounds=cluster_bounds,
                confidence=0.5  # Lower confidence for unidentified views
            ))

    # Sort by confidence (highest first)
    final_views.sort(key=lambda v: v.confidence, reverse=True)

    logger.info(f"Total detected views: {len(final_views)}")
    for i, view in enumerate(final_views):
        w = view.bounds[2] - view.bounds[0]
        h = view.bounds[3] - view.bounds[1]
        logger.info(f"  View {i+1}: {view.view_type.value} (confidence={view.confidence:.2f}, size={w:.2f}x{h:.2f}m)")

    return final_views


def find_primary_floor_plan(detected_views: List[DetectedView]) -> Optional[DetectedView]:
    """
    Find the primary floor plan from detected views.

    Priority:
    1. Explicitly labeled floor plan
    2. Largest view (if multiple unlabeled)

    Args:
        detected_views: List of detected views

    Returns:
        Primary floor plan view, or None if not found
    """
    # First, look for explicitly labeled floor plans
    floor_plans = [v for v in detected_views if v.view_type == ViewType.FLOOR_PLAN]

    if floor_plans:
        # Return highest confidence floor plan
        return floor_plans[0]

    # If no labeled floor plan, assume largest view is the plan
    if detected_views:
        largest = max(detected_views, key=lambda v: (v.bounds[2] - v.bounds[0]) * (v.bounds[3] - v.bounds[1]))
        logger.info(f"No labeled floor plan found, assuming largest view is the plan")
        return largest

    return None
