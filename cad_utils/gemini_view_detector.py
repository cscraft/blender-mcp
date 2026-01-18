"""
Gemini Vision API Integration for DXF View Detection

Uses Google Gemini to automatically detect architectural views (floor plans, elevations)
in DXF drawings rendered as images.
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image

logger = logging.getLogger("GeminiViewDetector")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not available - install with: pip install google-generativeai")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


GEMINI_PROMPT = """Analyze this architectural/engineering drawing and identify all distinct views.

IMPORTANT: This is a real architectural drawing following Japanese/international standards:
- There may NOT be text labels like "FLOOR PLAN" or "ELEVATION"
- Dimension lines and numbers are present (ignore these for view detection)
- Standard layout:
  * TOP area: Floor plan (largest view, top-down perspective)
  * BOTTOM CENTER: Front elevation (side view with height)
  * BOTTOM RIGHT: Side elevation or section view
- Sometimes only a single floor plan is present

How to identify each view type:
1. FLOOR_PLAN (平面図):
   - Top-down view of a building/room
   - Shows walls as lines forming closed shapes
   - May show doors, windows as blocks or symbols
   - Usually the LARGEST view
   - Typically positioned in the UPPER part of the drawing

2. FRONT_ELEVATION (正面図):
   - Side view showing height
   - Walls appear as rectangles
   - Windows/doors appear as rectangular openings
   - Usually positioned BELOW the floor plan
   - Shows vertical dimensions

3. SIDE_ELEVATION (側面図):
   - Another side view from different angle
   - Similar characteristics to front elevation
   - Usually positioned to the RIGHT of front elevation

4. SECTION (断面図):
   - Cut-through view showing internal structure
   - More detailed than elevations
   - May show floor slabs, structural elements

For each view you detect, provide:
1. view_type: One of [FLOOR_PLAN, FRONT_ELEVATION, SIDE_ELEVATION, SECTION]
2. bbox_normalized: Bounding box [min_x, min_y, max_x, max_y] in range 0-1
   - EXCLUDE dimension lines and text annotations from the bbox
   - Include only the main drawing elements (walls, openings)
3. confidence: Your confidence score 0-1
4. reasoning: Brief explanation of why you identified this view type

Return ONLY a JSON array (no markdown, no explanation):
[
  {
    "view_type": "FLOOR_PLAN",
    "bbox_normalized": [0.1, 0.55, 0.9, 0.95],
    "confidence": 0.95,
    "reasoning": "Largest view in upper portion, shows top-down layout with closed wall shapes"
  },
  {
    "view_type": "FRONT_ELEVATION",
    "bbox_normalized": [0.1, 0.1, 0.5, 0.45],
    "confidence": 0.85,
    "reasoning": "Below floor plan, shows vertical wall with window openings"
  }
]

If you only detect one view (single floor plan), return an array with one element."""


def detect_views_with_gemini(
    image_path: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash"
) -> List[Dict]:
    """
    Use Gemini Vision to detect architectural views in DXF image.

    Args:
        image_path: Path to rendered DXF image (PNG)
        api_key: Google Gemini API key
        model_name: Gemini model to use (default: gemini-2.0-flash-exp)

    Returns:
        List of detected views with structure:
        [
            {
                'view_type': str,  # FLOOR_PLAN, FRONT_ELEVATION, etc.
                'bbox_normalized': [float, float, float, float],  # [min_x, min_y, max_x, max_y] in 0-1
                'bbox_pixels': [int, int, int, int],  # [min_x, min_y, max_x, max_y] in pixels
                'confidence': float,  # 0-1
                'reasoning': str
            },
            ...
        ]

    Raises:
        RuntimeError: If google-generativeai not installed
        FileNotFoundError: If image not found
        ValueError: If API response is invalid
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai required. Install with: pip install google-generativeai")

    logger.info(f"Detecting views with Gemini: {image_path}")

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size
    logger.info(f"Image size: {img_width}x{img_height}")

    # Call Gemini
    try:
        response = model.generate_content([GEMINI_PROMPT, img])
        response_text = response.text
        logger.info(f"Gemini response received ({len(response_text)} chars)")

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    # Parse JSON response
    try:
        # Clean response (remove markdown code blocks if present)
        cleaned_text = response_text.strip()

        if '```json' in cleaned_text:
            cleaned_text = cleaned_text.split('```json')[1].split('```')[0]
        elif '```' in cleaned_text:
            cleaned_text = cleaned_text.split('```')[1].split('```')[0]

        cleaned_text = cleaned_text.strip()

        views = json.loads(cleaned_text)

        if not isinstance(views, list):
            raise ValueError("Response is not a JSON array")

        # Convert normalized coordinates to pixel coordinates
        for view in views:
            if 'bbox_normalized' not in view:
                raise ValueError(f"View missing 'bbox_normalized': {view}")

            bbox_norm = view['bbox_normalized']

            if len(bbox_norm) != 4:
                raise ValueError(f"Invalid bbox format: {bbox_norm}")

            # Convert to pixels
            view['bbox_pixels'] = [
                int(bbox_norm[0] * img_width),
                int(bbox_norm[1] * img_height),
                int(bbox_norm[2] * img_width),
                int(bbox_norm[3] * img_height)
            ]

            logger.info(f"Detected: {view['view_type']} (confidence={view['confidence']:.2f})")
            logger.info(f"  Reasoning: {view.get('reasoning', 'N/A')}")
            logger.info(f"  BBox pixels: {view['bbox_pixels']}")

        return views

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Response text: {response_text[:500]}...")
        raise ValueError(f"Invalid JSON response from Gemini: {e}")

    except Exception as e:
        logger.error(f"Error processing Gemini response: {e}")
        raise


def convert_image_coords_to_dxf_coords(
    bbox_pixels: List[int],
    image_path: str,
    dxf_bounds: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Convert image pixel coordinates back to DXF world coordinates.

    Args:
        bbox_pixels: [min_x, min_y, max_x, max_y] in pixels
        image_path: Path to rendered image (to get dimensions)
        dxf_bounds: (min_x, min_y, max_x, max_y) in DXF units from dxf_visualizer

    Returns:
        (min_x, min_y, max_x, max_y) in DXF units (meters)
    """
    # Load image to get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Convert pixel coords to normalized (0-1)
    norm_min_x = bbox_pixels[0] / img_width
    norm_max_x = bbox_pixels[2] / img_width

    # Y axis conversion:
    # In PIL images: Y=0 at top, Y=height at bottom
    # In matplotlib renders: depends on the data
    # In DXF space: Y increases upward (standard CAD convention)
    #
    # Testing shows matplotlib preserves DXF Y-axis direction in the PNG:
    # - DXF positive Y (floor plan) -> top of PNG image (low pixel Y)
    # - DXF negative Y (elevation) -> bottom of PNG image (high pixel Y)
    #
    # So we do NOT need to invert Y axis!
    norm_min_y = bbox_pixels[1] / img_height
    norm_max_y = bbox_pixels[3] / img_height

    # DXF bounds
    dxf_min_x, dxf_min_y, dxf_max_x, dxf_max_y = dxf_bounds
    dxf_width = dxf_max_x - dxf_min_x
    dxf_height = dxf_max_y - dxf_min_y

    # Convert to DXF world coordinates
    world_min_x = dxf_min_x + norm_min_x * dxf_width
    world_min_y = dxf_min_y + norm_min_y * dxf_height
    world_max_x = dxf_min_x + norm_max_x * dxf_width
    world_max_y = dxf_min_y + norm_max_y * dxf_height

    logger.info(f"Converted bbox_pixels {bbox_pixels} to DXF coords: ({world_min_x:.2f}, {world_min_y:.2f}, {world_max_x:.2f}, {world_max_y:.2f})")

    return (world_min_x, world_min_y, world_max_x, world_max_y)


def find_floor_plan_view(detected_views: List[Dict]) -> Optional[Dict]:
    """
    Find the floor plan view from detected views.

    Args:
        detected_views: List of views from detect_views_with_gemini()

    Returns:
        Floor plan view dict, or None if not found
    """
    # Look for explicitly labeled floor plan
    for view in detected_views:
        if view['view_type'] == 'FLOOR_PLAN' and view['confidence'] > 0.6:
            logger.info(f"Found floor plan with confidence {view['confidence']:.2f}")
            return view

    # If no floor plan found but views exist, assume largest is floor plan
    if detected_views:
        largest = max(detected_views, key=lambda v:
            (v['bbox_normalized'][2] - v['bbox_normalized'][0]) *
            (v['bbox_normalized'][3] - v['bbox_normalized'][1])
        )
        logger.warning(f"No labeled floor plan, assuming largest view is floor plan (type: {largest['view_type']})")
        return largest

    return None
