"""
DXF Visualizer Module

Renders DXF files to PNG images for AI analysis
"""

import logging
from typing import Tuple, List
import os

logger = logging.getLogger("DXFVisualizer")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualization requires matplotlib")

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logger.warning("ezdxf not available")


def render_dxf_to_image(
    dxf_path: str,
    output_path: str,
    dpi: int = 200,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[str, Tuple[float, float, float, float]]:
    """
    Render DXF file to PNG image for AI analysis.

    Args:
        dxf_path: Path to DXF file
        output_path: Path to save PNG image
        dpi: Image resolution (150-300 recommended)
        figsize: Figure size in inches (width, height)

    Returns:
        tuple: (image_path, dxf_bounds)
            - image_path: Path to saved PNG
            - dxf_bounds: (min_x, min_y, max_x, max_y) in DXF units

    Raises:
        RuntimeError: If matplotlib or ezdxf not available
        FileNotFoundError: If DXF file not found
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib required for DXF visualization. Install with: pip install matplotlib")

    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf required for DXF parsing")

    if not os.path.exists(dxf_path):
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    logger.info(f"Rendering DXF to image: {dxf_path}")

    # Load DXF
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('white')

    # Collect all drawing elements
    lines = []
    polylines = []
    texts = []
    blocks = []
    all_points = []

    # Extract LINE entities
    for entity in msp.query('LINE'):
        try:
            start = (entity.dxf.start[0], entity.dxf.start[1])
            end = (entity.dxf.end[0], entity.dxf.end[1])
            lines.append([start, end])
            all_points.extend([start, end])
        except Exception as e:
            logger.debug(f"Failed to process LINE: {e}")

    # Extract LWPOLYLINE entities
    for entity in msp.query('LWPOLYLINE'):
        try:
            points = list(entity.get_points('xy'))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    lines.append([points[i], points[i + 1]])
                all_points.extend(points)

                # Close polyline if needed
                if entity.closed and points[0] != points[-1]:
                    lines.append([points[-1], points[0]])

        except Exception as e:
            logger.debug(f"Failed to process LWPOLYLINE: {e}")

    # Extract POLYLINE entities
    for entity in msp.query('POLYLINE'):
        try:
            points = [(v.dxf.location[0], v.dxf.location[1]) for v in entity.vertices]
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    lines.append([points[i], points[i + 1]])
                all_points.extend(points)

                if entity.is_closed and points[0] != points[-1]:
                    lines.append([points[-1], points[0]])

        except Exception as e:
            logger.debug(f"Failed to process POLYLINE: {e}")

    # Extract ARC entities
    for entity in msp.query('ARC'):
        try:
            center = (entity.dxf.center[0], entity.dxf.center[1])
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle

            # Create arc as line segments
            import numpy as np
            theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 30)
            arc_points = [(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta]

            for i in range(len(arc_points) - 1):
                lines.append([arc_points[i], arc_points[i + 1]])

            all_points.extend(arc_points)

        except Exception as e:
            logger.debug(f"Failed to process ARC: {e}")

    # Extract CIRCLE entities
    for entity in msp.query('CIRCLE'):
        try:
            center = (entity.dxf.center[0], entity.dxf.center[1])
            radius = entity.dxf.radius

            # Create circle as line segments
            import numpy as np
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_points = [(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta]

            for i in range(len(circle_points)):
                lines.append([circle_points[i], circle_points[(i + 1) % len(circle_points)]])

            all_points.extend(circle_points)

        except Exception as e:
            logger.debug(f"Failed to process CIRCLE: {e}")

    # Extract TEXT entities
    for entity in msp.query('TEXT'):
        try:
            text = entity.dxf.text
            pos = (entity.dxf.insert[0], entity.dxf.insert[1])
            height = entity.dxf.height if hasattr(entity.dxf, 'height') else 0.2
            texts.append({'text': text, 'pos': pos, 'height': height})
            all_points.append(pos)
        except Exception as e:
            logger.debug(f"Failed to process TEXT: {e}")

    # Extract MTEXT entities
    for entity in msp.query('MTEXT'):
        try:
            text = entity.text
            pos = (entity.dxf.insert[0], entity.dxf.insert[1])
            height = entity.dxf.char_height if hasattr(entity.dxf, 'char_height') else 0.2
            texts.append({'text': text, 'pos': pos, 'height': height})
            all_points.append(pos)
        except Exception as e:
            logger.debug(f"Failed to process MTEXT: {e}")

    # Extract INSERT (block reference) entities
    for entity in msp.query('INSERT'):
        try:
            pos = (entity.dxf.insert[0], entity.dxf.insert[1])
            blocks.append(pos)
            all_points.append(pos)
        except Exception as e:
            logger.debug(f"Failed to process INSERT: {e}")

    # Draw all lines
    if lines:
        lc = LineCollection(lines, colors='black', linewidths=0.5)
        ax.add_collection(lc)
        logger.info(f"Drew {len(lines)} line segments")

    # Draw text annotations (important for Gemini)
    for text_info in texts:
        ax.text(
            text_info['pos'][0],
            text_info['pos'][1],
            text_info['text'],
            fontsize=6,
            color='red',
            weight='bold',
            ha='left',
            va='bottom'
        )
    logger.info(f"Drew {len(texts)} text annotations")

    # Draw blocks as markers
    if blocks:
        blocks_x = [b[0] for b in blocks]
        blocks_y = [b[1] for b in blocks]
        ax.plot(blocks_x, blocks_y, 'bs', markersize=2, label='Blocks (Windows/Doors)')
        logger.info(f"Drew {len(blocks)} block markers")

    # Calculate bounds
    if all_points:
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        dxf_bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        # Add margin (5%)
        margin_x = (dxf_bounds[2] - dxf_bounds[0]) * 0.05
        margin_y = (dxf_bounds[3] - dxf_bounds[1]) * 0.05

        ax.set_xlim(dxf_bounds[0] - margin_x, dxf_bounds[2] + margin_x)
        ax.set_ylim(dxf_bounds[1] - margin_y, dxf_bounds[3] + margin_y)
    else:
        dxf_bounds = (0, 0, 1, 1)
        logger.warning("No entities found in DXF, using default bounds")

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add title
    ax.set_title(f"DXF: {os.path.basename(dxf_path)}", fontsize=10, pad=10)

    # Save image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved image to: {output_path}")
    logger.info(f"DXF bounds: {dxf_bounds}")

    return output_path, dxf_bounds
