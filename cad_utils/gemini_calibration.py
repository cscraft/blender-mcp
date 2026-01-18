"""
Gemini Calibration Module - Phase 1 of 3-Phase Chain-of-Thought

Extracts calibration data from architectural drawings:
- Grid system (通り芯) detection: X1, X2, Y1, Y2...
- Scale factor estimation: 1px = ?? mm
- Structure type detection: RC造 (Reinforced Concrete) or S造 (Steel)

This is Phase 1 of the Gemini 3-phase prompting strategy as per Gemini_read.md.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger("GeminiCalibration")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")


def calibrate_drawing(
    image_path: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash"  # CHANGED: 最新の安定モデル
) -> Dict[str, Any]:
    """
    Phase 1: Calibration - Extract grid system, scale, and structure type.

    Args:
        image_path: Path to rendered DXF image
        api_key: Gemini API key
        model_name: Gemini model to use (default: gemini-2.0-flash)

    Returns:
        dict: {
            "grid_system": {
                "detected": bool,
                "x_axes": [{"label": "X1", "pixel_x": 150, "world_x_mm": 0}, ...],
                "y_axes": [{"label": "Y1", "pixel_y": 200, "world_y_mm": 0}, ...],
                "origin": "X1-Y1"
            },
            "scale_factor": float,  # 1px = ?? mm
            "structure_type": str,  # "RC_structure", "S_structure", or "unknown"
            "confidence": str,  # "high", "medium", or "low"
            "notes": str  # Additional observations
        }

    Raises:
        RuntimeError: If Gemini or PIL not available
        ValueError: If image cannot be loaded
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google.generativeai required for calibration")

    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required for image loading")

    logger.info(f"Starting calibration for: {image_path}")

    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        logger.info(f"Image size: {img_width} x {img_height}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Phase 1 Calibration Prompt (from plan document)
    prompt = """あなたは建築図面の専門家です。添付された図面画像を分析してください。

【タスク1: 通り芯（Grid Line）の検出】
- 一点鎖線で描かれたグリッド（X1, X2, Y1, Y2...またはA, B, C...と1, 2, 3...）を特定
- 各グリッド線のピクセル座標を記録
- グリッド間の寸法数値（mm）を読み取る
- グリッドが存在しない場合は detected: false を返してください

【タスク2: スケール推定】
- グリッド間隔の寸法（例: 7200mm）とピクセル距離から Scale Factor (1px = ?? mm) を計算
- 寸法線や縮尺表記（1:50など）があれば読み取る
- グリッドがない場合でも、寸法線から推定を試みる

【タスク3: 構造種別の判定】
- 柱の形状を観察:
  * H型や□型断面 → S造（鉄骨）
  * 塗りつぶし矩形/円 → RC造（鉄筋コンクリート）
- 壁厚を確認:
  * 150mm程度の薄壁 → S造+ALC
  * 200mm以上の厚壁 → RC造
- 判定できない場合は "unknown" を返す

以下のJSON形式で出力してください（markdownのコードブロック不要、純粋なJSONのみ）:
{
  "grid_system": {
    "detected": true または false,
    "x_axes": [
      {"label": "X1", "pixel_x": 150, "world_x_mm": 0, "confidence": "high"},
      {"label": "X2", "pixel_x": 450, "world_x_mm": 7200, "confidence": "high"}
    ],
    "y_axes": [
      {"label": "Y1", "pixel_y": 200, "world_y_mm": 0, "confidence": "high"},
      {"label": "Y2", "pixel_y": 600, "world_y_mm": 6400, "confidence": "high"}
    ],
    "origin": "X1-Y1"
  },
  "scale_factor": 0.5,
  "structure_type": "RC_structure" または "S_structure" または "unknown",
  "confidence": "high" または "medium" または "low",
  "notes": "観察された特徴や判定根拠"
}

注意:
- grid_system.detected が false の場合、x_axes と y_axes は空配列 [] にしてください
- 数値は必ず数値型（文字列ではない）で出力してください
- JSONのみを出力し、説明文やマークダウン記法は含めないでください
"""

    logger.info("Calling Gemini API for calibration...")

    try:
        response = model.generate_content([prompt, img])
        response_text = response.text.strip()

        logger.debug(f"Gemini response (raw): {response_text[:500]}...")

        # Parse JSON response
        calibration_data = _parse_gemini_response(response_text)

        # Validate and normalize
        calibration_data = _validate_calibration_data(calibration_data, img_width, img_height)

        logger.info(f"Calibration complete: grid_detected={calibration_data['grid_system']['detected']}, "
                   f"structure={calibration_data['structure_type']}, "
                   f"scale={calibration_data.get('scale_factor', 'N/A')}")

        return calibration_data

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        # Return default calibration (no grid detected)
        return _get_default_calibration()


def _parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Gemini's JSON response, handling markdown code blocks if present.

    Args:
        response_text: Raw response from Gemini

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON parsing fails
    """
    # Remove markdown code blocks if present
    text = response_text.strip()

    # Check for ```json ... ``` wrapper
    if text.startswith("```"):
        # Find the actual JSON content
        lines = text.split('\n')
        json_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                json_lines.append(line)

        text = '\n'.join(json_lines).strip()

    # Parse JSON
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response text: {text[:1000]}")
        raise ValueError(f"Invalid JSON response from Gemini: {e}")


def _validate_calibration_data(
    data: Dict[str, Any],
    img_width: int,
    img_height: int
) -> Dict[str, Any]:
    """
    Validate and normalize calibration data.

    Args:
        data: Raw calibration data from Gemini
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Validated and normalized calibration data
    """
    validated = {
        "grid_system": {
            "detected": False,
            "x_axes": [],
            "y_axes": [],
            "origin": None
        },
        "scale_factor": None,
        "structure_type": "unknown",
        "confidence": "low",
        "notes": ""
    }

    # Validate grid_system
    if "grid_system" in data:
        grid = data["grid_system"]
        validated["grid_system"]["detected"] = bool(grid.get("detected", False))

        # Validate x_axes
        if "x_axes" in grid and isinstance(grid["x_axes"], list):
            for axis in grid["x_axes"]:
                if _validate_axis(axis, "pixel_x", img_width):
                    validated["grid_system"]["x_axes"].append(axis)

        # Validate y_axes
        if "y_axes" in grid and isinstance(grid["y_axes"], list):
            for axis in grid["y_axes"]:
                if _validate_axis(axis, "pixel_y", img_height):
                    validated["grid_system"]["y_axes"].append(axis)

        # Validate origin
        if "origin" in grid and isinstance(grid["origin"], str):
            validated["grid_system"]["origin"] = grid["origin"]

    # Validate scale_factor
    if "scale_factor" in data:
        try:
            scale = float(data["scale_factor"])
            if 0.01 <= scale <= 100:  # Reasonable range: 0.01mm/px to 100mm/px
                validated["scale_factor"] = scale
            else:
                logger.warning(f"Scale factor {scale} outside reasonable range (0.01-100)")
        except (ValueError, TypeError):
            logger.warning(f"Invalid scale_factor: {data.get('scale_factor')}")

    # Validate structure_type
    if "structure_type" in data:
        structure = str(data["structure_type"]).upper()
        if "RC" in structure:
            validated["structure_type"] = "RC_structure"
        elif "S" in structure or "STEEL" in structure:
            validated["structure_type"] = "S_structure"
        else:
            validated["structure_type"] = "unknown"

    # Validate confidence
    if "confidence" in data:
        conf = str(data["confidence"]).lower()
        if conf in ["high", "medium", "low"]:
            validated["confidence"] = conf

    # Notes
    if "notes" in data:
        validated["notes"] = str(data["notes"])

    return validated


def _validate_axis(axis: Dict, pixel_key: str, max_pixel: int) -> bool:
    """
    Validate a single grid axis entry.

    Args:
        axis: Axis dict with label, pixel coordinate, world coordinate
        pixel_key: "pixel_x" or "pixel_y"
        max_pixel: Maximum valid pixel value

    Returns:
        True if valid
    """
    if not isinstance(axis, dict):
        return False

    # Check required fields
    if "label" not in axis or pixel_key not in axis or "world_x_mm" not in axis:
        # Try alternate key for world coordinate
        if "world_y_mm" not in axis:
            logger.warning(f"Axis missing required fields: {axis}")
            return False

    # Validate pixel coordinate
    try:
        pixel = float(axis[pixel_key])
        if not (0 <= pixel <= max_pixel):
            logger.warning(f"Pixel coordinate {pixel} outside image bounds (0-{max_pixel})")
            return False
    except (ValueError, TypeError, KeyError):
        logger.warning(f"Invalid pixel coordinate in axis: {axis}")
        return False

    return True


def _get_default_calibration() -> Dict[str, Any]:
    """
    Return default calibration when detection fails.

    Returns:
        Default calibration dict with no grid detected
    """
    return {
        "grid_system": {
            "detected": False,
            "x_axes": [],
            "y_axes": [],
            "origin": None
        },
        "scale_factor": None,
        "structure_type": "unknown",
        "confidence": "low",
        "notes": "Calibration failed or no grid system detected"
    }


def estimate_scale_from_grids(
    x_axes: List[Dict],
    y_axes: List[Dict]
) -> Optional[float]:
    """
    Estimate scale factor from grid spacing.

    Args:
        x_axes: List of X-axis grid definitions
        y_axes: List of Y-axis grid definitions

    Returns:
        Estimated scale factor (mm/pixel) or None if cannot estimate
    """
    if len(x_axes) < 2 and len(y_axes) < 2:
        return None

    scale_estimates = []

    # Estimate from X axes
    if len(x_axes) >= 2:
        for i in range(len(x_axes) - 1):
            pixel_distance = abs(x_axes[i+1]["pixel_x"] - x_axes[i]["pixel_x"])
            world_distance = abs(x_axes[i+1]["world_x_mm"] - x_axes[i]["world_x_mm"])

            if pixel_distance > 0:
                scale = world_distance / pixel_distance
                scale_estimates.append(scale)

    # Estimate from Y axes
    if len(y_axes) >= 2:
        for i in range(len(y_axes) - 1):
            pixel_distance = abs(y_axes[i+1]["pixel_y"] - y_axes[i]["pixel_y"])
            # Y axes use world_y_mm instead of world_x_mm
            world_key = "world_y_mm" if "world_y_mm" in y_axes[i] else "world_x_mm"
            world_distance = abs(y_axes[i+1].get(world_key, 0) - y_axes[i].get(world_key, 0))

            if pixel_distance > 0:
                scale = world_distance / pixel_distance
                scale_estimates.append(scale)

    if not scale_estimates:
        return None

    # Return median scale estimate
    scale_estimates.sort()
    median_scale = scale_estimates[len(scale_estimates) // 2]

    logger.info(f"Estimated scale from grids: {median_scale:.3f} mm/px "
               f"(from {len(scale_estimates)} grid intervals)")

    return median_scale
