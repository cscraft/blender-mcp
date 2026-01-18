"""
Section Analyzer Module - Phase 2 of Gemini Integration

Extracts level markers from architectural section drawings:
- GL (Ground Level): 地盤面
- FL (Floor Level): 仕上げ床面
- SL (Slab Level): 構造スラブ天端
- CH (Ceiling Height): 天井高
- ▽ markers with elevation values

This module uses Gemini Vision API to perform OCR on section drawings
and extract precise floor heights, slab levels, and ceiling heights.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger("SectionAnalyzer")

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


def parse_section_levels(
    image_path: str,
    section_bounds: Tuple[float, float, float, float],
    api_key: str,
    model_name: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Extract level markers from section drawing.

    Args:
        image_path: Path to rendered DXF image (PNG)
        section_bounds: (min_x, min_y, max_x, max_y) bounds of section view in DXF coords
        api_key: Gemini API key
        model_name: Gemini model to use (default: gemini-2.0-flash)

    Returns:
        dict: {
            "ground_level": 0,  # GL (mm)
            "floor_levels": [
                {
                    "floor": "1F",
                    "FL": 100,           # Floor Level (mm)
                    "SL": 50,            # Slab Level (mm)
                    "CH": 2700,          # Ceiling Height (mm)
                    "floor_to_floor": 4000,  # Total floor height (mm)
                    "confidence": "high"
                },
                {
                    "floor": "2F",
                    "FL": 4100,
                    "SL": 4050,
                    "CH": 2700,
                    "floor_to_floor": 4000,
                    "confidence": "medium"
                }
            ],
            "overall_confidence": "medium",
            "notes": "Additional observations"
        }

    Raises:
        RuntimeError: If Gemini or PIL not available
        ValueError: If image cannot be loaded
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google.generativeai required for section analysis")

    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required for image loading")

    logger.info(f"Starting section level analysis for: {image_path}")
    logger.info(f"Section bounds: {section_bounds}")

    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        logger.info(f"Image size: {img_width} x {img_height}")

        # Crop to section bounds if possible
        # Convert DXF bounds to pixel coordinates (simplified)
        # For now, use full image - cropping can be added later
        section_img = img

    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Phase 2: Section Level Marker Analysis Prompt (Japanese)
    prompt = """あなたは建築図面の専門家です。添付された断面図を分析してください。

【タスク1: レベルマーカーの検出】
以下のレベルマーカーを読み取ってください:

1. **GL (Ground Level)**: 地盤面の高さ（通常0mm基準）
2. **FL (Floor Level)**: 各階の仕上げ床面の高さ
3. **SL (Slab Level)**: 構造スラブ天端の高さ（通常FLより50-100mm下）
4. **CH (Ceiling Height)**: 天井高（例: CH=2700）
5. **▽記号**: 高さを示すマーカー（例: ▽+3500, ▽FL+100）

【タスク2: 階高の計算】
- 各階のFL間の距離を計算し、階高（Floor-to-Floor Height）を求めてください
- 寸法線に記載された数値も読み取ってください
- 1階、2階、3階...それぞれの情報を分離してください

【タスク3: 天井高の抽出】
- 各階の天井高（CH）を読み取ってください
- CHマーカーがない場合は、FLから天井面までの距離を推定してください

【タスク4: スラブレベルの推定】
- SLマーカーがある場合は読み取ってください
- ない場合は、FLから50-100mm下と推定してください

以下のJSON形式で出力してください（markdownのコードブロック不要、純粋なJSONのみ）:

{
  "ground_level": 0,
  "floor_levels": [
    {
      "floor": "1F",
      "FL": 100,
      "SL": 50,
      "CH": 2700,
      "floor_to_floor": 4000,
      "confidence": "high"
    },
    {
      "floor": "2F",
      "FL": 4100,
      "SL": 4050,
      "CH": 2700,
      "floor_to_floor": 4000,
      "confidence": "medium"
    }
  ],
  "overall_confidence": "high" または "medium" または "low",
  "notes": "観察された特徴や判定根拠"
}

注意事項:
- すべての寸法はmm単位で出力してください
- 数値は必ず数値型（文字列ではない）で出力してください
- JSONのみを出力し、説明文やマークダウン記法は含めないでください
- レベルマーカーが見つからない場合は、confidence: "low" として推定値を出力してください
- 階の順番は1F, 2F, 3F...の順に並べてください
"""

    logger.info("Calling Gemini API for section level analysis...")

    try:
        response = model.generate_content([prompt, section_img])
        response_text = response.text.strip()

        logger.debug(f"Gemini response (raw): {response_text[:500]}...")

        # Parse JSON response
        level_data = _parse_gemini_response(response_text)

        # Validate and normalize
        level_data = _validate_level_data(level_data)

        logger.info(f"Section analysis complete: {len(level_data.get('floor_levels', []))} floors detected, "
                   f"confidence={level_data.get('overall_confidence', 'unknown')}")

        return level_data

    except Exception as e:
        logger.error(f"Section analysis failed: {e}")
        # Return default level data (single floor with heuristics)
        return _get_default_level_data()


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


def _validate_level_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize level data.

    Args:
        data: Raw level data from Gemini

    Returns:
        Validated and normalized level data
    """
    validated = {
        "ground_level": 0,
        "floor_levels": [],
        "overall_confidence": "low",
        "notes": ""
    }

    # Validate ground_level
    if "ground_level" in data:
        try:
            validated["ground_level"] = float(data["ground_level"])
        except (ValueError, TypeError):
            logger.warning(f"Invalid ground_level: {data.get('ground_level')}, using 0")

    # Validate floor_levels
    if "floor_levels" in data and isinstance(data["floor_levels"], list):
        for floor_data in data["floor_levels"]:
            if not isinstance(floor_data, dict):
                continue

            floor_entry = {
                "floor": str(floor_data.get("floor", "1F")),
                "FL": 0,
                "SL": 0,
                "CH": 2700,  # Default ceiling height
                "floor_to_floor": 4000,  # Default floor-to-floor
                "confidence": "low"
            }

            # Validate FL (Floor Level)
            if "FL" in floor_data:
                try:
                    fl = float(floor_data["FL"])
                    if -1000 <= fl <= 100000:  # Reasonable range
                        floor_entry["FL"] = fl
                except (ValueError, TypeError):
                    logger.warning(f"Invalid FL: {floor_data.get('FL')}")

            # Validate SL (Slab Level)
            if "SL" in floor_data:
                try:
                    sl = float(floor_data["SL"])
                    if -1000 <= sl <= 100000:
                        floor_entry["SL"] = sl
                    else:
                        # Default: FL - 50mm
                        floor_entry["SL"] = floor_entry["FL"] - 50
                except (ValueError, TypeError):
                    floor_entry["SL"] = floor_entry["FL"] - 50
            else:
                floor_entry["SL"] = floor_entry["FL"] - 50

            # Validate CH (Ceiling Height)
            if "CH" in floor_data:
                try:
                    ch = float(floor_data["CH"])
                    if 2000 <= ch <= 5000:  # Reasonable ceiling height range
                        floor_entry["CH"] = ch
                except (ValueError, TypeError):
                    logger.warning(f"Invalid CH: {floor_data.get('CH')}")

            # Validate floor_to_floor
            if "floor_to_floor" in floor_data:
                try:
                    ftf = float(floor_data["floor_to_floor"])
                    if 2500 <= ftf <= 6000:  # Reasonable floor-to-floor range
                        floor_entry["floor_to_floor"] = ftf
                except (ValueError, TypeError):
                    logger.warning(f"Invalid floor_to_floor: {floor_data.get('floor_to_floor')}")

            # Validate confidence
            if "confidence" in floor_data:
                conf = str(floor_data["confidence"]).lower()
                if conf in ["high", "medium", "low"]:
                    floor_entry["confidence"] = conf

            validated["floor_levels"].append(floor_entry)

    # Validate overall_confidence
    if "overall_confidence" in data:
        conf = str(data["overall_confidence"]).lower()
        if conf in ["high", "medium", "low"]:
            validated["overall_confidence"] = conf

    # Notes
    if "notes" in data:
        validated["notes"] = str(data["notes"])

    return validated


def _get_default_level_data() -> Dict[str, Any]:
    """
    Return default level data when detection fails.

    Uses standard office building heuristics:
    - FL: 100mm (raised floor)
    - SL: 50mm (50mm below FL)
    - CH: 2700mm (standard ceiling height)
    - Floor-to-floor: 4000mm (standard office building)

    Returns:
        Default level data dict
    """
    return {
        "ground_level": 0,
        "floor_levels": [
            {
                "floor": "1F",
                "FL": 100,
                "SL": 50,
                "CH": 2700,
                "floor_to_floor": 4000,
                "confidence": "low"
            }
        ],
        "overall_confidence": "low",
        "notes": "Section analysis failed. Using default office building heuristics."
    }


def integrate_section_data_to_floors(
    floors: List[Dict[str, Any]],
    level_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Integrate section level data into floors structure.

    Args:
        floors: List of floor dictionaries from JSON schema v2.0
        level_data: Level data from parse_section_levels()

    Returns:
        Updated floors list with integrated level data
    """
    logger.info("Integrating section level data into floors structure")

    # Map floor IDs to level data
    level_map = {}
    for level_entry in level_data.get("floor_levels", []):
        floor_id = level_entry["floor"]
        level_map[floor_id] = level_entry

    # Update existing floors
    for floor in floors:
        floor_id = floor["id"]

        if floor_id in level_map:
            level = level_map[floor_id]

            # Update floor data with section information
            floor["elevation_mm"] = level["FL"]
            floor["floor_height_mm"] = level["floor_to_floor"]
            floor["ceiling_height_mm"] = level["CH"]
            floor["slab_level_mm"] = level["SL"]

            # Add metadata
            if "metadata" not in floor:
                floor["metadata"] = {}

            floor["metadata"]["section_analysis"] = {
                "FL": level["FL"],
                "SL": level["SL"],
                "CH": level["CH"],
                "floor_to_floor": level["floor_to_floor"],
                "confidence": level["confidence"]
            }

            logger.info(f"Updated {floor_id}: FL={level['FL']}mm, CH={level['CH']}mm, "
                       f"floor_to_floor={level['floor_to_floor']}mm")
        else:
            logger.warning(f"No section data found for {floor_id}, using default values")

    # Add new floors if section data has more floors than current structure
    existing_ids = {f["id"] for f in floors}
    for floor_id, level in level_map.items():
        if floor_id not in existing_ids:
            logger.info(f"Adding new floor {floor_id} from section data")

            new_floor = {
                "id": floor_id,
                "elevation_mm": level["FL"],
                "floor_height_mm": level["floor_to_floor"],
                "ceiling_height_mm": level["CH"],
                "slab_level_mm": level["SL"],
                "elements": {
                    "walls": [],
                    "openings": [],
                    "columns": [],
                    "beams": [],
                    "slabs": []
                },
                "metadata": {
                    "section_analysis": {
                        "FL": level["FL"],
                        "SL": level["SL"],
                        "CH": level["CH"],
                        "floor_to_floor": level["floor_to_floor"],
                        "confidence": level["confidence"]
                    }
                }
            }
            floors.append(new_floor)

    return floors
