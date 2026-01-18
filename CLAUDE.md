# Blender CAD Import - 建築図面から3Dモデル自動生成

## プロジェクト概要

AutoCAD図面（.dwg/.dxf）から建築用3Dモデルを自動生成するBlenderアドオン。
既存のblender-mcpアドオンに統合し、プレゼンテーション品質の3Dモデルを作成します。

**目的**: 建築CAD図面（平面図・立面図・断面図）からVR/プレゼンテーション用の高品質3Dモデルを自動生成

**対象ユーザー**: 建築設計者、プレゼンター、VR開発者

## アーキテクチャ

### 3段階処理フロー

```
┌─────────────┐
│  DXF File   │
│ (2D Drawing)│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Phase 1: 建築要素抽出 (cad_parser)   │
│ - 壁（位置・厚み・高さ）              │
│ - 床スラブ                           │
│ - 開口（窓・扉: 位置・W・H）         │
│ - 柱・梁                             │
│ - 屋根                               │
│ ↓                                    │
│ 出力: JSON (building_elements.json)  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Phase 2: 3D生成 (Geometry Nodes)     │
│ - 壁ジオメトリ生成                   │
│ - 床スラブジオメトリ生成             │
│ - 開口カット（Boolean）              │
│ - 柱・梁生成                         │
│ - 屋根生成                           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Phase 3: マテリアル割当               │
│ - レイヤー名からルールベース割当      │
│   例: WALL_CONCRETE → コンクリート   │
│       GLASS → ガラス                 │
│       FLOOR_WOOD → 木材床            │
│ - プロシージャルテクスチャ生成        │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│ 3D Model    │
│ (.blend)    │
└─────────────┘
```

## データ構造

### 建築要素JSON (Building Elements)

#### データ検証ルール

JSONに落とし込む際の必須ルール:

1. **単位系**
   - すべての寸法値は**メートル単位 (m)** に統一
   - DXFの元の単位（mm、inch等）から自動変換
   - `"units": "meters"` を明示

2. **外周壁（Perimeter Walls）**
   - **閉じたポリライン**として定義（最初と最後の点が一致）
   - 開いたポリラインは警告ログを出力
   - 最小頂点数: 3点（三角形以上）
   - 時計回り/反時計回りは自動判定（面積計算で補正）

3. **開口（Openings: Windows/Doors）**
   - **ブロック (INSERT)** として検出
   - 必須属性:
     - `position`: 中心点座標 [x, y, z] (m)
     - `width`: 幅 (m)
     - `height`: 高さ (m)
     - `rotation`: 回転角度（度）
   - ブロック名パターンマッチング:
     - 窓: `WINDOW*`, `WIN*`, `FENETRE*`
     - 扉: `DOOR*`, `TUR*`, `PORTE*`

4. **壁厚検証**
   - 二重線壁: 内外ポリライン間距離を自動計算
   - 有効範囲: 0.05m ≤ 厚み ≤ 0.5m
   - 範囲外は警告ログ

5. **高さ検証**
   - 壁高さ: 1.5m ≤ height ≤ 10.0m
   - 窓高さ: 0.3m ≤ height ≤ 3.0m
   - 扉高さ: 1.8m ≤ height ≤ 3.0m
   - 範囲外は警告ログ（値は保持）

6. **重複排除**
   - 同一座標の頂点を除去（許容誤差: 0.001m）
   - 重複する壁を統合
   - 重複する開口を除去

7. **座標精度**
   - 座標値は小数点以下3桁（mm精度）に丸める
   - 例: 1.234567 → 1.235

8. **レイヤー名正規化**
   - 大文字に統一（`wall_concrete` → `WALL_CONCRETE`）
   - 前後の空白を除去

9. **必須フィールド検証**
   - 壁: `id`, `type`, `height`, `thickness` 必須
   - 開口: `id`, `type`, `position`, `width`, `height` 必須
   - 欠損時はデフォルト値で補完＋警告ログ

10. **トポロジー検証**
    - 自己交差するポリラインを検出（警告）
    - 壁の接続関係を検証（オプション）

#### JSONスキーマ

```json
{
  "version": "1.0",
  "units": "meters",
  "scale": 1.0,
  "elements": {
    "walls": [
      {
        "id": "wall_001",
        "type": "DOUBLE_LINE_WALL",
        "layer": "WALL_CONCRETE",
        "outer_boundary": [[0, 0], [6, 0], [6, 5], [0, 5], [0, 0]],
        "inner_boundary": [[0.12, 0.12], [5.88, 0.12], [5.88, 4.88], [0.12, 4.88], [0.12, 0.12]],
        "thickness": 0.12,
        "height": 2.4,
        "z_offset": 0.0,
        "material_hint": "concrete"
      }
    ],
    "floors": [
      {
        "id": "floor_001",
        "layer": "FLOOR_CONCRETE",
        "boundary": [[0, 0], [6, 0], [6, 5], [0, 5], [0, 0]],
        "thickness": 0.2,
        "z_offset": 0.0,
        "material_hint": "concrete"
      }
    ],
    "openings": [
      {
        "id": "window_001",
        "type": "window",
        "block_name": "WINDOW_01",
        "layer": "WALL_CONCRETE",
        "position": [1.5, 0, 0],
        "width": 1.2,
        "height": 1.2,
        "sill_height": 0.9,
        "depth": 0.35,
        "rotation": 0
      },
      {
        "id": "door_001",
        "type": "door",
        "block_name": "DOOR_01",
        "layer": "WALL_CONCRETE",
        "position": [6, 2.5, 0],
        "width": 0.9,
        "height": 2.1,
        "depth": 0.40,
        "rotation": 90
      }
    ],
    "columns": [
      {
        "id": "column_001",
        "layer": "STRUCTURE",
        "position": [1, 1, 0],
        "width": 0.3,
        "depth": 0.3,
        "height": 2.4
      }
    ],
    "beams": [
      {
        "id": "beam_001",
        "layer": "STRUCTURE",
        "start": [0, 0, 2.4],
        "end": [6, 0, 2.4],
        "width": 0.2,
        "height": 0.4
      }
    ],
    "roofs": [
      {
        "id": "roof_001",
        "layer": "ROOF",
        "boundary": [[0, 0], [6, 0], [6, 5], [0, 5], [0, 0]],
        "thickness": 0.15,
        "z_offset": 2.4,
        "pitch": 0,
        "material_hint": "concrete"
      }
    ]
  },
  "metadata": {
    "source_file": "test_complete_drawing.dxf",
    "import_date": "2026-01-17",
    "total_floors": 1,
    "building_height": 2.4
  }
}
```

## コンポーネント構成

### 1. CAD Parser (`cad_utils/cad_parser.py`)

**責務**: DXFファイルを解析し、建築要素JSONを生成

**主要機能**:
- `parse_dxf_to_elements()`: DXFファイル → 建築要素JSON
- `_extract_walls()`: 壁の検出（単線/二重線両対応）
- `_extract_floors()`: 床スラブの検出
- `_extract_openings()`: 窓・扉ブロックの検出
- `_extract_columns()`: 柱の検出
- `_extract_beams()`: 梁の検出
- `_extract_roofs()`: 屋根の検出

**入力**: DXFファイルパス、スケール設定
**出力**: 建築要素JSON (保存先: `{filename}_elements.json`)

### 2. Geometry Builder (`cad_utils/geometry_builder.py`)

**責務**: 建築要素JSONからGeometry Nodesベースの3Dジオメトリを生成

**主要機能**:
- `build_from_elements()`: JSON → 3Dモデル
- `_create_wall_geometry_nodes()`: 壁用Geometry Nodesセットアップ
- `_create_floor_geometry_nodes()`: 床用Geometry Nodesセットアップ
- `_create_opening_cutouts()`: Booleanで開口カット
- `_create_column_geometry()`: 柱ジオメトリ生成
- `_create_beam_geometry()`: 梁ジオメトリ生成
- `_create_roof_geometry()`: 屋根ジオメトリ生成

**Geometry Nodes利用理由**:
- パラメトリック編集が可能
- 後から高さ・厚みを調整可能
- 複数階の一括生成が容易
- プロシージャル生成で軽量

### 3. Material Mapper (`cad_utils/material_mapper.py`)

**責務**: レイヤー名からルールベースでマテリアルを自動割当

**マテリアルルール**:
```python
MATERIAL_RULES = {
    'WALL_CONCRETE': {'type': 'concrete', 'color': (0.7, 0.7, 0.7)},
    'WALL_BRICK': {'type': 'brick', 'color': (0.6, 0.3, 0.2)},
    'FLOOR_WOOD': {'type': 'wood', 'color': (0.4, 0.25, 0.1)},
    'FLOOR_TILE': {'type': 'tile', 'color': (0.9, 0.9, 0.85)},
    'GLASS': {'type': 'glass', 'color': (0.8, 0.9, 1.0), 'transmission': 0.95},
    'ROOF': {'type': 'concrete', 'color': (0.5, 0.5, 0.5)},
}
```

**主要機能**:
- `assign_materials_from_json()`: JSONのlayer情報からマテリアル割当
- `create_procedural_material()`: Principled BSDFベースのプロシージャルマテリアル生成
- `parse_layer_name()`: レイヤー名から用途推定

## ファイル構成

```
blender-mcp/
├── addon.py                        # メインアドオン（UI・オペレーター）
├── cad_utils/
│   ├── __init__.py
│   ├── cad_parser.py              # DXF → JSON変換
│   ├── geometry_builder.py        # JSON → 3Dモデル (Geometry Nodes)
│   └── material_mapper.py         # マテリアル自動割当
├── data/
│   └── building_elements.json     # 建築要素データ（生成物）
└── tests/
    ├── test_cad_parser.py
    └── test_geometry_builder.py

drawing/                            # サンプルDXFファイル
├── test_complete_drawing.dxf      # 平面図+立面図統合
├── test_architectural_standard.dxf # 二重線壁表記
└── create_*.py                     # DXF生成スクリプト
```

## 開発フェーズ

### Phase 1: 建築要素抽出 ✅ (一部完了)
- [x] DXF読込（ezdxf）
- [x] 単線壁検出
- [x] 二重線壁検出（建築標準）
- [x] 窓・扉ブロック検出
- [ ] **JSON出力機能** ← 次のタスク
- [ ] 床スラブ検出
- [ ] 柱・梁検出
- [ ] 屋根検出
- [ ] 複数階検出

### Phase 2: Geometry Nodes統合 (未着手)
- [ ] 壁生成用Geometry Nodesテンプレート作成
- [ ] 床生成用Geometry Nodesテンプレート作成
- [ ] JSONからGeometry Nodesパラメータ設定
- [ ] Boolean開口カット
- [ ] 柱・梁・屋根Geometry Nodes

### Phase 3: マテリアル自動割当 (未着手)
- [ ] レイヤー名パースルール実装
- [ ] プロシージャルマテリアル生成（コンクリート、木材、ガラス等）
- [ ] 既存PolyHaven統合との連携

### Phase 4: UI統合 (一部完了)
- [x] Import CADボタン
- [x] スケール設定UI
- [ ] JSON表示パネル
- [ ] 建築要素編集UI
- [ ] エラーログ表示

## 技術スタック

- **DXF解析**: ezdxf 1.3.0+
- **3D生成**: Blender Geometry Nodes 4.2+
- **マテリアル**: Principled BSDF (PBR)
- **データ形式**: JSON
- **Python**: 3.11+ (Blender内蔵Python)

## サンプルDXFファイル

| ファイル名 | 内容 | 用途 |
|-----------|------|------|
| `test_complete_drawing.dxf` | 平面図+南側立面図+東側立面図 | 統合図面テスト |
| `test_architectural_standard.dxf` | 二重線壁表記（建築標準） | 壁厚検出テスト |
| `test_room_with_openings.dxf` | 単線壁+窓・扉 | 基本インポートテスト |
| `test_south_elevation.dxf` | 南側立面図のみ | 窓高さ検出テスト |
| `test_east_elevation.dxf` | 東側立面図のみ | 扉高さ検出テスト |
| `test_section_view.dxf` | 断面図 | 床・天井厚検出テスト |

## 使用方法

### 1. DXFインポート（現在の実装）

```python
# Blender内で
1. アドオン有効化
2. サイドパネル → Blender MCP → CAD Import
3. 設定:
   - Import Scale: 0.1 (大きすぎる図面用)
   - Wall Height: 2.4m
   - Wall Thickness: 0.12m
4. "Import DXF File" ボタンクリック
5. DXFファイル選択
```

### 2. 将来の実装（Phase 2以降）

```python
# 建築要素JSONを編集してから3D生成
1. DXFインポート → JSON自動生成
2. JSON編集パネルで建築要素を調整
   - 壁高さ変更
   - 開口位置調整
   - マテリアル指定
3. "Generate 3D Model" ボタンで3D化
4. Geometry Nodesでパラメトリック編集可能
```

## パフォーマンス目標

- 一般的なアパート図面（5,000エンティティ）: <30秒
- 大規模DXF（50,000+エンティティ）: 1-2分
- JSON生成: <5秒
- Geometry Nodes 3D生成: <10秒

## エラー処理戦略

**ベストエフォート方式**:
- 壁検出失敗 → 警告ログ、空の壁リストで継続
- 開口検出失敗 → 警告、プレースホルダーボックスで代替
- JSON生成失敗 → デフォルト値で補完
- Geometry Nodes失敗 → フォールバックで通常メッシュ生成

## 今後の拡張

1. **BIM統合**: IFC形式出力対応
2. **複数ファイル統合**: 平面図+立面図+断面図の情報マージ
3. **AI支援**: 曖昧な要素をClaude APIで分類
4. **Sketchfab統合**: 窓・扉モデル自動配置（既存機能）
5. **パラメトリック編集**: JSON編集で即座に3D更新

## 参考資料

- [ezdxf documentation](https://ezdxf.readthedocs.io/)
- [Blender Geometry Nodes](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/)
- [AutoCAD DXF Reference](https://help.autodesk.com/view/OARX/2024/ENU/)
- [建築CAD標準](https://www.mlit.go.jp/jutakukentiku/build/cals.html)

## ライセンス

MIT License (blender-mcpプロジェクトに準拠)

## 開発者向けメモ

### 次の実装タスク (優先順位順)

1. ✅ **窓・扉深さ修正** - 壁貫通のため0.35m/0.4mに増加
2. **JSON出力機能** - `cad_parser.py`に`save_elements_json()`追加
3. **床スラブ検出** - LWPOLYLINE内部領域を床として認識
4. **Geometry Nodesテンプレート** - 壁生成用ノードグループ作成
5. **JSON → Geometry Nodes連携** - JSONデータをノードパラメータに反映

### デバッグ用スクリプト

```python
# 建築要素JSON検証
import json
with open('data/building_elements.json') as f:
    elements = json.load(f)
print(f"Walls: {len(elements['elements']['walls'])}")
print(f"Openings: {len(elements['elements']['openings'])}")
```

---

**最終更新**: 2026-01-17
**バージョン**: 0.2.0 (Phase 1 進行中)
