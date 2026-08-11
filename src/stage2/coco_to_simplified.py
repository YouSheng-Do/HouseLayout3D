"""COCO panoptic 類別 → MULTIFLOOR3D 簡化類別（Appendix Table 5）。

簡化詞彙表刻意保留 Surfaces 內的細分（cabinet/door/curtain/window_blind），
因為 Stage 4 窗偵測要用 {window, window_blind, curtain}（附錄 B 的兩點修改），
而 skeleton 保留類（keep-classes）可用名稱列表自由組合。
"""
from typing import Dict, List

import numpy as np

# 簡化類別（uint8 id 順序即 labels.txt 行序）
SIMPLIFIED_LABELS: List[str] = [
    "wall",          # 0
    "ceiling",       # 1
    "floor",         # 2
    "cabinet",       # 3  (Surfaces)
    "door",          # 4  (Surfaces: door-stuff)
    "curtain",       # 5  (Surfaces；窗偵測時併入 window)
    "window_blind",  # 6  (Surfaces；窗偵測時併入 window)
    "window",        # 7  (幾何不準表面)
    "mirror",        # 8  (幾何不準表面)
    "outdoor",       # 9  (幾何不準表面/雜訊)
    "stairs",        # 10
    "object",        # 11 (其餘全部)
]

# extract_skeleton 的 --keep-classes 建議值（Table 5 的 Structure 全家 + stairs）
KEEP_CLASSES = "ceiling,wall,floor,cabinet,door,curtain,window_blind,stairs"
WINDOW_CLASSES_SKELETON = "window"                       # Stage 2 skeleton 階段
WINDOW_CLASSES_DETECTION = "window,window_blind,curtain"  # Stage 4 窗偵測（附錄 B）

# COCO panoptic 類別名（HF id2label 的 value）→ 簡化類別名
COCO_NAME_TO_SIMPLIFIED: Dict[str, str] = {
    # Structure / Wall
    "wall-brick": "wall", "wall-stone": "wall", "wall-tile": "wall",
    "wall-wood": "wall", "wall-other-merged": "wall",
    # Structure / Ceiling
    "ceiling-merged": "ceiling",
    # Structure / Floor
    "floor-wood": "floor", "floor-other-merged": "floor", "rug-merged": "floor",
    # Structure / Surfaces
    "cabinet-merged": "cabinet", "door-stuff": "door",
    "curtain": "curtain", "window-blind": "window_blind",
    # 幾何不準表面
    "window-other": "window", "mirror-stuff": "mirror",
    # Outdoor / Noise
    "gravel": "outdoor", "tree-merged": "outdoor", "sky-other-merged": "outdoor",
    "pavement-merged": "outdoor", "grass-merged": "outdoor", "dirt-merged": "outdoor",
    # Stairs
    "stairs": "stairs",
    # 其餘 → object（在 build_lut 中處理）
}


def build_lut(id2label: Dict[int, str]) -> np.ndarray:
    """model.config.id2label → LUT: coco_id -> simplified_id（uint8）。"""
    n = max(int(k) for k in id2label.keys()) + 1
    lut = np.full(n, SIMPLIFIED_LABELS.index("object"), dtype=np.uint8)
    matched = []
    for k, name in id2label.items():
        simp = COCO_NAME_TO_SIMPLIFIED.get(name)
        if simp is not None:
            lut[int(k)] = SIMPLIFIED_LABELS.index(simp)
            matched.append(name)
    missing = set(COCO_NAME_TO_SIMPLIFIED) - set(matched)
    assert not missing, f"Table 5 類別在 id2label 中找不到: {missing}"
    return lut
