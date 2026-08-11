"""載入 HouseLayout3D GT 標註（HF 資料集格式）。"""
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import open3d as o3d

DATA_ROOT = "/home/ado/storage/HouseLayout3D/external/houselayout3d/data"


@dataclass
class SceneGT:
    scene: str
    structure_entities: List[o3d.geometry.TriangleMesh] = field(default_factory=list)
    doors: List[np.ndarray] = field(default_factory=list)      # 每個 (4,3)
    windows: List[np.ndarray] = field(default_factory=list)    # 每個 (4,3)
    stair_meshes: List[o3d.geometry.TriangleMesh] = field(default_factory=list)
    poses: Dict = field(default_factory=dict)
    structures_mesh: o3d.geometry.TriangleMesh = None          # 整體 layout mesh（深度評估用）


def _load_rects(path: str) -> List[np.ndarray]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    key = list(data.keys())[0]
    rects = []
    for e in data[key]:
        v = np.asarray(e["vertices"], dtype=np.float64)
        assert v.ndim == 2 and v.shape[0] >= 3 and v.shape[1] == 3, f"{path}: {v.shape}"
        rects.append(v)  # 多數是 (4,3)，少數標註為 >4 角的多邊形
    return rects


def load_scene_gt(scene: str, root: str = DATA_ROOT, load_poses: bool = True) -> SceneGT:
    gt = SceneGT(scene=scene)

    for f in sorted(glob.glob(f"{root}/structures/layouts_split_by_entity/{scene}/*.ply"),
                    key=lambda p: int(os.path.basename(p)[:-4])):
        m = o3d.io.read_triangle_mesh(f)
        if len(m.vertices) >= 3 and len(m.triangles) >= 1:
            gt.structure_entities.append(m)

    gt.doors = _load_rects(f"{root}/doors/{scene}.json")
    gt.windows = _load_rects(f"{root}/windows/{scene}.json")

    for f in sorted(glob.glob(f"{root}/stairs/{scene}/*.ply")):
        m = o3d.io.read_triangle_mesh(f)
        if len(m.vertices) >= 3:
            gt.stair_meshes.append(m)

    obj_path = f"{root}/structures/{scene}.obj"
    if os.path.exists(obj_path):
        gt.structures_mesh = o3d.io.read_triangle_mesh(obj_path)

    if load_poses:
        with open(f"{root}/poses/{scene}.json") as f:
            gt.poses = json.load(f)
    return gt


ALL_SCENES = [
    "2t7WUuJeko7", "WYY7iVyf5p8", "TbHJrupSAjP", "YFuZgdQ5vWj", "jtcxE69GiFV",
    "1LXtFkjw3qL", "5LpN3gDmAk7", "e9zR4mvMWw7", "i5noydFURQK", "HxpKQynjfin",
    "JeFG25nYj2p", "JmbYfDe2QKZ", "p5wJjkQkbXX", "r47D5H71a5s", "S9hNv5qa7GM",
    "17DRP5sb8fy",
]
