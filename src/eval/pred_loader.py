"""解析官方預測 combined.ply → 實體集合。

顏色語意（已驗證）：灰(191,191,191)=結構殼、綠(0,255,0)=門框、藍(0,0,255)=窗。
- 結構：以「共享邊＋共面＋同法向」做 region growing → 平面實體（wall/floor/ceiling 不分類，Table 2 合併評估）。
- 門/窗：以共享頂點做 connected components → 每組擬合平面矩形（PCA）。
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import open3d as o3d

GRAY, GREEN, BLUE = (191, 191, 191), (0, 255, 0), (0, 0, 255)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _face_groups_to_meshes(verts: np.ndarray, faces: np.ndarray, groups: List[np.ndarray]) -> List[o3d.geometry.TriangleMesh]:
    out = []
    for g in groups:
        f = faces[g]
        used = np.unique(f)
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts[used]),
            o3d.utility.Vector3iVector(remap[f]))
        out.append(m)
    return out


def _components(faces: np.ndarray, keep_edge) -> List[np.ndarray]:
    """faces 上的 union-find；keep_edge(f1,f2) 決定兩共享邊的面是否相連。"""
    uf = _UF(len(faces))
    edge_map = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in ((a, b), (b, c), (c, a)):
            key = (min(e), max(e))
            other = edge_map.get(key)
            if other is None:
                edge_map[key] = fi
            else:
                if keep_edge(other, fi):
                    uf.union(other, fi)
    roots = {}
    for fi in range(len(faces)):
        roots.setdefault(uf.find(fi), []).append(fi)
    return [np.asarray(v) for v in roots.values()]


def _vertex_share_components(faces: np.ndarray) -> List[np.ndarray]:
    uf = _UF(len(faces))
    v2f = {}
    for fi, f in enumerate(faces):
        for v in f:
            if v in v2f:
                uf.union(v2f[v], fi)
            else:
                v2f[v] = fi
    roots = {}
    for fi in range(len(faces)):
        roots.setdefault(uf.find(fi), []).append(fi)
    return [np.asarray(v) for v in roots.values()]


def fit_rectangle(pts: np.ndarray) -> np.ndarray:
    """點集 → 平面 PCA → 主軸極值 → 4 角 (4,3)。"""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    ax_u, ax_v = vt[0], vt[1]
    u = q @ ax_u
    v = q @ ax_v
    corners = []
    for su, sv in ((u.min(), v.min()), (u.min(), v.max()), (u.max(), v.max()), (u.max(), v.min())):
        corners.append(c + su * ax_u + sv * ax_v)
    return np.asarray(corners)


@dataclass
class ScenePred:
    scene: str
    structure_entities: List[o3d.geometry.TriangleMesh] = field(default_factory=list)
    doors: List[np.ndarray] = field(default_factory=list)
    windows: List[np.ndarray] = field(default_factory=list)
    full_mesh: o3d.geometry.TriangleMesh = None


def load_scene_pred(ply_path: str, scene: str,
                    normal_tol_deg: float = 1.0, coplanar_tol: float = 0.005,
                    signed_normals: bool = False, min_area: float = 0.1) -> ScenePred:
    """min_area：結構實體最小面積 m²（過濾三角化殘留碎片；校準實驗定為 0.1）。"""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = (np.asarray(mesh.vertex_colors) * 255).round().astype(int)

    # 面類別 = 三頂點顏色一致（取第一個頂點色）
    fcol = colors[faces[:, 0]]
    is_gray = np.all(fcol == GRAY, axis=1)
    is_green = np.all(fcol == GREEN, axis=1)
    is_blue = np.all(fcol == BLUE, axis=1)

    pred = ScenePred(scene=scene, full_mesh=mesh)

    # --- 結構：先抽灰色子網格並 weld 重複頂點（combined.ply 未 weld，會斷開共享邊）---
    gidx = np.where(is_gray)[0]
    sub = o3d.geometry.TriangleMesh(mesh)
    sub.remove_triangles_by_mask(~is_gray)
    sub.remove_unreferenced_vertices()
    sub = sub.merge_close_vertices(1e-3)
    sub.remove_degenerate_triangles()
    verts_g = np.asarray(sub.vertices)
    gfaces = np.asarray(sub.triangles)
    verts, gfaces_src = verts, faces  # 門窗仍用原始（未 weld）網格
    fn = np.cross(verts_g[gfaces[:, 1]] - verts_g[gfaces[:, 0]], verts_g[gfaces[:, 2]] - verts_g[gfaces[:, 0]])
    norm = np.linalg.norm(fn, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-12
    fn = np.where(norm > 1e-12, fn / np.maximum(norm, 1e-12), 0.0)
    fcent = verts_g[gfaces].mean(axis=1)
    cos_tol = np.cos(np.deg2rad(normal_tol_deg))

    def keep(f1, f2):
        if not (valid[f1] and valid[f2]):
            return False
        dot = float(fn[f1] @ fn[f2])
        if (dot if signed_normals else abs(dot)) < cos_tol:
            return False
        return abs(float(fn[f1] @ (fcent[f2] - fcent[f1]))) < coplanar_tol

    groups = _components(gfaces, keep)
    groups = [g for g in groups if valid[g].any()]
    ents = _face_groups_to_meshes(verts_g, gfaces, groups)
    pred.structure_entities = [m for m in ents if m.get_surface_area() >= min_area]

    # --- 門 / 窗：頂點共享 CC → 矩形 ---
    for mask, out in ((is_green, pred.doors), (is_blue, pred.windows)):
        cfaces = faces[np.where(mask)[0]]
        if len(cfaces) == 0:
            continue
        for g in _vertex_share_components(cfaces):
            pts = verts[np.unique(cfaces[g])]
            out.append(fit_rectangle(pts))
    return pred
