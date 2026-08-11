"""Stage 4 輸入適配：從 Stage 3 的 fitted_mesh.ply 重建 polygon 結構。

官方 fit_prototype 只輸出三角化 ply（頂點色＝polygon 顏色，來自我們 init 的隨機色），
未輸出最終 polygon 類別/平面 JSON。此模組：
  1. 以「頂點色」分組三角形 → polygon
  2. 每 polygon SVD 擬合平面
  3. 分類：幾何規則（法向 vs up，附錄 fit 內同款 10° 規則）＋ Stage 2 骨架頂點語意投票
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

UP = np.array([0.0, 0.0, 1.0])  # FARO/MP3D 皆 Z-up


@dataclass
class ProtoPolygon:
    pid: int
    tri_ix: np.ndarray            # 索引 fitted mesh 三角形
    plane: np.ndarray             # (4,) n·x + d = 0，n 單位化
    verts: np.ndarray             # (k,3) 此 polygon 用到的頂點
    tris3d: np.ndarray = None     # (m,3,3) 三角形座標（BEV 精確足跡用）
    cls: str = "unknown"
    area: float = 0.0

    @property
    def elevation(self) -> float:
        return float(self.verts[:, 2].mean())


@dataclass
class Prototype:
    mesh: o3d.geometry.TriangleMesh
    polygons: List[ProtoPolygon] = field(default_factory=list)

    def by_class(self, cls: str) -> List[ProtoPolygon]:
        return [p for p in self.polygons if p.cls == cls]


def _fit_plane(pts: np.ndarray):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return np.array([*n, -float(n @ c)])


def load_prototype(fitted_ply: str, skeleton_ply: str = None, skeleton_probs: str = None,
                   labels_npy: str = None, sem_vote_dist: float = 0.15,
                   min_area: float = 0.05) -> Prototype:
    mesh = o3d.io.read_triangle_mesh(fitted_ply)
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    C = (np.asarray(mesh.vertex_colors) * 255).round().astype(int)

    tri_color = C[T[:, 0]]
    keys = tri_color[:, 0] * 1000000 + tri_color[:, 1] * 1000 + tri_color[:, 2]
    proto = Prototype(mesh=mesh)

    # 語意投票資料（Stage 2 骨架頂點的粗粒度機率）
    sem = None
    if skeleton_ply and skeleton_probs and labels_npy:
        skel = o3d.io.read_triangle_mesh(skeleton_ply)
        sp = np.asarray(skel.vertices)
        probs = np.load(skeleton_probs).astype(np.float32)
        labels = [str(x) for x in np.load(labels_npy)]
        kd = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sp)))
        sem = (sp, probs, labels, kd)

    for pid, key in enumerate(np.unique(keys)):
        tri_ix = np.where(keys == key)[0]
        vids = np.unique(T[tri_ix])
        verts = V[vids]
        if len(verts) < 3:
            continue
        plane = _fit_plane(verts)
        e1 = V[T[tri_ix, 1]] - V[T[tri_ix, 0]]
        e2 = V[T[tri_ix, 2]] - V[T[tri_ix, 0]]
        area = float(np.linalg.norm(np.cross(e1, e2), axis=1).sum() / 2)
        if area < min_area:
            continue

        # 幾何角度（同官方 fit 的 10° 規則）
        cos_up = abs(float(plane[:3] @ UP))
        ang_from_horiz = np.rad2deg(np.arccos(np.clip(cos_up, -1, 1)))  # 0=水平面
        can_floor_ceiling = ang_from_horiz <= 10
        can_wall = ang_from_horiz >= 80

        # 語意投票
        vote = None
        if sem is not None:
            sp, probs, labels, kd = sem
            centers = V[T[tri_ix]].mean(axis=1)
            samp = centers[:: max(1, len(centers) // 200)]
            acc = np.zeros(probs.shape[1])
            hit = 0
            for q in samp:
                k, idx, d2 = kd.search_knn_vector_3d(q, 1)
                if k > 0 and d2[0] < sem_vote_dist ** 2:
                    acc += probs[idx[0]]
                    hit += 1
            if hit > 0:
                vote = acc / hit

        cls = "unknown"
        if can_floor_ceiling:
            # 上下由語意優先，否則由法向 z 分量與高度啟發
            # 2t7W 診斷（2026-07-14）：櫃頂/簾頂等 surface 水平面此前必成 floor/ceiling，
            # 再被 extrude「最低天花板獲勝」放大成房間高度的假天花 → front 失敗主體。
            # 全類別 argmax：top 非 {floor,ceiling} 且 ≥0.15 → surface（不進樓層幾何）。
            if vote is not None and labels.index("floor") < len(vote):
                order = np.argsort(vote)[::-1]
                top_all = labels[order[0]]
                if top_all not in ("floor", "ceiling") and vote[order[0]] >= 0.15:
                    cls = "surface"
                else:
                    fl, ce = vote[labels.index("floor")], vote[labels.index("ceiling")]
                    cls = "floor" if fl >= ce else "ceiling"
            else:
                cls = "floor" if plane[2] > 0 else "ceiling"
        elif can_wall:
            cls = "wall"
            if vote is not None:
                # 官方 fit 的 polygon 分類是語意式：surface（窗簾/櫃/門）不是 wall，
                # 不得參與 D.2 牆選擇與 D.3 刻蝕（否則假隔牆封視線/切房間）。
                # 2t7W 診斷（2026-07-14）：舊版只比 {wall,surface,door} 三類，
                # window/curtain 等其他類別的直立面繞過過濾照樣當牆 → 改全類別 argmax。
                order = np.argsort(vote)[::-1]
                top_all = labels[order[0]]
                if top_all == "door" and vote[order[0]] >= 0.15:
                    cls = "wall"          # 門面維持牆身分（門洞由 D.3/D.6 開）
                elif top_all != "wall" and vote[order[0]] >= 0.15:
                    cls = "surface"       # 窗簾/櫃/窗/鏡…等非牆直立面
        else:
            cls = "slanted"  # 斜面（斜天花板等），暫留
            if vote is not None:
                order = np.argsort(vote)[::-1]
                top = labels[order[0]]
                if top in ("ceiling", "floor", "wall"):
                    cls = top
        proto.polygons.append(ProtoPolygon(pid=pid, tri_ix=tri_ix, plane=plane,
                                           verts=verts, tris3d=V[T[tri_ix]],
                                           cls=cls, area=area))
    return proto


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted", required=True)
    ap.add_argument("--skeleton", default=None)
    ap.add_argument("--probs", default=None)
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    p = load_prototype(a.fitted, a.skeleton, a.probs, a.labels)
    print(f"polygons: {len(p.polygons)}")
    print("類別分布:", dict(Counter(x.cls for x in p.polygons)))
    for cls in ("floor", "ceiling"):
        for poly in p.by_class(cls):
            print(f"  {cls} pid={poly.pid} area={poly.area:.1f}m² elev={poly.elevation:.2f}m")
