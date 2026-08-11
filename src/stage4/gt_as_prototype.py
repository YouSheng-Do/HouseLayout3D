"""把 HF GT layout 實體轉成 Prototype——供 Stage 4 在真實多樓層建築上乾跑（不需 MP3D）。

GT 實體（layouts_split_by_entity/*.ply）無類別標籤：
  - 近水平面（≤10°）：用「相機位置」判 floor/ceiling——人在室內行走，相機多位於地板上方
    0.5~2.5m；對每個水平面統計其 BEV 足跡內、上方/下方帶內的相機數。
  - 近垂直面（≥80°）：wall。其餘：slanted（依相機在下方→ceiling）。
"""
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_prototype import Prototype, ProtoPolygon, _fit_plane  # noqa: E402

DATA = "/home/ado/storage/HouseLayout3D/external/houselayout3d/data"


def _cameras(scene: str) -> np.ndarray:
    with open(f"{DATA}/poses/{scene}.json") as f:
        poses = json.load(f)
    return np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in poses["frames"]])


def gt_prototype(scene: str) -> Prototype:
    cams = _cameras(scene)
    files = sorted(glob.glob(f"{DATA}/structures/layouts_split_by_entity/{scene}/*.ply"),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    all_v, all_t, polys = [], [], []
    voff = 0
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.geometry import Polygon as ShPolygon

    for f in files:
        m = o3d.io.read_triangle_mesh(f)
        V = np.asarray(m.vertices)
        T = np.asarray(m.triangles)
        if len(V) < 3 or len(T) < 1:
            continue
        plane = _fit_plane(V)
        e1 = V[T[:, 1]] - V[T[:, 0]]
        e2 = V[T[:, 2]] - V[T[:, 0]]
        area = float(np.linalg.norm(np.cross(e1, e2), axis=1).sum() / 2)
        if area < 0.05:
            continue
        ang = np.rad2deg(np.arccos(np.clip(abs(plane[2]), -1, 1)))  # 0=水平
        z = float(V[:, 2].mean())

        cls = "wall"
        if ang <= 10 or ang < 80:
            # BEV 足跡內的相機（粗略：以頂點 xy 的 bbox 近似，足跡小實體用 buffer）
            tris = [ShPolygon(V[t][:, :2]) for t in T]
            tris = [t.buffer(1e-6) for t in tris if t.area > 1e-8]
            fp = unary_union(tris) if tris else None
            above = below = 0
            if fp is not None and fp.area > 0.02:
                sel = cams[np.abs(cams[:, 2] - z) < 3.5]
                for c in sel:
                    if fp.buffer(0.3).contains(Point(c[0], c[1])):
                        dz = c[2] - z
                        if 0.3 < dz < 2.6:
                            above += 1
                        elif -2.6 < dz < -0.3:
                            below += 1
            if ang <= 10:
                cls = "floor" if above >= below and above > 0 else \
                      "ceiling" if below > above else \
                      ("floor" if plane[2] > 0 else "ceiling")  # 無相機證據→法向
            else:
                cls = "ceiling" if below > above else "slanted"

        polys.append((V, T, plane, cls, area, voff))
        all_v.append(V)
        all_t.append(T + voff)
        voff += len(V)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(all_v)),
        o3d.utility.Vector3iVector(np.concatenate(all_t)))
    proto = Prototype(mesh=mesh)
    tcount = 0
    for pid, (V, T, plane, cls, area, voff) in enumerate(polys):
        proto.polygons.append(ProtoPolygon(
            pid=pid, tri_ix=np.arange(tcount, tcount + len(T)), plane=plane,
            verts=V, tris3d=V[T], cls=cls, area=area))
        tcount += len(T)
    return proto


if __name__ == "__main__":
    from collections import Counter
    scene = sys.argv[1] if len(sys.argv) > 1 else "1LXtFkjw3qL"
    p = gt_prototype(scene)
    print(scene, dict(Counter(x.cls for x in p.polygons)), f"共 {len(p.polygons)} 實體")
