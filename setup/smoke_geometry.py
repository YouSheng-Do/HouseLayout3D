"""Stage 0 smoke check: geometry env 功能測試（CDT/DBSCAN/LOF/RDP/open3d/shapely）。"""
import numpy as np

# 1) constrained Delaunay via triangle（帶洞方形，Stage 4 room extrusion 用法的迷你版）
import triangle as tr
square = {
    "vertices": np.array([[0, 0], [4, 0], [4, 4], [0, 4],
                          [1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]], dtype=float),
    "segments": np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4]]),
    "holes": np.array([[2.0, 2.0]]),
}
t = tr.triangulate(square, "p")
assert "triangles" in t and len(t["triangles"]) > 0
print(f"CDT: {len(t['triangles'])} triangles (帶洞方形) OK")

# 2) DBSCAN + LOF（窗偵測用）
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
rng = np.random.RandomState(0)
pts = np.vstack([rng.normal([0, 0], 0.05, (60, 2)),
                 rng.normal([2, 2], 0.05, (60, 2)),
                 rng.uniform(-1, 3, (10, 2))])
labels = DBSCAN(eps=0.2, min_samples=10).fit_predict(pts)
lof = LocalOutlierFactor(n_neighbors=15).fit_predict(pts)
print(f"DBSCAN: {len(set(labels) - {-1})} clusters | LOF outliers: {(lof == -1).sum()} OK")

# 3) RDP 簡化（vertex merging 用）
from rdp import rdp
line = np.array([[0, 0], [1, 0.02], [2, -0.01], [3, 0.01], [4, 0], [4, 4]], dtype=float)
simp = rdp(line, epsilon=0.1)
assert len(simp) == 3
print(f"RDP: {len(line)} -> {len(simp)} points OK")

# 4) open3d Poisson + shapely union（補地板洞用）
import open3d as o3d
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(2000)
pcd.estimate_normals()
m2, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
print(f"open3d Poisson: {len(m2.vertices)} verts OK")

from shapely.geometry import Polygon
from shapely.ops import unary_union
u = unary_union([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                 Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
print(f"shapely union area: {u.area:.1f} (期望 7.0) OK")

import networkx, skfmm, trimesh
print("networkx/skfmm/trimesh imports OK — geometry env 全部通過")
