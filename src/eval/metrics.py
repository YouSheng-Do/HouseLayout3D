"""MULTIFLOOR3D 評估指標：d_E（矩形，Hungarian 角點）、d_H（廣義 Hausdorff）、F1@τ。

依論文 §5：
- d_E(E,E') = max_i ||c_i - c'_{π(i)}||，π 為 Hungarian 最佳角點排列（同類矩形）。
- d_H(P,P') = max{ max_{v∈V} D_pp(v,P'), max_{v'∈V'} D_pp(v',P) }（式 6）。
- F1@τ：實體間以 cost 矩陣做 Hungarian 匹配，d ≤ τ 者算命中；P=M/|pred|、R=M/|gt|。
"""
from typing import Callable, List, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment


def rect_entity_distance(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """d_E：兩個 4 角矩形，Hungarian 最佳角點對應後取最大角距。"""
    assert corners_a.shape == (4, 3) and corners_b.shape == (4, 3)
    cost = np.linalg.norm(corners_a[:, None, :] - corners_b[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].max())


class MeshDistanceField:
    """對單一實體 mesh 的點到面距離查詢（open3d RaycastingScene BVH）。"""

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def dist(self, pts: np.ndarray) -> np.ndarray:
        q = o3d.core.Tensor(np.asarray(pts, dtype=np.float32))
        return self.scene.compute_distance(q).numpy()


def pairwise_hausdorff(
    gt_meshes: Sequence[o3d.geometry.TriangleMesh],
    pred_meshes: Sequence[o3d.geometry.TriangleMesh],
) -> np.ndarray:
    """d_H 成對矩陣 (n_gt, n_pred)。

    每個實體只建一次距離場；GT 全部頂點一次查向每個 pred 實體、反之亦然。
    """
    gt_verts = [np.asarray(m.vertices) for m in gt_meshes]
    pr_verts = [np.asarray(m.vertices) for m in pred_meshes]
    gt_all = np.concatenate(gt_verts, axis=0)
    pr_all = np.concatenate(pr_verts, axis=0)
    gt_slices = np.cumsum([0] + [len(v) for v in gt_verts])
    pr_slices = np.cumsum([0] + [len(v) for v in pr_verts])

    # dist(所有 GT 頂點 → pred_j 表面)
    d_gtv_to_pred = np.stack([MeshDistanceField(m).dist(gt_all) for m in pred_meshes])  # (n_pred, n_gt_pts)
    # dist(所有 pred 頂點 → gt_i 表面)
    d_prv_to_gt = np.stack([MeshDistanceField(m).dist(pr_all) for m in gt_meshes])      # (n_gt, n_pr_pts)

    n_gt, n_pr = len(gt_meshes), len(pred_meshes)
    dh = np.empty((n_gt, n_pr), dtype=np.float64)
    for i in range(n_gt):
        gi = slice(gt_slices[i], gt_slices[i + 1])
        for j in range(n_pr):
            pj = slice(pr_slices[j], pr_slices[j + 1])
            a = d_gtv_to_pred[j, gi].max()   # GT_i 頂點 → pred_j 面
            b = d_prv_to_gt[i, pj].max()     # pred_j 頂點 → GT_i 面
            dh[i, j] = max(a, b)
    return dh


def polygon_to_mesh(poly: np.ndarray) -> o3d.geometry.TriangleMesh:
    """平面多邊形 (n,3) → 質心扇形三角化 mesh。"""
    n = len(poly)
    c = poly.mean(axis=0, keepdims=True)
    verts = np.vstack([poly, c])
    tris = np.asarray([[i, (i + 1) % n, n] for i in range(n)], dtype=np.int64)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))


def generalized_entity_distance(a: np.ndarray, b: np.ndarray) -> float:
    """兩邊皆 4 角 → d_E；否則廣義 d_H（頂點 ↔ 多邊形面，式 6）。"""
    if a.shape == (4, 3) and b.shape == (4, 3):
        return rect_entity_distance(a, b)
    ma, mb = polygon_to_mesh(a), polygon_to_mesh(b)
    d1 = MeshDistanceField(mb).dist(a).max()
    d2 = MeshDistanceField(ma).dist(b).max()
    return float(max(d1, d2))


def pairwise_rect(gt_rects: Sequence[np.ndarray], pred_rects: Sequence[np.ndarray]) -> np.ndarray:
    de = np.empty((len(gt_rects), len(pred_rects)), dtype=np.float64)
    for i, g in enumerate(gt_rects):
        for j, p in enumerate(pred_rects):
            de[i, j] = generalized_entity_distance(g, p)
    return de


def stair_entity_distance(gt_mesh: o3d.geometry.TriangleMesh, pred_rect: np.ndarray) -> float:
    """樓梯：GT 是 mesh（可能 4 角矩形或 7–14 頂點非矩形），pred 是 4 角矩形。
    兩邊皆矩形 → d_E；GT 非矩形 → 廣義 d_H，且 **用 GT mesh 的真實三角面**（非重新扇形化，
    因非矩形 GT 頂點未必是有序邊界環）。"""
    gv = np.asarray(gt_mesh.vertices, dtype=np.float64)
    pr = np.asarray(pred_rect, dtype=np.float64)
    if gv.shape == (4, 3) and pr.shape == (4, 3):
        return rect_entity_distance(gv, pr)
    pred_mesh = polygon_to_mesh(pr)                       # 4 角 → 扇形（合法有序）
    a = MeshDistanceField(pred_mesh).dist(gv).max()       # GT 頂點 → pred 面
    b = MeshDistanceField(gt_mesh).dist(pr).max()         # pred 角 → GT 真實面
    return float(max(a, b))


def pairwise_stairs(gt_meshes: Sequence[o3d.geometry.TriangleMesh],
                    pred_rects: Sequence[np.ndarray]) -> np.ndarray:
    """全 34 GT 樓梯（mesh）× pred 矩形 的單一 distance matrix，d_E/d_H 混用。"""
    d = np.empty((len(gt_meshes), len(pred_rects)), dtype=np.float64)
    for i, gm in enumerate(gt_meshes):
        for j, prj in enumerate(pred_rects):
            d[i, j] = stair_entity_distance(gm, prj)
    return d


def f1_at_tau(dist_matrix: np.ndarray, tau: float) -> Tuple[float, int, float, float]:
    """Hungarian 匹配後，d ≤ τ 的配對數 → (F1, matches, precision, recall)。"""
    n_gt, n_pr = dist_matrix.shape
    if n_gt == 0 and n_pr == 0:
        return 1.0, 0, 1.0, 1.0
    if n_gt == 0 or n_pr == 0:
        return 0.0, 0, 0.0, 0.0
    BIG = 1e6
    cost = np.where(dist_matrix <= tau, dist_matrix, BIG)
    ri, ci = linear_sum_assignment(cost)
    m = int((cost[ri, ci] < BIG).sum())
    prec = m / n_pr
    rec = m / n_gt
    f1 = 0.0 if m == 0 else 2 * prec * rec / (prec + rec)
    return f1, m, prec, rec


def f1_curve(dist_matrix: np.ndarray, taus: Sequence[float]) -> List[float]:
    return [f1_at_tau(dist_matrix, t)[0] for t in taus]
