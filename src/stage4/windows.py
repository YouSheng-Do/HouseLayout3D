"""Stage 4 窗偵測（論文 §4.4＋Appendix B）：
window(+window_blind+curtain+outdoor) 類 rays → 與 layout 牆面求交 → LOF 濾離群 →
按牆實例分組 → DBSCAN → ≥k=10 的群擬合牆平面內 AABB 矩形 → 高寬 >30cm 判窗。
"""
import sys
from collections import Counter
from typing import List, Sequence

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

K_MIN = 10          # 正文
MIN_SIZE = 0.30     # 正文 30cm
DBSCAN_EPS = 0.15   # TODO: tune with MP3D
DBSCAN_MIN = 5      # TODO: tune with MP3D
LEGACY_WINDOW_RAY_CLASSES = ("window", "window_blind", "curtain")
WINDOW_RAY_CLASSES = (*LEGACY_WINDOW_RAY_CLASSES, "outdoor")  # Appendix B


def _record_from_cluster(points, wall, wall_index, ray_names,
                         included_ray_classes, cluster_id):
    """Fit one deterministic wall-plane rectangle and its 2D annotation."""
    points = np.asarray(points, dtype=np.float64)
    normal = wall.plane[:3]
    horizontal = np.cross(normal, [0, 0, 1.0])
    if np.linalg.norm(horizontal) < 1e-6:
        return None
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    along = points @ horizontal
    height = points @ vertical
    width_m = float(along.max() - along.min())
    height_m = float(height.max() - height.min())
    if width_m < MIN_SIZE or height_m < MIN_SIZE:
        return None
    centre = points.mean(axis=0)
    base = (centre - (centre @ horizontal) * horizontal
            - (centre @ vertical) * vertical)
    corners = np.array([
        base + a * horizontal + b * vertical
        for a, b in ((along.min(), height.min()),
                     (along.min(), height.max()),
                     (along.max(), height.max()),
                     (along.max(), height.min()))
    ])
    class_counts = dict(sorted(Counter(map(str, ray_names)).items()))
    return {
        "corners": corners,
        "segment_2d": corners[[0, 3], :2],
        "wall_index": int(wall_index),
        "wall_pid": int(wall.pid),
        "width_m": width_m,
        "height_m": height_m,
        "ray_count": int(len(points)),
        "ray_class_counts": class_counts,
        "included_ray_classes": list(map(str, included_ray_classes)),
        "cluster_id": int(cluster_id),
        "confidence": float(min(1.0, len(points) / 50.0)),
        "provenance": "paper_window_raycast_appendix_b_local_clustering",
    }


def detect_window_records(
        proto, stage2_dir: str,
        ray_classes: Sequence[str] = WINDOW_RAY_CLASSES,
        max_ray_len_slack: float = 0.3) -> List[dict]:
    labels = [str(x) for x in np.load(f"{stage2_dir}/simplified_segmentation_labels.npy")]
    ray_cls = np.load(f"{stage2_dir}/hard_labels_simplified_segmentations.npy")
    origins = np.load(f"{stage2_dir}/full_ray_origins.npy").astype(np.float64)
    dests = np.load(f"{stage2_dir}/full_ray_dests.npy").astype(np.float64)
    valid = np.load(f"{stage2_dir}/ray_is_valid.npy")

    win_ids = [labels.index(c) for c in ray_classes if c in labels]
    id_to_name = {labels.index(c): c for c in ray_classes if c in labels}
    sel = np.isin(ray_cls, win_ids) & valid
    o, e, selected_classes = origins[sel], dests[sel], ray_cls[sel]
    if len(o) == 0:
        return []

    # 牆 submesh（帶 polygon id 對映）
    walls = proto.by_class("wall")
    V = np.asarray(proto.mesh.vertices)
    T = np.asarray(proto.mesh.triangles)
    tri_list, tri2wall = [], []
    for wi, w in enumerate(walls):
        tri_list.append(T[w.tri_ix])
        tri2wall.extend([wi] * len(w.tri_ix))
    if not tri_list:
        return []
    wall_T = np.concatenate(tri_list)
    tri2wall = np.array(tri2wall)
    wm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                   o3d.utility.Vector3iVector(wall_T))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wm))

    d = e - o
    L = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(L, 1e-9)
    rays = o3d.core.Tensor(np.concatenate([o, d], axis=1), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    prim = ans["primitive_ids"].numpy()
    ok = np.isfinite(t_hit) & (t_hit <= L[:, 0] + max_ray_len_slack)
    pts = o[ok] + d[ok] * t_hit[ok, None]
    wall_of = tri2wall[prim[ok].astype(int)]
    hit_classes = selected_classes[ok]
    if len(pts) < K_MIN:
        return []

    # LOF 濾離群（全域）
    if len(pts) > 50:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(pts) - 1)).fit_predict(pts)
        keep = lof == 1
        pts, wall_of, hit_classes = pts[keep], wall_of[keep], hit_classes[keep]

    records = []
    for wi in np.unique(wall_of):
        P = pts[wall_of == wi]
        if len(P) < K_MIN:
            continue
        cl = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(P)
        for cid in set(cl) - {-1}:
            Q = P[cl == cid]
            if len(Q) < K_MIN:
                continue
            names = [id_to_name[int(value)]
                     for value in hit_classes[wall_of == wi][cl == cid]]
            record = _record_from_cluster(
                Q, walls[wi], wi, names, ray_classes, cid)
            if record is not None:
                records.append(record)
    records.sort(key=lambda record: (
        record["wall_pid"],
        tuple(np.round(np.asarray(record["segment_2d"]).mean(axis=0), 9)),
        record["cluster_id"],
    ))
    return records


def detect_windows(proto, stage2_dir: str, max_ray_len_slack: float = 0.3,
                   ray_classes: Sequence[str] = WINDOW_RAY_CLASSES) -> List[np.ndarray]:
    """Backward-compatible 3D rectangle view used by Stage 4 extrusion."""
    return [record["corners"] for record in detect_window_records(
        proto, stage2_dir, ray_classes=ray_classes,
        max_ray_len_slack=max_ray_len_slack)]


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/stage4")
    from load_prototype import load_prototype
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stage2-dir", required=True)
    a = ap.parse_args()
    proto = load_prototype(a.fitted, a.skeleton, a.probs, a.labels)
    rects = detect_windows(proto, a.stage2_dir)
    print(f"偵測到 {len(rects)} 扇窗")
    for r in rects[:10]:
        e = [np.linalg.norm(r[(i + 1) % 4] - r[i]) for i in range(4)]
        print(f"  中心 {np.round(r.mean(0), 2)}, 尺寸 {sorted(set(np.round(e, 2)))} m")
