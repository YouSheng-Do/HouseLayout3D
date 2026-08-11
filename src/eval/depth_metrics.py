"""深度一致性 Δτ（論文式 7）：以 GT poses 分別 render GT layout 與預測 layout 的深度，
計算 |D_pred - D_GT| ≤ τ cm 的像素比例。z-depth（相機座標 z），非 ray 距離。"""
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d


def _make_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def _z_factor(h: int, w: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """t_hit（ray 距離）→ z-depth 的每像素轉換因子（= 單位 ray 的 z 分量）。"""
    u = np.arange(w) + 0.5
    v = np.arange(h) + 0.5
    uu, vv = np.meshgrid(u, v)
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    return 1.0 / np.sqrt(dx * dx + dy * dy + 1.0)


def render_z(scene: o3d.t.geometry.RaycastingScene, w2c: np.ndarray, h: int, w: int,
             fx: float, fy: float, cx: float, cy: float, zfac: np.ndarray) -> np.ndarray:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(K), o3d.core.Tensor(w2c.astype(np.float64)), w, h)
    t = scene.cast_rays(rays)["t_hit"].numpy()
    return t * zfac  # inf 保持 inf


def scene_delta_tau(gt_mesh: o3d.geometry.TriangleMesh, pred_mesh: o3d.geometry.TriangleMesh,
                    poses: Dict, taus_cm: Tuple[float, ...] = (5.0, 10.0),
                    stride: int = 1, seg_dir: str = None, labels_path: str = None,
                    exclude_classes=("object", "window", "mirror", "outdoor")) -> Dict[float, float]:
    """Δτ（式 7）。seg_dir 給定時套論文協定：用 GT 語意 PNG 剔除 object/window 等像素
    （Table 3 的「use ground truth semantic annotations to ignore those 3D points ...
    as well as points on windows」），使 layout 深度不被家具/穿透面懲罰。"""
    import os
    from PIL import Image
    sg = _make_scene(gt_mesh)
    sp = _make_scene(pred_mesh)
    frames = poses["frames"][::stride]

    exclude_ids = None
    if seg_dir is not None and labels_path is not None:
        labels = [str(x) for x in np.load(labels_path)]
        exclude_ids = {labels.index(c) for c in exclude_classes if c in labels}

    f0 = frames[0]
    zfac_cache = {}
    hit_counts = {t: 0 for t in taus_cm}
    valid_total = 0
    for fr in frames:
        h, w = fr["h"], fr["w"]
        fx, fy, cx, cy = fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]
        key = (h, w, fx, fy, cx, cy)
        if key not in zfac_cache:
            zfac_cache[key] = _z_factor(h, w, fx, fy, cx, cy)
        zfac = zfac_cache[key]

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w[0:3, 1:3] *= -1  # OpenGL(nerfstudio) → OpenCV
        w2c = np.linalg.inv(c2w)

        zg = render_z(sg, w2c, h, w, fx, fy, cx, cy, zfac)
        zp = render_z(sp, w2c, h, w, fx, fy, cx, cy, zfac)

        valid = np.isfinite(zg)
        if exclude_ids is not None:
            stem = os.path.splitext(os.path.basename(fr["file_path"]))[0]
            seg_path = os.path.join(seg_dir, stem + ".png")
            if os.path.exists(seg_path):
                seg = np.array(Image.open(seg_path))
                if seg.shape != (h, w):
                    seg = np.array(Image.fromarray(seg).resize((w, h), Image.NEAREST))
                valid = valid & ~np.isin(seg, list(exclude_ids))
        n = int(valid.sum())
        if n == 0:
            continue
        valid_total += n
        diff = np.abs(np.where(np.isfinite(zp), zp, 1e9) - zg)[valid]
        for t in taus_cm:
            hit_counts[t] += int((diff <= t / 100.0).sum())
    return {t: 100.0 * hit_counts[t] / max(valid_total, 1) for t in taus_cm}


if __name__ == "__main__":
    sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
    from gt_loader import load_scene_gt
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="1LXtFkjw3qL")
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    gt = load_scene_gt(args.scene, load_poses=True)
    pred_mesh = o3d.io.read_triangle_mesh(
        f"/home/ado/storage/HouseLayout3D/docs/supplementary/supplementary/predictions-ours/{args.scene}/combined.ply")
    t0 = time.time()
    res = scene_delta_tau(gt.structures_mesh, pred_mesh, gt.poses, stride=args.stride)
    n_fr = len(gt.poses["frames"][::args.stride])
    print(f"[{args.scene}] stride={args.stride} ({n_fr} frames, {time.time()-t0:.0f}s): "
          f"Δ5={res[5.0]:.1f}  Δ10={res[10.0]:.1f}   ← Table2 平均: Δ5=61.1±9.2, Δ10=76.3±7.9")
