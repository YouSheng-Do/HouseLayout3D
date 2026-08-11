"""16 場景完整深度校準（stride=1）。"""
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from depth_metrics import scene_delta_tau
from gt_loader import ALL_SCENES, load_scene_gt

PRED_ROOT = "/home/ado/storage/HouseLayout3D/docs/supplementary/supplementary/predictions-ours"

d5s, d10s = [], []
for scene in ALL_SCENES:
    t0 = time.time()
    gt = load_scene_gt(scene, load_poses=True)
    pred_mesh = o3d.io.read_triangle_mesh(f"{PRED_ROOT}/{scene}/combined.ply")
    res = scene_delta_tau(gt.structures_mesh, pred_mesh, gt.poses, stride=1)
    d5s.append(res[5.0]); d10s.append(res[10.0])
    print(f"{scene:>12s}  Δ5={res[5.0]:5.1f}  Δ10={res[10.0]:5.1f}  "
          f"({len(gt.poses['frames'])} frames, {time.time()-t0:.0f}s)", flush=True)

print(f"\nmean Δ5 = {np.mean(d5s):.1f}±{np.std(d5s):.1f}   ← Table2: 61.1±9.2")
print(f"mean Δ10 = {np.mean(d10s):.1f}±{np.std(d10s):.1f}  ← Table2: 76.3±7.9")
