"""每棟診斷圖：pred layout（紅）vs GT（藍）三視圖，存 outputs/mp3d_reports/<scene>.png。
全量跑時每棟自動產一張，供逐棟目視（即使中間產物被清理）。"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from gt_loader import load_scene_gt


def render(scene: str, pred_dir: str, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = load_scene_gt(scene, load_poses=False)
    pred = o3d.io.read_triangle_mesh(f"{pred_dir}/combined.ply")

    def pts(m, n=120000):
        return np.asarray(m.sample_points_uniformly(n).points) if len(m.triangles) else np.asarray(m.vertices)
    gp, pp = pts(gt.structures_mesh), pts(pred)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (a, b, nm) in zip(axes, [(0, 1, "top XY"), (0, 2, "front XZ"), (1, 2, "side YZ")]):
        ax.scatter(gp[:, a], gp[:, b], s=0.4, c="tab:blue", linewidths=0, label="GT")
        ax.scatter(pp[:, a], pp[:, b], s=0.4, c="tab:red", alpha=0.5, linewidths=0, label="our pred")
        ax.set_title(f"{scene} — {nm}"); ax.set_aspect("equal"); ax.legend(markerscale=10)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=85, bbox_inches="tight"); plt.close()
    print(f"[report] {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    render(a.scene, a.pred_dir, a.out)
