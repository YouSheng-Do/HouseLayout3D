"""[FROZEN eval_v1_buggy — 2026-08-11] 保留供追溯：stairs 用 len(v)==4 靜默丟 5 個非矩形 GT
(只評 29/34，灌高 F1)。勿改；新評估用 eval_scene.py(v2_nonrect)。"""
"""單場景評估入口：pipeline 原生實體輸出 vs HF GT → F1@0.5／Avg F1／（可選）Δτ。

pred 目錄格式（鏡射 HF GT 格式，由 stage4 匯出）：
  entities/structure_*.ply   平面實體 mesh（牆/地板/天花板，不分類，Table 2 合併評估）
  doors.json / windows.json  {"doors"|"windows": [{"vertices": [[x,y,z]×4]}...]}
  stairs.json                {"stairs": [{"vertices": [[x,y,z]×4]}...]}（矩形，走 d_E）
  combined.ply（可選，深度評估用）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from gt_loader import load_scene_gt
from metrics import f1_at_tau, pairwise_hausdorff, pairwise_rect

TAUS = np.round(np.arange(0.05, 1.01, 0.05), 2)  # Avg F1 工作假設


def load_pred_entities(pred_dir: str):
    structs = []
    for f in sorted(glob.glob(f"{pred_dir}/entities/structure_*.ply")):
        m = o3d.io.read_triangle_mesh(f)
        if len(m.vertices) >= 3 and len(m.triangles) >= 1:
            structs.append(m)
    def rects(name, key):
        p = f"{pred_dir}/{name}.json"
        if not os.path.exists(p):
            return []
        with open(p) as fh:
            data = json.load(fh)
        return [np.asarray(e["vertices"], dtype=np.float64) for e in data.get(key, [])]
    return structs, rects("doors", "doors"), rects("windows", "windows"), rects("stairs", "stairs")


def evaluate(scene: str, pred_dir: str, classes=("structures", "doors", "windows", "stairs")):
    gt = load_scene_gt(scene, load_poses=False)
    ps, pd, pw, pst = load_pred_entities(pred_dir)
    # GT 樓梯（4 頂點矩形 ply → 角點）
    gt_stairs = []
    for m in gt.stair_meshes:
        v = np.asarray(m.vertices)
        if len(v) == 4:
            gt_stairs.append(v)
    out = {}
    pairs = {
        "structures": (gt.structure_entities, ps, pairwise_hausdorff),
        "doors": (gt.doors, pd, pairwise_rect),
        "windows": (gt.windows, pw, pairwise_rect),
        "stairs": (gt_stairs, pst, pairwise_rect),
    }
    for cls in classes:
        g, p, fn = pairs[cls]
        if len(g) == 0 and len(p) == 0:
            out[cls] = {"f1@0.5": 1.0, "avg_f1": 1.0, "n_gt": 0, "n_pred": 0}
            continue
        if len(g) == 0 or len(p) == 0:
            out[cls] = {"f1@0.5": 0.0, "avg_f1": 0.0, "n_gt": len(g), "n_pred": len(p)}
            continue
        d = fn(g, p)
        f1, m, prec, rec = f1_at_tau(d, 0.5)
        avg = float(np.mean([f1_at_tau(d, t)[0] for t in TAUS]))
        out[cls] = {"f1@0.5": round(f1, 3), "avg_f1": round(avg, 3),
                    "n_gt": len(g), "n_pred": len(p), "P": round(prec, 3), "R": round(rec, 3)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pred-dir", required=True)
    a = ap.parse_args()
    res = evaluate(a.scene, a.pred_dir)
    for k, v in res.items():
        print(f"{k:<11s} {v}")
