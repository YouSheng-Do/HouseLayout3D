"""彙總 16 場景我方 pipeline 的 F1 與 Δτ → mean±std 對照論文 Table 2/3。"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from eval_scene import evaluate
from depth_metrics import scene_delta_tau
from gt_loader import load_scene_gt

TABLE2 = {"structures": (0.40, 0.38), "doors": (0.55, 0.44),
          "windows": (0.43, 0.38), "stairs": (0.42, 0.41)}
TABLE_DEPTH = {"d5": 61.1, "d10": 76.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, help="outputs/mp3d，含 <scene>/stage4/")
    ap.add_argument("--depth-stride", type=int, default=1)
    args = ap.parse_args()

    acc = {c: [] for c in TABLE2}
    d5s, d10s = [], []
    scenes = sorted(os.path.basename(os.path.dirname(p))
                    for p in glob.glob(f"{args.pred_root}/*/stage4"))
    if not scenes:
        print(f"找不到任何場景結果於 {args.pred_root}/*/stage4"); return

    print(f"{'scene':>12s} | Struct  Doors  Wind  Stair |  Δ5   Δ10")
    for s in scenes:
        pred_dir = f"{args.pred_root}/{s}/stage4"
        res = evaluate(s, pred_dir)
        for c in TABLE2:
            acc[c].append(res[c]["f1@0.5"])
        try:
            gt = load_scene_gt(s)
            pred = o3d.io.read_triangle_mesh(f"{pred_dir}/combined.ply")
            dr = scene_delta_tau(gt.structures_mesh, pred, gt.poses, stride=args.depth_stride)
            d5s.append(dr[5.0]); d10s.append(dr[10.0])
            dstr = f"{dr[5.0]:5.1f} {dr[10.0]:5.1f}"
        except Exception as e:
            dstr = f"(depth err: {type(e).__name__})"
        print(f"{s:>12s} | {res['structures']['f1@0.5']:.2f}   {res['doors']['f1@0.5']:.2f}  "
              f"{res['windows']['f1@0.5']:.2f}  {res['stairs']['f1@0.5']:.2f}  | {dstr}")

    print("\n===== 我方 pipeline vs 論文 Table 2/3 (mean±std) =====")
    for c in TABLE2:
        a = np.array(acc[c]); f1, avg = TABLE2[c]
        print(f"{c:<11s} F1@0.5 = {a.mean():.3f}±{a.std():.2f}   ← 論文 {f1:.2f}")
    if d5s:
        print(f"{'depth':<11s} Δ5 = {np.mean(d5s):.1f}±{np.std(d5s):.1f} "
              f"Δ10 = {np.mean(d10s):.1f}±{np.std(d10s):.1f}   ← 論文 61.1 / 76.3")


if __name__ == "__main__":
    main()
