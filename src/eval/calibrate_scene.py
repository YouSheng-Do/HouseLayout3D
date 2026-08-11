"""單場景校準：官方預測 vs HF GT → F1 對照 Table 2。"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from gt_loader import load_scene_gt
from metrics import f1_at_tau, f1_curve, pairwise_hausdorff, pairwise_rect
from pred_loader import load_scene_pred

PRED_ROOT = "/home/ado/storage/HouseLayout3D/docs/supplementary/supplementary/predictions-ours"
TAUS = np.round(np.arange(0.05, 1.01, 0.05), 2)


def run(scene: str):
    t0 = time.time()
    gt = load_scene_gt(scene, load_poses=False)
    pred = load_scene_pred(f"{PRED_ROOT}/{scene}/combined.ply", scene)
    print(f"[{scene}] GT: {len(gt.structure_entities)} structures, {len(gt.doors)} doors, "
          f"{len(gt.windows)} windows, {len(gt.stair_meshes)} stairs")
    print(f"[{scene}] Pred: {len(pred.structure_entities)} structures, {len(pred.doors)} doors, "
          f"{len(pred.windows)} windows  (載入 {time.time()-t0:.1f}s)")

    results = {}
    t0 = time.time()
    d_struct = pairwise_hausdorff(gt.structure_entities, pred.structure_entities)
    f1, m, p, r = f1_at_tau(d_struct, 0.5)
    results["structures"] = (d_struct, f1, m, p, r)
    print(f"Structures  F1@0.5 = {f1:.3f}  (match {m}, P {p:.3f}, R {r:.3f})  "
          f"[d_H 矩陣 {d_struct.shape}, {time.time()-t0:.1f}s]  ← Table2: 0.40")

    if gt.doors and pred.doors:
        d_door = pairwise_rect(gt.doors, pred.doors)
        f1, m, p, r = f1_at_tau(d_door, 0.5)
        results["doors"] = (d_door, f1, m, p, r)
        print(f"Doors       F1@0.5 = {f1:.3f}  (match {m}, P {p:.3f}, R {r:.3f})  ← Table2: 0.55")
    if gt.windows and pred.windows:
        d_win = pairwise_rect(gt.windows, pred.windows)
        f1, m, p, r = f1_at_tau(d_win, 0.5)
        results["windows"] = (d_win, f1, m, p, r)
        print(f"Windows     F1@0.5 = {f1:.3f}  (match {m}, P {p:.3f}, R {r:.3f})  ← Table2: 0.43")

    print("\nAvg-F1 候選（找出官方 Avg F1 的門檻集合；Table2 Avg: struct .38 / doors .44 / windows .38）")
    for name in ("structures", "doors", "windows"):
        if name not in results:
            continue
        d = results[name][0]
        curve = f1_curve(d, TAUS)
        avg_all = np.mean(curve)
        sub = [t for t in (0.25, 0.5, 0.75, 1.0)]
        avg_4 = np.mean([f1_at_tau(d, t)[0] for t in sub])
        avg_10 = np.mean([f1_at_tau(d, t)[0] for t in np.arange(0.1, 1.01, 0.1)])
        print(f"  {name:<11s} mean(τ=.05:.05:1)={avg_all:.3f} | mean(.25,.5,.75,1)={avg_4:.3f} | mean(.1:.1:1)={avg_10:.3f}")
        print(f"      F1 曲線: " + " ".join(f"{t:.2f}:{v:.2f}" for t, v in zip(TAUS[1::4], curve[1::4])))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="1LXtFkjw3qL")
    args = ap.parse_args()
    run(args.scene)
