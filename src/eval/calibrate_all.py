"""16 場景全量校準：官方預測 vs HF GT → mean±std 對照 Table 2。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from gt_loader import ALL_SCENES, load_scene_gt
from metrics import f1_at_tau, pairwise_hausdorff, pairwise_rect
from pred_loader import load_scene_pred

PRED_ROOT = "/home/ado/storage/HouseLayout3D/docs/supplementary/supplementary/predictions-ours"

TAU_SETS = {
    "F1@0.5": [0.5],
    "avg(.05:.05:1)": np.round(np.arange(0.05, 1.01, 0.05), 2),
    "avg(.1:.1:1)": np.round(np.arange(0.1, 1.01, 0.1), 2),
    "avg(.25,.5,.75,1)": [0.25, 0.5, 0.75, 1.0],
}
TABLE2 = {"structures": (0.40, 0.38), "doors": (0.55, 0.44), "windows": (0.43, 0.38)}


def main():
    per_scene = {k: {s: [] for s in TAU_SETS} for k in ("structures", "doors", "windows")}
    vert_counts = []
    for scene in ALL_SCENES:
        t0 = time.time()
        gt = load_scene_gt(scene, load_poses=False)
        pred = load_scene_pred(f"{PRED_ROOT}/{scene}/combined.ply", scene)
        nv = sum(len(np.asarray(m.vertices)) for m in pred.structure_entities) \
            + 4 * len(pred.windows) + 8 * len(pred.doors)
        vert_counts.append(nv)

        dmats = {}
        dmats["structures"] = pairwise_hausdorff(gt.structure_entities, pred.structure_entities)
        dmats["doors"] = pairwise_rect(gt.doors, pred.doors) if gt.doors and pred.doors else None
        dmats["windows"] = pairwise_rect(gt.windows, pred.windows) if gt.windows and pred.windows else None

        line = [f"{scene:>12s}"]
        for cls in ("structures", "doors", "windows"):
            d = dmats[cls]
            for set_name, taus in TAU_SETS.items():
                if d is None:
                    v = 0.0 if (len(getattr(gt, cls)) if cls != "structures" else 1) else 1.0
                else:
                    v = float(np.mean([f1_at_tau(d, t)[0] for t in taus]))
                per_scene[cls][set_name].append(v)
            line.append(f"{cls[0].upper()}@.5={per_scene[cls]['F1@0.5'][-1]:.2f}")
        line.append(f"({time.time()-t0:.0f}s, ~verts {nv})")
        print("  ".join(line), flush=True)

    print("\n===== 彙總（mean±std over 16 scenes）vs Table 2 =====")
    for cls in ("structures", "doors", "windows"):
        t_f1, t_avg = TABLE2[cls]
        row = [f"{cls:<11s}"]
        for set_name in TAU_SETS:
            arr = np.array(per_scene[cls][set_name])
            row.append(f"{set_name}={arr.mean():.3f}±{arr.std():.2f}")
        print("  ".join(row) + f"   ← Table2: F1@0.5={t_f1:.2f}, Avg={t_avg:.2f}")
    print(f"\n~#Vertices(近似) mean = {np.mean(vert_counts):.0f}  ← Table2: 1957")


if __name__ == "__main__":
    main()
