"""端到端自檢：把 GT 自己匯出成 pred 格式 → eval → 全類別 F1 應為 1.0。
驗證「原生實體匯出格式 ↔ eval 載入 ↔ 匹配」整條路無損。"""
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from eval_scene import evaluate
from gt_loader import load_scene_gt


def export_gt_as_pred(scene: str, out_dir: str):
    gt = load_scene_gt(scene, load_poses=False)
    os.makedirs(f"{out_dir}/entities", exist_ok=True)
    for i, m in enumerate(gt.structure_entities):
        o3d.io.write_triangle_mesh(f"{out_dir}/entities/structure_{i:04d}.ply", m)
    with open(f"{out_dir}/doors.json", "w") as f:
        json.dump({"doors": [{"vertices": r.tolist()} for r in gt.doors]}, f)
    with open(f"{out_dir}/windows.json", "w") as f:
        json.dump({"windows": [{"vertices": r.tolist()} for r in gt.windows]}, f)
    stairs = [np.asarray(m.vertices) for m in gt.stair_meshes if len(m.vertices) == 4]
    with open(f"{out_dir}/stairs.json", "w") as f:
        json.dump({"stairs": [{"vertices": r.tolist()} for r in stairs]}, f)


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "1LXtFkjw3qL"
    out = f"/home/ado/storage/HouseLayout3D/outputs/identity_test/{scene}"
    export_gt_as_pred(scene, out)
    res = evaluate(scene, out)
    ok = True
    for k, v in res.items():
        good = v["f1@0.5"] >= 0.999 and v["avg_f1"] >= 0.999
        ok &= good
        print(f"{k:<11s} {v}  {'✓' if good else '✗ 應為 1.0！'}")
    print("IDENTITY TEST", "PASS" if ok else "FAIL")
