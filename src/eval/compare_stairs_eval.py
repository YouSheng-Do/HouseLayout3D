"""P0 stairs evaluator 修正對照：eval_v1_buggy vs eval_v2_nonrect，同一組凍結 predictions。
數字變化只來自 evaluator 修正（不重跑 pipeline）。含 regression 斷言（n_gt 必須全載）。"""
import glob, importlib, os, sys
import numpy as np
sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
import open3d as o3d
from gt_loader import load_scene_gt

v1 = importlib.import_module("eval_v1_buggy")
v2 = importlib.import_module("eval_scene")
ROOT = "/home/ado/storage/HouseLayout3D"

scenes = sorted(p.split("/outputs/mp3d/")[1].split("/")[0]
                for p in glob.glob(f"{ROOT}/outputs/mp3d/*/stage4/scene_graph.json"))

# GT 樓梯真實總數（供 regression）
gt_total = {}
for S in scenes:
    gt = load_scene_gt(S, load_poses=False)
    gt_total[S] = len(gt.stair_meshes)
GT_STAIRS_ALL = sum(gt_total.values())

print(f"{'scene':>12s} | {'GT實':>4s} | {'v1 ngt':>6s} {'v1 F1':>6s} | {'v2 ngt':>6s} {'v2 F1':>6s} {'v2 P/R':>9s}")
rows = []
for S in scenes:
    pd = f"{ROOT}/outputs/mp3d/{S}/stage4"
    r1 = v1.evaluate(S, pd, classes=("stairs",))["stairs"]
    r2 = v2.evaluate(S, pd, classes=("stairs",))["stairs"]
    rows.append((S, r1, r2))
    pr = f"{r2.get('P','-')}/{r2.get('R','-')}"
    print(f"{S:>12s} | {gt_total[S]:>4d} | {r1['n_gt']:>6d} {r1['f1@0.5']:>6.3f} | "
          f"{r2['n_gt']:>6d} {r2['f1@0.5']:>6.3f} {pr:>9s}")

# 均值（與 aggregate 同法：全 16 場景平均，含無樓梯場景=1.0）
m1 = np.mean([r1["f1@0.5"] for _, r1, _ in rows])
m2 = np.mean([r2["f1@0.5"] for _, _, r2 in rows])
print(f"\nStairs F1@0.5 均值(16 場景): v1_buggy {m1:.3f}  →  v2_nonrect {m2:.3f}   (論文 0.42)")
# 僅有樓梯的場景
has = [(S, r1, r2) for S, r1, r2 in rows if gt_total[S] > 0]
h1 = np.mean([r1["f1@0.5"] for _, r1, _ in has]); h2 = np.mean([r2["f1@0.5"] for _, _, r2 in has])
print(f"僅計 {len(has)} 有樓梯場景: v1 {h1:.3f} → v2 {h2:.3f}")

# ---- regression 斷言 ----
v1_ngt = sum(r1["n_gt"] for _, r1, _ in rows)
v2_ngt = sum(r2["n_gt"] for _, _, r2 in rows)
print(f"\n[regression] GT 樓梯總數={GT_STAIRS_ALL}  v1 載入={v1_ngt}(丟{GT_STAIRS_ALL-v1_ngt})  v2 載入={v2_ngt}")
assert GT_STAIRS_ALL == 34, f"GT 樓梯總數應 34，得 {GT_STAIRS_ALL}"
assert v2_ngt == GT_STAIRS_ALL, f"v2 必須全載 34，得 {v2_ngt}"
# 非矩形場景（e9zR 有 2 個 8 頂點）必須在 v2 n_gt 反映
er = next(r2 for S, _, r2 in rows if S == "e9zR4mvMWw7")
assert er["n_gt"] == gt_total["e9zR4mvMWw7"], "e9zR 非矩形樓梯未全載"
print("[regression] ✅ 全部通過：v2 載入全部 34 GT，非矩形場景無漏。")
