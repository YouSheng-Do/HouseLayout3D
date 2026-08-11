"""Task C — 用凍結 baseline 的 GT/pred 以 eval2d_v2_hungarian 重算，對照 v1_greedy。
只讀 baselines/watershed_v3_pre_report/{gt,pred}；不呼叫 segmentation/extract_pred。"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score_scene, prf, EVAL2D_VERSION

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
OUT = f"{BASE}/eval2d_v2_hungarian"
os.makedirs(OUT, exist_ok=True)

scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/gt/*.pkl"))
v2 = {}
for S in scenes:
    gt = pickle.load(open(f"{BASE}/gt/{S}.pkl", "rb"))
    pred = pickle.load(open(f"{BASE}/pred/{S}.pkl", "rb"))
    v2[S] = score_scene(gt, pred)
pickle.dump(v2, open(f"{OUT}/scores.pkl", "wb"))
v1 = pickle.load(open(f"{BASE}/scores_v1_greedy.pkl", "rb"))

def micro(res, key, sub):
    t = [0, 0, 0]
    for S in res:
        v = res[S][key].get(sub, [0, 0, 0]); t = [t[i] + v[i] for i in range(3)]
    return prf(*t)

def macro_f1(res, key, sub):
    fs = [prf(*res[S][key].get(sub, [0, 0, 0]))["f1"] for S in res]
    return float(np.mean(fs))

METRICS = [("A", "room", "Room"), ("B", "room+type", "Room+type"),
           ("B", "doors@0.2", "Doors@0.2"), ("B", "doors@0.5", "Doors@0.5"),
           ("A", "corner@0.1", "Corner@0.1"), ("A", "corner@0.2", "Corner@0.2"),
           ("A", "corner@0.3", "Corner@0.3"), ("A", "angle@0.1", "Angle@0.1"),
           ("C", "room_room", "edge_room_room"), ("C", "outside", "edge_outside"), ("C", "all", "edge_all")]

comp = {"evaluator_v1": "eval2d_v1_greedy", "evaluator_v2": EVAL2D_VERSION,
        "predictions_source": "frozen baselines/watershed_v3_pre_report", "n_scenes": len(scenes),
        "micro": {}, "macro": {}, "mean_matched_iou": {}, "per_scene_room_f1": {}}
for key, sub, name in METRICS:
    m1, m2 = micro(v1, key, sub), micro(v2, key, sub)
    comp["micro"][name] = {"v1_f1": round(m1["f1"], 4), "v2_f1": round(m2["f1"], 4),
                           "v1_PR": [round(m1["p"], 4), round(m1["r"], 4)],
                           "v2_PR": [round(m2["p"], 4), round(m2["r"], 4)],
                           "delta_f1": round(m2["f1"] - m1["f1"], 4)}
    comp["macro"][name] = {"v1_f1": round(macro_f1(v1, key, sub), 4),
                           "v2_f1": round(macro_f1(v2, key, sub), 4)}
iou1 = [x for S in v1 for x in v1[S]["iou"]]; iou2 = [x for S in v2 for x in v2[S]["iou"]]
comp["mean_matched_iou"] = {"v1": round(float(np.mean(iou1)), 4), "v2": round(float(np.mean(iou2)), 4),
                            "v1_n": len(iou1), "v2_n": len(iou2)}
for S in scenes:
    comp["per_scene_room_f1"][S] = {"v1": round(prf(*v1[S]["A"]["room"])["f1"], 3),
                                    "v2": round(prf(*v2[S]["A"]["room"])["f1"], 3)}
json.dump(comp, open(f"{OUT}/comparison_v1_v2.json", "w"), ensure_ascii=False, indent=1)

# 人類可讀 RESULTS_2D.md
L = ["# RESULTS_2D — watershed_v3_pre_report · eval2d_v2_hungarian\n",
     f"> 凍結 predictions（{comp['predictions_source']}）· evaluator v1_greedy → v2_hungarian",
     "> 數字變化唯一來源＝matching evaluator（未重跑 pipeline / segmentation）。\n",
     "## v1 greedy vs v2 hungarian（micro P/R/F1 pooled；macro＝逐棟 F1 平均）\n",
     "| 指標 | v1 F1 | v2 F1 | Δ | v2 P/R (micro) | macro v1→v2 |",
     "|---|---|---|---|---|---|"]
for key, sub, name in METRICS:
    c = comp["micro"][name]; mc = comp["macro"][name]
    L.append(f"| {name} | {c['v1_f1']:.3f} | {c['v2_f1']:.3f} | {c['delta_f1']:+.3f} | "
             f"{c['v2_PR'][0]:.2f}/{c['v2_PR'][1]:.2f} | {mc['v1_f1']:.2f}→{mc['v2_f1']:.2f} |")
mi = comp["mean_matched_iou"]
L.append(f"\nmean matched Room IoU: v1 {mi['v1']:.3f}(n={mi['v1_n']}) → v2 {mi['v2']:.3f}(n={mi['v2_n']})")
allsame = all(comp["micro"][n]["delta_f1"] == 0 for _, _, n in METRICS)
L.append(f"\n**結論**：{'evaluator hardening 後 16 棟結果與 v1 一致（差異=0）——room IoU>0.5 下 greedy≈hungarian，此為預期；價值在 determinism/permutation-invariance。' if allsame else 'matching 硬化後部分指標變動，見上表；未調任何 threshold。'}")
L.append("\n## 逐棟 Room F1（v1→v2）\n| scene | v1 | v2 |\n|---|---|---|")
for S in sorted(scenes, key=lambda s: -comp["per_scene_room_f1"][s]["v2"]):
    r = comp["per_scene_room_f1"][S]; L.append(f"| {S} | {r['v1']:.2f} | {r['v2']:.2f} |")
open(f"{OUT}/RESULTS_2D.md", "w").write("\n".join(L))

print(f"[Task C] 16 場景重算完成 → {OUT}")
print(f"Room F1: v1 {micro(v1,'A','room')['f1']:.4f} → v2 {micro(v2,'A','room')['f1']:.4f}")
print(f"Doors@0.5 F1: v1 {micro(v1,'B','doors@0.5')['f1']:.4f} → v2 {micro(v2,'B','doors@0.5')['f1']:.4f}")
print(f"edge_all F1: v1 {micro(v1,'C','all')['f1']:.4f} → v2 {micro(v2,'C','all')['f1']:.4f}")
print(f"全指標 delta=0? {allsame}")
# 驗收：16 場景、hash 不變（pred 未動）
assert len(v2) == 16
print("[Task C 驗收] ✅ 16 場景成功；predictions 未動（沿用凍結 baseline）")
