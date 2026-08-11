"""2D eval suite — Task 4: RESULTS_2D.md ＋ 最佳/中/最差視覺化。"""
import glob, os, pickle, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ado/storage/HouseLayout3D"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import prf, align_levels  # noqa: E402

GT = f"{ROOT}/outputs/eval2d/gt"; PRED = f"{ROOT}/outputs/eval2d/pred"
scores = pickle.load(open(f"{ROOT}/outputs/eval2d/scores.pkl", "rb"))


def f1_of(res, key, sub):
    v = res[key].get(sub, [0, 0, 0]); return prf(*v)["f1"]


def draw(ax, floor, title, edges=None):
    """畫一層：房間多邊形（填色）＋門（紅段）＋access 邊（灰線，房心連線）。"""
    import matplotlib.cm as cm
    cents = {}
    for i, r in enumerate(floor["rooms"]):
        p = np.asarray(r["poly"])
        ax.fill(p[:, 0], p[:, 1], alpha=.45, color=cm.tab20(i % 20), lw=.5, ec="k")
        cents[r["idx"]] = p.mean(0)
    for d in floor["doors"]:
        s = np.asarray(d["seg"]); ax.plot(s[:, 0], s[:, 1], "-", c="crimson", lw=3, zorder=5)
    for e in (edges or []):
        a, b = e["rooms"]
        if e["kind"] == "room-room" and a in cents and b in cents:
            ca, cb = cents[a], cents[b]; ax.plot([ca[0], cb[0]], [ca[1], cb[1]], "-", c="0.35", lw=1, zorder=3)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.axis("off")


def viz(scene, tag):
    gt = pickle.load(open(f"{GT}/{scene}.pkl", "rb")); pr = pickle.load(open(f"{PRED}/{scene}.pkl", "rb"))
    # 取第一層（最主要）畫
    def one_level(data, lvl_key):
        lv = sorted(set(r["level"] for r in data["rooms"]))[0]
        return {"rooms": [r for r in data["rooms"] if r["level"] == lv],
                "doors": [d for d in data["doors"] if d["level"] == lv],
                "edges": [e for e in data.get("edges", []) if e.get("level") == lv]}
    g1, p1 = one_level(gt, "lvl_z"), one_level(pr, "level_z")
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    draw(ax[0], g1, f"GT — {scene} (L0)", g1["edges"]); draw(ax[1], p1, f"PRED — {scene} (L0)", p1["edges"])
    fig.suptitle(f"[{tag}] {scene}  Room-F1={f1_of(scores[scene],'A','room'):.2f}  IoU={np.mean(scores[scene]['iou'] or [0]):.2f}", fontsize=12)
    fig.tight_layout(); out = f"{ROOT}/outputs/eval2d/_viz_{tag}_{scene}.png"
    fig.savefig(out, dpi=110); plt.close(fig); return out


# 逐棟 Room F1 → best/median/worst
per = {S: f1_of(scores[S], "A", "room") for S in scores}
order = sorted(per, key=per.get)
worst, med, best = order[0], order[len(order)//2], order[-1]
vp = {t: viz(s, t) for t, s in [("best", best), ("median", med), ("worst", worst)]}


def agg(key, sub):
    t = [0, 0, 0]
    for S in scores:
        v = scores[S][key].get(sub, [0, 0, 0]); t = [t[i]+v[i] for i in range(3)]
    return prf(*t)


# 全域房型混淆（top 錯配）
conf = Counter()
for S in scores:
    conf.update(scores[S]["confusion"])

lines = []
W = lines.append
W("# RESULTS — 2D Floorplan Evaluation (HouseLayout3D, watershed v3)\n")
W("> 依 `docs/eval_2d_metric.md`。GT＝MP3D region（濾鏡：排 x/Z/stairs → 324 房≈論文 317）＋HL3D 門；")
W("> 連通性 **derived**（portals 全 0），用移植自 Structured3D 的 point-in-polygon 探針規則")
W("> （on-GT 內門準確率 78.9%，外門佔比 25%≈S3D 23%——見下）。預測＝Stage 4a **擠出前** floorplan。\n")

W("## Tier A — 幾何（可比對已發表工作）\n")
W("| 指標 | 值 |\n|---|---|")
m = agg("A", "room"); W(f"| **Room P/R/F1 @IoU>0.5** | {m['p']:.3f} / {m['r']:.3f} / **{m['f1']:.3f}** |")
alliou = [x for S in scores for x in scores[S]["iou"]]
W(f"| mean Room IoU (matched, n={len(alliou)}) | {np.mean(alliou):.3f} |")
for thr in (0.1, 0.2, 0.3):
    W(f"| Corner F1 @{thr}m / Angle F1 @{thr}m | {agg('A',f'corner@{thr}')['f1']:.3f} / {agg('A',f'angle@{thr}')['f1']:.3f} |")
rce = [x for S in scores for x in scores[S]["rc_err"]]
W(f"| 房數誤差/層 (pred−GT) | 均 {np.mean(rce):+.1f}（{sum(x>0 for x in rce)} 層過分割 / {sum(x<0 for x in rce)} 欠）|")
W("\n**判讀**：Room F1 0.61、IoU 0.79 → 房間層級大致正確；但 Corner/Angle 低 → 邊界**精度**差")
W("（watershed＋光柵→輪廓的鋸齒；且 GT 房多邊形亦自 region mesh 光柵化，非乾淨 CAD，會壓低 Corner/Angle，屬下界）。\n")

W("## Tier B — 語意\n")
W("| 指標 | 值 |\n|---|---|")
W(f"| Room+type F1 | {agg('B','room+type')['f1']:.3f} |")
for thr in (0.2, 0.5):
    d = agg("B", f"doors@{thr}"); W(f"| Doors P/R/F1 @{thr}m | {d['p']:.3f} / {d['r']:.3f} / {d['f1']:.3f} |")
W("\n**房型混淆（GT→pred, top 6）**：")
for (gt_t, pr_t), n in conf.most_common(6):
    W(f"- {gt_t} → {pr_t}: {n}")
W("→ 房型 F1 0.15：CLIP 房型近乎失效（多數房被判單一型），與 Stage 4a 觀察一致。\n")

W("## Tier C — 拓撲（連通；GT 為 derived，權重下調）\n")
W("| edge 集合 | P | R | F1 |\n|---|---|---|---|")
for k in ("room_room", "outside", "all"):
    m = agg("C", k); W(f"| {k} | {m['p']:.3f} | {m['r']:.3f} | {m['f1']:.3f} |")
W("\n**注**：GT-side 推導在 HL3D 真掃描上內門準 78.9%（非 S3D 的 98.2%，因 region tiling 不完美）。")
W("故 Tier C 是「預測 vs 79% 可信的 derived GT」，讀數要打折。\n")

W("## 逐棟分布（Room F1）\n")
W("| scene | Room F1 | IoU | Doors@0.5 F1 | edge_all F1 |\n|---|---|---|---|---|")
for S in sorted(scores, key=lambda s: -per[s]):
    W(f"| {S} | {per[S]:.2f} | {np.mean(scores[S]['iou'] or [0]):.2f} | "
      f"{f1_of(scores[S],'B','doors@0.5'):.2f} | {f1_of(scores[S],'C','all'):.2f} |")
good = sum(v >= 0.5 for v in per.values())
W(f"\n**分布**：{good}/16 棟 Room F1 ≥ 0.5。中位 {np.median(list(per.values())):.2f}，"
  f"範圍 {min(per.values()):.2f}（{worst}）–{max(per.values()):.2f}（{best}）。"
  f"（管線變異 ±6 Δ5 已知，逐棟差異須據此讀。）\n")

W("## 視覺化（best / median / worst，第一層）\n")
for t in ("best", "median", "worst"):
    W(f"- **{t}**：`{os.path.basename(vp[t])}`")
W("")

W("## 直接評估：floorplan 夠不夠好撐 attribute layer？\n")
W(f"- **房間層級（撐得住）**：Room F1 {agg('A','room')['f1']:.2f}、IoU 0.79——多數房間位置/範圍對，足以掛「房級」屬性（房型雖爛但幾何在）。")
W(f"- **邊界精度（撐不住）**：Corner@0.1m F1 {agg('A','corner@0.1')['f1']:.2f}——需精確角點/牆線的屬性（門寬、開向貼牆）**不可靠**。")
W(f"- **連通/導航（撐不住）**：edge_all F1 {agg('C','all')['f1']:.2f}——access graph 不足以直接當導航圖。")
W("- **房型（撐不住）**：0.15，需另做房型分類。\n")
W("**結論**：目前 floorplan 適合「房級、粗粒度」屬性；不足以支撐需要精確邊界或可靠連通的導航屬性。")
W("要往導航級，優先補：①邊界正則化（角點精度）②直接門偵測（連通 recall）③房型分類。\n")

open(f"{ROOT}/RESULTS_2D.md", "w").write("\n".join(lines))
print("寫出 RESULTS_2D.md")
print("viz:", {k: os.path.basename(v) for k, v in vp.items()})
print(f"\nbest={best}({per[best]:.2f}) median={med}({per[med]:.2f}) worst={worst}({per[worst]:.2f})")
