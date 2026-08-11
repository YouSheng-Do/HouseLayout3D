"""Consolidated 2D comparison — one evaluator, one GT, four prediction sets.

Everything is (re-)scored with ``score_scene`` (eval2d_v3_strict_levels)
against the official ``mp3d_house_floor_v0_1`` GT (325 rooms / 32 levels):

  * watershed_v3      — our full pipeline (predicts levels itself); frozen
                        scores loaded from the existing official325 A/B run.
  * roomformer per_floor — RoomFormer stru3d semantic-rich, oracle .house
                        levels, re-scored here from its schema pred pkls.
  * roomformer per_room  — same checkpoint, oracle GT room crops (strongest
                        oracle), re-scored here.
  * CAGE zero-shot    — CAGE stru3d SwinV2-L, oracle .house levels; scores
                        loaded from the cage_stru3d_swinv2_zeroshot baseline.

Writes only into a new ``outputs/eval2d/comparison_strict_v3`` directory.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import EVAL2D_VERSION, prf, score_scene  # noqa: E402

ROOT = Path("/home/ado/storage/HouseLayout3D")
GT_DIR = ROOT / "outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
RF_DIR = ROOT / "outputs/eval2d/baselines/roomformer_hl3d_2d_v1"
WS_SCORES = GT_DIR / "eval_frozen_watershed_v3/scores_house_official325.pkl"
CAGE_SCORES = ROOT / "outputs/eval2d/baselines/cage_stru3d_swinv2_zeroshot/scores.pkl"
OUT = ROOT / "outputs/eval2d/comparison_strict_v3"

METRIC_ROWS = [
    ("A", "room", "Room F1 @IoU>0.5"),
    (None, "iou", "matched mean IoU (n)"),
    ("A", "corner@0.1", "Corner@0.1 F1"),
    ("A", "corner@0.2", "Corner@0.2 F1"),
    ("A", "corner@0.3", "Corner@0.3 F1"),
    ("A", "angle@0.1", "Angle@0.1 F1"),
    ("B", "room+type", "Room+type F1"),
    ("B", "doors@0.2", "Doors@0.2 F1"),
    ("B", "doors@0.5", "Doors@0.5 F1"),
    ("C", "room_room", "edge room-room F1"),
    ("C", "outside", "edge outside F1"),
    ("C", "all", "edge all F1"),
]

METHODS = [
    ("watershed_v3", "watershed_v3（ours, full pipeline）", "自行預測 levels；無 oracle"),
    ("rf_per_floor", "RoomFormer per-floor", "oracle `.house` levels＋region 點雲"),
    ("cage", "CAGE per-floor zero-shot", "oracle `.house` levels＋region 點雲"),
    ("rf_per_room", "RoomFormer per-room", "oracle GT room crops（最強 oracle）"),
]


def rescore(mode):
    scores = {}
    for pkl in sorted((RF_DIR / mode / "pred").glob("*.pkl")):
        scene = pkl.stem
        with open(GT_DIR / f"{scene}.pkl", "rb") as s:
            gt = pickle.load(s)
        with open(pkl, "rb") as s:
            pred = pickle.load(s)
        scores[scene] = score_scene(gt, pred)
    if len(scores) != 16:
        raise RuntimeError(f"{mode}: expected 16 scenes, got {len(scores)}")
    return scores


def agg(allres, tier, key):
    t = [0, 0, 0]
    for r in allres.values():
        v = r[tier].get(key, [0, 0, 0])
        t = [t[i] + v[i] for i in range(3)]
    return prf(*t)


def cell(allres, tier, key):
    if key == "iou":
        ious = [x for s in allres for x in allres[s]["iou"]]
        return f"{np.mean(ious):.3f} ({len(ious)})" if ious else "- (0)"
    m = agg(allres, tier, key)
    return f"{m['f1']:.3f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(WS_SCORES, "rb") as s:
        ws = pickle.load(s)
    with open(CAGE_SCORES, "rb") as s:
        cage = pickle.load(s)
    rf_floor = rescore("per_floor")
    rf_room = rescore("per_room")
    for name, scores in [("roomformer_per_floor", rf_floor), ("roomformer_per_room", rf_room)]:
        with open(OUT / f"scores_{name}.pkl", "wb") as s:
            pickle.dump(scores, s, protocol=4)

    by_key = {"watershed_v3": ws, "rf_per_floor": rf_floor,
              "cage": cage, "rf_per_room": rf_room}

    lines = [
        "# HouseLayout3D 16-scene — 2D 方法比較（單一協定）",
        "",
        f"> evaluator=`{EVAL2D_VERSION}`；GT=`mp3d_house_floor_v0_1`（official 325 rooms / 32 levels）。",
        "> RoomFormer 兩個 mode 由其 schema pred 重新以本 evaluator 評分（原 RESULTS_2D.md 是",
        "> `roomformer2d_eval_v1` 自家指標，數字不同屬正常）。所有 checkpoint 都是 Structured3D",
        "> 訓練、MP3D zero-shot。**輸入條件不同（見下），比較時必須連同條件一起引用。**",
        "",
        "| 方法 | 條件 |",
        "|---|---|",
    ]
    for key, label, cond in METHODS:
        lines.append(f"| {label} | {cond} |")
    lines += [
        "",
        "| 指標 | " + " | ".join(label for _, label, _ in METHODS) + " |",
        "|---|" + "---:|" * len(METHODS),
    ]
    for tier, key, label in METRIC_ROWS:
        row = [cell(by_key[k], tier, key) for k, _, _ in METHODS]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    lines += ["", "## 逐棟 Room F1", "",
              "| scene | " + " | ".join(label for _, label, _ in METHODS) + " |",
              "|---|" + "---:|" * len(METHODS)]
    for scene in sorted(ws):
        row = [f"{prf(*by_key[k][scene]['A']['room'])['f1']:.3f}" for k, _, _ in METHODS]
        lines.append(f"| {scene} | " + " | ".join(row) + " |")

    report = "\n".join(lines) + "\n"
    with open(OUT / "COMPARISON_2D.md", "w") as s:
        s.write(report)
    print(report)
    print(f"-> {OUT}/COMPARISON_2D.md")


if __name__ == "__main__":
    main()
