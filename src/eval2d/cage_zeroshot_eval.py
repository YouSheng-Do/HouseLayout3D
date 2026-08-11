"""CAGE zero-shot baseline -> eval2d_v3_strict_levels scoring.

Converts CAGE per-scene predictions (/home/ado/storage/CAGE/mp3d_inference/,
polygons already in world XY) into the eval2d prediction schema, then scores
them against the official ``mp3d_house_floor_v0_1`` GT with the frozen
``score_scene`` evaluator.  Writes to a NEW baseline directory only.

Disclosure (NOT comparable 1:1 with watershed_v3, which predicts levels itself):
  * ORACLE LEVELS — floors sliced via official ``.house`` region->level mapping.
  * ORACLE REGION MASK — density-map input uses region_segmentations vertices
    (excludes outdoor 'x' / junk 'Z' geometry).
  * Geometry only: no doors, no room types, no connectivity edges predicted.
"""
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import EVAL2D_VERSION, prf, score_scene  # noqa: E402

ROOT = "/home/ado/storage/HouseLayout3D"
CAGE_DIR = Path("/home/ado/storage/CAGE/mp3d_inference")
MP3D_SCANS = Path(f"{ROOT}/data/mp3d/v1/scans")
GT_DIR = Path(f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1")
OUT_DIR = Path(f"{ROOT}/outputs/eval2d/baselines/cage_stru3d_swinv2_zeroshot")

REGION_EXCLUDE = {"x", "Z"}

# frozen watershed_v3 headline (strict v3 + official325 GT) for the report table
WATERSHED = {
    "Room": (0.611, 0.594, 0.602), "iou": (0.785, 193),
    "Corner@0.1": 0.195, "Corner@0.2": 0.342, "Corner@0.3": 0.436,
    "Angle@0.1": 0.112, "Room+type": 0.144,
    "Doors@0.2": 0.190, "Doors@0.5": 0.243,
    "edge_room_room": 0.161, "edge_outside": 0.025, "edge_all": 0.131,
}


def house_level_z(scene):
    """{level: median region-bbox floor z} from the official .house R lines."""
    zs = defaultdict(list)
    path = MP3D_SCANS / scene / scene / "house_segmentations" / f"{scene}.house"
    with open(path, errors="ignore") as stream:
        for line in stream:
            t = line.split()
            if not t or t[0] != "R" or t[5] in REGION_EXCLUDE:
                continue
            zs[int(t[2])].append(float(t[11]))
    return {lvl: float(np.median(v)) for lvl, v in zs.items()}


def edges_to_poly(edges):
    """Ordered CAGE edges (K,4) -> (K,2) vertex loop: corner k = midpoint of
    edge k's start and edge k-1's end (adjacent endpoints are near-duplicates)."""
    starts = edges[:, 0:2]
    prev_ends = np.roll(edges[:, 2:4], 1, axis=0)
    return (starts + prev_ends) / 2.0


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def convert_scene(scene):
    with open(CAGE_DIR / scene / "predictions.pkl", "rb") as stream:
        cage = pickle.load(stream)
    level_z = house_level_z(scene)
    rooms, dropped = [], 0
    idx = 0
    for lv in cage["levels"]:
        for edges in lv["polygons_world"]:
            poly = edges_to_poly(np.asarray(edges, float))
            if len(poly) < 3 or shoelace(poly) <= 0:
                dropped += 1
                continue
            rooms.append({"idx": idx, "level": lv["level"], "poly": poly,
                          "type": "?", "area": shoelace(poly), "in_roomset": True})
            idx += 1
    pred = {
        "scene": scene, "rooms": rooms, "doors": [], "edges": [],
        "n_levels": len(cage["levels"]),
        "level_z": {lv["level"]: level_z[lv["level"]] for lv in cage["levels"]},
        "provenance": {
            "method": "CAGE zero-shot (Structured3D SwinV2-L official checkpoint)",
            "checkpoint": cage.get("checkpoint"),
            "level_source": "oracle_.house",
            "input": "region_segmentations vertices per level (x/Z excluded)",
        },
    }
    return pred, dropped


def agg(allres, tier, key):
    total = [0, 0, 0]
    for result in allres.values():
        v = result[tier].get(key, [0, 0, 0])
        total = [total[i] + v[i] for i in range(3)]
    return prf(*total)


def main():
    scenes = sorted(p.name for p in CAGE_DIR.iterdir()
                    if (p / "predictions.pkl").exists())
    if len(scenes) != 16:
        raise RuntimeError(f"expected 16 scenes, found {len(scenes)}")
    (OUT_DIR / "pred").mkdir(parents=True, exist_ok=True)

    allres = {}
    total_rooms = total_dropped = 0
    for scene in scenes:
        pred, dropped = convert_scene(scene)
        total_rooms += len(pred["rooms"]); total_dropped += dropped
        with open(OUT_DIR / "pred" / f"{scene}.pkl", "wb") as stream:
            pickle.dump(pred, stream, protocol=4)
        with open(GT_DIR / f"{scene}.pkl", "rb") as stream:
            gt = pickle.load(stream)
        allres[scene] = score_scene(gt, pred)
        diag = allres[scene]["level_diagnostics"]
        if diag["unmatched_gt"] or diag["unmatched_pred"]:
            print(f"  {scene} level diag: aligned={diag['aligned']} "
                  f"unmatched_gt={diag['unmatched_gt']} unmatched_pred={diag['unmatched_pred']}")

    with open(OUT_DIR / "scores.pkl", "wb") as stream:
        pickle.dump(allres, stream, protocol=4)

    room = agg(allres, "A", "room")
    ious = [x for s in allres for x in allres[s]["iou"]]
    lines = [
        "# CAGE zero-shot（Stru3D SwinV2-L 官方 checkpoint）— eval2d_v3_strict_levels",
        "",
        f"> evaluator=`{EVAL2D_VERSION}`；GT=`mp3d_house_floor_v0_1`（official325）。",
        "> **ORACLE LEVELS + ORACLE REGION MASK**（切層與輸入點雲用官方 `.house`／region ply），",
        "> watershed_v3 是自行預測 levels 的 full pipeline — 非同條件，只作參考對照。",
        "> CAGE 只出幾何：doors/types/edges 未預測，該三類指標按 0 分計入。",
        "",
        f"predicted rooms={total_rooms}（degenerate dropped={total_dropped}）",
        "",
        "| 指標 | CAGE zero-shot (oracle lv) | watershed_v3 (full pipeline) |",
        "|---|---:|---:|",
        (f"| Room P/R/F1 @IoU>0.5 | {room['p']:.3f}/{room['r']:.3f}/**{room['f1']:.3f}** "
         f"(TP/FP/FN={room['tp']}/{room['fp']}/{room['fn']}) | "
         f"{WATERSHED['Room'][0]:.3f}/{WATERSHED['Room'][1]:.3f}/**{WATERSHED['Room'][2]:.3f}** |"),
        (f"| matched mean IoU (n) | {np.mean(ious):.3f} ({len(ious)}) | "
         f"{WATERSHED['iou'][0]:.3f} ({WATERSHED['iou'][1]}) |"),
    ]
    for key, label in [("corner@0.1", "Corner@0.1"), ("corner@0.2", "Corner@0.2"),
                       ("corner@0.3", "Corner@0.3"), ("angle@0.1", "Angle@0.1")]:
        m = agg(allres, "A", key)
        lines.append(f"| {label} F1 | {m['f1']:.3f} | {WATERSHED[label]:.3f} |")
    m = agg(allres, "B", "room+type")
    lines.append(f"| Room+type F1 | {m['f1']:.3f} (no types) | {WATERSHED['Room+type']:.3f} |")
    for key, label in [("doors@0.2", "Doors@0.2"), ("doors@0.5", "Doors@0.5")]:
        m = agg(allres, "B", key)
        lines.append(f"| {label} F1 | {m['f1']:.3f} (no doors) | {WATERSHED[label]:.3f} |")
    for key, label in [("room_room", "edge_room_room"), ("outside", "edge_outside"),
                       ("all", "edge_all")]:
        m = agg(allres, "C", key)
        lines.append(f"| {label} F1 | {m['f1']:.3f} (no doors) | {WATERSHED[label]:.3f} |")

    lines += ["", "## 逐棟 Room F1", "", "| scene | Room F1 | matched IoU (n) |", "|---|---:|---:|"]
    for scene in scenes:
        f1 = prf(*allres[scene]["A"]["room"])["f1"]
        si = allres[scene]["iou"]
        lines.append(f"| {scene} | {f1:.3f} | "
                     f"{np.mean(si):.3f} ({len(si)}) |" if si else f"| {scene} | {f1:.3f} | - (0) |")

    report = "\n".join(lines) + "\n"
    with open(OUT_DIR / "RESULTS.md", "w") as stream:
        stream.write(report)
    print(report)
    print(f"scores -> {OUT_DIR}/scores.pkl")


if __name__ == "__main__":
    main()
