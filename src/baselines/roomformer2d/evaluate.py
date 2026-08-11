#!/usr/bin/env python
"""Evaluate per-floor and per-room RoomFormer predictions in the 2D suite."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL2D = HERE.parents[1] / "eval2d"
sys.path.insert(0, str(EVAL2D))
from house_floor_gt import build_scene
from metrics import align_levels, match_doors, prf, score_scene

from common import ROOMFORMER_TYPE_NAMES, load_openings, scene_ids


def match_openings_by_level(pred_items: list, gt_items: list, gt_z: dict,
                            pred_z: dict, threshold: float) -> list:
    """Match openings and count predictions on GT-empty levels as FP."""
    gt_by_level = defaultdict(list)
    pred_by_level = defaultdict(list)
    for item in gt_items:
        gt_by_level[item["level"]].append(item)
    for item in pred_items:
        pred_by_level[item["level"]].append(item)
    level_map = align_levels(gt_z, pred_z)
    total = [0, 0, 0]
    # Iterate every GT level, not only levels containing a GT opening.
    for gt_level in gt_z:
        pred_level = level_map.get(gt_level)
        values = pred_by_level.get(pred_level, []) if pred_level is not None else []
        counts = match_doors(values, gt_by_level.get(gt_level, []), threshold)
        total = [a + b for a, b in zip(total, counts)]
    for pred_level in set(pred_by_level) - set(level_map.values()):
        total[1] += len(pred_by_level[pred_level])
    return total


def add_window_scores(result: dict, gt: dict, pred: dict) -> None:
    gt_windows = load_openings(gt["scene"], "windows", gt["lvl_z"])
    pred_windows = pred.get("windows", [])
    for threshold in (0.2, 0.5):
        result["B"][f"windows@{threshold}"] = match_openings_by_level(
            pred_windows, gt_windows, gt["lvl_z"], pred.get("level_z", {}), threshold)


def aggregate(results: dict, tier: str, name: str) -> dict:
    total = [0, 0, 0]
    for result in results.values():
        values = result[tier].get(name, [0, 0, 0])
        total = [a + b for a, b in zip(total, values)]
    return prf(*total)


def make_summary(mode: str, results: dict, run_manifest: dict) -> dict:
    names = {
        "room": ("A", "room"), "corner@0.2m": ("A", "corner@0.2"),
        "angle@0.2m": ("A", "angle@0.2"),
        "room+type": ("B", "room+type"),
        "doors@0.2m": ("B", "doors@0.2"),
        "doors@0.5m": ("B", "doors@0.5"),
        "windows@0.2m": ("B", "windows@0.2"),
        "windows@0.5m": ("B", "windows@0.5"),
        "topology": ("C", "all"),
    }
    metrics = {name: aggregate(results, *key) for name, key in names.items()}
    ious = [value for result in results.values() for value in result["iou"]]
    per_scene = []
    for scene, result in sorted(results.items()):
        room = prf(*result["A"]["room"])
        per_scene.append({
            "scene": scene, "room_p": room["p"], "room_r": room["r"],
            "room_f1": room["f1"],
            "matched_room_iou": float(np.mean(result["iou"])) if result["iou"] else None,
            "matched_rooms": len(result["iou"]),
        })
    return {
        "version": "roomformer2d_eval_v1", "mode": mode,
        "scene_count": len(results), "input_count": run_manifest["sample_count"],
        "inference_seconds": run_manifest["elapsed_seconds"],
        "checkpoint_sha256": run_manifest["checkpoint_sha256"],
        "metrics": metrics,
        "matched_room_iou_mean": float(np.mean(ious)) if ious else None,
        "matched_room_count": len(ious), "per_scene": per_scene,
    }


def markdown(combined: dict) -> str:
    lines = [
        "# RoomFormer × HouseLayout3D — 2D results",
        "",
        "Micro-averaged over the 16 local Matterport3D scenes. Room matches use "
        "IoU > 0.5; corner/angle distances are in metres.",
        "",
        "| mode | inputs | inference-loop sec* | Room P/R/F1 | matched IoU | Corner@.2 F1 | "
        "Room+type F1 | Door@.5 F1 | Window@.5 F1 | Topology F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("per_floor", "per_room"):
        s = combined[mode]
        m = s["metrics"]
        room = m["room"]
        lines.append(
            f"| {mode} | {s['input_count']} | {s['inference_seconds']:.1f} | "
            f"{room['p']:.3f}/{room['r']:.3f}/{room['f1']:.3f} | "
            f"{(s['matched_room_iou_mean'] or 0):.3f} | "
            f"{m['corner@0.2m']['f1']:.3f} | {m['room+type']['f1']:.3f} | "
            f"{m['doors@0.5m']['f1']:.3f} | {m['windows@0.5m']['f1']:.3f} | "
            f"{m['topology']['f1']:.3f} |")
    lines += [
        "", "## Per-scene Room F1", "",
        "| scene | per-floor | per-room |",
        "|---|---:|---:|",
    ]
    floor_rows = {r["scene"]: r for r in combined["per_floor"]["per_scene"]}
    room_rows = {r["scene"]: r for r in combined["per_room"]["per_scene"]}
    for scene in sorted(floor_rows):
        lines.append(f"| {scene} | {floor_rows[scene]['room_f1']:.3f} | "
                     f"{room_rows[scene]['room_f1']:.3f} |")
    lines += [
        "", "## Reproduction scope", "",
        "- GT level/region partition: official Matterport3D `.house` metadata.",
        "- Evaluated rooms: indoor non-stair regions (x, Z and s excluded).",
        "- Local GT totals: 32 levels, 325 rooms, 292 doors and 379 windows.",
        "- The paper reports 33 levels/317 rooms; the local MP3D metadata mismatch remains unresolved.",
        "- Input: points sampled uniformly by triangle area from `regionN.ply` surfaces.",
        "- Checkpoint and postprocessing are recorded in each run manifest.",
        "- Point count is a disclosed reproduction assumption because the paper does not specify it.",
        "- `inference-loop sec` excludes Python/CUDA/model/checkpoint startup.",
        "- These are project 2D diagnostic metrics, not HouseLayout3D Table 2's 3D metrics.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing per_floor/ and per_room/")
    args = parser.parse_args()
    combined = {}
    all_scenes = scene_ids()
    for mode in ("per_floor", "per_room"):
        mode_root = args.root / mode
        with (mode_root / "run_manifest.json").open() as stream:
            run_manifest = json.load(stream)
        results = {}
        for scene in all_scenes:
            gt = build_scene(scene)
            with (mode_root / "pred" / f"{scene}.pkl").open("rb") as stream:
                pred = pickle.load(stream)
            # Keep inference artifacts immutable while applying the documented
            # Structured3D -> MP3D semantic crosswalk during evaluation.
            for room in pred["rooms"]:
                room["type"] = ROOMFORMER_TYPE_NAMES[room["type_id"]]
            result = score_scene(gt, pred)
            add_window_scores(result, gt, pred)
            results[scene] = result
        with (mode_root / "scores.pkl").open("wb") as stream:
            pickle.dump(results, stream, protocol=4)
        summary = make_summary(mode, results, run_manifest)
        with (mode_root / "summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        combined[mode] = summary
    with (args.root / "summary.json").open("w") as stream:
        json.dump(combined, stream, indent=2)
    report = markdown(combined)
    (args.root / "RESULTS_2D.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
