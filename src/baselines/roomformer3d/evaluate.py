#!/usr/bin/env python
"""Evaluate lifted RoomFormer predictions with the calibrated 3D evaluator."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval"))
from eval_scene import evaluate
from gt_loader import ALL_SCENES


PAPER = {
    "per_floor": {
        "structures": {"f1@0.5": 0.24, "avg_f1": 0.22},
        "doors": {"f1@0.5": 0.23, "avg_f1": 0.20},
        "windows": {"f1@0.5": 0.07, "avg_f1": 0.07},
        "vertices": 764.9,
    },
    "per_room": {
        "structures": {"f1@0.5": 0.18, "avg_f1": 0.16},
        "doors": {"f1@0.5": 0.18, "avg_f1": 0.16},
        "windows": {"f1@0.5": 0.08, "avg_f1": 0.09},
        "vertices": 1134.5,
    },
}
PAPER_STD = {
    "per_floor": {
        "structures": {"f1@0.5": 0.06, "avg_f1": 0.06},
        "doors": {"f1@0.5": 0.10, "avg_f1": 0.09},
        "windows": {"f1@0.5": 0.06, "avg_f1": 0.04},
    },
    "per_room": {
        "structures": {"f1@0.5": 0.14, "avg_f1": 0.12},
        "doors": {"f1@0.5": 0.14, "avg_f1": 0.12},
        "windows": {"f1@0.5": 0.08, "avg_f1": 0.07},
    },
}


def summarize(mode: str, results: dict, elapsed: float) -> dict:
    classes = ("structures", "doors", "windows")
    metrics = {}
    for cls in classes:
        metrics[cls] = {}
        for key in ("f1@0.5", "avg_f1"):
            values = np.asarray([results[s][cls][key] for s in results])
            metrics[cls][key] = {
                "mean": float(values.mean()), "std": float(values.std()),
                "paper_mean": PAPER[mode][cls][key],
                "paper_std": PAPER_STD[mode][cls][key],
            }
    vertices = np.asarray([results[s]["vertices"] for s in results])
    return {
        "version": "roomformer3d_eval_v1", "mode": mode,
        "scene_count": len(results), "elapsed_seconds": elapsed,
        "metrics": metrics,
        "vertices": {"mean": float(vertices.mean()), "std": float(vertices.std()),
                     "paper_mean": PAPER[mode]["vertices"]},
        "stairs": "not predicted by RoomFormer; paper reports dash",
        "per_scene": results,
    }


def report(combined: dict, depth: dict = None) -> str:
    lines = [
        "# RoomFormer × HouseLayout3D — 3D results", "",
        "Macro mean±population-std across 16 scenes, using the calibrated "
        "HouseLayout3D entity evaluator. Values in parentheses are Table 2.", "",
        "| mode | Structures F1@.5 | Structures Avg | Doors F1@.5 | Doors Avg | "
        "Windows F1@.5 | Windows Avg | vertices/scene |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("per_floor", "per_room"):
        s = combined[mode]
        m = s["metrics"]
        def cell(cls, key):
            x = m[cls][key]
            return (f"{x['mean']:.3f}±{x['std']:.3f} "
                    f"({x['paper_mean']:.2f}±{x['paper_std']:.2f})")
        v = s["vertices"]
        lines.append(
            f"| {mode} | {cell('structures', 'f1@0.5')} | "
            f"{cell('structures', 'avg_f1')} | {cell('doors', 'f1@0.5')} | "
            f"{cell('doors', 'avg_f1')} | {cell('windows', 'f1@0.5')} | "
            f"{cell('windows', 'avg_f1')} | {v['mean']:.1f} ({v['paper_mean']:.1f}) |")
    if depth is not None:
        lines += [
            "", "## Layout depth", "",
            "| mode | Δ5 | Δ10 |",
            "|---|---:|---:|",
        ]
        for mode in ("per_floor", "per_room"):
            values = depth["modes"][mode]
            def dcell(key):
                x = values[key]
                return (f"{x['mean']:.1f}±{x['std']:.1f} "
                        f"({x['paper_mean']:.1f}±{x['paper_std']:.1f})")
            lines.append(f"| {mode} | {dcell('delta_5')} | {dcell('delta_10')} |")
        lines.append(f"\nDepth was rendered at frame stride {depth['stride']}.")
        lines += [
            "", "## Per-scene layout depth", "",
            "| scene | floor Δ5 | floor Δ10 | room Δ5 | room Δ10 |",
            "|---|---:|---:|---:|---:|",
        ]
        for scene in sorted(depth["per_scene"]):
            scores = depth["per_scene"][scene]["scores"]
            lines.append(
                f"| {scene} | {scores['per_floor']['delta_5']:.1f} | "
                f"{scores['per_floor']['delta_10']:.1f} | "
                f"{scores['per_room']['delta_5']:.1f} | "
                f"{scores['per_room']['delta_10']:.1f} |")
    lines += ["", "## Per-scene F1@0.5", "",
              "| scene | floor struct | room struct | floor door | room door | floor window | room window |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for scene in sorted(combined["per_floor"]["per_scene"]):
        f = combined["per_floor"]["per_scene"][scene]
        r = combined["per_room"]["per_scene"][scene]
        lines.append(f"| {scene} | {f['structures']['f1@0.5']:.3f} | "
                     f"{r['structures']['f1@0.5']:.3f} | "
                     f"{f['doors']['f1@0.5']:.3f} | {r['doors']['f1@0.5']:.3f} | "
                     f"{f['windows']['f1@0.5']:.3f} | {r['windows']['f1@0.5']:.3f} |")
    lines += [
        "", "## Protocol", "",
        "- Structure entities are the lifted floor, ceiling and wall polygons.",
        "- Doors/windows use the paper's d_E; structures use generalized d_H; threshold is 0.5 m.",
        "- Avg F1 is averaged over τ=0.05,0.10,…,1.00 (local calibrated working assumption).",
        "- RoomFormer predicts no stairs, matching the dash in Table 2.",
        "- Depth Δ5/Δ10 renders the GT and prediction layouts from the supplied camera poses.",
        "- Local MP3D partition is 32 levels/325 rooms versus the paper's 33/317.",
        "- Point-sampling count is a disclosed reproduction assumption; lifting itself follows Appendix E.1.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()
    scenes = args.scenes or sorted(ALL_SCENES)
    combined = {}
    for mode in ("per_floor", "per_room"):
        started = time.time()
        results = {}
        for index, scene in enumerate(scenes, 1):
            pred_dir = args.pred_root / mode / scene
            values = evaluate(scene, str(pred_dir),
                              classes=("structures", "doors", "windows"))
            mesh = o3d.io.read_triangle_mesh(str(pred_dir / "combined.ply"))
            values["vertices"] = len(mesh.vertices)
            results[scene] = values
            print(f"{mode} [{index}/{len(scenes)}] {scene}: "
                  f"S={values['structures']['f1@0.5']:.3f} "
                  f"D={values['doors']['f1@0.5']:.3f} "
                  f"W={values['windows']['f1@0.5']:.3f}", flush=True)
        summary = summarize(mode, results, time.time() - started)
        with (args.pred_root / mode / "scores.json").open("w") as stream:
            json.dump(results, stream, indent=2)
        with (args.pred_root / mode / "summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        combined[mode] = summary
    with (args.pred_root / "summary.json").open("w") as stream:
        json.dump(combined, stream, indent=2)
    depth_path = args.pred_root / "depth_summary.json"
    depth = None
    if depth_path.exists():
        with depth_path.open() as stream:
            depth = json.load(stream)
        combined["depth"] = depth
        with (args.pred_root / "summary.json").open("w") as stream:
            json.dump(combined, stream, indent=2)
    text = report(combined, depth)
    (args.pred_root / "RESULTS_3D.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
