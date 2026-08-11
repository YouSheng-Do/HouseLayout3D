"""Artifact-only 2D window evaluator using released HouseLayout3D rectangles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from metrics import align_levels, match_segments_endpoint, prf  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
WINDOW_GT_DIR = ROOT / "external" / "houselayout3d" / "data" / "windows"
EVAL_VERSION = "eval2d_windows_v0_2_endpoint_hungarian"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory):
    path = Path(directory) / "collection_manifest.json"
    manifest = json.load(path.open())
    failures = [relative for relative, expected in manifest["files_sha256"].items()
                if not (Path(directory) / relative).exists()
                or sha256(Path(directory) / relative) != expected]
    if failures:
        raise RuntimeError(f"prediction hash failure: {failures}")
    return manifest, path


def official_windows(scene, level_z):
    path = WINDOW_GT_DIR / f"{scene}.json"
    payload = json.load(path.open())
    result = []
    for index, item in enumerate(payload["windows"]):
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        order = np.argsort(vertices[:, 2], kind="stable")
        bottom = vertices[order[:2]]
        bottom_z = float(bottom[:, 2].mean())
        level = min(level_z, key=lambda candidate: (
            abs(float(level_z[candidate]) - bottom_z), str(candidate)))
        result.append({
            "idx": index,
            "level": level,
            "seg": bottom[:, :2],
            "normal": item.get("normal"),
            "source": "released_houselayout3d_window_rectangle_bottom_edge",
        })
    return result


def canonical_windows(artifact):
    level_z, windows = {}, []
    for level in artifact["levels"]:
        level_z[level["id"]] = float(level["elevation"])
        for window in level["windows"]:
            windows.append({
                "idx": window["id"],
                "level": level["id"],
                "seg": np.asarray(window["segment"], dtype=np.float64),
            })
    return level_z, windows


def score_scene(gt_level_z, gt_windows, pred_level_z, pred_windows):
    mapping = align_levels(gt_level_z, pred_level_z)
    gt_by_level = defaultdict(list)
    pred_by_level = defaultdict(list)
    for window in gt_windows:
        gt_by_level[window["level"]].append(window)
    for window in pred_windows:
        pred_by_level[window["level"]].append(window)

    counts = {threshold: [0, 0, 0] for threshold in (0.2, 0.5)}
    per_level = []
    for gt_level in sorted(gt_level_z, key=str):
        pred_level = mapping.get(gt_level)
        gt_items = gt_by_level.get(gt_level, [])
        pred_items = (pred_by_level.get(pred_level, [])
                      if pred_level is not None else [])
        row = {
            "gt_level": gt_level, "pred_level": pred_level,
            "gt_windows": len(gt_items), "pred_windows": len(pred_items),
        }
        for threshold in counts:
            value = match_segments_endpoint(pred_items, gt_items, threshold)
            counts[threshold] = [counts[threshold][i] + value[i]
                                 for i in range(3)]
            row[f"counts@{threshold}"] = list(value)
        per_level.append(row)

    unmatched_pred = sorted(set(pred_by_level) - set(mapping.values()), key=str)
    for pred_level in unmatched_pred:
        n = len(pred_by_level[pred_level])
        for threshold in counts:
            counts[threshold][1] += n
        per_level.append({
            "gt_level": None, "pred_level": pred_level,
            "gt_windows": 0, "pred_windows": n,
            "counts@0.2": [0, n, 0], "counts@0.5": [0, n, 0],
        })
    return {
        "counts": {str(threshold): value
                   for threshold, value in counts.items()},
        "level_alignment": {str(key): value for key, value in mapping.items()},
        "unmatched_pred_levels": unmatched_pred,
        "per_level": per_level,
    }


def aggregate(scores, scenes):
    metrics = {}
    for threshold in ("0.2", "0.5"):
        counts = [sum(scores[scene]["counts"][threshold][index]
                      for scene in scenes) for index in range(3)]
        metrics[f"windows@{threshold}"] = prf(*counts)
    return {
        "n_scenes": len(scenes),
        "gt_windows": sum(
            sum(row["gt_windows"] for row in scores[scene]["per_level"])
            for scene in scenes),
        "pred_windows": sum(
            sum(row["pred_windows"] for row in scores[scene]["per_level"])
            for scene in scenes),
        "metrics": metrics,
        "per_scene_f1@0.5": {
            scene: prf(*scores[scene]["counts"]["0.5"])["f1"]
            for scene in scenes
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    manifest, manifest_path = verify_manifest(args.canonical_dir)
    if not set(scenes) <= set(manifest["scene_ids"]):
        raise RuntimeError("prediction collection does not cover requested scenes")
    validator = Draft202012Validator(json.load(args.schema.open()))
    scores = {}
    gt_hashes = {}
    for scene in scenes:
        artifact_path = args.canonical_dir / f"{scene}.json"
        artifact = json.load(artifact_path.open())
        validator.validate(artifact)
        with (args.gt_dir / f"{scene}.pkl").open("rb") as stream:
            gt = pickle.load(stream)
        gt_path = WINDOW_GT_DIR / f"{scene}.json"
        gt_hashes[os.path.relpath(gt_path, ROOT)] = sha256(gt_path)
        pred_level_z, pred = canonical_windows(artifact)
        official = official_windows(scene, gt["lvl_z"])
        scores[scene] = score_scene(gt["lvl_z"], official,
                                    pred_level_z, pred)

    summary = {
        "status": "artifact_only_released_window_gt_evaluation",
        "evaluator": EVAL_VERSION,
        "prediction_collection": os.path.relpath(args.canonical_dir, ROOT),
        "gt_source": "released HouseLayout3D data/windows 3D rectangles projected to bottom 2D edge",
        "partition": args.partition,
        "checkpoint_name": args.checkpoint_name,
        "metric": (
            "Hungarian matching; segment distance is the smaller endpoint "
            "orientation's maximum endpoint L2 error; <= threshold"),
        "selected": aggregate(scores, scenes),
    }
    if args.partition == "all":
        summary["dev"] = aggregate(scores, split["dev"])
        summary["held_out"] = aggregate(scores, split["held_out"])
    out = args.out or args.canonical_dir / f"{EVAL_VERSION}_{args.partition}"
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.json"
    summary_path = out / "summary.json"
    scores_path.write_text(json.dumps(
        scores, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    output_manifest = {
        "inputs": {
            os.path.relpath(manifest_path, ROOT): sha256(manifest_path),
            os.path.relpath(args.split, ROOT): sha256(args.split),
            os.path.relpath(args.schema, ROOT): sha256(args.schema),
            **gt_hashes,
        },
        "outputs": {
            "scores.json": sha256(scores_path),
            "summary.json": sha256(summary_path),
        },
    }
    (out / "manifest.json").write_text(json.dumps(
        output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for name in ["selected"] + (["dev", "held_out"]
                                if args.partition == "all" else []):
        value = summary[name]
        metric = value["metrics"]["windows@0.5"]
        print(f"{name}: windows={value['pred_windows']} GT={value['gt_windows']} "
              f"P/R/F1@0.5={metric['p']:.4f}/{metric['r']:.4f}/{metric['f1']:.4f}")
    print(f"window evaluation -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
