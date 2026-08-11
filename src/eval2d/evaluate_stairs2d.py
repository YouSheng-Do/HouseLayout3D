"""Artifact-only 2D stair-footprint evaluator.

The released 34 stair meshes support a projected footprint metric.  They do
not contain official inter-level or adjacent-room links, so this evaluator
reports prediction-side link coverage but deliberately does not manufacture a
Stair-link accuracy score from mesh heights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from geometry_v2 import to_shapely  # noqa: E402
from metrics import align_levels, match_rooms_iou, prf  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402
from stairs2d import (STAIR_LEVEL_Z_TOLERANCE_M,
                      official_stair_entities)  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
STAIR_GT_DIR = ROOT / "external" / "houselayout3d" / "data" / "stairs"
EVAL_VERSION = "eval2d_stairs_v0_1_footprint_iou_hungarian"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory):
    manifest_path = Path(directory) / "collection_manifest.json"
    manifest = json.load(manifest_path.open())
    failures = [
        relative for relative, expected in manifest["files_sha256"].items()
        if not (Path(directory) / relative).exists()
        or sha256(Path(directory) / relative) != expected
    ]
    if failures:
        raise RuntimeError(f"prediction hash failure: {failures}")
    return manifest, manifest_path


def canonical_stairs(artifact):
    level_z, stairs = {}, []
    for level in artifact["levels"]:
        level_id = int(level["id"])
        level_z[level_id] = float(level["elevation"])
        for stair in level["stairs"]:
            if int(stair["from_level"]) != level_id:
                raise ValueError(
                    f"stair {stair['id']} is not stored on from_level")
            geometry = to_shapely(stair["geometry"])
            if geometry.is_empty or geometry.area <= 0:
                raise ValueError(f"stair {stair['id']} has empty footprint")
            stairs.append({
                "idx": stair["id"],
                "level": level_id,
                "geometry": stair["geometry"],
                "from_level": stair["from_level"],
                "to_level": stair["to_level"],
                "adjacent_rooms": stair["adjacent_rooms"],
            })
    return level_z, stairs


def score_scene(gt_level_z, gt_stairs, pred_level_z, pred_stairs):
    mapping = align_levels(gt_level_z, pred_level_z)
    gt_by_level = defaultdict(list)
    pred_by_level = defaultdict(list)
    for stair in gt_stairs:
        gt_by_level[stair["level"]].append(stair)
    for stair in pred_stairs:
        pred_by_level[stair["level"]].append(stair)

    counts = [0, 0, 0]
    per_level = []
    matches = []
    for gt_level in sorted(gt_level_z, key=str):
        pred_level = mapping.get(gt_level)
        gt_items = gt_by_level.get(gt_level, [])
        pred_items = (pred_by_level.get(pred_level, [])
                      if pred_level is not None else [])
        matched, iou_of = match_rooms_iou(pred_items, gt_items, thresh=0.5)
        value = [len(matched), len(pred_items) - len(matched),
                 len(gt_items) - len(matched)]
        counts = [counts[index] + value[index] for index in range(3)]
        per_level.append({
            "gt_level": gt_level,
            "pred_level": pred_level,
            "gt_stairs": len(gt_items),
            "pred_stairs": len(pred_items),
            "counts": value,
        })
        for pred_id, (gt_id, iou) in sorted(
                iou_of.items(), key=lambda item: str(item[0])):
            matches.append({
                "pred_id": pred_id,
                "gt_id": gt_id,
                "gt_level": gt_level,
                "pred_level": pred_level,
                "iou": iou,
            })

    unmatched_pred = sorted(
        set(pred_by_level) - set(mapping.values()), key=str)
    for pred_level in unmatched_pred:
        n_pred = len(pred_by_level[pred_level])
        counts[1] += n_pred
        per_level.append({
            "gt_level": None,
            "pred_level": pred_level,
            "gt_stairs": 0,
            "pred_stairs": n_pred,
            "counts": [0, n_pred, 0],
        })

    complete_level_links = sum(
        stair["to_level"] is not None for stair in pred_stairs)
    complete_room_links = sum(
        len(stair["adjacent_rooms"]) == 2 for stair in pred_stairs)
    return {
        "counts": counts,
        "level_alignment": {str(key): value for key, value in mapping.items()},
        "unmatched_pred_levels": unmatched_pred,
        "per_level": per_level,
        "matches": matches,
        "prediction_link_coverage": {
            "predicted_stairs": len(pred_stairs),
            "complete_level_links": complete_level_links,
            "complete_room_links": complete_room_links,
            "level_link_coverage": (
                complete_level_links / len(pred_stairs)
                if pred_stairs else None),
            "room_link_coverage": (
                complete_room_links / len(pred_stairs)
                if pred_stairs else None),
        },
        "stair_link_metric": None,
        "stair_link_status": "N/A: released GT has no inter-level link labels",
    }


def aggregate(scores, scenes):
    counts = [sum(scores[scene]["counts"][index] for scene in scenes)
              for index in range(3)]
    metric = prf(*counts)
    predicted = sum(
        scores[scene]["prediction_link_coverage"]["predicted_stairs"]
        for scene in scenes)
    level_links = sum(
        scores[scene]["prediction_link_coverage"]["complete_level_links"]
        for scene in scenes)
    room_links = sum(
        scores[scene]["prediction_link_coverage"]["complete_room_links"]
        for scene in scenes)
    return {
        "n_scenes": len(scenes),
        "scorable_scenes": sum(
            sum(scores[scene]["counts"]) > 0 for scene in scenes),
        "gt_stair_entities": counts[0] + counts[2],
        "predicted_stairs": counts[0] + counts[1],
        "metrics": {"stair_footprint_iou>0.5": metric},
        "per_scene_f1": {
            scene: (prf(*scores[scene]["counts"])["f1"]
                    if sum(scores[scene]["counts"]) else None)
            for scene in scenes
        },
        "prediction_link_coverage": {
            "predicted_stairs": predicted,
            "complete_level_links": level_links,
            "complete_room_links": room_links,
            "level_link_coverage": level_links / predicted if predicted else None,
            "room_link_coverage": room_links / predicted if predicted else None,
        },
        "stair_link_metric": None,
        "stair_link_status": "N/A: released GT has no inter-level link labels",
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
    parser.add_argument("--stair-gt-dir", type=Path, default=STAIR_GT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    manifest, manifest_path = verify_manifest(args.canonical_dir)
    if not set(scenes) <= set(manifest["scene_ids"]):
        raise RuntimeError("prediction collection does not cover requested scenes")
    validator = Draft202012Validator(json.load(args.schema.open()))
    scores, gt_hashes = {}, {}
    for scene in scenes:
        artifact_path = args.canonical_dir / f"{scene}.json"
        artifact = json.load(artifact_path.open())
        validator.validate(artifact)
        if artifact["limitations"]["stairs_included"] is not True:
            raise ValueError(f"{scene}: stairs capability is not enabled")
        with (args.gt_dir / f"{scene}.pkl").open("rb") as stream:
            gt = pickle.load(stream)
        gt_scene_dir = args.stair_gt_dir / scene
        for path in sorted(gt_scene_dir.glob("*.ply")) \
                if gt_scene_dir.exists() else []:
            gt_hashes[os.path.relpath(path, ROOT)] = sha256(path)
        official = official_stair_entities(
            scene, gt["lvl_z"], args.stair_gt_dir)
        pred_level_z, predicted = canonical_stairs(artifact)
        scores[scene] = score_scene(
            gt["lvl_z"], official, pred_level_z, predicted)

    summary = {
        "status": "artifact_only_released_stair_footprint_evaluation",
        "evaluator": EVAL_VERSION,
        "prediction_collection": os.path.relpath(args.canonical_dir, ROOT),
        "gt_source": (
            "all 34 released HouseLayout3D stair meshes; exact XY triangle "
            "union; no fragment grouping"),
        "owning_level_rule": (
            "highest official floor elevation <= min(mesh_z)+"
            f"{STAIR_LEVEL_Z_TOLERANCE_M:.2f}m; does not infer to_level"),
        "metric": (
            "strict aligned levels; deterministic one-to-one Hungarian; "
            "footprint IoU > 0.5"),
        "partition": args.partition,
        "checkpoint_name": args.checkpoint_name,
        "selected": aggregate(scores, scenes),
        "stair_link_metric": None,
        "stair_link_status": (
            "N/A: released meshes have no from/to-level or adjacent-room labels"),
    }
    if args.partition == "all":
        summary["dev"] = aggregate(scores, split["dev"])
        summary["held_out"] = aggregate(scores, split["held_out"])
    output = args.out or args.canonical_dir / f"{EVAL_VERSION}_{args.partition}"
    output.mkdir(parents=True, exist_ok=True)
    scores_path, summary_path = output / "scores.json", output / "summary.json"
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
    (output / "manifest.json").write_text(json.dumps(
        output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for name in ["selected"] + (["dev", "held_out"]
                                if args.partition == "all" else []):
        value = summary[name]
        metric = value["metrics"]["stair_footprint_iou>0.5"]
        print(f"{name}: pred={value['predicted_stairs']} "
              f"GT={value['gt_stair_entities']} "
              f"P/R/F1={metric['p']:.4f}/{metric['r']:.4f}/{metric['f1']:.4f}")
    print(f"stair-link: {summary['stair_link_status']}")
    print(f"stairs2d evaluation -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
