"""Artifact-only Room+type evaluator with an explicit ontology crosswalk."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from metrics import align_levels, match_rooms_iou, prf  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
CROSSWALK = ROOT / "configs" / "eval2d" / "room_type_crosswalk_v0_1.json"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
EVAL_VERSION = "eval2d_room_types_v0_1_explicit_crosswalk"


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


def canonical_rooms(artifact):
    level_z, rooms = {}, []
    for level in artifact["levels"]:
        level_id = int(level["id"])
        level_z[level_id] = float(level["elevation"])
        for room in level["rooms"]:
            rooms.append({
                "idx": room["id"],
                "level": level_id,
                "geometry": room["geometry"],
                "type": room.get("type_raw", "unknown"),
                "type_scores": room.get("type_scores"),
            })
    return level_z, rooms


def score_scene(gt, pred_level_z, pred_rooms, crosswalk):
    gt_by_level, pred_by_level = defaultdict(list), defaultdict(list)
    for room in gt["rooms"]:
        if room["in_roomset"] and not room.get("is_stairs"):
            gt_by_level[room["level"]].append(room)
    for room in pred_rooms:
        pred_by_level[room["level"]].append(room)
    mapping = align_levels(gt["lvl_z"], pred_level_z)
    room_counts = [0, 0, 0]
    typed_counts = [0, 0, 0]
    gt_rooms = gt_mappable = matched_mappable = 0
    top1_correct = top3_correct = feature_classified = 0
    confusion = Counter()
    per_level = []

    for gt_level in sorted(gt["lvl_z"], key=str):
        pred_level = mapping.get(gt_level)
        gt_items = gt_by_level.get(gt_level, [])
        pred_items = (pred_by_level.get(pred_level, [])
                      if pred_level is not None else [])
        matches, _ = match_rooms_iou(pred_items, gt_items, thresh=0.5)
        room_tp = len(matches)
        room_counts = [room_counts[0] + room_tp,
                       room_counts[1] + len(pred_items) - room_tp,
                       room_counts[2] + len(gt_items) - room_tp]
        gt_map = {room["idx"]: room for room in gt_items}
        pred_map = {room["idx"]: room for room in pred_items}
        type_tp = 0
        for pred_id, gt_id in matches.items():
            gt_target = crosswalk.get(gt_map[gt_id]["type"])
            predicted = pred_map[pred_id].get("type", "unknown")
            if predicted != "unknown":
                feature_classified += 1
            if gt_target is None:
                continue
            matched_mappable += 1
            scores = pred_map[pred_id].get("type_scores") or {}
            top3 = [name for name, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0]))[:3]]
            correct = predicted == gt_target
            top1_correct += int(correct)
            top3_correct += int(gt_target in top3)
            type_tp += int(correct)
            confusion[(gt_target, predicted)] += 1
        typed_counts = [typed_counts[0] + type_tp,
                        typed_counts[1] + len(pred_items) - type_tp,
                        typed_counts[2] + len(gt_items) - type_tp]
        gt_rooms += len(gt_items)
        gt_mappable += sum(crosswalk.get(room["type"]) is not None
                           for room in gt_items)
        per_level.append({
            "gt_level": gt_level, "pred_level": pred_level,
            "gt_rooms": len(gt_items), "pred_rooms": len(pred_items),
            "geometry_matches": room_tp, "type_correct_matches": type_tp,
        })

    unmatched_pred = sorted(
        set(pred_by_level) - set(mapping.values()), key=str)
    for pred_level in unmatched_pred:
        n_pred = len(pred_by_level[pred_level])
        room_counts[1] += n_pred
        typed_counts[1] += n_pred
        per_level.append({
            "gt_level": None, "pred_level": pred_level,
            "gt_rooms": 0, "pred_rooms": n_pred,
            "geometry_matches": 0, "type_correct_matches": 0,
        })
    return {
        "room_counts": room_counts,
        "room_type_counts": typed_counts,
        "gt_rooms": gt_rooms,
        "gt_mappable": gt_mappable,
        "matched_mappable": matched_mappable,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "feature_classified_matched": feature_classified,
        "confusion": [
            {"gt": gt_type, "pred": pred_type, "count": count}
            for (gt_type, pred_type), count in sorted(confusion.items())
        ],
        "level_alignment": {str(key): value for key, value in mapping.items()},
        "unmatched_pred_levels": unmatched_pred,
        "per_level": per_level,
    }


def aggregate(scores, scenes):
    room_counts = [sum(scores[scene]["room_counts"][index]
                       for scene in scenes) for index in range(3)]
    type_counts = [sum(scores[scene]["room_type_counts"][index]
                       for scene in scenes) for index in range(3)]
    gt_rooms = sum(scores[scene]["gt_rooms"] for scene in scenes)
    gt_mappable = sum(scores[scene]["gt_mappable"] for scene in scenes)
    matched_mappable = sum(
        scores[scene]["matched_mappable"] for scene in scenes)
    top1 = sum(scores[scene]["top1_correct"] for scene in scenes)
    top3 = sum(scores[scene]["top3_correct"] for scene in scenes)
    confusion = Counter()
    for scene in scenes:
        for row in scores[scene]["confusion"]:
            confusion[(row["gt"], row["pred"])] += row["count"]
    return {
        "n_scenes": len(scenes),
        "room": prf(*room_counts),
        "room+type_all_rooms": prf(*type_counts),
        "gt_rooms": gt_rooms,
        "gt_mappable": gt_mappable,
        "crosswalk_coverage": gt_mappable / gt_rooms if gt_rooms else None,
        "matched_mappable": matched_mappable,
        "conditional_mappable_top1_accuracy": (
            top1 / matched_mappable if matched_mappable else None),
        "conditional_mappable_top3_accuracy": (
            top3 / matched_mappable if matched_mappable else None),
        "top1_correct": top1,
        "top3_correct": top3,
        "confusion": [
            {"gt": gt_type, "pred": pred_type, "count": count}
            for (gt_type, pred_type), count in sorted(confusion.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    manifest, manifest_path = verify_manifest(args.canonical_dir)
    if not set(scenes) <= set(manifest["scene_ids"]):
        raise RuntimeError("prediction collection does not cover requested scenes")
    validator = Draft202012Validator(json.load(args.schema.open()))
    crosswalk_payload = json.load(args.crosswalk.open())
    crosswalk = crosswalk_payload["mapping"]
    scores = {}
    for scene in scenes:
        artifact = json.load((args.canonical_dir / f"{scene}.json").open())
        validator.validate(artifact)
        with (args.gt_dir / f"{scene}.pkl").open("rb") as stream:
            gt = pickle.load(stream)
        pred_level_z, pred_rooms = canonical_rooms(artifact)
        scores[scene] = score_scene(gt, pred_level_z, pred_rooms, crosswalk)

    summary = {
        "status": "artifact_only_room_type_evaluation",
        "evaluator": EVAL_VERSION,
        "prediction_collection": os.path.relpath(args.canonical_dir, ROOT),
        "partition": args.partition,
        "checkpoint_name": args.checkpoint_name,
        "crosswalk": os.path.relpath(args.crosswalk, ROOT),
        "crosswalk_policy": crosswalk_payload["policy"],
        "metric_contract": (
            "strict level alignment; room IoU>0.5 Hungarian; end-to-end "
            "Room+type over all rooms plus conditional top1/top3 accuracy "
            "only on geometry-matched rooms with a non-null GT crosswalk"),
        "selected": aggregate(scores, scenes),
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
            os.path.relpath(args.crosswalk, ROOT): sha256(args.crosswalk),
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
        metric = value["room+type_all_rooms"]
        print(f"{name}: Room+type P/R/F1="
              f"{metric['p']:.4f}/{metric['r']:.4f}/{metric['f1']:.4f}; "
              f"matched-mappable top1/top3="
              f"{value['conditional_mappable_top1_accuracy']:.4f}/"
              f"{value['conditional_mappable_top3_accuracy']:.4f} "
              f"(n={value['matched_mappable']})")
    print(f"room-type evaluation -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
