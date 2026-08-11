"""Artifact-only evaluation for canonical annotated-floorplan v0.2.

Prediction JSON and frozen GT PKLs are the only scene inputs.  This module does
not import Stage 4, segmentation, or extraction code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from jsonschema import Draft202012Validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_io import SCHEMA_VERSION, load_canonical  # noqa: E402
from metrics import EVAL2D_VERSION, prf, score_scene  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


ROOT = "/home/ado/storage/HouseLayout3D"
CANONICAL = f"{ROOT}/outputs/eval2d/canonical/watershed_v3_pre_report_v0_2"
GT_DIR = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
SPLIT = f"{ROOT}/configs/eval2d/split_v0_1.json"
SCHEMA = f"{ROOT}/configs/eval2d/annotated_floorplan_v0_2.schema.json"
EXPECTED = (f"{GT_DIR}/eval_frozen_watershed_v3/"
            "scores_house_official325.pkl")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory, manifest_name, hash_field):
    path = os.path.join(directory, manifest_name)
    with open(path) as stream:
        manifest = json.load(stream)
    failures = []
    for relative, expected in manifest[hash_field].items():
        candidate = os.path.join(directory, relative)
        if not os.path.exists(candidate) or sha256(candidate) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"hash verification failed: {failures}")
    return manifest, path


def aggregate(scores, scenes):
    result = {"n_scenes": len(scenes), "metrics": {}}
    for tier in ("A", "B", "C"):
        keys = sorted(set().union(*(scores[scene][tier] for scene in scenes)))
        result["metrics"][tier] = {}
        for key in keys:
            counts = [0, 0, 0]
            for scene in scenes:
                value = scores[scene][tier].get(key, [0, 0, 0])
                counts = [counts[i] + value[i] for i in range(3)]
            result["metrics"][tier][key] = prf(*counts)
    ious = [value for scene in scenes for value in scores[scene]["iou"]]
    result["matched_iou"] = {
        "mean": float(np.mean(ious)) if ious else None,
        "n": len(ious),
    }
    result["per_scene_room_f1"] = {
        scene: prf(*scores[scene]["A"]["room"])["f1"]
        for scene in scenes
    }
    return result


def assert_expected(scores, expected_path):
    with open(expected_path, "rb") as stream:
        expected = pickle.load(stream)
    failures = []
    for scene, result in scores.items():
        reference = expected[scene]
        for tier in ("A", "B", "C"):
            if result[tier] != reference[tier]:
                failures.append(f"{scene}:{tier}")
        if not np.allclose(result["iou"], reference["iou"], atol=1e-12,
                           rtol=0):
            failures.append(f"{scene}:iou")
    if failures:
        raise RuntimeError("canonical/source score mismatch: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", default=CANONICAL)
    parser.add_argument("--gt-dir", default=GT_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--schema", default=SCHEMA)
    parser.add_argument("--expected", default=EXPECTED)
    parser.add_argument(
        "--skip-expected-equivalence", action="store_true",
        help="Required for a newly generated method whose scores should differ from the frozen baseline.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(
        args.canonical_dir,
        f"{EVAL2D_VERSION}_official325_{args.partition}")

    canonical_manifest, canonical_manifest_path = verify_manifest(
        args.canonical_dir, "collection_manifest.json", "files_sha256")
    gt_manifest, gt_manifest_path = verify_manifest(
        args.gt_dir, "candidate_manifest.json", "files_sha256")
    split = load_split(args.split)
    with open(args.schema) as stream:
        validator = Draft202012Validator(json.load(stream))
    collection_scenes = canonical_manifest["scene_ids"]
    if not set(collection_scenes) <= set(split["dev"]) | set(split["held_out"]):
        raise RuntimeError("canonical collection contains scenes outside the split")
    if canonical_manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("canonical schema mismatch")

    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    if not set(scenes) <= set(collection_scenes):
        missing = sorted(set(scenes) - set(collection_scenes))
        raise RuntimeError(f"canonical collection is missing requested scenes: {missing}")
    scores = {}
    for scene in scenes:
        with open(os.path.join(args.canonical_dir, f"{scene}.json")) as stream:
            canonical = json.load(stream)
        validator.validate(canonical)
        prediction = load_canonical(canonical)
        with open(os.path.join(args.gt_dir, f"{scene}.pkl"), "rb") as stream:
            gt = pickle.load(stream)
        scores[scene] = score_scene(gt, prediction)
    if not args.skip_expected_equivalence:
        assert_expected(scores, args.expected)

    summary = {
        "status": "artifact_only_frozen_evaluation",
        "evaluator": EVAL2D_VERSION,
        "canonical_schema": SCHEMA_VERSION,
        "prediction_collection": os.path.relpath(args.canonical_dir, ROOT),
        "gt_collection": os.path.relpath(args.gt_dir, ROOT),
        "split": os.path.relpath(args.split, ROOT),
        "partition": args.partition,
        "checkpoint_name": args.checkpoint_name,
        "source_score_equivalence": (
            "not_applicable_new_method" if args.skip_expected_equivalence
            else f"{len(scenes)}/{len(scenes)} exact A/B/C; IoU atol=1e-12"),
        "selected": aggregate(scores, scenes),
    }
    if args.partition == "all":
        summary["dev"] = aggregate(scores, split["dev"])
        summary["held_out"] = aggregate(scores, split["held_out"])
    os.makedirs(out, exist_ok=True)
    scores_path = os.path.join(out, "scores.pkl")
    summary_path = os.path.join(out, "summary.json")
    with open(scores_path, "wb") as stream:
        pickle.dump(scores, stream, protocol=4)
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
        stream.write("\n")
    manifest = {
        "inputs": {
            os.path.relpath(canonical_manifest_path, ROOT): sha256(canonical_manifest_path),
            os.path.relpath(gt_manifest_path, ROOT): sha256(gt_manifest_path),
            os.path.relpath(args.split, ROOT): sha256(args.split),
            os.path.relpath(args.schema, ROOT): sha256(args.schema),
        },
        "outputs": {
            "scores.pkl": sha256(scores_path),
            "summary.json": sha256(summary_path),
        },
        "n_verified_prediction_artifacts": len(canonical_manifest["files_sha256"]),
        "n_verified_gt_artifacts": len(gt_manifest["files_sha256"]),
    }
    if not args.skip_expected_equivalence:
        manifest["inputs"][os.path.relpath(args.expected, ROOT)] = sha256(
            args.expected)
    with open(os.path.join(out, "manifest.json"), "w") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
        stream.write("\n")

    names = ["selected"] + (["dev", "held_out"]
                            if args.partition == "all" else [])
    for name in names:
        room = summary[name]["metrics"]["A"]["room"]
        iou = summary[name]["matched_iou"]
        print(f"{name}: Room P/R/F1={room['p']:.4f}/{room['r']:.4f}/"
              f"{room['f1']:.4f}; matched IoU={iou['mean']:.4f} n={iou['n']}")
    print(f"artifact-only results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
