"""Export canonical annotated-floorplan v0.2 from immutable prediction PKLs.

This command never imports or reruns Stage 4.  Existing v0.1 artifacts are not
overwritten; the default target is a separately versioned collection.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys

from jsonschema import Draft202012Validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_io import SCHEMA_VERSION, build_canonical, load_canonical  # noqa: E402


ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
DEFAULT_OUT = f"{ROOT}/outputs/eval2d/canonical/watershed_v3_pre_report_v0_2"
SPLIT = f"{ROOT}/configs/eval2d/split_v0_1.json"
SCHEMA = f"{ROOT}/configs/eval2d/annotated_floorplan_v0_2.schema.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(pred_dir):
    manifest_path = f"{BASE}/manifest.json"
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    expected = {os.path.basename(relative): digest
                for relative, digest in manifest["files_sha256"].items()
                if relative.startswith("pred/")}
    paths = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 prediction PKLs, found {len(paths)}")
    for path in paths:
        name = os.path.basename(path)
        if expected.get(name) != sha256(path):
            raise RuntimeError(f"frozen prediction hash mismatch: {name}")
    return paths, manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=f"{BASE}/pred")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()

    paths, source_manifest = verify_sources(args.pred_dir)
    with open(args.split) as stream:
        split = json.load(stream)
    with open(SCHEMA) as stream:
        schema = json.load(stream)
    validator = Draft202012Validator(schema)
    split_scenes = set(split["dev"]) | set(split["held_out"])
    scenes = [os.path.basename(path)[:-4] for path in paths]
    if set(scenes) != split_scenes or len(split["dev"]) != 8 or len(split["held_out"]) != 8:
        raise RuntimeError("frozen split does not exactly partition the 16 scenes")

    os.makedirs(args.out, exist_ok=True)
    output_hashes = {}
    source_hashes = {}
    topology = {"legacy_single_exterior_ring": 0,
                "structured_polygon_topology": 0}
    for path, scene in zip(paths, scenes):
        source_hash = sha256(path)
        with open(path, "rb") as stream:
            prediction = pickle.load(stream)
        canonical = build_canonical(
            prediction,
            source_artifact=os.path.relpath(path, ROOT),
            source_sha256=source_hash,
        )
        validator.validate(canonical)
        loaded = load_canonical(canonical)
        if len(loaded["rooms"]) != len(prediction["rooms"]):
            raise RuntimeError(f"{scene}: room count changed during export")
        for room in canonical["levels"]:
            for item in room["rooms"]:
                topology[item["source_geometry_capability"]] += 1
        text = json.dumps(canonical, ensure_ascii=False, indent=2,
                          sort_keys=True) + "\n"
        output_path = os.path.join(args.out, f"{scene}.json")
        with open(output_path, "w") as stream:
            stream.write(text)
        output_hashes[f"{scene}.json"] = sha256(output_path)
        source_hashes[os.path.relpath(path, ROOT)] = source_hash

    source_files = [
        __file__,
        os.path.join(os.path.dirname(__file__), "canonical_io.py"),
        os.path.join(os.path.dirname(__file__), "geometry_v2.py"),
        SCHEMA,
    ]
    manifest = {
        "collection": "canonical_annotated_floorplan_v0.2",
        "schema_version": SCHEMA_VERSION,
        "baseline_id": "watershed_v3_pre_report",
        "source": "immutable frozen prediction PKLs; no segmentation rerun",
        "source_manifest": os.path.relpath(source_manifest, ROOT),
        "source_manifest_sha256": sha256(source_manifest),
        "split": os.path.relpath(args.split, ROOT),
        "split_sha256": sha256(args.split),
        "n_scenes": len(scenes),
        "scene_ids": scenes,
        "source_files_sha256": source_hashes,
        "builder_files_sha256": {
            os.path.relpath(path, ROOT): sha256(path) for path in source_files
        },
        "files_sha256": output_hashes,
        "source_geometry_capability_counts": topology,
        "known_source_limitation": (
            "all frozen v0.1 PKLs contain one exterior ring per room; v0.2 "
            "supports holes/MultiPolygons but cannot recover topology absent "
            "from the frozen source"),
        "deterministic_manifest": True,
    }
    manifest_path = os.path.join(args.out, "collection_manifest.json")
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
        stream.write("\n")

    print(f"exported {len(scenes)} {SCHEMA_VERSION} artifacts -> {args.out}")
    print(f"source topology: {topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
