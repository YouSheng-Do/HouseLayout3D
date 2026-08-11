"""Generate a governed annotated-best direct semantic door A/B.

The generator consumes only frozen canonical predictions and retained Stage-2
OneFormer ray endpoints/labels.  It never reads GT or evaluator scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from access_derive import (ONE_OUTSIDE, SUCCESS, Door, Room,  # noqa: E402
                           derive_access_graph)
from direct_semantic_doors import extract_candidates  # noqa: E402
from geometry_v2 import to_shapely  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
BASE_CANONICAL = (ROOT / "outputs" / "eval2d" / "experiments" /
                  "room_types_ab_v0_1" / "all" /
                  "sample_point_room_mean")
DEFAULT_OUT = (ROOT / "outputs" / "eval2d" / "experiments" /
               "direct_doors_ab_v0_1")
VARIANTS = ("watershed_control", "direct_semantic_only", "fused_union")
FUSION_DUPLICATE_MIDPOINT_M = 0.50


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_paths(scene):
    root = ROOT / "outputs" / "mp3d" / scene / "skeleton"
    return {
        "ray_points": root / "full_ray_dests.npy",
        "ray_labels": root / "hard_labels_simplified_segmentations.npy",
        "ray_valid": root / "ray_is_valid.npy",
        "semantic_labels": root / "simplified_segmentation_labels.npy",
    }


def room_inputs(level):
    return [Room(room["id"], room.get("type_raw", "unknown"),
                 to_shapely(room["geometry"]))
            for room in level["rooms"]]


def semantic_door_records(level, candidates):
    records = []
    for index, candidate in enumerate(candidates):
        door_id = f"L{level['id']}_SD{index:04d}"
        records.append({
            "id": door_id,
            "segment": candidate["segment"].tolist(),
            "width_m": float(candidate["width_m"]),
            "room_a": None,
            "room_b": None,
            "association_status": "unresolved",
            "confidence": None,
            "provenance": candidate["provenance"],
        })
    if not records:
        return records, []
    access_doors = [
        Door(record["id"], np.asarray(record["segment"])[0],
             np.asarray(record["segment"])[1])
        for record in records
    ]
    _, probes = derive_access_graph(
        room_inputs(level), access_doors, d=0.30)
    probe_by_id = {probe["door_id"]: probe for probe in probes}
    edges = []
    for record in records:
        probe = probe_by_id[record["id"]]
        if probe["outcome"] == SUCCESS:
            room_a, room_b = probe["probe_a"][0], probe["probe_b"][0]
            status = "semantic_geometry_derived_room_room_candidate"
            kind = "room-room"
        elif probe["outcome"] == ONE_OUTSIDE:
            room_a = (probe["probe_a"] or probe["probe_b"])[0]
            room_b = "OUTSIDE"
            status = "semantic_geometry_derived_exterior_candidate"
            kind = "room-outside-candidate"
        else:
            continue
        record["room_a"], record["room_b"] = room_a, room_b
        record["association_status"] = status
        edges.append({
            "door_id": record["id"], "rooms": [room_a, room_b],
            "kind": kind, "status": status,
        })
    return records, edges


def midpoint(door):
    return np.asarray(door["segment"], dtype=np.float64).mean(axis=0)


def augment(base, scene_candidates, variant, source_path, source_hash):
    artifact = copy.deepcopy(base)
    artifact["artifact_provenance"]["source_artifact"] = os.path.relpath(
        source_path, ROOT)
    artifact["artifact_provenance"]["source_sha256"] = source_hash
    artifact["method"]["baseline_id"] = f"annotated_best_{variant}_doors_v0_1"
    artifact["method"].setdefault("annotation_layers", {})["doors"] = {
        "variant": variant,
        "status": ("local_direct_semantic_extension"
                   if variant != "watershed_control"
                   else "frozen_watershed_control"),
        "semantic_source": "retained OneFormer door-labeled valid ray endpoints",
        "fusion_duplicate_midpoint_m": (
            FUSION_DUPLICATE_MIDPOINT_M if variant == "fused_union" else None),
        "gt_accessed": False,
    }
    artifact["limitations"]["connectivity"] = (
        "door_geometry_associations_are_derived_not_independent_truth")

    diagnostics = {"variant": variant, "gt_accessed": False, "levels": []}
    candidates_by_level = {}
    for candidate in scene_candidates:
        candidates_by_level.setdefault(candidate["level"], []).append(candidate)
    for level in artifact["levels"]:
        level_id = int(level["id"])
        candidates = candidates_by_level.get(level_id, [])
        semantic_doors, semantic_edges = semantic_door_records(
            level, candidates)
        base_count = len(level["doors"])
        if variant == "watershed_control":
            added = 0
        elif variant == "direct_semantic_only":
            level["doors"] = semantic_doors
            level["graph"]["edges"] = [
                edge for edge in level["graph"]["edges"]
                if edge["kind"] == "stair"] + semantic_edges
            added = len(semantic_doors)
        else:
            base_midpoints = [midpoint(door) for door in level["doors"]]
            novel_indices = [
                index for index, door in enumerate(semantic_doors)
                if not any(np.linalg.norm(midpoint(door) - base_midpoint) <
                           FUSION_DUPLICATE_MIDPOINT_M
                           for base_midpoint in base_midpoints)
            ]
            novel_ids = {semantic_doors[index]["id"] for index in novel_indices}
            level["doors"].extend(semantic_doors[index]
                                  for index in novel_indices)
            level["graph"]["edges"].extend(
                edge for edge in semantic_edges
                if edge["door_id"] in novel_ids)
            added = len(novel_indices)
        level["graph"]["edge_status"] = (
            "mixed_watershed_and_semantic_derived_candidates"
            if variant == "fused_union" else
            "semantic_geometry_derived_candidates"
            if variant == "direct_semantic_only" else
            level["graph"]["edge_status"])
        diagnostics["levels"].append({
            "level": level_id,
            "base_doors": base_count,
            "semantic_candidates": len(semantic_doors),
            "semantic_added": added,
            "output_doors": len(level["doors"]),
            "resolved_semantic_associations": sum(
                door["association_status"] != "unresolved"
                for door in semantic_doors),
        })
    diagnostics["summary"] = {
        key: sum(row[key] for row in diagnostics["levels"])
        for key in ("base_doors", "semantic_candidates", "semantic_added",
                    "output_doors", "resolved_semantic_associations")
    }
    return artifact, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--base-canonical", type=Path, default=BASE_CANONICAL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    output_root = args.out_root / args.partition
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    base_manifest_path = args.base_canonical / "collection_manifest.json"
    base_manifest = json.load(base_manifest_path.open())
    files = {variant: {} for variant in VARIANTS}
    diagnostic_files = {variant: {} for variant in VARIANTS}
    source_hashes, semantic_hashes = {}, {}

    for scene_index, scene in enumerate(scenes, 1):
        source_path = args.base_canonical / f"{scene}.json"
        paths = semantic_paths(scene)
        missing = [str(path) for path in (source_path, *paths.values())
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing {missing}")
        source_hash = sha256(source_path)
        if base_manifest["files_sha256"].get(source_path.name) != source_hash:
            raise RuntimeError(f"{scene}: frozen base canonical hash mismatch")
        source_hashes[os.path.relpath(source_path, ROOT)] = source_hash
        semantic_hashes[scene] = {
            os.path.relpath(path, ROOT): sha256(path)
            for path in paths.values()
        }
        base = json.load(source_path.open())
        candidates, extraction = extract_candidates(
            np.load(paths["ray_points"], mmap_mode="r"),
            np.load(paths["ray_labels"], mmap_mode="r"),
            np.load(paths["ray_valid"], mmap_mode="r"),
            np.load(paths["semantic_labels"]), base)
        print(f"[{scene_index}/{len(scenes)}] {scene}: semantic candidates="
              f"{len(candidates)}", flush=True)
        for variant in VARIANTS:
            artifact, diagnostics = augment(
                base, candidates, variant, source_path, source_hash)
            diagnostics["scene"] = scene
            diagnostics["extraction"] = extraction
            validator.validate(artifact)
            out_dir = output_root / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = out_dir / f"{scene}.json"
            diagnostics_path = out_dir / f"{scene}.diagnostics.json"
            artifact_path.write_text(json.dumps(
                artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            diagnostics_path.write_text(json.dumps(
                diagnostics, ensure_ascii=False, indent=2,
                sort_keys=True) + "\n")
            files[variant][artifact_path.name] = sha256(artifact_path)
            diagnostic_files[variant][diagnostics_path.name] = sha256(
                diagnostics_path)

    builders = [Path(__file__), HERE / "direct_semantic_doors.py",
                SCHEMA, args.split]
    for variant in VARIANTS:
        out_dir = output_root / variant
        manifest = {
            "collection": "canonical_annotated_floorplan_v0.2_direct_door_ab",
            "schema_version": "annotated_floorplan_v0.2",
            "variant": variant,
            "partition": args.partition,
            "checkpoint_name": args.checkpoint_name,
            "scene_ids": scenes,
            "n_scenes": len(scenes),
            "same_input_variants": list(VARIANTS),
            "base_collection_manifest_sha256": sha256(base_manifest_path),
            "base_canonical_files_sha256": source_hashes,
            "semantic_inputs_sha256": semantic_hashes,
            "builder_files_sha256": {
                os.path.relpath(path, ROOT): sha256(path) for path in builders
            },
            "files_sha256": files[variant],
            "diagnostics_files_sha256": diagnostic_files[variant],
            "execution": (
                "CPU-only retained OneFormer ray aggregation; no GT/score "
                "access, OneFormer inference, Stage 2/3/4 rerun, room "
                "segmentation, pipeline, extrusion, or GPU"),
        }
        (out_dir / "collection_manifest.json").write_text(json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"direct-door A/B -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
