"""Generate same-input paper_spec_two_stage vs watershed_v3 canonical A/B.

CPU-only Stage-4a experiment: each scene's fitted prototype and semantic
skeleton are loaded once, level identification is run once, and deep-copied
levels are passed to the two segmentation methods.  No Stage 2/3 computation,
extrusion, GPU, GT information, or evaluator score is used during generation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE4 = ROOT / "src" / "stage4"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STAGE4))

import scene_graph as stage4  # noqa: E402
from canonical_io import SCHEMA_VERSION, build_canonical  # noqa: E402
from geometry_v2 import mask_to_json_geometry, to_shapely  # noqa: E402
from load_prototype import load_prototype  # noqa: E402
from paper_spec_two_stage import segment_level as segment_paper  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
DEFAULT_OUT = ROOT / "outputs" / "eval2d" / "experiments" / "paper_two_stage_ab_v0_1"
METHODS = ("paper_spec_two_stage", "watershed_v3_rerun")
RDP_M = 0.10
WALL_BUFFER_M = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(scene: str):
    base = ROOT / "outputs" / "mp3d" / scene
    return {
        "fitted_mesh": base / "stage3" / "fit" / "fitted_mesh.ply",
        "semantic_skeleton": base / "skeleton" / "ceiling_wall_floor_mesh.ply",
        "coarse_probabilities": base / "stage3" / "coarse" / "cwf_classes.npy",
        "coarse_labels": base / "stage3" / "coarse" / "labels.npy",
    }


def _prediction(levels, scene: str, method: str):
    rooms, doors = [], []
    level_z = {}
    diagnostics = {
        "scene": scene,
        "method": method,
        "rdp_tolerance_m": RDP_M,
        "levels": [],
    }
    for level in levels:
        level_z[level.idx] = float(level.elevation)
        level_diag = {
            "level": int(level.idx),
            "elevation_m": float(level.elevation),
            "room_count": len(level.rooms),
            "opening_count": len(level.openings),
            "door_count": sum(bool(opening.get("is_door"))
                              for opening in level.openings),
            "bottlenecks": [],
        }
        if method == "paper_spec_two_stage":
            level_diag.update(level._paper_spec_diagnostics)

        retained_room_ids = set()
        for room in level.rooms:
            geometry = mask_to_json_geometry(
                room["mask"], level._grid_origin, stage4.RES,
                rdp_m=RDP_M)
            if geometry is None:
                continue
            rooms.append({
                "idx": room["id"],
                "level": level.idx,
                "geometry": geometry,
                "type": "unknown",
                "area": float(to_shapely(geometry).area),
                "source_mask_area": float(
                    np.asarray(room["mask"], bool).sum()
                    * stage4.RES * stage4.RES),
                "geometry_provenance": (
                    "paper_spec_two_stage_mask_polygonization_v0.2"
                    if method == "paper_spec_two_stage"
                    else "watershed_v3_rerun_mask_polygonization_v0.2"),
                "in_roomset": True,
            })
            retained_room_ids.add(room["id"])
        level_diag["canonical_room_count"] = len(retained_room_ids)
        level_diag["dropped_room_ids"] = sorted(
            {room["id"] for room in level.rooms} - retained_room_ids,
            key=str)
        canonical_doors = 0

        for opening_index, opening in enumerate(level.openings):
            row = {
                "id": f"L{level.idx}_B{opening_index:04d}",
                "rooms": list(opening["rooms"]),
                "width_m": float(opening["width"]),
                "segment": np.asarray(opening["seg"], float).tolist(),
                "oriented_rectangle": (
                    np.asarray(opening["oriented_rectangle"], float).tolist()
                    if opening.get("oriented_rectangle") is not None else None),
                "is_door": bool(opening.get("is_door")),
                "bottleneck_stage": opening.get("bottleneck_stage"),
                "provenance": opening.get(
                    "provenance",
                    "watershed_v3_shared_boundary_local_rule"),
            }
            if row["is_door"]:
                missing_rooms = sorted(
                    set(opening["rooms"]) - retained_room_ids, key=str)
                if missing_rooms:
                    row["canonical_export_status"] = (
                        "dropped_missing_room_geometry")
                    row["missing_canonical_room_ids"] = missing_rooms
                    level_diag["bottlenecks"].append(row)
                    continue
                row["canonical_export_status"] = "exported_direct_door"
                doors.append({
                    "idx": row["id"],
                    "level": level.idx,
                    "seg": np.asarray(opening["seg"], dtype=np.float64),
                    "width_m": float(opening["width"]),
                    "room_a": opening["rooms"][0],
                    "room_b": opening["rooms"][1],
                    "association_status": "direct_bottleneck_boundary",
                    "confidence": None,
                    "provenance": row["provenance"],
                })
                canonical_doors += 1
            else:
                row["canonical_export_status"] = "diagnostic_opening_only"
            level_diag["bottlenecks"].append(row)
        level_diag["canonical_door_count"] = canonical_doors
        diagnostics["levels"].append(level_diag)

    return {
        "scene": scene,
        "rooms": rooms,
        "doors": doors,
        "level_z": level_z,
        "n_levels": len(levels),
    }, diagnostics


def _method_metadata(method: str):
    if method == "paper_spec_two_stage":
        return {
            "name": method,
            "kind": "paper_spec_interpretation_not_author_code_port",
            "source": "HouseLayout3D Appendix D.3; same Stage-4 inputs",
            "paper_bottlenecks_m": [2.5, 1.5],
            "door_rule": "bottleneck width < 1.5m",
            "wall_raster_adapter_m": WALL_BUFFER_M,
            "wall_adapter_status": "shared_local_adapter_to_isolate_D3",
        }
    return {
        "name": method,
        "kind": "local_engineering_variant_same_input_rerun",
        "source": "Stage-4a scene_graph.segment_rooms; same Stage-4 inputs",
        "room_core_min_m": float(stage4.ROOM_CORE_MIN),
        "door_width_range_m": [float(stage4.DOOR_W_MIN),
                               float(stage4.DOOR_W_MAX)],
        "wall_raster_adapter_m": WALL_BUFFER_M,
    }


def _segment_pair(base_levels):
    levels_by_method = {method: copy.deepcopy(base_levels)
                        for method in METHODS}
    for level in levels_by_method["paper_spec_two_stage"]:
        segment_paper(
            level,
            wall_buffer_m=WALL_BUFFER_M,
            resolution_m=stage4.RES,
            first_bottleneck_m=2.5,
            second_bottleneck_m=1.5,
            door_max_width_m=1.5,
        )
    for level in levels_by_method["watershed_v3_rerun"]:
        stage4.segment_rooms(level, wall_buffer=WALL_BUFFER_M)
    return levels_by_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    partition_root = args.out_root / args.partition
    input_manifest_dir = partition_root / "shared_input_manifests"
    input_manifest_dir.mkdir(parents=True, exist_ok=True)
    with SCHEMA.open() as stream:
        validator = Draft202012Validator(json.load(stream))

    files_by_method = {method: {} for method in METHODS}
    diagnostics_by_method = {method: {} for method in METHODS}
    topology_by_method = {
        method: {"legacy_single_exterior_ring": 0,
                 "structured_polygon_topology": 0}
        for method in METHODS
    }
    input_manifest_hashes = {}

    for scene_index, scene in enumerate(scenes, 1):
        paths = input_paths(scene)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing same-input files {missing}")
        input_hashes = {
            os.path.relpath(path, ROOT): sha256(path)
            for path in paths.values()
        }
        print(f"[{scene_index}/{len(scenes)}] load once: {scene}", flush=True)
        prototype = load_prototype(
            str(paths["fitted_mesh"]),
            str(paths["semantic_skeleton"]),
            str(paths["coarse_probabilities"]),
            str(paths["coarse_labels"]),
        )
        base_levels = stage4.identify_levels(prototype)
        input_payload = {
            "scene": scene,
            "same_input_pair": list(METHODS),
            "files_sha256": input_hashes,
            "level_count_before_segmentation": len(base_levels),
            "level_elevations_m": [float(level.elevation)
                                     for level in base_levels],
            "generation_contract": (
                "prototype loaded once; identify_levels run once; deep-copied "
                "levels supplied to both D.3 methods"),
        }
        input_manifest_path = input_manifest_dir / f"{scene}.json"
        input_manifest_path.write_text(
            json.dumps(input_payload, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n")
        input_manifest_hash = sha256(input_manifest_path)
        input_manifest_hashes[os.path.relpath(
            input_manifest_path, partition_root)] = input_manifest_hash

        segmented = _segment_pair(base_levels)
        for method in METHODS:
            method_dir = partition_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            prediction, diagnostics = _prediction(
                segmented[method], scene, method)
            canonical = build_canonical(
                prediction,
                baseline_id=f"{method}_same_input_ab_v0_1",
                source_artifact=os.path.relpath(input_manifest_path, ROOT),
                source_sha256=input_manifest_hash,
                method=_method_metadata(method),
                segmentation_rerun=True,
            )
            validator.validate(canonical)
            canonical_path = method_dir / f"{scene}.json"
            canonical_path.write_text(
                json.dumps(canonical, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n")
            files_by_method[method][canonical_path.name] = sha256(
                canonical_path)
            for level in canonical["levels"]:
                for room in level["rooms"]:
                    topology_by_method[method][
                        room["source_geometry_capability"]] += 1

            diagnostics_path = method_dir / f"{scene}.diagnostics.json"
            diagnostics_path.write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n")
            diagnostics_by_method[method][diagnostics_path.name] = sha256(
                diagnostics_path)
            print(
                f"  {method}: levels={len(segmented[method])} "
                f"rooms={len(prediction['rooms'])} doors={len(prediction['doors'])}",
                flush=True)

        del segmented, base_levels, prototype
        gc.collect()

    builder_files = [
        Path(__file__), HERE / "paper_spec_two_stage.py",
        HERE / "canonical_io.py", HERE / "geometry_v2.py",
        STAGE4 / "scene_graph.py", STAGE4 / "load_prototype.py",
        SCHEMA, args.split,
    ]
    for method in METHODS:
        method_dir = partition_root / method
        manifest = {
            "collection": "canonical_annotated_floorplan_v0.2_same_input_ab",
            "schema_version": SCHEMA_VERSION,
            "baseline_id": f"{method}_same_input_ab_v0_1",
            "method": _method_metadata(method),
            "partition": args.partition,
            "checkpoint_name": args.checkpoint_name,
            "split": os.path.relpath(args.split, ROOT),
            "split_sha256": sha256(args.split),
            "scene_ids": scenes,
            "n_scenes": len(scenes),
            "same_input_pair": list(METHODS),
            "shared_input_manifests_sha256": input_manifest_hashes,
            "builder_files_sha256": {
                os.path.relpath(path, ROOT): sha256(path)
                for path in builder_files
            },
            "files_sha256": files_by_method[method],
            "diagnostics_files_sha256": diagnostics_by_method[method],
            "source_geometry_capability_counts": topology_by_method[method],
            "execution": "CPU-only Stage-4a segmentation; no Stage 2/3 rerun, extrusion, GPU, GT, or score access",
            "paper_faithfulness_boundary": (
                "Appendix D.3 width/order/door rule implemented; public paper "
                "does not specify the morphology internals and released "
                "Stage-4 author code is unavailable"),
            "deterministic_manifest": True,
        }
        manifest_path = method_dir / "collection_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n")

    print(f"same-input A/B artifacts -> {partition_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
