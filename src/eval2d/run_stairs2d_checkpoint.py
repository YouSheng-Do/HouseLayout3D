"""Augment frozen annotated-best floorplans with Stage-4 D.5 stairs.

This is an artifact-only conversion.  It reads existing ``scene_graph.json``
and ``stairs.json`` outputs and does not rerun stair detection, room
segmentation, Stage 2/3, extrusion, or GPU work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from geometry_v2 import to_json_geometry  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402
from stairs2d import normalize_stage4_link, stair_rect_footprint  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
BASE_CANONICAL = (ROOT / "outputs" / "eval2d" / "experiments" /
                  "window_rays_ab_v0_1" / "all" /
                  "legacy_three_classes")
DEFAULT_OUT = (ROOT / "outputs" / "eval2d" / "experiments" /
               "stairs2d_v0_1")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage4_paths(scene):
    root = ROOT / "outputs" / "mp3d" / scene / "stage4"
    return {
        "scene_graph": root / "scene_graph.json",
        "stair_entities": root / "stairs.json",
    }


def _validate_entity_mirror(graph_stairs, entity_stairs):
    if len(graph_stairs) != len(entity_stairs):
        raise ValueError(
            f"scene_graph/stairs.json count mismatch: {len(graph_stairs)} "
            f"!= {len(entity_stairs)}")
    import numpy as np

    for index, (graph_stair, entity_stair) in enumerate(
            zip(graph_stairs, entity_stairs)):
        graph_vertices = np.asarray(graph_stair["rect"], dtype=float)
        entity_vertices = np.asarray(entity_stair["vertices"], dtype=float)
        if graph_vertices.shape != (4, 3) or not np.allclose(
                graph_vertices, entity_vertices, atol=1e-9, rtol=0):
            raise ValueError(f"stair entity mirror mismatch at index {index}")


def augment(base, graph, entity_payload, source_path, source_hash,
            graph_path, graph_hash):
    artifact = copy.deepcopy(base)
    artifact["artifact_provenance"]["source_artifact"] = os.path.relpath(
        source_path, ROOT)
    artifact["artifact_provenance"]["source_sha256"] = source_hash
    artifact["method"]["baseline_id"] = "annotated_best_stairs_v0_1"
    layers = artifact["method"].setdefault("annotation_layers", {})
    layers["stairs"] = {
        "variant": "stage4_d5_existing_outputs",
        "status": "local_best_variant_with_direct_prediction_links",
        "component_merge_distance_m": 0.4,
        "minimum_rectangle_area_m2": 0.3,
        "minimum_rise_m": 0.4,
        "room_assignment_reject_distance_m": 0.5,
        "geometry": "horizontal OBB; storage endpoints 01/23 reordered to perimeter 0132",
        "source_scene_graph": os.path.relpath(graph_path, ROOT),
        "source_scene_graph_sha256": graph_hash,
    }
    artifact["limitations"]["stairs_included"] = True
    artifact["limitations"]["connectivity"] = (
        artifact["limitations"]["connectivity"] +
        "; stairs preserve direct Stage-4 D.5 prediction associations; "
        "released GT has no link labels")

    graph_stairs = graph.get("stairs", [])
    entity_stairs = entity_payload.get("stairs", [])
    _validate_entity_mirror(graph_stairs, entity_stairs)
    levels = {int(level["id"]): level for level in artifact["levels"]}
    level_z = {level_id: float(level["elevation"])
               for level_id, level in levels.items()}
    room_ids = {
        level_id: {room["id"] for room in level["rooms"]}
        for level_id, level in levels.items()
    }
    for level in levels.values():
        level["stairs"] = []
        level["graph"]["edges"] = [
            edge for edge in level["graph"]["edges"]
            if edge["kind"] != "stair"]

    diagnostics = {
        "scene": artifact["scene_id"],
        "source": "existing Stage-4 D.5 outputs",
        "gt_accessed": False,
        "assumptions": layers["stairs"],
        "records": [],
    }
    for index, stair in enumerate(graph_stairs):
        link = normalize_stage4_link(stair, level_z)
        from_level = link["from_level"]
        to_level = link["to_level"]
        adjacent_rooms = link["adjacent_rooms"]
        if adjacent_rooms[0] not in room_ids[from_level]:
            raise ValueError(
                f"stair {index} missing from-room L{from_level}/"
                f"{adjacent_rooms[0]}")
        if adjacent_rooms[1] not in room_ids[to_level]:
            raise ValueError(
                f"stair {index} missing to-room L{to_level}/"
                f"{adjacent_rooms[1]}")
        geometry = stair_rect_footprint(stair["rect"])
        stair_id = f"S{index:04d}"
        record = {
            "id": stair_id,
            "geometry": to_json_geometry(geometry),
            "from_level": from_level,
            "to_level": to_level,
            "adjacent_rooms": adjacent_rooms,
            "confidence": None,
            "provenance": (
                "stage4_d5_nearby_merge_obb_direct_room_link_local_best_variant"),
        }
        levels[from_level]["stairs"].append(record)
        levels[from_level]["graph"]["edges"].append({
            # v0.2 retained this historical field name for all edge kinds.
            "door_id": stair_id,
            "rooms": adjacent_rooms,
            "kind": "stair",
            "status": "direct_stage4_d5_prediction_association",
        })
        levels[from_level]["graph"]["edge_status"] = (
            "mixed_door_and_direct_stair_prediction_associations")
        diagnostics["records"].append({
            "stair_id": stair_id,
            "raw_rooms": stair["rooms"],
            **link,
            "footprint_area_m2": float(geometry.area),
            "rectangle_storage_order": "endpoint_edges_01_and_23",
            "polygon_perimeter_order": [0, 1, 3, 2],
        })
    diagnostics["summary"] = {
        "predicted_stairs": len(graph_stairs),
        "complete_level_links": sum(
            row["to_level"] is not None for row in diagnostics["records"]),
        "complete_room_links": sum(
            len(row["adjacent_rooms"]) == 2 for row in diagnostics["records"]),
        "non_adjacent_level_index_links": sum(
            abs(row["to_level"] - row["from_level"]) != 1
            for row in diagnostics["records"]),
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
    output = args.out_root / args.partition / "annotated_best"
    output.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    base_manifest_path = args.base_canonical / "collection_manifest.json"
    base_manifest = json.load(base_manifest_path.open())
    files, diagnostics_files = {}, {}
    source_hashes, stage4_hashes = {}, {}

    for scene in scenes:
        source_path = args.base_canonical / f"{scene}.json"
        paths = stage4_paths(scene)
        missing = [str(path) for path in (source_path, *paths.values())
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing {missing}")
        expected = base_manifest["files_sha256"].get(source_path.name)
        source_hash = sha256(source_path)
        if expected != source_hash:
            raise RuntimeError(f"{scene}: frozen base canonical hash mismatch")
        graph_hash = sha256(paths["scene_graph"])
        base = json.load(source_path.open())
        graph = json.load(paths["scene_graph"].open())
        entity_payload = json.load(paths["stair_entities"].open())
        artifact, diagnostics = augment(
            base, graph, entity_payload, source_path, source_hash,
            paths["scene_graph"], graph_hash)
        validator.validate(artifact)
        artifact_path = output / f"{scene}.json"
        diagnostics_path = output / f"{scene}.diagnostics.json"
        artifact_path.write_text(json.dumps(
            artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        diagnostics_path.write_text(json.dumps(
            diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        files[artifact_path.name] = sha256(artifact_path)
        diagnostics_files[diagnostics_path.name] = sha256(diagnostics_path)
        source_hashes[os.path.relpath(source_path, ROOT)] = source_hash
        stage4_hashes[scene] = {
            os.path.relpath(path, ROOT): sha256(path)
            for path in paths.values()
        }
        print(f"{scene}: stairs={diagnostics['summary']['predicted_stairs']} "
              f"links={diagnostics['summary']['complete_level_links']}")

    builder_paths = [Path(__file__), HERE / "stairs2d.py", SCHEMA, args.split]
    manifest = {
        "collection": "canonical_annotated_floorplan_v0.2_stairs2d",
        "schema_version": "annotated_floorplan_v0.2",
        "variant": "annotated_best",
        "partition": args.partition,
        "checkpoint_name": args.checkpoint_name,
        "scene_ids": scenes,
        "n_scenes": len(scenes),
        "base_collection_manifest_sha256": sha256(base_manifest_path),
        "base_canonical_files_sha256": source_hashes,
        "stage4_stair_inputs_sha256": stage4_hashes,
        "builder_files_sha256": {
            os.path.relpath(path, ROOT): sha256(path)
            for path in builder_paths
        },
        "files_sha256": files,
        "diagnostics_files_sha256": diagnostics_files,
        "execution": (
            "CPU-only conversion of existing Stage-4 D.5 outputs; no GT or "
            "score access; no stair detection, Stage 2/3, room segmentation, "
            "extrusion, pipeline, or GPU rerun"),
    }
    (output / "collection_manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"stairs2d canonical checkpoint -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
