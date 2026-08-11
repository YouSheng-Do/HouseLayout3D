"""Generate a same-input window-ray A/B on frozen watershed room artifacts.

The two variants differ only in ray classes:

* ``legacy_three_classes``: window, window_blind, curtain
* ``paper_plus_outdoor``: the same classes plus outdoor

Each scene's Stage-3 prototype is loaded once.  No GT, evaluator scores,
Stage 2/3 rerun, room segmentation, extrusion, or GPU is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE4 = ROOT / "src" / "stage4"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STAGE4))

from geometry_v2 import to_shapely  # noqa: E402
from load_prototype import load_prototype  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402
from windows import (  # noqa: E402
    LEGACY_WINDOW_RAY_CLASSES, WINDOW_RAY_CLASSES, detect_window_records,
)


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
BASE_CANONICAL = (ROOT / "outputs" / "eval2d" / "experiments" /
                  "paper_two_stage_ab_v0_1" / "all" /
                  "watershed_v3_rerun")
DEFAULT_OUT = (ROOT / "outputs" / "eval2d" / "experiments" /
               "window_rays_ab_v0_1")
VARIANTS = {
    "legacy_three_classes": LEGACY_WINDOW_RAY_CLASSES,
    "paper_plus_outdoor": WINDOW_RAY_CLASSES,
}
ROOM_ASSOC_MAX_M = 0.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(scene):
    base = ROOT / "outputs" / "mp3d" / scene
    skeleton = base / "skeleton"
    return {
        "fitted_mesh": base / "stage3" / "fit" / "fitted_mesh.ply",
        "semantic_skeleton": skeleton / "ceiling_wall_floor_mesh.ply",
        "coarse_probabilities": base / "stage3" / "coarse" / "cwf_classes.npy",
        "coarse_labels": base / "stage3" / "coarse" / "labels.npy",
        "ray_labels": skeleton / "hard_labels_simplified_segmentations.npy",
        "ray_label_names": skeleton / "simplified_segmentation_labels.npy",
        "ray_origins": skeleton / "full_ray_origins.npy",
        "ray_destinations": skeleton / "full_ray_dests.npy",
        "ray_validity": skeleton / "ray_is_valid.npy",
    }


def _jsonable(record):
    output = {}
    for key, value in record.items():
        output[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return output


def _level_for_record(record, levels):
    bottom_z = float(np.asarray(record["corners"])[:, 2].min())
    return min(levels, key=lambda level: (
        abs(float(level["elevation"]) - bottom_z), int(level["id"])))


def _wall_polyline(wall):
    points = np.asarray(wall.verts, dtype=np.float64)[:, :2]
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    direction = vh[0]
    positions = (points - centre) @ direction
    return np.array([
        centre + positions.min() * direction,
        centre + positions.max() * direction,
    ])


def _associate_room(level, segment):
    line = LineString(np.asarray(segment, dtype=np.float64))
    candidates = []
    for room in level["rooms"]:
        geometry = to_shapely(room["geometry"])
        candidates.append((float(geometry.distance(line)), str(room["id"]),
                           room["id"]))
    if not candidates:
        return None, None
    distance, _, room_id = min(candidates)
    return (room_id, distance) if distance <= ROOM_ASSOC_MAX_M else (None, distance)


def augment(base, records, proto, variant, source_path, source_hash):
    artifact = copy.deepcopy(base)
    artifact["artifact_provenance"]["source_artifact"] = os.path.relpath(
        source_path, ROOT)
    artifact["artifact_provenance"]["source_sha256"] = source_hash
    artifact["method"]["baseline_id"] = f"window_rays_{variant}_v0_1"
    artifact["method"]["annotation_layers"] = {
        "windows": {
            "variant": variant,
            "ray_classes": list(VARIANTS[variant]),
            "room_association": f"nearest room geometry <= {ROOM_ASSOC_MAX_M:.2f}m",
            "wall_association": "direct Stage-3 prototype wall PID",
            "status": ("paper_class_set_with_local_clustering_assumptions"
                       if variant == "paper_plus_outdoor"
                       else "legacy_local_class_set_control"),
        }
    }
    artifact["limitations"]["windows_included"] = True
    artifact["limitations"]["connectivity"] = (
        artifact["limitations"]["connectivity"] +
        "; window-room association nearest-geometry candidate")

    by_level = defaultdict(list)
    for record in records:
        level = _level_for_record(record, artifact["levels"])
        by_level[int(level["id"])].append(record)

    diagnostics = {
        "scene": artifact["scene_id"],
        "variant": variant,
        "ray_classes": list(VARIANTS[variant]),
        "room_association_max_m": ROOM_ASSOC_MAX_M,
        "records": [],
    }
    for level in artifact["levels"]:
        level_id = int(level["id"])
        level["windows"] = []
        # Current room artifacts do not yet carry complete walls.  Add every
        # referenced prototype wall as an explicit 2D polyline.
        existing_walls = {wall["id"]: wall for wall in level["walls"]}
        associated_by_wall = defaultdict(set)
        for index, record in enumerate(by_level.get(level_id, [])):
            segment = np.asarray(record["segment_2d"], dtype=np.float64)
            room_id, room_distance = _associate_room(level, segment)
            wall_id = f"L{level_id}_PW{record['wall_pid']}"
            if room_id is not None:
                associated_by_wall[wall_id].add(room_id)
            window_id = f"L{level_id}_W{index:04d}"
            level["windows"].append({
                "id": window_id,
                "segment": segment.tolist(),
                "wall_id": wall_id,
                "room_id": room_id,
                "confidence": float(record["confidence"]),
                "provenance": record["provenance"],
            })
            row = _jsonable(record)
            row.update({
                "window_id": window_id,
                "level": level_id,
                "wall_id": wall_id,
                "room_id": room_id,
                "room_distance_m": room_distance,
            })
            diagnostics["records"].append(row)

        for wall_id in sorted({window["wall_id"] for window in level["windows"]}):
            if wall_id in existing_walls:
                continue
            pid = int(wall_id.rsplit("PW", 1)[1])
            wall = next(candidate for candidate in proto.by_class("wall")
                        if candidate.pid == pid)
            level["walls"].append({
                "id": wall_id,
                "polyline": _wall_polyline(wall).tolist(),
                # v0.2 wall contract allows at most two adjacent rooms.  A
                # long prototype plane can span several rooms, so retain a
                # deterministic local candidate pair and disclose the source.
                "adjacent_rooms": sorted(
                    associated_by_wall[wall_id], key=str)[:2],
                "provenance": "stage3_prototype_wall_projection_for_window_link",
            })
        level["walls"].sort(key=lambda wall: str(wall["id"]))
    diagnostics["summary"] = {
        "windows": sum(len(level["windows"]) for level in artifact["levels"]),
        "referenced_walls": len({row["wall_id"] for row in diagnostics["records"]}),
        "room_associated": sum(row["room_id"] is not None
                               for row in diagnostics["records"]),
        "outdoor_supported_records": sum(
            row["ray_class_counts"].get("outdoor", 0) > 0
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
    partition_root = args.out_root / args.partition
    with SCHEMA.open() as stream:
        validator = Draft202012Validator(json.load(stream))
    files = {variant: {} for variant in VARIANTS}
    diagnostics_files = {variant: {} for variant in VARIANTS}
    source_hashes = {}
    input_hashes = {}

    for scene_index, scene in enumerate(scenes, 1):
        paths = input_paths(scene)
        missing = [str(path) for path in paths.values() if not path.exists()]
        source_path = args.base_canonical / f"{scene}.json"
        if not source_path.exists():
            missing.append(str(source_path))
        if missing:
            raise FileNotFoundError(f"{scene}: missing {missing}")
        source_hash = sha256(source_path)
        source_hashes[os.path.relpath(source_path, ROOT)] = source_hash
        input_hashes[scene] = {
            os.path.relpath(path, ROOT): sha256(path)
            for path in paths.values()
        }
        base = json.load(source_path.open())
        print(f"[{scene_index}/{len(scenes)}] load prototype once: {scene}",
              flush=True)
        proto = load_prototype(
            str(paths["fitted_mesh"]), str(paths["semantic_skeleton"]),
            str(paths["coarse_probabilities"]), str(paths["coarse_labels"]))

        for variant, ray_classes in VARIANTS.items():
            records = detect_window_records(
                proto, str(paths["ray_origins"].parent),
                ray_classes=ray_classes)
            artifact, diagnostics = augment(
                base, records, proto, variant, source_path, source_hash)
            validator.validate(artifact)
            out_dir = partition_root / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = out_dir / f"{scene}.json"
            diagnostics_path = out_dir / f"{scene}.diagnostics.json"
            artifact_path.write_text(json.dumps(
                artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            diagnostics_path.write_text(json.dumps(
                diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            files[variant][artifact_path.name] = sha256(artifact_path)
            diagnostics_files[variant][diagnostics_path.name] = sha256(
                diagnostics_path)
            print(f"  {variant}: windows={diagnostics['summary']['windows']} "
                  f"room-linked={diagnostics['summary']['room_associated']} "
                  f"outdoor-evidence={diagnostics['summary']['outdoor_supported_records']}",
                  flush=True)

    builder_paths = [Path(__file__), STAGE4 / "windows.py", SCHEMA, args.split]
    for variant in VARIANTS:
        out_dir = partition_root / variant
        manifest = {
            "collection": "canonical_annotated_floorplan_v0.2_window_ray_ab",
            "schema_version": "annotated_floorplan_v0.2",
            "variant": variant,
            "ray_classes": list(VARIANTS[variant]),
            "partition": args.partition,
            "checkpoint_name": args.checkpoint_name,
            "scene_ids": scenes,
            "n_scenes": len(scenes),
            "same_input_pair": list(VARIANTS),
            "base_canonical_files_sha256": source_hashes,
            "stage4_inputs_sha256": input_hashes,
            "builder_files_sha256": {
                os.path.relpath(path, ROOT): sha256(path)
                for path in builder_paths
            },
            "files_sha256": files[variant],
            "diagnostics_files_sha256": diagnostics_files[variant],
            "execution": (
                "CPU-only window annotation; prototype loaded once per scene; "
                "no GT, score access, Stage 2/3 rerun, room segmentation, "
                "extrusion, or GPU"),
        }
        (out_dir / "collection_manifest.json").write_text(json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"window ray A/B -> {partition_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
