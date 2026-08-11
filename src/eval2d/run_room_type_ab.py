"""Generate same-base Room-type A/B from retained OpenSeg feature samples.

No OpenSeg inference or GPU work is performed.  Both variants use identical
canonical room geometry, feature samples, 15 CLIP text embeddings, and
argmax-cosine classification.  They differ only in whether room features are
averaged directly from assigned samples or first projected to structural mesh
vertices using k=5 KNN as stated by Appendix D.4.
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

from room_types import (KNN_K, ROOM_TYPES, aggregate_mesh_vertices,  # noqa: E402
                        aggregate_sample_points,
                        annotate_pruning_candidates,
                        classify_room_features, read_ply_vertices)
from split_io import load_split, select_scenes  # noqa: E402


SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
BASE_CANONICAL = (ROOT / "outputs" / "eval2d" / "experiments" /
                  "stairs2d_v0_1" / "all" / "annotated_best")
DEFAULT_OUT = (ROOT / "outputs" / "eval2d" / "experiments" /
               "room_types_ab_v0_1")
TEXT_EMBEDDINGS = ROOT / ".cache" / "models" / "room_text_emb.npy"
VARIANTS = ("sample_point_room_mean", "mesh_vertex_k5_room_mean")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(scene):
    root = ROOT / "outputs" / "mp3d" / scene
    return {
        "openseg_points": root / "stage4" / "openseg_points.npy",
        "openseg_features": root / "stage4" / "openseg_feats.npy",
        "structural_mesh": root / "skeleton" /
        "ceiling_wall_floor_mesh.ply",
    }


def augment(base, annotations, records, aggregation, variant,
            source_path, source_hash):
    artifact = copy.deepcopy(base)
    artifact["artifact_provenance"]["source_artifact"] = os.path.relpath(
        source_path, ROOT)
    artifact["artifact_provenance"]["source_sha256"] = source_hash
    artifact["method"]["baseline_id"] = f"annotated_best_{variant}_v0_1"
    layers = artifact["method"].setdefault("annotation_layers", {})
    layers["room_types"] = {
        "variant": variant,
        "classes": ROOM_TYPES,
        "classification": "15-way CLIP cosine argmax",
        "full_scores_preserved": True,
        "feature_projection": aggregation["feature_projection"],
        "knn_k": aggregation["knn_k"],
        "status": (
            "paper_spec_vertex_aggregation_reproducible_interpretation"
            if variant == "mesh_vertex_k5_room_mean"
            else "local_direct_sample_point_ablation"),
        "outdoor_policy": (
            "paper last-five + graph degree<=1 recorded as pruning candidate; "
            "room geometry retained"),
    }
    artifact["limitations"]["room_types_reliable"] = False

    room_lookup = {
        f"L{level['id']}_R{room['id']}": room
        for level in artifact["levels"] for room in level["rooms"]
    }
    if set(room_lookup) != set(annotations):
        raise ValueError("classification/canonical room key mismatch")
    rows = []
    mean_distances = aggregation.get(
        "mean_nearest_sample_distance_m_by_room")
    for index, record in enumerate(records):
        key = record["key"]
        annotation = annotations[key]
        room = room_lookup[key]
        room["type_raw"] = annotation["type"]
        room["type_scores"] = annotation["scores"]
        room["type_confidence"] = annotation["confidence"]
        row = {
            "room_key": key,
            "level": record["level"],
            "room_id": record["room_id"],
            **annotation,
        }
        if mean_distances is not None:
            row["mean_nearest_sample_distance_m"] = mean_distances[index]
        rows.append(row)
    diagnostics = {
        "scene": artifact["scene_id"],
        "variant": variant,
        "gt_accessed": False,
        "aggregation": aggregation,
        "rooms": rows,
        "summary": {
            "rooms": len(rows),
            "classified": sum(row["status"] == "classified_argmax_cosine"
                              for row in rows),
            "unknown_no_features": sum(row["status"] ==
                                       "no_assigned_features" for row in rows),
            "outdoor_candidates": sum(row["is_outdoor_candidate"]
                                      for row in rows),
            "paper_pruning_candidates_retained": sum(
                row["is_paper_pruning_candidate"] for row in rows),
        },
    }
    return artifact, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("dev", "held_out", "all"),
                        default="dev")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--base-canonical", type=Path, default=BASE_CANONICAL)
    parser.add_argument("--text-embeddings", type=Path,
                        default=TEXT_EMBEDDINGS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, args.partition, args.checkpoint_name)
    output_root = args.out_root / args.partition
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    base_manifest_path = args.base_canonical / "collection_manifest.json"
    base_manifest = json.load(base_manifest_path.open())
    text = np.load(args.text_embeddings).astype(np.float32)
    files = {variant: {} for variant in VARIANTS}
    diagnostics_files = {variant: {} for variant in VARIANTS}
    source_hashes, input_hashes = {}, {}

    for scene_index, scene in enumerate(scenes, 1):
        source_path = args.base_canonical / f"{scene}.json"
        paths = input_paths(scene)
        missing = [str(path) for path in (source_path, *paths.values(),
                                          args.text_embeddings)
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing {missing}")
        source_hash = sha256(source_path)
        if base_manifest["files_sha256"].get(source_path.name) != source_hash:
            raise RuntimeError(f"{scene}: frozen base canonical hash mismatch")
        source_hashes[os.path.relpath(source_path, ROOT)] = source_hash
        input_hashes[scene] = {
            os.path.relpath(path, ROOT): sha256(path)
            for path in paths.values()
        }
        base = json.load(source_path.open())
        points = np.load(paths["openseg_points"], mmap_mode="r")
        features = np.load(paths["openseg_features"], mmap_mode="r")
        if len(points) != len(features) or features.shape[1] != text.shape[1]:
            raise ValueError(f"{scene}: OpenSeg point/feature shape mismatch")
        print(f"[{scene_index}/{len(scenes)}] {scene}: "
              f"samples={len(points)}", flush=True)

        sums, counts, records, aggregation = aggregate_sample_points(
            points, features, base)
        sample_annotations = annotate_pruning_candidates(
            classify_room_features(sums, counts, records, text),
            records, base)

        vertices = read_ply_vertices(paths["structural_mesh"])
        print(f"  structural vertices={len(vertices)}; project k={KNN_K}",
              flush=True)
        sums, counts, vertex_records, vertex_aggregation = \
            aggregate_mesh_vertices(
                points, features, vertices, base, k=KNN_K)
        if [record["key"] for record in records] != [
                record["key"] for record in vertex_records]:
            raise ValueError(f"{scene}: room ordering changed between variants")
        vertex_annotations = annotate_pruning_candidates(
            classify_room_features(sums, counts, vertex_records, text),
            vertex_records, base)

        for variant, annotations, variant_records, variant_aggregation in (
            ("sample_point_room_mean", sample_annotations, records,
             aggregation),
            ("mesh_vertex_k5_room_mean", vertex_annotations, vertex_records,
             vertex_aggregation),
        ):
            artifact, diagnostics = augment(
                base, annotations, variant_records, variant_aggregation,
                variant, source_path, source_hash)
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
            diagnostics_files[variant][diagnostics_path.name] = sha256(
                diagnostics_path)
            print(f"  {variant}: classified="
                  f"{diagnostics['summary']['classified']}/"
                  f"{diagnostics['summary']['rooms']} prune-candidates="
                  f"{diagnostics['summary']['paper_pruning_candidates_retained']}",
                  flush=True)
        del vertices, points, features

    builder_paths = [Path(__file__), HERE / "room_types.py", SCHEMA,
                     args.split]
    for variant in VARIANTS:
        out_dir = output_root / variant
        manifest = {
            "collection": "canonical_annotated_floorplan_v0.2_room_type_ab",
            "schema_version": "annotated_floorplan_v0.2",
            "variant": variant,
            "partition": args.partition,
            "checkpoint_name": args.checkpoint_name,
            "scene_ids": scenes,
            "n_scenes": len(scenes),
            "same_input_pair": list(VARIANTS),
            "base_collection_manifest_sha256": sha256(base_manifest_path),
            "base_canonical_files_sha256": source_hashes,
            "room_type_inputs_sha256": input_hashes,
            "text_embeddings_sha256": sha256(args.text_embeddings),
            "builder_files_sha256": {
                os.path.relpath(path, ROOT): sha256(path)
                for path in builder_paths
            },
            "files_sha256": files[variant],
            "diagnostics_files_sha256": diagnostics_files[variant],
            "execution": (
                "CPU-only retained-feature aggregation; no OpenSeg/CLIP "
                "inference, Stage 2/3/4 rerun, room segmentation, pruning, "
                "pipeline, extrusion, or GPU"),
        }
        (out_dir / "collection_manifest.json").write_text(json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"room-type A/B -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
