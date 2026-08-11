"""Formal invariants for the 16-scene Room-type A/B checkpoint."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "room_types_ab_v0_1" / "all")
SOURCE = (ROOT / "outputs" / "eval2d" / "experiments" /
          "stairs2d_v0_1" / "all" / "annotated_best")
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
CROSSWALK = ROOT / "configs" / "eval2d" / "room_type_crosswalk_v0_1.json"
EVAL = "eval2d_room_types_v0_1_explicit_crosswalk_all"
VARIANTS = ("sample_point_room_mean", "mesh_vertex_k5_room_mean")
ROOM_TYPES = (
    "bathroom", "bedroom", "living room", "garage", "entrance",
    "kitchen", "office", "stairs", "gym", "classroom", "spa/sauna",
    "mirror", "grass/bushes/trees", "driveway",
    "veranda/terrace/balcony",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entity_signature(artifact):
    """Fields that a room-type-only augmentation must not alter."""
    signature = []
    for level in artifact["levels"]:
        signature.append({
            "id": level["id"],
            "elevation": level["elevation"],
            "rooms": [
                {key: value for key, value in room.items()
                 if key not in {"type_raw", "type_scores", "type_confidence"}}
                for room in level["rooms"]
            ],
            "walls": level["walls"],
            "doors": level["doors"],
            "windows": level["windows"],
            "stairs": level["stairs"],
            "graph": level["graph"],
        })
    return signature


def main():
    checks = {}
    split = json.load(SPLIT.open())
    scenes = sorted(split["dev"] + split["held_out"])
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    source_manifest_path = SOURCE / "collection_manifest.json"
    source_manifest = json.load(source_manifest_path.open())
    manifests = {
        variant: json.load((BASE / variant / "collection_manifest.json").open())
        for variant in VARIANTS
    }

    checks["formal split and named checkpoint"] = all(
        sorted(manifest["scene_ids"]) == scenes
        and manifest["checkpoint_name"] == "phase3_room_types_ab_v0_1"
        and manifest["partition"] == "all"
        for manifest in manifests.values())
    checks["same base and retained-feature hashes"] = (
        manifests[VARIANTS[0]]["base_canonical_files_sha256"] ==
        manifests[VARIANTS[1]]["base_canonical_files_sha256"]
        and manifests[VARIANTS[0]]["room_type_inputs_sha256"] ==
        manifests[VARIANTS[1]]["room_type_inputs_sha256"]
        and manifests[VARIANTS[0]]["text_embeddings_sha256"] ==
        manifests[VARIANTS[1]]["text_embeddings_sha256"])
    checks["frozen source collection manifest verifies"] = all(
        manifest["base_collection_manifest_sha256"] ==
        sha256(source_manifest_path) for manifest in manifests.values())

    schema_ok = hashes_ok = diagnostics_hashes_ok = True
    lineage_ok = unchanged_ok = score_contract_ok = candidate_retained = True
    gt_not_accessed = provenance_ok = True
    totals = {variant: {"rooms": 0, "unknown": 0, "candidates": 0}
              for variant in VARIANTS}
    for scene in scenes:
        source_path = SOURCE / f"{scene}.json"
        source = json.load(source_path.open())
        source_hash = sha256(source_path)
        for variant in VARIANTS:
            directory = BASE / variant
            path = directory / f"{scene}.json"
            diagnostics_path = directory / f"{scene}.diagnostics.json"
            artifact = json.load(path.open())
            diagnostics = json.load(diagnostics_path.open())
            manifest = manifests[variant]
            try:
                validator.validate(artifact)
            except Exception:
                schema_ok = False
            hashes_ok &= manifest["files_sha256"].get(path.name) == sha256(path)
            diagnostics_hashes_ok &= (
                manifest["diagnostics_files_sha256"].get(diagnostics_path.name)
                == sha256(diagnostics_path))
            lineage_ok &= (
                artifact["artifact_provenance"]["source_artifact"] ==
                str(source_path.relative_to(ROOT))
                and artifact["artifact_provenance"]["source_sha256"] ==
                source_hash
                and source_manifest["files_sha256"].get(source_path.name) ==
                source_hash)
            unchanged_ok &= entity_signature(artifact) == entity_signature(source)
            gt_not_accessed &= diagnostics["gt_accessed"] is False
            provenance_ok &= (
                artifact["limitations"]["room_types_reliable"] is False
                and artifact["method"]["annotation_layers"]["room_types"]
                ["variant"] == variant)
            candidate_keys = {
                row["room_key"] for row in diagnostics["rooms"]
                if row["is_paper_pruning_candidate"]
            }
            artifact_keys = {
                f"L{level['id']}_R{room['id']}"
                for level in artifact["levels"] for room in level["rooms"]
            }
            candidate_retained &= candidate_keys <= artifact_keys
            for level in artifact["levels"]:
                for room in level["rooms"]:
                    totals[variant]["rooms"] += 1
                    if room["type_raw"] == "unknown":
                        totals[variant]["unknown"] += 1
                        score_contract_ok &= (
                            room["type_scores"] is None
                            and room["type_confidence"] is None)
                    else:
                        score_contract_ok &= (
                            set(room["type_scores"]) == set(ROOM_TYPES)
                            and all(math.isfinite(value)
                                    for value in room["type_scores"].values())
                            and room["type_raw"] in ROOM_TYPES
                            and math.isfinite(room["type_confidence"]))
            totals[variant]["candidates"] += len(candidate_keys)

    checks["32 canonical artifacts pass v0.2 schema"] = schema_ok
    checks["all canonical and diagnostic hashes verify"] = (
        hashes_ok and diagnostics_hashes_ok)
    checks["artifact lineage resolves to frozen stairs base"] = lineage_ok
    checks["room geometry and other annotation layers unchanged"] = unchanged_ok
    checks["classified rooms preserve all 15 finite scores"] = score_contract_ok
    checks["paper pruning candidates are retained"] = candidate_retained
    checks["generator diagnostics explicitly deny GT access"] = gt_not_accessed
    checks["variant provenance and unreliable flag explicit"] = provenance_ok
    checks["both variants preserve all 316 predicted rooms"] = all(
        totals[variant]["rooms"] == 316 for variant in VARIANTS)
    checks["feature coverage is explicit"] = (
        totals["sample_point_room_mean"]["unknown"] == 1
        and totals["mesh_vertex_k5_room_mean"]["unknown"] == 0)
    checks["paper candidate counts are stable"] = (
        totals["sample_point_room_mean"]["candidates"] == 45
        and totals["mesh_vertex_k5_room_mean"]["candidates"] == 41)

    summaries = {
        variant: json.load((BASE / variant / EVAL / "summary.json").open())
        for variant in VARIANTS
    }
    checks["crosswalk denominator is frozen at 221/325"] = all(
        summary["selected"]["gt_rooms"] == 325
        and summary["selected"]["gt_mappable"] == 221
        and summary["selected"]["matched_mappable"] == 139
        for summary in summaries.values())
    sample = summaries["sample_point_room_mean"]["selected"]
    vertex = summaries["mesh_vertex_k5_room_mean"]["selected"]
    checks["formal all-room type counts are stable"] = (
        tuple(sample["room+type_all_rooms"][key]
              for key in ("tp", "fp", "fn")) == (60, 256, 265)
        and tuple(vertex["room+type_all_rooms"][key]
                  for key in ("tp", "fp", "fn")) == (53, 263, 272))
    checks["same geometry matching counts"] = (
        sample["room"] == vertex["room"]
        and tuple(sample["room"][key] for key in ("tp", "fp", "fn")) ==
        (192, 124, 133))

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
