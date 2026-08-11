"""Formal invariants for the 16-scene direct semantic door checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "direct_doors_ab_v0_1" / "all")
SOURCE = (ROOT / "outputs" / "eval2d" / "experiments" /
          "room_types_ab_v0_1" / "all" / "sample_point_room_mean")
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
EVAL = "eval2d_v3_strict_levels_official325_all"
VARIANTS = ("watershed_control", "direct_semantic_only", "fused_union")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def non_door_signature(artifact):
    return [{
        "id": level["id"], "elevation": level["elevation"],
        "footprint": level["footprint"], "rooms": level["rooms"],
        "walls": level["walls"], "windows": level["windows"],
        "stairs": level["stairs"],
    } for level in artifact["levels"]]


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
        and manifest["checkpoint_name"] == "phase3_direct_doors_ab_v0_1"
        and manifest["partition"] == "all"
        for manifest in manifests.values())
    checks["same canonical and semantic inputs"] = all(
        manifests[variant]["base_canonical_files_sha256"] ==
        manifests[VARIANTS[0]]["base_canonical_files_sha256"]
        and manifests[variant]["semantic_inputs_sha256"] ==
        manifests[VARIANTS[0]]["semantic_inputs_sha256"]
        for variant in VARIANTS[1:])
    checks["frozen source collection manifest verifies"] = all(
        manifest["base_collection_manifest_sha256"] ==
        sha256(source_manifest_path) for manifest in manifests.values())

    schema_ok = hashes_ok = diagnostics_hashes_ok = True
    lineage_ok = unchanged_ok = references_ok = provenance_ok = True
    gt_not_accessed = control_exact = True
    totals = {variant: {key: 0 for key in (
        "base_doors", "semantic_candidates", "semantic_added",
        "output_doors", "resolved_semantic_associations")}
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
            unchanged_ok &= non_door_signature(artifact) == \
                non_door_signature(source)
            gt_not_accessed &= (
                diagnostics["gt_accessed"] is False
                and artifact["method"]["annotation_layers"]["doors"]
                ["gt_accessed"] is False)
            if variant == "watershed_control":
                control_exact &= all(
                    level["doors"] == source_level["doors"]
                    and level["graph"] == source_level["graph"]
                    for level, source_level in zip(
                        artifact["levels"], source["levels"]))
            for level in artifact["levels"]:
                room_ids = {room["id"] for room in level["rooms"]}
                door_ids = {door["id"] for door in level["doors"]}
                edge_ids = {
                    edge["door_id"] for edge in level["graph"]["edges"]
                    if edge["kind"] != "stair"
                }
                references_ok &= edge_ids <= door_ids
                for door in level["doors"]:
                    for room_id in (door["room_a"], door["room_b"]):
                        references_ok &= (room_id is None or
                                          room_id == "OUTSIDE" or
                                          room_id in room_ids)
                    if "SD" in door["id"]:
                        provenance_ok &= door["provenance"] == (
                            "OneFormer_door_ray_wall_snap_local_extension")
            for key in totals[variant]:
                totals[variant][key] += diagnostics["summary"][key]

    checks["48 canonical artifacts pass v0.2 schema"] = schema_ok
    checks["all canonical and diagnostic hashes verify"] = (
        hashes_ok and diagnostics_hashes_ok)
    checks["artifact lineage resolves to frozen typed base"] = lineage_ok
    checks["non-door annotation layers are unchanged"] = unchanged_ok
    checks["control doors and graph are exact source copies"] = control_exact
    checks["all room/door/graph references resolve"] = references_ok
    checks["semantic door provenance is explicit"] = provenance_ok
    checks["generator and artifacts deny GT access"] = gt_not_accessed
    checks["formal door totals are stable"] = (
        totals["watershed_control"]["output_doors"] == 119
        and totals["direct_semantic_only"]["output_doors"] == 115
        and totals["fused_union"]["output_doors"] == 218
        and totals["fused_union"]["semantic_added"] == 99
        and totals["fused_union"]["resolved_semantic_associations"] == 94)

    summaries = {
        variant: json.load((BASE / variant / EVAL / "summary.json").open())
        for variant in VARIANTS
    }
    checks["formal all-16 Doors@0.5 counts are stable"] = (
        tuple(summaries["watershed_control"]["selected"]["metrics"]["B"]
              ["doors@0.5"][key] for key in ("tp", "fp", "fn")) ==
        (50, 69, 242)
        and tuple(summaries["fused_union"]["selected"]["metrics"]["B"]
                  ["doors@0.5"][key] for key in ("tp", "fp", "fn")) ==
        (82, 136, 210))
    checks["dev and held-out both improve without retuning"] = all(
        summaries["fused_union"][partition]["metrics"]["B"]["doors@0.5"]
        ["f1"] >
        summaries["watershed_control"][partition]["metrics"]["B"]
        ["doors@0.5"]["f1"]
        for partition in ("dev", "held_out"))

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
