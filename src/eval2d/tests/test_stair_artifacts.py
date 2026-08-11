"""Formal invariants for the 16-scene stairs2d checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "stairs2d_v0_1" / "all" / "annotated_best")
SOURCE = (ROOT / "outputs" / "eval2d" / "experiments" /
          "window_rays_ab_v0_1" / "all" / "legacy_three_classes")
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
EVAL = "eval2d_stairs_v0_1_footprint_iou_hungarian_all"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    checks = {}
    split = json.load(SPLIT.open())
    scenes = sorted(split["dev"] + split["held_out"])
    manifest = json.load((BASE / "collection_manifest.json").open())
    source_manifest_path = SOURCE / "collection_manifest.json"
    source_manifest = json.load(source_manifest_path.open())
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    checks["formal split and checkpoint"] = (
        sorted(manifest["scene_ids"]) == scenes
        and manifest["checkpoint_name"] == "phase3_stairs2d_v0_1")
    checks["frozen base collection manifest verifies"] = (
        manifest["base_collection_manifest_sha256"] ==
        sha256(source_manifest_path))

    schema_ok = hashes_ok = diagnostics_hashes_ok = True
    references_ok = graph_ok = provenance_ok = lineage_ok = True
    total_stairs = total_windows = total_non_adjacent = 0
    for scene in scenes:
        path = BASE / f"{scene}.json"
        diagnostics_path = BASE / f"{scene}.diagnostics.json"
        artifact = json.load(path.open())
        diagnostics = json.load(diagnostics_path.open())
        try:
            validator.validate(artifact)
        except Exception:
            schema_ok = False
        hashes_ok &= manifest["files_sha256"].get(path.name) == sha256(path)
        diagnostics_hashes_ok &= (
            manifest["diagnostics_files_sha256"].get(diagnostics_path.name)
            == sha256(diagnostics_path))
        source_path = SOURCE / f"{scene}.json"
        source_hash = sha256(source_path)
        lineage_ok &= (
            artifact["artifact_provenance"]["source_artifact"] ==
            str(source_path.relative_to(ROOT))
            and artifact["artifact_provenance"]["source_sha256"] == source_hash
            and source_manifest["files_sha256"].get(source_path.name) == source_hash)
        levels = {int(level["id"]): level for level in artifact["levels"]}
        room_ids = {
            level_id: {room["id"] for room in level["rooms"]}
            for level_id, level in levels.items()
        }
        for from_level, level in levels.items():
            total_windows += len(level["windows"])
            stair_edges = {
                edge["door_id"]: edge for edge in level["graph"]["edges"]
                if edge["kind"] == "stair"
            }
            for stair in level["stairs"]:
                total_stairs += 1
                to_level = stair["to_level"]
                rooms = stair["adjacent_rooms"]
                references_ok &= (
                    stair["from_level"] == from_level
                    and to_level in levels
                    and len(rooms) == 2
                    and rooms[0] in room_ids[from_level]
                    and rooms[1] in room_ids[to_level])
                edge = stair_edges.get(stair["id"])
                graph_ok &= (
                    edge is not None
                    and edge["rooms"] == rooms
                    and edge["status"] ==
                    "direct_stage4_d5_prediction_association")
                provenance_ok &= stair["provenance"].endswith(
                    "local_best_variant")
        total_non_adjacent += diagnostics["summary"][
            "non_adjacent_level_index_links"]
        provenance_ok &= (
            artifact["limitations"]["stairs_included"] is True
            and artifact["limitations"]["windows_included"] is True
            and diagnostics["gt_accessed"] is False)

    checks["16 canonical artifacts pass v0.2 schema"] = schema_ok
    checks["all artifact hashes verify"] = hashes_ok
    checks["all diagnostic hashes verify"] = diagnostics_hashes_ok
    checks["artifact provenance points to frozen window+room base"] = lineage_ok
    checks["all ordered level/room references resolve"] = references_ok
    checks["stair graph edges exactly mirror direct links"] = graph_ok
    checks["capabilities and local provenance explicit"] = provenance_ok
    checks["12 existing D.5 predictions retained"] = total_stairs == 12
    checks["existing windows survive stair augmentation"] = total_windows == 637
    checks["one non-adjacent predicted-level link audited"] = (
        total_non_adjacent == 1)

    summary = json.load((BASE / EVAL / "summary.json").open())
    metric = summary["selected"]["metrics"]["stair_footprint_iou>0.5"]
    checks["formal evaluator retains all 34 released entities"] = (
        summary["selected"]["gt_stair_entities"] == 34)
    checks["formal footprint counts are stable"] = (
        (metric["tp"], metric["fp"], metric["fn"]) == (5, 7, 29))
    checks["Stair-link remains explicitly N/A"] = (
        summary["stair_link_metric"] is None
        and summary["selected"]["stair_link_metric"] is None)

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
