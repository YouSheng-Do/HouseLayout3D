"""Formal invariants for the 16-scene window ray-class checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "window_rays_ab_v0_1" / "all")
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"
SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
EVAL = "eval2d_windows_v0_2_endpoint_hungarian_all"
VARIANTS = ("legacy_three_classes", "paper_plus_outdoor")


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
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    manifests = {
        variant: json.load((BASE / variant / "collection_manifest.json").open())
        for variant in VARIANTS
    }
    checks["formal split and checkpoint"] = all(
        sorted(manifest["scene_ids"]) == scenes
        and manifest["checkpoint_name"] == "phase3_window_rays_ab_v0_1"
        for manifest in manifests.values())
    checks["same Stage-4 input hashes"] = (
        manifests[VARIANTS[0]]["stage4_inputs_sha256"] ==
        manifests[VARIANTS[1]]["stage4_inputs_sha256"])
    checks["same frozen room artifact hashes"] = (
        manifests[VARIANTS[0]]["base_canonical_files_sha256"] ==
        manifests[VARIANTS[1]]["base_canonical_files_sha256"])
    checks["only paper class set adds outdoor"] = (
        manifests[VARIANTS[0]]["ray_classes"] ==
        ["window", "window_blind", "curtain"]
        and manifests[VARIANTS[1]]["ray_classes"] ==
        ["window", "window_blind", "curtain", "outdoor"])

    schema_ok = hashes_ok = references_ok = provenance_ok = True
    for variant in VARIANTS:
        manifest = manifests[variant]
        for scene in scenes:
            path = BASE / variant / f"{scene}.json"
            artifact = json.load(path.open())
            try:
                validator.validate(artifact)
            except Exception:
                schema_ok = False
            hashes_ok &= manifest["files_sha256"].get(path.name) == sha256(path)
            for level in artifact["levels"]:
                room_ids = {room["id"] for room in level["rooms"]}
                wall_ids = {wall["id"] for wall in level["walls"]}
                for window in level["windows"]:
                    references_ok &= window["wall_id"] in wall_ids
                    references_ok &= (window["room_id"] is None
                                      or window["room_id"] in room_ids)
                    provenance_ok &= window["provenance"] == (
                        "paper_window_raycast_appendix_b_local_clustering")
            provenance_ok &= artifact["limitations"]["windows_included"] is True
    checks["32 artifacts validate against canonical v0.2"] = schema_ok
    checks["all canonical hashes verify"] = hashes_ok
    checks["window wall/room references resolve"] = references_ok
    checks["window provenance and coverage explicit"] = provenance_ok

    evaluation_ok = True
    totals = {}
    for variant in VARIANTS:
        summary = json.load((BASE / variant / EVAL / "summary.json").open())
        evaluation_ok &= summary["checkpoint_name"] == "phase3_window_rays_ab_v0_1"
        evaluation_ok &= summary["partition"] == "all"
        totals[variant] = summary["selected"]["gt_windows"]
    checks["formal artifact-only evaluations cover all"] = evaluation_ok
    checks["released GT total is stable and shared"] = (
        totals == {variant: 379 for variant in VARIANTS})

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
