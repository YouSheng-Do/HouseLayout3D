"""Formal checkpoint invariants for the same-input D.3 A/B artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from split_io import load_split  # noqa: E402


BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "paper_two_stage_ab_v0_1" / "all")
METHODS = ("paper_spec_two_stage", "watershed_v3_rerun")
EVAL = "eval2d_v3_strict_levels_official325_all"
SPLIT = ROOT / "configs" / "eval2d" / "split_v0_1.json"
SCHEMA = ROOT / "configs" / "eval2d" / "annotated_floorplan_v0_2.schema.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    checks = {}
    failures = []

    def check(name, condition, detail=""):
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            failures.append(f"{name}: {detail}")
        print(f"  {'PASS' if passed else 'FAIL'} {name}")

    split = load_split(SPLIT)
    scenes = sorted(split["dev"] + split["held_out"])
    validator = Draft202012Validator(json.load(SCHEMA.open()))
    manifests = {
        method: json.load((BASE / method / "collection_manifest.json").open())
        for method in METHODS
    }
    check("both manifests contain exactly frozen 16-scene split",
          all(sorted(value["scene_ids"]) == scenes and value["n_scenes"] == 16
              for value in manifests.values()))
    check("both methods reference identical shared input manifests",
          manifests[METHODS[0]]["shared_input_manifests_sha256"] ==
          manifests[METHODS[1]]["shared_input_manifests_sha256"])
    check("formal named checkpoint recorded",
          all(value["checkpoint_name"] == "phase2_paper_two_stage_ab_v0_1"
              for value in manifests.values()))

    all_valid = True
    files_hash = True
    provenance = True
    same_levels = True
    unknown_types = True
    paper_door_rule = True
    for scene in scenes:
        artifacts = {}
        for method in METHODS:
            path = BASE / method / f"{scene}.json"
            artifact = json.load(path.open())
            artifacts[method] = artifact
            try:
                validator.validate(artifact)
            except Exception as error:  # detailed failure retained below
                failures.append(f"{scene}:{method}:schema:{error}")
                all_valid = False
            provenance &= artifact["artifact_provenance"]["segmentation_rerun"] is True
            provenance &= artifact["artifact_provenance"]["source_artifact"].endswith(
                f"/all/shared_input_manifests/{scene}.json")
            unknown_types &= all(
                room["type_raw"] == "unknown"
                for level in artifact["levels"] for room in level["rooms"])
            expected_hash = manifests[method]["files_sha256"].get(path.name)
            files_hash &= expected_hash == sha256(path)

        paper_levels = [(level["id"], level["elevation"])
                        for level in artifacts[METHODS[0]]["levels"]]
        water_levels = [(level["id"], level["elevation"])
                        for level in artifacts[METHODS[1]]["levels"]]
        same_levels &= paper_levels == water_levels

        diagnostics = json.load(
            (BASE / "paper_spec_two_stage" /
             f"{scene}.diagnostics.json").open())
        for level in diagnostics["levels"]:
            for opening in level["bottlenecks"]:
                paper_door_rule &= (
                    opening["is_door"] == (opening["width_m"] < 1.5))
                paper_door_rule &= opening["bottleneck_stage"] in {"2.5m", "1.5m"}

    check("32 canonical artifacts pass v0.2 schema", all_valid)
    check("manifest hashes verify all canonical files", files_hash)
    check("new-method provenance says segmentation_rerun=true", provenance)
    check("paired methods preserve identical predicted levels/elevations", same_levels)
    check("room type disabled symmetrically", unknown_types)
    check("paper diagnostics obey <1.5m door rule", paper_door_rule)

    evaluator_ok = True
    for method in METHODS:
        summary = json.load((BASE / method / EVAL / "summary.json").open())
        with (BASE / method / EVAL / "scores.pkl").open("rb") as stream:
            scores = pickle.load(stream)
        evaluator_ok &= summary["checkpoint_name"] == "phase2_paper_two_stage_ab_v0_1"
        evaluator_ok &= summary["partition"] == "all"
        evaluator_ok &= sorted(scores) == scenes
        evaluator_ok &= np.isfinite(summary["selected"]["metrics"]["A"]["room"]["f1"])
    check("both artifact-only evaluations cover formal all checkpoint", evaluator_ok)

    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    if failures:
        for failure in failures:
            print("  ", failure)
    return 0 if all(checks.values()) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
