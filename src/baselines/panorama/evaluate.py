#!/usr/bin/env python3
"""Per-view IoU, duplicate merging, and frozen strict-v3 evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from common import (
    BASE_OUT,
    GT_ROOT,
    ROOM_METRIC_EXCLUDE,
    json_ready,
    parse_house,
    scene_ids,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "src/eval2d"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from metrics import EVAL2D_VERSION, prf, score_scene  # noqa: E402


MODELS = ("dopnet", "horizonnet")
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
PREDICTION_ROOT = BASE_OUT / "predictions"
EVALUATION_ROOT = BASE_OUT / "evaluation"
SPLIT_PATH = ROOT / "configs/eval2d/split_v0_1.json"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def polygon_iou(left: Polygon, right: Polygon) -> float:
    if not left.intersects(right):
        return 0.0
    intersection = left.intersection(right).area
    union = left.area + right.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def load_view_predictions(model: str, scene: str) -> list:
    results = []
    for path in sorted((PREDICTION_ROOT / model / scene).glob("*.json")):
        payload = load_json(path)
        if payload.get("status") != "ok":
            continue
        polygon = Polygon(payload["polygon_world_xy_m"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda item: item.area)
        if polygon.geom_type != "Polygon" or polygon.is_empty or polygon.area <= 0:
            continue
        results.append({"payload": payload, "polygon": polygon})
    return results


def merge_level(candidates: list, threshold: float) -> list:
    """Connected IoU components; retain their deterministic agreement medoid."""
    count = len(candidates)
    dsu = DisjointSet(count)
    ious = np.eye(count, dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = polygon_iou(
                candidates[left]["polygon"], candidates[right]["polygon"])
            ious[left, right] = ious[right, left] = value
            if value >= threshold:
                dsu.union(left, right)
    components = defaultdict(list)
    for index in range(count):
        components[dsu.find(index)].append(index)

    merged = []
    for indices in sorted(components.values(), key=lambda group: min(group)):
        # The method checkpoints expose no calibrated layout confidence.
        # Agreement with other views is therefore the only non-GT confidence.
        scores = {
            index: float(np.mean([ious[index, other] for other in indices]))
            for index in indices
        }
        medoid = sorted(
            indices,
            key=lambda index: (
                -scores[index],
                candidates[index]["payload"]["pano_id"],
            ),
        )[0]
        merged.append({
            "medoid": candidates[medoid],
            "members": [candidates[index] for index in indices],
            "agreement_confidence": scores[medoid],
        })
    return merged


def build_scene_prediction(model: str, scene: str, threshold: float,
                           gt: dict) -> dict:
    candidates = load_view_predictions(model, scene)
    by_level = defaultdict(list)
    for candidate in candidates:
        by_level[candidate["payload"]["level"]].append(candidate)
    rooms = []
    merge_audit = []
    for level in sorted(by_level):
        level_candidates = sorted(
            by_level[level], key=lambda item: item["payload"]["pano_id"])
        for component_index, component in enumerate(
                merge_level(level_candidates, threshold)):
            medoid = component["medoid"]
            source_ids = sorted(
                item["payload"]["pano_id"] for item in component["members"])
            idx = f"panorama_L{level}_C{component_index:03d}"
            rooms.append({
                "idx": idx,
                "level": level,
                "poly": np.asarray(
                    medoid["polygon"].exterior.coords[:-1]).tolist(),
                "type": "N/A",
                "source_pano_ids": source_ids,
                "representative_pano_id": medoid["payload"]["pano_id"],
                "agreement_confidence": component["agreement_confidence"],
            })
            merge_audit.append({
                "idx": idx,
                "level": level,
                "component_size": len(source_ids),
                "source_pano_ids": source_ids,
                "representative_pano_id": medoid["payload"]["pano_id"],
                "agreement_confidence": component["agreement_confidence"],
            })
    return {
        "scene": scene,
        "rooms": rooms,
        "doors": [],
        "edges": [],
        "n_levels": len(gt["lvl_z"]),
        "level_z": dict(gt["lvl_z"]),
        "provenance": {
            "method": model,
            "duplicate_rule": (
                "connected components of pairwise polygon IoU >= threshold; "
                "keep highest mean-IoU agreement medoid"),
            "iou_threshold": threshold,
            "camera_poses": "official MP3D GT poses",
            "semantic_outputs": "not available",
        },
        "merge_audit": merge_audit,
    }


def aggregate_room(scores: dict, subset: list) -> dict:
    counts = [0, 0, 0]
    ious = []
    room_count_error = []
    for scene in subset:
        values = scores[scene]["A"].get("room", [0, 0, 0])
        counts = [counts[index] + values[index] for index in range(3)]
        ious.extend(scores[scene]["iou"])
        room_count_error.extend(scores[scene]["rc_err"])
    result = prf(*counts)
    result.update({
        "matched_iou_mean": float(np.mean(ious)) if ious else None,
        "matched_iou_n": len(ious),
        "room_count_error_mean": (
            float(np.mean(room_count_error)) if room_count_error else None),
    })
    return result


def per_view_iou(model: str, scenes: list) -> dict:
    rows = []
    successful = failed = eligible = 0
    by_room = defaultdict(list)
    for scene in scenes:
        parsed = parse_house(scene)
        pano_by_id = {item["pano_id"]: item for item in parsed["panoramas"]}
        paths = sorted((PREDICTION_ROOT / model / scene).glob("*.json"))
        for path in paths:
            payload = load_json(path)
            successful += int(payload.get("status") == "ok")
            failed += int(payload.get("status") != "ok")
            pano = pano_by_id[payload["pano_id"]]
            if (pano["region_id"] is None
                    or pano["region_label"] in ROOM_METRIC_EXCLUDE):
                continue
            eligible += 1
            row = {
                "model": model,
                "scene": scene,
                "pano_id": pano["pano_id"],
                "region_id": pano["region_id"],
                "level": pano["level"],
                "status": payload.get("status"),
                "iou": None,
            }
            if payload.get("status") == "ok":
                pred = Polygon(payload["polygon_world_xy_m"])
                gt = Polygon(parsed["floors"][pano["region_id"]]["points"])
                value = polygon_iou(pred, gt)
                row["iou"] = value
                by_room[(scene, pano["region_id"])].append(value)
            rows.append(row)
    values = [row["iou"] for row in rows if row["iou"] is not None]
    room_rows = []
    for (scene, region), room_values in sorted(by_room.items()):
        room_rows.append({
            "scene": scene,
            "region_id": region,
            "views": len(room_values),
            "mean_iou": float(np.mean(room_values)),
            "median_iou": float(np.median(room_values)),
            "best_iou": float(np.max(room_values)),
        })
    return {
        "rows": rows,
        "room_rows": room_rows,
        "summary": {
            "prediction_success": successful,
            "prediction_failed": failed,
            "gt_room_eligible_views": eligible,
            "scored_views": len(values),
            "covered_gt_rooms": len(room_rows),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "q25": float(np.quantile(values, 0.25)) if values else None,
            "q75": float(np.quantile(values, 0.75)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        },
    }


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()
    scenes = scene_ids()
    split = load_json(SPLIT_PATH)
    dev = split["dev"]
    held_out = split["held_out"]
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    all_summary = {
        "evaluator": EVAL2D_VERSION,
        "gt": "mp3d_house_floor_v0_1",
        "split": str(SPLIT_PATH),
        "selection_policy": (
            "maximize dev micro Room F1; then dev matched IoU; then prefer "
            "the lower threshold; held-out is not used for selection"),
        "thresholds": args.thresholds,
        "models": {},
    }
    for model in args.models:
        view_eval = per_view_iou(model, scenes)
        write_csv(EVALUATION_ROOT / model / "per_view_iou.csv", view_eval["rows"])
        write_csv(EVALUATION_ROOT / model / "per_room_iou.csv", view_eval["room_rows"])
        with (EVALUATION_ROOT / model / "per_view_iou_summary.json").open("w") as stream:
            json.dump(json_ready(view_eval["summary"]), stream, indent=2)

        threshold_rows = []
        scores_by_threshold = {}
        for threshold in args.thresholds:
            threshold_name = f"iou_{threshold:.2f}"
            prediction_dir = EVALUATION_ROOT / model / threshold_name / "pred"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            scores = {}
            for scene in scenes:
                with (GT_ROOT / f"{scene}.pkl").open("rb") as stream:
                    gt = pickle.load(stream)
                pred = build_scene_prediction(model, scene, threshold, gt)
                with (prediction_dir / f"{scene}.pkl").open("wb") as stream:
                    pickle.dump(pred, stream, protocol=4)
                scores[scene] = score_scene(gt, pred)
            with (EVALUATION_ROOT / model / threshold_name / "scores.pkl").open("wb") as stream:
                pickle.dump(scores, stream, protocol=4)
            scores_by_threshold[threshold] = scores
            row = {
                "threshold": threshold,
                "dev": aggregate_room(scores, dev),
                "held_out": aggregate_room(scores, held_out),
                "all": aggregate_room(scores, scenes),
            }
            threshold_rows.append(row)
            print(model, json.dumps(row), flush=True)

        selected = max(
            args.thresholds,
            key=lambda threshold: (
                aggregate_room(scores_by_threshold[threshold], dev)["f1"],
                aggregate_room(scores_by_threshold[threshold], dev)[
                    "matched_iou_mean"] or -1.0,
                -threshold,
            ),
        )
        selected_scores = scores_by_threshold[selected]
        selected_dir = EVALUATION_ROOT / model / f"iou_{selected:.2f}"
        model_summary = {
            "per_view": view_eval["summary"],
            "threshold_sweep": threshold_rows,
            "selected_threshold": selected,
            "selected_dev": aggregate_room(selected_scores, dev),
            "selected_held_out": aggregate_room(selected_scores, held_out),
            "selected_all": aggregate_room(selected_scores, scenes),
            "selected_scores": str(selected_dir / "scores.pkl"),
            "doors_types_connectivity": "N/A (models do not predict them)",
        }
        all_summary["models"][model] = model_summary
        with (EVALUATION_ROOT / model / "summary.json").open("w") as stream:
            json.dump(json_ready(model_summary), stream, indent=2)

    with (EVALUATION_ROOT / "summary.json").open("w") as stream:
        json.dump(json_ready(all_summary), stream, indent=2)
    print(json.dumps(json_ready(all_summary), indent=2))


if __name__ == "__main__":
    main()
