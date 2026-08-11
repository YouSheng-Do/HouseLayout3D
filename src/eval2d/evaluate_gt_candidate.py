"""CPU-only frozen-prediction A/B for evaluator v3 and official MP3D GT.

Produces four auditable variants:

1. legacy v2 + raster-derived 324-room GT (existing frozen score),
2. strict v3 + the same raster-derived 324-room GT,
3. strict v3 + direct .house polygons on the exact same 324 room IDs,
4. strict v3 + direct .house polygons with the explicit 325-room roomset.

Variants 2->3 isolate geometry.  Variants 3->4 isolate the one valid official
closet that the raster/area path dropped.  No prediction or pipeline stage runs.
"""
import copy
import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import EVAL2D_VERSION, prf, score_scene  # noqa: E402

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
CANDIDATE = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
STRICT_OUT = f"{BASE}/eval2d_v3_strict_levels"
AB_OUT = f"{CANDIDATE}/eval_frozen_watershed_v3"

METRICS = [
    ("A", "room", "Room"),
    ("B", "room+type", "Room+type"),
    ("B", "doors@0.2", "Doors@0.2"),
    ("B", "doors@0.5", "Doors@0.5"),
    ("A", "corner@0.1", "Corner@0.1"),
    ("A", "corner@0.2", "Corner@0.2"),
    ("A", "corner@0.3", "Corner@0.3"),
    ("A", "angle@0.1", "Angle@0.1"),
    ("C", "room_room", "edge_room_room"),
    ("C", "outside", "edge_outside"),
    ("C", "all", "edge_all"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs():
    with open(f"{BASE}/manifest.json") as stream:
        manifest = json.load(stream)
    failures = []
    for relative, expected in manifest["files_sha256"].items():
        if not (relative.startswith("gt/") or relative.startswith("pred/")):
            continue
        path = f"{BASE}/{relative}"
        actual = sha256(path)
        if actual != expected:
            failures.append(f"{relative}: {actual} != {expected}")
    if failures:
        raise RuntimeError("frozen input hash mismatch:\n" + "\n".join(failures))
    return len([k for k in manifest["files_sha256"]
                if k.startswith("gt/") or k.startswith("pred/")])


def common_roomset(old_gt, candidate_gt):
    """Direct geometry, but exactly the legacy 324 evaluated room IDs."""
    result = copy.deepcopy(candidate_gt)
    allowed = {room["idx"] for room in old_gt["rooms"]
               if room["in_roomset"] and not room.get("is_stairs")}
    for room in result["rooms"]:
        if not room.get("is_stairs"):
            room["in_roomset"] = bool(room["in_roomset"] and room["idx"] in allowed)
    result["roomset_definition"] = "legacy raster GT 324 evaluated room IDs; direct .house geometry"
    result["candidate_version"] = "mp3d_house_floor_common324"
    return result


def micro(scores, tier, key):
    total = [0, 0, 0]
    for result in scores.values():
        value = result[tier].get(key, [0, 0, 0])
        total = [total[i] + value[i] for i in range(3)]
    return prf(*total)


def macro_f1(scores, tier, key):
    return float(np.mean([prf(*result[tier].get(key, [0, 0, 0]))["f1"]
                          for result in scores.values()]))


def metric_rooms(gt):
    return sum(room["in_roomset"] and not room.get("is_stairs")
               for room in gt["rooms"])


def overlap_ratio(gt):
    excess = union_area = 0.0
    for level in sorted(gt["lvl_z"]):
        geoms = [Polygon(room["poly"]) for room in gt["rooms"]
                 if room["level"] == level and room["in_roomset"]
                 and not room.get("is_stairs")]
        if not geoms:
            continue
        union = unary_union(geoms).area
        excess += sum(g.area for g in geoms) - union
        union_area += union
    return float(excess / union_area) if union_area else 0.0


def summarize(name, scores, gt_by_scene, evaluator):
    summary = {
        "name": name,
        "evaluator": evaluator,
        "gt_rooms": sum(metric_rooms(gt) for gt in gt_by_scene.values()),
        "gt_overlap_ratio": (
            sum(overlap_ratio(gt) for gt in gt_by_scene.values()) / len(gt_by_scene)),
        "micro": {},
        "macro_f1": {},
        "mean_matched_iou": float(np.mean(
            [iou for result in scores.values() for iou in result["iou"]])),
        "n_matched_rooms": sum(len(result["iou"]) for result in scores.values()),
        "per_scene_room_f1": {},
    }
    for tier, key, label in METRICS:
        summary["micro"][label] = micro(scores, tier, key)
        summary["macro_f1"][label] = macro_f1(scores, tier, key)
    for scene, result in scores.items():
        summary["per_scene_room_f1"][scene] = prf(*result["A"]["room"])["f1"]
    return summary


def pooled_overlap(gt_by_scene):
    excess = union_area = 0.0
    for gt in gt_by_scene.values():
        for level in sorted(gt["lvl_z"]):
            geoms = [Polygon(room["poly"]) for room in gt["rooms"]
                     if room["level"] == level and room["in_roomset"]
                     and not room.get("is_stairs")]
            if not geoms:
                continue
            union = unary_union(geoms).area
            excess += sum(g.area for g in geoms) - union
            union_area += union
    return float(excess / union_area) if union_area else 0.0


def render_markdown(payload):
    order = payload["variant_order"]
    variants = payload["variants"]
    lines = [
        "# Frozen watershed_v3：evaluator／GT A/B",
        "",
        "> CPU-only；未重跑 pipeline；predictions 通過 frozen SHA-256 驗證。",
        "",
        "## 核心結果",
        "",
        "| variant | GT rooms | overlap | Room P/R/F1 | matched IoU (n) | Doors@0.5 | edge_all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        item = variants[key]
        room = item["micro"]["Room"]
        lines.append(
            f"| {key} | {item['gt_rooms']} | {100*item['gt_overlap_ratio_pooled']:.3f}% | "
            f"{room['p']:.3f}/{room['r']:.3f}/**{room['f1']:.3f}** | "
            f"{item['mean_matched_iou']:.3f} ({item['n_matched_rooms']}) | "
            f"{item['micro']['Doors@0.5']['f1']:.3f} | "
            f"{item['micro']['edge_all']['f1']:.3f} |")
    lines.extend([
        "",
        "## 可識別的差異",
        "",
        "- `legacy_v2_raster324 → strict_v3_raster324`：只改 extra predicted levels 的 FP 漏算。",
        "- `strict_v3_raster324 → strict_v3_house_common324`：room IDs/denominator/evaluator 固定，只換成官方 `.house` floor polygons。",
        "- `strict_v3_house_common324 → strict_v3_house_official325`：只加入舊 raster path 因 `<0.5 m²` 門檻漏掉的有效 closet。",
        "- 論文的 317 rooms／33 levels 仍無公開 manifest 可精確還原；325／32 是明確可重現的 MP3D `.house` roomset，不宣稱等同論文 subset。",
        "",
        "## 逐棟 Room F1",
        "",
        "| scene | v2 raster324 | v3 raster324 | v3 house common324 | v3 house official325 |",
        "|---|---:|---:|---:|---:|",
    ])
    scenes = sorted(variants[order[0]]["per_scene_room_f1"])
    for scene in scenes:
        values = [variants[key]["per_scene_room_f1"][scene] for key in order]
        lines.append(f"| {scene} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def main():
    verified = verify_frozen_inputs()
    scenes = sorted(os.path.basename(path)[:-4]
                    for path in glob.glob(f"{BASE}/pred/*.pkl"))
    if len(scenes) != 16:
        raise RuntimeError(f"expected 16 frozen scenes, found {len(scenes)}")

    legacy_scores = pickle.load(open(f"{BASE}/eval2d_v2_hungarian/scores.pkl", "rb"))
    old_gt = {}
    candidate_gt = {}
    common_gt = {}
    strict_old_scores = {}
    common_scores = {}
    candidate_scores = {}
    extra_pred_rooms = 0

    for scene in scenes:
        old_gt[scene] = pickle.load(open(f"{BASE}/gt/{scene}.pkl", "rb"))
        candidate_gt[scene] = pickle.load(open(f"{CANDIDATE}/{scene}.pkl", "rb"))
        common_gt[scene] = common_roomset(old_gt[scene], candidate_gt[scene])
        pred = pickle.load(open(f"{BASE}/pred/{scene}.pkl", "rb"))
        strict_old_scores[scene] = score_scene(old_gt[scene], pred)
        common_scores[scene] = score_scene(common_gt[scene], pred)
        candidate_scores[scene] = score_scene(candidate_gt[scene], pred)
        extra_levels = set(strict_old_scores[scene]["level_diagnostics"]["unmatched_pred"])
        extra_pred_rooms += sum(room["level"] in extra_levels for room in pred["rooms"])

    variants_raw = {
        "legacy_v2_raster324": (legacy_scores, old_gt, "eval2d_v2_hungarian"),
        "strict_v3_raster324": (strict_old_scores, old_gt, EVAL2D_VERSION),
        "strict_v3_house_common324": (common_scores, common_gt, EVAL2D_VERSION),
        "strict_v3_house_official325": (candidate_scores, candidate_gt, EVAL2D_VERSION),
    }
    order = list(variants_raw)
    variants = {}
    for name, (scores, gt_map, evaluator) in variants_raw.items():
        variants[name] = summarize(name, scores, gt_map, evaluator)
        variants[name]["gt_overlap_ratio_pooled"] = pooled_overlap(gt_map)

    payload = {
        "status": "validated_candidate_not_silently_promoted",
        "prediction_source": "frozen watershed_v3_pre_report",
        "frozen_gt_pred_hashes_verified": verified,
        "n_scenes": len(scenes),
        "unmatched_pred_rooms_newly_counted_as_fp": extra_pred_rooms,
        "variant_order": order,
        "variants": variants,
        "unresolved": "paper reports 317 rooms / 33 levels; public exact subset manifest not found",
    }

    os.makedirs(STRICT_OUT, exist_ok=True)
    os.makedirs(AB_OUT, exist_ok=True)
    with open(f"{STRICT_OUT}/scores.pkl", "wb") as stream:
        pickle.dump(strict_old_scores, stream, protocol=4)
    with open(f"{AB_OUT}/scores_house_common324.pkl", "wb") as stream:
        pickle.dump(common_scores, stream, protocol=4)
    with open(f"{AB_OUT}/scores_house_official325.pkl", "wb") as stream:
        pickle.dump(candidate_scores, stream, protocol=4)
    with open(f"{AB_OUT}/comparison.json", "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    report = render_markdown(payload)
    with open(f"{AB_OUT}/RESULTS.md", "w") as stream:
        stream.write(report)
    with open(f"{STRICT_OUT}/RESULTS_2D.md", "w") as stream:
        stream.write(report)

    for name in order:
        item = variants[name]
        room = item["micro"]["Room"]
        print(f"{name}: Room P/R/F1={room['p']:.4f}/{room['r']:.4f}/{room['f1']:.4f} | "
              f"matched IoU={item['mean_matched_iou']:.4f} n={item['n_matched_rooms']} | "
              f"overlap={100*item['gt_overlap_ratio_pooled']:.4f}%")
    print(f"extra predicted rooms newly counted as FP: {extra_pred_rooms}")
    print(f"results -> {AB_OUT}")


if __name__ == "__main__":
    main()
