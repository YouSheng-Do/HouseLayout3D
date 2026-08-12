#!/usr/bin/env python3
"""Generate RESULTS_PANORAMA.md and best/median/worst floorplan figures."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from common import (
    BASE_OUT,
    DATA_OUT,
    GT_ROOT,
    ROOM_METRIC_EXCLUDE,
    parse_house,
    scene_ids,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "src/eval2d"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from metrics import align_levels, match_rooms_iou, prf  # noqa: E402


EVALUATION_ROOT = BASE_OUT / "evaluation"
REPORT_PATH = ROOT / "RESULTS_PANORAMA.md"
VISUAL_ROOT = BASE_OUT / "visualizations"
MODELS = ("dopnet", "horizonnet")
MODEL_LABEL = {"dopnet": "DOPNet", "horizonnet": "HorizonNet"}


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def metric(scores: dict, tier: str, name: str, scenes=None) -> dict:
    selected = scenes or sorted(scores)
    counts = [0, 0, 0]
    for scene in selected:
        values = scores[scene][tier].get(name, [0, 0, 0])
        counts = [counts[index] + values[index] for index in range(3)]
    return prf(*counts)


def matched_iou(scores: dict, scenes=None) -> tuple:
    selected = scenes or sorted(scores)
    values = [value for scene in selected for value in scores[scene]["iou"]]
    return (float(np.mean(values)) if values else None, len(values))


def fmt(value, digits=3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def selected_artifacts(model: str, summary: dict) -> tuple:
    threshold = summary["models"][model]["selected_threshold"]
    directory = EVALUATION_ROOT / model / f"iou_{threshold:.2f}"
    return threshold, load_pickle(directory / "scores.pkl"), directory / "pred"


def manhattan_residual(points) -> float:
    """Length-weighted residual to the best single Manhattan orientation."""
    points = np.asarray(points, dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > 1e-6
    angles = np.mod(np.degrees(np.arctan2(
        edges[valid, 1], edges[valid, 0])), 90.0)
    weights = lengths[valid]
    if not len(angles):
        return float("nan")
    best = float("inf")
    for offset in np.linspace(0.0, 90.0, 901, endpoint=False):
        difference = np.abs(angles - offset)
        residual = np.minimum(difference, 90.0 - difference)
        best = min(best, float(np.average(residual, weights=weights)))
    return best


def diagnostic_groups(model: str) -> dict:
    room_meta = {}
    areas = []
    covered = set()
    uncovered = set()
    for scene in scene_ids():
        parsed = parse_house(scene)
        view_regions = Counter(
            pano["region_id"] for pano in parsed["panoramas"]
            if pano["region_id"] is not None)
        for region_id, region in parsed["regions"].items():
            if region["label"] in ROOM_METRIC_EXCLUDE:
                continue
            floor = parsed["floors"][region_id]
            key = (scene, region_id)
            room_meta[key] = {
                "area": floor["area"],
                "label": region["label"],
                "manhattan_residual": manhattan_residual(floor["points"]),
            }
            areas.append(floor["area"])
            (covered if view_regions[region_id] else uncovered).add(key)
    large_threshold = float(np.quantile(areas, 0.75))
    groups = defaultdict(list)
    with (EVALUATION_ROOT / model / "per_view_iou.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["iou"] in ("", "None"):
                continue
            key = (row["scene"], int(row["region_id"]))
            metadata = room_meta[key]
            value = float(row["iou"])
            groups["all"].append(value)
            if metadata["area"] >= large_threshold:
                groups["large_area_q4"].append(value)
            if metadata["label"] == "h":
                groups["hallway"].append(value)
            if metadata["manhattan_residual"] > 10.0:
                groups["non_manhattan"].append(value)
    return {
        "large_area_threshold_m2": large_threshold,
        "covered_rooms": len(covered),
        "uncovered_rooms": len(uncovered),
        "uncovered_keys": sorted(f"{scene}:R{region}" for scene, region in uncovered),
        "groups": {
            key: {
                "n_views": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for key, values in groups.items()
        },
    }


def coverage_match_audit(pred_dir: Path) -> dict:
    covered = Counter()
    for scene in scene_ids():
        gt = load_pickle(GT_ROOT / f"{scene}.pkl")
        pred = load_pickle(pred_dir / f"{scene}.pkl")
        parsed = parse_house(scene)
        views = Counter(
            pano["region_id"] for pano in parsed["panoramas"]
            if pano["region_id"] is not None)
        level_map = align_levels(gt["lvl_z"], pred["level_z"])
        for gt_level, pred_level in level_map.items():
            gt_rooms = [
                room for room in gt["rooms"]
                if room["level"] == gt_level and room["in_roomset"]
                and not room["is_stairs"]
            ]
            pred_rooms = [room for room in pred["rooms"]
                          if room["level"] == pred_level]
            mapping, _ = match_rooms_iou(pred_rooms, gt_rooms, 0.5)
            matched_gt = set(mapping.values())
            for room in gt_rooms:
                key = "covered" if views[room["idx"]] else "uncovered"
                covered[f"{key}_total"] += 1
                covered[f"{key}_matched"] += int(room["idx"] in matched_gt)
    return dict(covered)


def component_stats(pred_dir: Path) -> dict:
    sizes = []
    confidences = []
    output_rooms = 0
    source_views = 0
    for scene in scene_ids():
        pred = load_pickle(pred_dir / f"{scene}.pkl")
        output_rooms += len(pred["rooms"])
        for item in pred["merge_audit"]:
            sizes.append(item["component_size"])
            source_views += item["component_size"]
            confidences.append(item["agreement_confidence"])
    return {
        "source_views": source_views,
        "output_rooms": output_rooms,
        "component_size_median": float(np.median(sizes)),
        "component_size_q25": float(np.quantile(sizes, 0.25)),
        "component_size_q75": float(np.quantile(sizes, 0.75)),
        "singletons": int(sum(size == 1 for size in sizes)),
        "agreement_mean": float(np.mean(confidences)),
    }


def plot_building(scene: str, role: str, room_f1: float,
                  pred_dir: Path) -> Path:
    gt = load_pickle(GT_ROOT / f"{scene}.pkl")
    pred = load_pickle(pred_dir / f"{scene}.pkl")
    parsed = parse_house(scene)
    levels = sorted(gt["lvl_z"])
    figure, axes = plt.subplots(
        1, len(levels), figsize=(6.0 * len(levels), 6.2), squeeze=False)
    for column, level in enumerate(levels):
        axis = axes[0, column]
        gt_rooms = [room for room in gt["rooms"]
                    if room["level"] == level and room["in_roomset"]
                    and not room["is_stairs"]]
        pred_rooms = [room for room in pred["rooms"] if room["level"] == level]
        for room_index, room in enumerate(gt_rooms):
            points = np.asarray(room["poly"])
            axis.fill(points[:, 0], points[:, 1], color="#4c78a8", alpha=0.13)
            axis.plot(*np.vstack([points, points[0]]).T,
                      color="#2d5f8b", linewidth=1.5,
                      label="GT" if room_index == 0 else None)
        for room_index, room in enumerate(pred_rooms):
            points = np.asarray(room["poly"])
            axis.fill(points[:, 0], points[:, 1], color="#e45756", alpha=0.08)
            axis.plot(*np.vstack([points, points[0]]).T,
                      color="#c83f49", linewidth=1.25,
                      label="panorama" if room_index == 0 else None)
        positions = np.asarray([
            pano["position"][:2] for pano in parsed["panoramas"]
            if pano["level"] == level
        ])
        if len(positions):
            axis.scatter(positions[:, 0], positions[:, 1], s=8,
                         color="#333333", alpha=0.4, label="viewpoint")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(
            f"level {level} · pred/GT rooms {len(pred_rooms)}/{len(gt_rooms)}")
        axis.grid(alpha=0.15)
        if column == 0:
            axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{role.upper()} · DOPNet · {scene} · Room F1={room_f1:.3f}")
    figure.tight_layout()
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    path = VISUAL_ROOT / f"{role}_{scene}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def score_cell(scores: dict, tier: str, name: str) -> str:
    return fmt(metric(scores, tier, name)["f1"])


def main() -> None:
    summary = load_json(EVALUATION_ROOT / "summary.json")
    manifest = load_json(DATA_OUT / "manifest.json")
    split = load_json(ROOT / "configs/eval2d/split_v0_1.json")
    artifacts = {}
    for model in MODELS:
        artifacts[model] = selected_artifacts(model, summary)

    dop_threshold, dop_scores, dop_pred_dir = artifacts["dopnet"]
    scene_f1 = {
        scene: prf(*dop_scores[scene]["A"]["room"])["f1"]
        for scene in dop_scores
    }
    ordered = sorted(scene_f1, key=lambda scene: (scene_f1[scene], scene))
    chosen = {
        "worst": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "best": ordered[-1],
    }
    visual_paths = {
        role: plot_building(scene, role, scene_f1[scene], dop_pred_dir)
        for role, scene in chosen.items()
    }

    diagnostics = {model: diagnostic_groups(model) for model in MODELS}
    merge_stats = {
        model: component_stats(artifacts[model][2]) for model in MODELS}
    coverage_audits = {
        model: coverage_match_audit(artifacts[model][2]) for model in MODELS}
    with (EVALUATION_ROOT / "failure_analysis.json").open("w") as stream:
        json.dump({
            "per_view_categories": diagnostics,
            "merge": merge_stats,
            "coverage_matching": coverage_audits,
        }, stream, indent=2)

    lines = [
        "# Panorama-based layout baselines — HouseLayout3D",
        "",
        "> Measurement-only experiment: published MP3D checkpoints, no training or fine-tuning. "
        "Building scoring uses the unchanged `eval2d_v3_strict_levels` + "
        "`mp3d_house_floor_v0_1` protocol (325 rooms / 32 levels).",
        "",
        "## Executive result",
        "",
    ]
    for model in MODELS:
        threshold, scores, _ = artifacts[model]
        room = metric(scores, "A", "room")
        iou_mean, iou_n = matched_iou(scores)
        per_view = summary["models"][model]["per_view"]
        lines.append(
            f"- **{MODEL_LABEL[model]}**: per-view GT-room IoU mean/median "
            f"**{per_view['mean']:.3f}/{per_view['median']:.3f}** "
            f"(n={per_view['scored_views']}); assembled Room P/R/F1 "
            f"**{room['p']:.3f}/{room['r']:.3f}/{room['f1']:.3f}** at the "
            f"dev-selected merge threshold {threshold:.2f}; conditional matched IoU "
            f"**{iou_mean:.3f}** (n={iou_n}).")
    lines += [
        "",
        "The local-layout score and assembled score are reported separately: the former tests "
        "the model; the latter also includes duplicate merging and building coverage. "
        "Doors, room types, and connectivity are **N/A** for both panorama models.",
        "",
        "The published ~79% MatterportLayout number is 3D IoU against that dataset's dedicated "
        "physical-enclosure layout labels. Our diagnostic is 2D IoU against MP3D `.house` "
        "semantic region polygons, which can split visually open spaces. It is useful for this "
        "downstream floorplan task but is not a like-for-like reproduction of the 79% metric.",
        "",
        "## 1. Data preparation and pose verification",
        "",
        f"All **{manifest['scenes']}** scenes contain skyboxes: "
        f"**{manifest['totals']['viewpoints']} viewpoints / "
        f"{manifest['totals']['skybox_files']} JPEG faces**, exactly six 1024×1024 faces per "
        "viewpoint. Faces were stitched as MP3D U/L/F/R/B/D to 1024×512 equirectangular PNG, "
        "then Manhattan-VP aligned with HorizonNet's published preprocessing.",
        "",
        f"Room coverage is **{manifest['totals']['covered_metric_rooms']}/"
        f"{manifest['totals']['metric_rooms']} = "
        f"{100*manifest['metric_room_coverage']:.2f}%** for Room-metric regions; including "
        f"stairs it is {manifest['totals']['covered_indoor_regions']}/"
        f"{manifest['totals']['indoor_regions_including_stairs']} = "
        f"{100*manifest['indoor_region_coverage']:.2f}%. Therefore the direct-observation "
        f"Room recall ceiling is at most {manifest['metric_room_coverage']:.3f} before any "
        "layout or merge error.",
        "",
        "Level assignment used nearest robust camera-z center, computed per scene from "
        "non-stair official panorama records. This avoids copying the single region-level of "
        "stair regions that physically span floors. Non-stair disagreements are disclosed below.",
        "",
        "Orientation uses `pose_1_5`, whose forward direction is skybox image 1 / cubemap left "
        "under the [EDM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/"
        "papers/Jung_EDM_Equirectangular_Projection-Oriented_Dense_Kernelized_Feature_"
        "Matching_CVPR_2025_paper.pdf) Matterport3D convention. Therefore panorama "
        "right/forward/up maps to -camera-forward/camera-right/-camera-down. A first-scene "
        "smoke test caught and rejected the incorrect `pose_1_2 = skybox face 2` assumption; "
        "the corrected mapping is global and frozen, not fitted per panorama.",
        "",
        "| scene | views | views by z-level | rooms covered | indoor+stairs covered | "
        "exact pose-in-room | outside >2cm / unassigned | non-stair level disagreements |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["per_scene"]:
        distribution = ", ".join(
            f"L{level}:{count}" for level, count in row["viewpoints_by_level"].items())
        containment = row["containment"]
        lines.append(
            f"| {row['scene']} | {row['viewpoints']} | {distribution} | "
            f"{row['covered_metric_rooms']}/{row['metric_rooms']} | "
            f"{row['covered_indoor_regions']}/{row['indoor_regions_including_stairs']} | "
            f"{containment['inside_exact']}/{containment['n']} | "
            f"{containment['outside_2cm']} / {containment['unassigned']} | "
            f"{row['nonstair_region_vs_z_level_disagreements']} |")
    exact = sum(row["containment"]["inside_exact"] for row in manifest["per_scene"])
    near = sum(row["containment"]["inside_or_within_2cm"] for row in manifest["per_scene"])
    outside = sum(row["containment"]["outside_2cm"] for row in manifest["per_scene"])
    unassigned = sum(row["containment"]["unassigned"] for row in manifest["per_scene"])
    max_pose_delta = max(
        row["pose_vs_house_position_max_delta_m"] for row in manifest["per_scene"])
    lines += [
        "",
        f"Empirical frame check: {exact}/{manifest['totals']['viewpoints']} positions are "
        f"inside their official region exactly, {near}/{manifest['totals']['viewpoints']} "
        f"inside or within 2 cm; {outside} are farther than 2 cm and {unassigned} official "
        f"records remain unassigned. Pose-file vs `.house` position max delta is "
        f"{max_pose_delta:.3f} m. Per-scene overlays are in "
        f"`{DATA_OUT.relative_to(ROOT)}/pose_overlays/`. Visual seam/pole checks were made "
        "before full inference; sample paths are recorded in the run README.",
        "",
        "## 2. Per-viewpoint room-layout quality",
        "",
        "Only viewpoints assigned to a non-stair Room-metric GT region enter this diagnostic. "
        "The GT polygon is used only for scoring, never for inference or scaling.",
        "",
        "| model | successful / failed views | scored in-room views | GT rooms represented | "
        "mean | q25 | median | q75 | min–max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = summary["models"][model]["per_view"]
        lines.append(
            f"| {MODEL_LABEL[model]} | {values['prediction_success']} / "
            f"{values['prediction_failed']} | {values['scored_views']} | "
            f"{values['covered_gt_rooms']} | {fmt(values['mean'])} | "
            f"{fmt(values['q25'])} | {fmt(values['median'])} | "
            f"{fmt(values['q75'])} | {fmt(values['min'])}–{fmt(values['max'])} |")

    lines += [
        "",
        "Room-balanced view: first average all viewpoint IoUs within each represented GT "
        "room, then summarize those room means (so heavily sampled rooms do not dominate).",
        "",
        "| model | represented GT rooms | mean of room means | median of room means |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        with (EVALUATION_ROOT / model / "per_room_iou.csv").open() as stream:
            room_means = [float(row["mean_iou"]) for row in csv.DictReader(stream)]
        lines.append(
            f"| {MODEL_LABEL[model]} | {len(room_means)} | "
            f"{fmt(float(np.mean(room_means)))} | "
            f"{fmt(float(np.median(room_means)))} |")

    failures = []
    for model in MODELS:
        for path in sorted((BASE_OUT / "predictions" / model).glob("*/*.json")):
            record = load_json(path)
            if record.get("status") == "failed":
                failures.append(
                    f"- {MODEL_LABEL[model]} `{record['scene']}/{record['pano_id']}`: "
                    f"`{record['error_type']}: {record['error']}`")
    lines += [
        "",
        "**Serialized inference failures:**",
        "",
        *(failures or ["- None."]),
        "",
        "Full auditable distributions: `evaluation/<model>/per_view_iou.csv` and "
        "`per_room_iou.csv` under the panorama output directory.",
        "",
        "## 3. Building-scale results",
        "",
        f"Thresholds were selected independently per model using only the frozen dev scenes "
        f"({', '.join(split['dev'])}). The held-out scenes were not used in selection. "
        "This split is prospective only: all 16 scenes had been observed before it was frozen.",
        "",
        "| model / subset | merge IoU | Room P | Room R | Room F1 | matched IoU (n) | "
        "Corner .1/.2/.3 F1 | mean room-count error / level |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        threshold, scores, _ = artifacts[model]
        for subset_name, subset in (("dev", split["dev"]),
                                    ("held-out", split["held_out"]),
                                    ("all", sorted(scores))):
            room = metric(scores, "A", "room", subset)
            iou_mean, iou_n = matched_iou(scores, subset)
            corners = "/".join(
                fmt(metric(scores, "A", f"corner@{distance}", subset)["f1"])
                for distance in (0.1, 0.2, 0.3))
            count_error = [
                value for scene in subset for value in scores[scene]["rc_err"]]
            lines.append(
                f"| {MODEL_LABEL[model]} / {subset_name} | {threshold:.2f} | "
                f"{room['p']:.3f} | {room['r']:.3f} | {room['f1']:.3f} | "
                f"{fmt(iou_mean)} ({iou_n}) | {corners} | "
                f"{np.mean(count_error):+.2f} |")
    lines += [
        "",
        "### Per building (selected thresholds)",
        "",
        "| scene | DOPNet P/R/F1 · IoU | HorizonNet P/R/F1 · IoU | DOPNet pred/GT rooms |",
        "|---|---:|---:|---:|",
    ]
    for scene in sorted(dop_scores):
        cells = []
        for model in MODELS:
            scores = artifacts[model][1]
            room = prf(*scores[scene]["A"]["room"])
            values = scores[scene]["iou"]
            cells.append(
                f"{room['p']:.3f}/{room['r']:.3f}/{room['f1']:.3f} · "
                f"{fmt(float(np.mean(values)) if values else None)}")
        pred = load_pickle(dop_pred_dir / f"{scene}.pkl")
        gt = load_pickle(GT_ROOT / f"{scene}.pkl")
        gt_count = sum(room["in_roomset"] and not room["is_stairs"]
                       for room in gt["rooms"])
        lines.append(
            f"| {scene} | {cells[0]} | {cells[1]} | "
            f"{len(pred['rooms'])}/{gt_count} |")
    lines += [
        "",
        "### Room count per level (selected thresholds)",
        "",
        "| scene | level | GT | DOPNet | HorizonNet |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene in sorted(dop_scores):
        gt = load_pickle(GT_ROOT / f"{scene}.pkl")
        predictions = {
            model: load_pickle(artifacts[model][2] / f"{scene}.pkl")
            for model in MODELS
        }
        for level in sorted(gt["lvl_z"]):
            gt_count = sum(
                room["level"] == level and room["in_roomset"]
                and not room["is_stairs"] for room in gt["rooms"])
            pred_counts = {
                model: sum(room["level"] == level
                           for room in predictions[model]["rooms"])
                for model in MODELS
            }
            lines.append(
                f"| {scene} | {level} | {gt_count} | "
                f"{pred_counts['dopnet']} | {pred_counts['horizonnet']} |")
    lines += [
        "",
        "The aggregate mean count errors above are pred−GT, including zero-view and "
        "stair-view failure effects.",
        "",
        "## 4. Merge-threshold sensitivity",
        "",
        "Rule: within each z-level, form connected components where pairwise polygon IoU is "
        "at least the threshold; retain the component member with highest mean IoU agreement "
        "(deterministic pano-ID tie break). No GT room ID, union cleanup, or learned merger is used.",
        "",
        "| model | threshold | dev F1 | held-out F1 | all F1 | all P/R | all matched IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        selected = summary["models"][model]["selected_threshold"]
        for row in summary["models"][model]["threshold_sweep"]:
            marker = " **(selected)**" if row["threshold"] == selected else ""
            all_values = row["all"]
            lines.append(
                f"| {MODEL_LABEL[model]}{marker} | {row['threshold']:.2f} | "
                f"{row['dev']['f1']:.3f} | {row['held_out']['f1']:.3f} | "
                f"{all_values['f1']:.3f} | {all_values['p']:.3f}/"
                f"{all_values['r']:.3f} | {fmt(all_values['matched_iou_mean'])} |")
    lines += [
        "",
        "## 5. Failure analysis",
        "",
        f"Operational checks use large rooms = top area quartile (≥"
        f"{diagnostics['dopnet']['large_area_threshold_m2']:.2f} m²), corridor = MP3D "
        "`h`/hallway, and non-Manhattan = length-weighted residual >10° from the best single "
        "orthogonal frame. These labels are diagnostic slices, not tuned filters.",
        "",
        "| model | all view IoU (n) | large-area IoU (n) | hallway IoU (n) | "
        "non-Manhattan IoU (n) | merge source→output; singleton clusters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        groups = diagnostics[model]["groups"]
        def group_cell(name):
            values = groups.get(name, {"mean": None, "n_views": 0})
            return f"{fmt(values['mean'])} ({values['n_views']})"
        merge = merge_stats[model]
        lines.append(
            f"| {MODEL_LABEL[model]} | {group_cell('all')} | "
            f"{group_cell('large_area_q4')} | {group_cell('hallway')} | "
            f"{group_cell('non_manhattan')} | {merge['source_views']}→"
            f"{merge['output_rooms']}; {merge['singletons']} |")
    lines += [
        "",
        f"**Rooms without a viewpoint:** {diagnostics['dopnet']['uncovered_rooms']} of 325: "
        f"`{', '.join(diagnostics['dopnet']['uncovered_keys'])}`. They are retained in GT. "
        "A neighbouring panorama prediction can accidentally overlap one, but that is not a "
        "direct observation. Covered-vs-uncovered match counts are in "
        "`evaluation/failure_analysis.json`.",
        "",
        "The main assembly stress signal is the number of duplicate components and singletons: "
        "a high threshold leaves multiple views of one room as false-positive rooms; a low "
        "threshold can transitively collapse adjacent rooms. This experiment deliberately stops "
        "at that simple merger, as required.",
        "",
        "Manhattan VP alignment is part of both official inference paths. Large/open, hallway, "
        "and non-Manhattan slices above show where that prior and single-room framing help or "
        "hurt; no category-specific repair was added.",
        "",
        "## 6. Best / median / worst visualisations",
        "",
        "Ranked by selected-threshold DOPNet per-building Room F1 (blue=GT, red=assembled "
        "panorama polygons, black dots=viewpoints):",
        "",
    ]
    for role in ("best", "median", "worst"):
        scene = chosen[role]
        lines.append(
            f"- {role}: `{scene}`, F1={scene_f1[scene]:.3f} — "
            f"`{visual_paths[role].relative_to(ROOT)}`")

    ws_scores = load_pickle(
        GT_ROOT / "eval_frozen_watershed_v3/scores_house_official325.pkl")
    rf_scores = load_pickle(
        ROOT / "outputs/eval2d/comparison_strict_v3/"
        "scores_roomformer_per_floor.pkl")
    cage_scores = load_pickle(
        ROOT / "outputs/eval2d/baselines/cage_stru3d_swinv2_zeroshot/scores.pkl")
    methods = [
        ("MULTIFLOOR3D reimpl", ws_scores, "full pipeline; predicts levels"),
        ("RoomFormer per-floor", rf_scores, "oracle `.house` levels + region points"),
        ("CAGE per-floor", cage_scores, "oracle `.house` levels + region points"),
        (f"Panorama DOPNet (IoU {dop_threshold:.2f})", dop_scores,
         "RGB panoramas + GT poses; z-levels"),
    ]
    lines += [
        "",
        "## 7. Four-way comparison under the same evaluator",
        "",
        "All layout numbers below use strict-v3 + official325. Input conditions differ and "
        "must be quoted with the score.",
        "",
        "| method | condition | Room F1 | matched IoU (n) | Corner .1/.2/.3 | "
        "Room+type | Door@.5 | edge-all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scores, condition in methods:
        room = score_cell(scores, "A", "room")
        iou_mean, iou_n = matched_iou(scores)
        corners = "/".join(
            score_cell(scores, "A", f"corner@{distance}")
            for distance in (0.1, 0.2, 0.3))
        if name.startswith("CAGE") or name.startswith("Panorama"):
            semantic = ("N/A", "N/A", "N/A")
        else:
            semantic = (
                score_cell(scores, "B", "room+type"),
                score_cell(scores, "B", "doors@0.5"),
                score_cell(scores, "C", "all"),
            )
        lines.append(
            f"| {name} | {condition} | {room} | {fmt(iou_mean)} ({iou_n}) | "
            f"{corners} | {semantic[0]} | {semantic[1]} | {semantic[2]} |")
    lines += [
        "",
        "HorizonNet is retained as the classic panorama reference in Sections 2–4; DOPNet is "
        "the spec-designated primary panorama row in the four-way table.",
        "",
        "## Reproducibility and deviations",
        "",
        "- Official repos/checkpoints: DOPNet MP3D `model_best_mp3d.pkl` "
        "(SHA-256 `0dcc7929...38cb`); HorizonNet MP3D "
        "`resnet50_rnn__mp3d.pth` (SHA-256 `bdef2e33...0c96`).",
        "- No parameter update, training, fine-tuning, GT-shape scaling, or pose estimation was "
        "performed.",
        "- Compatibility-only changes are documented in `src/baselines/panorama/README.md`: "
        "device-agnostic DOPNet sampling buffer, no redundant ImageNet initialization, OpenCV "
        "contour-list adaptation, and NumPy SVD replacing HorizonNet's one-component sklearn PCA.",
        "- Four official `.house` panorama records remain unassigned rather than being forced "
        "into a GT room. All per-view failures are serialized with exception type and message.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
