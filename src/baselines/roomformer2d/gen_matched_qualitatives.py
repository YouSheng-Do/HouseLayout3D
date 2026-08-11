#!/usr/bin/env python
"""Render RoomFormer on the qualitative cases selected by our 2D method.

Two outputs are produced for each fixed case:

* ``roomformer_same_format`` keeps the report's original two-panel layout
  (official GT | prediction), with the prediction replaced by RoomFormer.
* ``direct_comparison`` uses one shared coordinate transform for
  official GT | our prediction | RoomFormer, so shapes can be compared at
  exactly the same scale.

The labels best/representative/worst refer to the frozen watershed result that
selected the report examples.  They are not RoomFormer's ranking.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
EVAL2D = ROOT / "src" / "eval2d"
sys.path.insert(0, str(EVAL2D))

from gen_v3_visuals_pillow import (  # noqa: E402
    HEIGHT,
    WIDTH,
    bounds,
    doors_at,
    draw_panel,
    font,
    pair_metrics,
    rooms_at,
)
from metrics import align_levels, prf  # noqa: E402


OURS_ROOT = (
    ROOT / "outputs/eval2d/baselines/watershed_v3_pre_report"
)
GT_ROOT = ROOT / "outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
ROOMFORMER_BASE = (
    ROOT
    / "outputs/eval2d/baselines/roomformer_hl3d_2d_v1"
)

# These are the exact scene/GT-level choices shown in the current report.
CASES = (
    ("best", "e9zR4mvMWw7", 2),
    ("representative", "i5noydFURQK", 1),
    ("worst", "HxpKQynjfin", 0),
)


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def prediction_level(gt: dict, pred: dict, gt_level: int) -> int:
    mapping = align_levels(gt["lvl_z"], pred.get("level_z", {}))
    if gt_level not in mapping:
        raise RuntimeError(
            f"GT level {gt_level} has no aligned prediction level; mapping={mapping}"
        )
    return int(mapping[gt_level])


def aggregate_metrics(scores: dict, scene: str) -> tuple[float, float]:
    result = scores[scene]
    room_f1 = float(prf(*result["A"]["room"])["f1"])
    matched_iou = float(np.mean(result["iou"] or [0.0]))
    return room_f1, matched_iou


def unmatched_levels(gt: dict, pred: dict) -> list[int]:
    mapping = align_levels(gt["lvl_z"], pred.get("level_z", {}))
    return sorted(set(pred.get("level_z", {})) - set(mapping.values()))


def render_same_format(
    tag: str,
    scene: str,
    gt_level: int,
    gt: dict,
    roomformer: dict,
    roomformer_level: int,
    roomformer_scores: dict,
    mode: str,
    output_root: Path,
) -> tuple[Path, dict]:
    gt_rooms = rooms_at(gt, gt_level, gt=True)
    gt_doors = doors_at(gt, gt_level)
    pred_rooms = rooms_at(roomformer, roomformer_level)
    pred_doors = doors_at(roomformer, roomformer_level)
    lo, hi = bounds([gt_rooms, pred_rooms], [gt_doors, pred_doors])

    tp, shown_f1, shown_iou = pair_metrics(gt_rooms, pred_rooms)
    scene_f1, scene_iou = aggregate_metrics(roomformer_scores, scene)
    extras = unmatched_levels(gt, roomformer)

    image = Image.new("RGB", (WIDTH, HEIGHT), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    mode_label = mode.replace("_", "-")
    title = (
        f"[ours-{tag} case] {scene} · RoomFormer {mode_label} · "
        f"GT L{gt_level} ↔ PRED L{roomformer_level}"
    )
    draw.text(
        (WIDTH / 2, 38),
        title,
        anchor="ma",
        font=font(30, True),
        fill=(18, 23, 28),
    )
    draw_panel(
        draw,
        (55, 105, 935, 930),
        f"GT L{gt_level} · {len(gt_rooms)} rooms（stairs excluded）",
        gt_rooms,
        gt_doors,
        lo,
        hi,
    )
    draw_panel(
        draw,
        (985, 105, 1865, 930),
        f"ROOMFORMER PRED L{roomformer_level} · {len(pred_rooms)} rooms",
        pred_rooms,
        pred_doors,
        lo,
        hi,
    )
    footer = (
        f"shown pair: TP={tp}, F1={shown_f1:.3f}, matched IoU={shown_iou:.3f}  |  "
        f"RoomFormer scene aggregate: Room F1={scene_f1:.3f}, "
        f"matched IoU={scene_iou:.3f}  |  unmatched PRED levels={extras or 'none'}"
    )
    draw.text(
        (WIDTH / 2, 962),
        footer,
        anchor="ma",
        font=font(20),
        fill=(50, 58, 66),
    )
    note = (
        f"CASE SELECTION: our frozen watershed_v3 result · RoomFormer {mode_label} semantic-rich · "
        "red=doors · gray=derived room-room edges · colors are independent"
    )
    draw.text(
        (WIDTH / 2, 1010),
        note,
        anchor="ma",
        font=font(17),
        fill=(150, 91, 10),
    )
    output = output_root / "roomformer_same_format" / f"{tag}_{scene}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return output, {
        "tp": tp,
        "f1": shown_f1,
        "matched_iou": shown_iou,
        "scene_room_f1": scene_f1,
        "scene_matched_iou": scene_iou,
    }


def render_direct_comparison(
    tag: str,
    scene: str,
    gt_level: int,
    gt: dict,
    ours: dict,
    ours_level: int,
    roomformer: dict,
    roomformer_level: int,
    mode: str,
    output_root: Path,
) -> tuple[Path, dict]:
    width, height = 2400, 1080
    gt_rooms = rooms_at(gt, gt_level, gt=True)
    gt_doors = doors_at(gt, gt_level)
    our_rooms = rooms_at(ours, ours_level)
    our_doors = doors_at(ours, ours_level)
    rf_rooms = rooms_at(roomformer, roomformer_level)
    rf_doors = doors_at(roomformer, roomformer_level)
    lo, hi = bounds(
        [gt_rooms, our_rooms, rf_rooms],
        [gt_doors, our_doors, rf_doors],
    )
    our_tp, our_f1, our_iou = pair_metrics(gt_rooms, our_rooms)
    rf_tp, rf_f1, rf_iou = pair_metrics(gt_rooms, rf_rooms)

    image = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    mode_label = mode.replace("_", "-")
    title = (
        f"[ours-{tag} case] {scene} · same GT level and shared scale · "
        f"ours vs RoomFormer {mode_label}"
    )
    draw.text(
        (width / 2, 38),
        title,
        anchor="ma",
        font=font(30, True),
        fill=(18, 23, 28),
    )
    panels = (
        (
            (45, 105, 785, 930),
            f"GT L{gt_level} · {len(gt_rooms)} rooms",
            gt_rooms,
            gt_doors,
        ),
        (
            (830, 105, 1570, 930),
            f"OURS PRED L{ours_level} · {len(our_rooms)} rooms",
            our_rooms,
            our_doors,
        ),
        (
            (1615, 105, 2355, 930),
            f"ROOMFORMER {mode_label.upper()} L{roomformer_level} · {len(rf_rooms)} rooms",
            rf_rooms,
            rf_doors,
        ),
    )
    for panel, panel_title, rooms, doors in panels:
        draw_panel(draw, panel, panel_title, rooms, doors, lo, hi)

    footer = (
        f"shown pair · OURS: TP={our_tp}, F1={our_f1:.3f}, IoU={our_iou:.3f}  |  "
        f"ROOMFORMER: TP={rf_tp}, F1={rf_f1:.3f}, IoU={rf_iou:.3f}"
    )
    draw.text(
        (width / 2, 962),
        footer,
        anchor="ma",
        font=font(21),
        fill=(50, 58, 66),
    )
    note = (
        "official MP3D floor GT · red=doors · gray=derived room-room edges · "
        "all panels share bounds/scale · room colors are independent"
    )
    draw.text(
        (width / 2, 1010),
        note,
        anchor="ma",
        font=font(18),
        fill=(150, 91, 10),
    )
    output = output_root / "direct_comparison" / f"{tag}_{scene}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return output, {
        "ours": {"tp": our_tp, "f1": our_f1, "matched_iou": our_iou},
        "roomformer": {"tp": rf_tp, "f1": rf_f1, "matched_iou": rf_iou},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("per_room", "per_floor"),
        default="per_room",
        help="RoomFormer inference mode to visualize (default: per_room).",
    )
    args = parser.parse_args()
    roomformer_root = ROOMFORMER_BASE / args.mode
    output_root = (
        ROOMFORMER_BASE / "qualitative_matched_to_ours" / args.mode
    )
    ours_scores = load_pickle(
        GT_ROOT / "eval_frozen_watershed_v3/scores_house_official325.pkl"
    )
    roomformer_scores = load_pickle(roomformer_root / "scores.pkl")
    run_manifest = json.loads((roomformer_root / "run_manifest.json").read_text())
    del ours_scores  # Pair metrics are recomputed on exactly the rendered levels.

    records = []
    for tag, scene, gt_level in CASES:
        gt = load_pickle(GT_ROOT / f"{scene}.pkl")
        ours = load_pickle(OURS_ROOT / "pred" / f"{scene}.pkl")
        roomformer = load_pickle(roomformer_root / "pred" / f"{scene}.pkl")
        ours_level = prediction_level(gt, ours, gt_level)
        roomformer_level = prediction_level(gt, roomformer, gt_level)

        same_path, same_metrics = render_same_format(
            tag,
            scene,
            gt_level,
            gt,
            roomformer,
            roomformer_level,
            roomformer_scores,
            args.mode,
            output_root,
        )
        comparison_path, comparison_metrics = render_direct_comparison(
            tag,
            scene,
            gt_level,
            gt,
            ours,
            ours_level,
            roomformer,
            roomformer_level,
            args.mode,
            output_root,
        )
        records.append(
            {
                "tag": tag,
                "selection_source": "ours_frozen_watershed_v3",
                "scene": scene,
                "gt_level": gt_level,
                "ours_pred_level": ours_level,
                "roomformer_pred_level": roomformer_level,
                "roomformer_same_format": str(same_path.relative_to(ROOT)),
                "direct_comparison": str(comparison_path.relative_to(ROOT)),
                "same_format_metrics": same_metrics,
                "comparison_pair_metrics": comparison_metrics,
            }
        )

    manifest = {
        "version": "roomformer_qualitative_matched_to_ours_v1",
        "case_labels_are_ranked_by": "ours_frozen_watershed_v3_scene_room_f1",
        "roomformer_mode": args.mode,
        "checkpoint": run_manifest.get("checkpoint"),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
        "coordinate_policy": (
            "Each direct-comparison image uses one shared world-coordinate "
            "bounds/scale for GT, ours, and RoomFormer."
        ),
        "cases": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for record in records:
        print(ROOT / record["direct_comparison"])
        print(ROOT / record["roomformer_same_format"])
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
