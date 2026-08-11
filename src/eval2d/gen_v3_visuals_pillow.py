"""Generate current v3/official-GT qualitative figures without Matplotlib.

The base environment currently has a NumPy/Matplotlib ABI mismatch, so this
small deterministic Pillow renderer keeps report generation CPU-only and avoids
changing the Python environment.  It draws official MP3D floor polygons,
frozen predictions, HouseLayout3D doors, and separately derived access edges.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import align_levels, derive_edges, prf  # noqa: E402

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
GT_DIR = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
SCORES_PATH = f"{GT_DIR}/eval_frozen_watershed_v3/scores_house_official325.pkl"
OUT = f"{BASE}/eval2d_v3_strict_levels/visualizations"
WIDTH, HEIGHT = 1920, 1080

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    path = FONT_BOLD if bold and os.path.exists(FONT_BOLD) else FONT_REGULAR
    return ImageFont.truetype(path, size=size)


PALETTE = [
    (169, 201, 229), (246, 186, 150), (174, 218, 166), (242, 166, 166),
    (198, 186, 218), (230, 205, 148), (184, 220, 218), (244, 202, 228),
    (205, 205, 205), (214, 228, 154), (152, 193, 218), (255, 218, 155),
]


def rooms_at(data, level, gt=False):
    rooms = [room for room in data["rooms"] if room["level"] == level]
    if gt:
        rooms = [room for room in rooms
                 if room.get("in_roomset") and not room.get("is_stairs")]
    return rooms


def doors_at(data, level):
    return [door for door in data["doors"] if door["level"] == level]


def bounds(room_groups, door_groups):
    points = []
    for rooms in room_groups:
        points.extend(np.asarray(room["poly"], float) for room in rooms)
    for doors in door_groups:
        points.extend(np.asarray(door["seg"], float) for door in doors)
    all_points = np.vstack(points)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    lo -= 0.06 * span
    hi += 0.06 * span
    return lo, hi


def transform_for(panel, lo, hi):
    x0, y0, x1, y1 = panel
    scale = min((x1 - x0) / (hi[0] - lo[0]),
                (y1 - y0) / (hi[1] - lo[1]))
    used_w = (hi[0] - lo[0]) * scale
    used_h = (hi[1] - lo[1]) * scale
    ox = x0 + ((x1 - x0) - used_w) / 2
    oy = y0 + ((y1 - y0) - used_h) / 2

    def tf(point):
        point = np.asarray(point, float)
        return (int(round(ox + (point[0] - lo[0]) * scale)),
                int(round(oy + used_h - (point[1] - lo[1]) * scale)))
    return tf


def draw_panel(draw, panel, title, rooms, doors, lo, hi):
    x0, y0, x1, y1 = panel
    draw.rounded_rectangle(panel, radius=12, fill=(250, 251, 252),
                           outline=(175, 182, 190), width=2)
    draw.text((x0 + 20, y0 + 12), title, font=font(24, True), fill=(28, 33, 38))
    plot = (x0 + 28, y0 + 55, x1 - 28, y1 - 28)
    tf = transform_for(plot, lo, hi)

    centroids = {}
    for index, room in enumerate(rooms):
        polygon = [tf(point) for point in np.asarray(room["poly"])]
        color = PALETTE[index % len(PALETTE)]
        draw.polygon(polygon, fill=color, outline=(70, 75, 80))
        if polygon:
            draw.line(polygon + [polygon[0]], fill=(70, 75, 80), width=2)
        centroid = np.asarray(room["poly"], float).mean(axis=0)
        centroids[room["idx"]] = tf(centroid)

    room_edges, outside_edges = derive_edges(rooms, doors)
    for edge in room_edges:
        a, b = tuple(edge)
        if a in centroids and b in centroids:
            draw.line([centroids[a], centroids[b]], fill=(85, 90, 96), width=3)
    # Exterior links are intentionally not extended to an arbitrary outside point.
    del outside_edges

    for index, room in enumerate(rooms):
        center = centroids[room["idx"]]
        label = f"R{room['idx']}\n{str(room.get('type', '?'))[:10]}"
        box = draw.multiline_textbbox((0, 0), label, font=font(17, True),
                                      spacing=1, align="center")
        width = box[2] - box[0]
        height = box[3] - box[1]
        draw.multiline_text((center[0] - width / 2, center[1] - height / 2),
                            label, font=font(17, True), fill=(20, 24, 28),
                            spacing=1, align="center")
    for door in doors:
        segment = [tf(point) for point in np.asarray(door["seg"])]
        draw.line(segment, fill=(220, 30, 65), width=7)


def pair_metrics(gt_rooms, pred_rooms):
    # Import locally to keep the drawing function independent from matching.
    from metrics import match_rooms_iou
    matches, ious = match_rooms_iou(pred_rooms, gt_rooms, 0.5)
    tp = len(matches)
    f1 = 2 * tp / (len(gt_rooms) + len(pred_rooms)) if gt_rooms or pred_rooms else 0.0
    mean_iou = float(np.mean([value[1] for value in ious.values()])) if ious else 0.0
    return tp, f1, mean_iou


def render(scene, tag, scores, prefer_upper=False):
    with open(f"{GT_DIR}/{scene}.pkl", "rb") as stream:
        gt = pickle.load(stream)
    with open(f"{BASE}/pred/{scene}.pkl", "rb") as stream:
        pred = pickle.load(stream)
    mapping = align_levels(gt["lvl_z"], pred.get("level_z", {}))
    pairs = sorted(mapping.items())
    if prefer_upper and len(pairs) > 1:
        gt_level, pred_level = max(
            pairs[1:], key=lambda pair: len(rooms_at(gt, pair[0], gt=True)))
    else:
        gt_level, pred_level = max(
            pairs, key=lambda pair: len(rooms_at(gt, pair[0], gt=True)))

    gt_rooms = rooms_at(gt, gt_level, gt=True)
    pred_rooms = rooms_at(pred, pred_level)
    gt_doors = doors_at(gt, gt_level)
    pred_doors = doors_at(pred, pred_level)
    lo, hi = bounds([gt_rooms, pred_rooms], [gt_doors, pred_doors])
    pair_tp, pair_f1, pair_iou = pair_metrics(gt_rooms, pred_rooms)
    scene_f1 = prf(*scores[scene]["A"]["room"])["f1"]
    scene_iou = float(np.mean(scores[scene]["iou"] or [0]))
    unmatched_pred = sorted(set(pred.get("level_z", {})) - set(mapping.values()))

    image = Image.new("RGB", (WIDTH, HEIGHT), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    title = (f"[{tag}] {scene} · official MP3D floor GT · "
             f"GT L{gt_level} ↔ PRED L{pred_level} "
             f"(GT {len(gt['lvl_z'])} levels / PRED {len(pred.get('level_z', {}))} levels)")
    draw.text((WIDTH / 2, 38), title, anchor="ma", font=font(30, True),
              fill=(18, 23, 28))
    draw_panel(draw, (55, 105, 935, 930),
               f"GT L{gt_level} · {len(gt_rooms)} rooms（stairs excluded）",
               gt_rooms, gt_doors, lo, hi)
    draw_panel(draw, (985, 105, 1865, 930),
               f"PRED L{pred_level} · {len(pred_rooms)} rooms",
               pred_rooms, pred_doors, lo, hi)
    footer = (f"shown pair: TP={pair_tp}, F1={pair_f1:.3f}, matched IoU={pair_iou:.3f}  |  "
              f"scene aggregate: Room F1={scene_f1:.3f}, matched IoU={scene_iou:.3f}  |  "
              f"unmatched PRED levels={unmatched_pred or 'none'}")
    draw.text((WIDTH / 2, 962), footer, anchor="ma", font=font(21),
              fill=(50, 58, 66))
    note = ("CURRENT: eval2d_v3_strict_levels + mp3d_house_floor_v0_1 · "
            "red=doors · gray=derived room-room edges · colors are independent (not matches) · "
            "room types unreliable")
    draw.text((WIDTH / 2, 1010), note, anchor="ma", font=font(18),
              fill=(150, 91, 10))
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{tag}_{scene}.png"
    image.save(path, format="PNG", optimize=True)
    return path


def _metric_f1(result, tier, key):
    return prf(*result[tier].get(key, [0, 0, 0]))["f1"]


def render_error_analysis(scores):
    """Render the current 16-scene diagnostic summary and persist its inputs."""
    rows = []
    for scene, result in scores.items():
        with open(f"{GT_DIR}/{scene}.pkl", "rb") as stream:
            gt = pickle.load(stream)
        with open(f"{BASE}/pred/{scene}.pkl", "rb") as stream:
            pred = pickle.load(stream)
        mapping = align_levels(gt["lvl_z"], pred.get("level_z", {}))
        gt_rooms = sum(room.get("in_roomset") and not room.get("is_stairs")
                       for room in gt["rooms"])
        pred_rooms = len(pred["rooms"])
        room_f1 = _metric_f1(result, "A", "room")
        matched_iou = float(np.mean(result["iou"] or [0]))
        door_f1 = _metric_f1(result, "B", "doors@0.5")
        edge_f1 = _metric_f1(result, "C", "all")
        flags = {
            "geometry_good": bool(room_f1 >= 0.65 and matched_iou >= 0.75),
            "under_segmented": bool(pred_rooms <= gt_rooms - 3),
            "level_mismatch_or_extra": bool(
                set(gt["lvl_z"]) - set(mapping)
                or set(pred.get("level_z", {})) - set(mapping.values())),
            "door_or_topology_weak": bool(door_f1 < 0.25 or edge_f1 < 0.15),
        }
        rows.append({
            "scene": scene, "room_f1": room_f1,
            "matched_room_iou": matched_iou, "gt_rooms": gt_rooms,
            "pred_rooms": pred_rooms, "doors_0.5_f1": door_f1,
            "edge_all_f1": edge_f1, "flags": flags,
        })

    rules = {
        "geometry_good": "Room F1>=0.65 and matched IoU>=0.75",
        "under_segmented": "pred room count <= GT-3",
        "level_mismatch_or_extra": "at least one GT/PRED level is unmatched",
        "door_or_topology_weak": "Doors@0.5<0.25 or edge_all<0.15",
    }
    os.makedirs(OUT, exist_ok=True)
    json_path = f"{OUT}/error_analysis.json"
    with open(json_path, "w") as stream:
        json.dump({
            "status": "current_v3_official_gt_diagnostic_nonexclusive_flags",
            "protocol": "eval2d_v3_strict_levels+mp3d_house_floor_v0_1",
            "rules": rules, "per_scene": sorted(rows, key=lambda row: row["scene"]),
        }, stream, ensure_ascii=False, indent=2)

    image = Image.new("RGB", (WIDTH, HEIGHT), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.text((WIDTH / 2, 38),
              "16-scene error analysis · strict v3 + official MP3D floor GT",
              anchor="ma", font=font(31, True), fill=(18, 23, 28))

    # Left: Room-F1 distribution.
    left = (55, 105, 930, 985)
    draw.rounded_rectangle(left, radius=12, fill=(250, 251, 252),
                           outline=(175, 182, 190), width=2)
    draw.text((80, 125), "Room F1 distribution", font=font(24, True),
              fill=(28, 33, 38))
    x0, x1 = 315, 880
    y0, gap, bar_h = 182, 47, 25
    draw.line((x0 + 0.5 * (x1 - x0), y0 - 15,
               x0 + 0.5 * (x1 - x0), y0 + 15 * gap + bar_h),
              fill=(110, 117, 124), width=2)
    for index, row in enumerate(sorted(rows, key=lambda item: item["room_f1"])):
        y = y0 + index * gap
        value = row["room_f1"]
        color = ((64, 124, 78) if value >= 0.65 else
                 (192, 130, 47) if value >= 0.5 else (169, 74, 66))
        draw.text((80, y + bar_h / 2), row["scene"][:8], anchor="lm",
                  font=font(16), fill=(50, 58, 66))
        draw.rectangle((x0, y, x0 + value * (x1 - x0), y + bar_h), fill=color)
        draw.text((x0 + value * (x1 - x0) + 8, y + bar_h / 2),
                  f"{value:.2f}", anchor="lm", font=font(15), fill=(45, 52, 58))
    draw.text(((x0 + x1) / 2, 955), "Room F1 @ IoU>0.5（dashed reference = 0.5）",
              anchor="mm", font=font(16), fill=(65, 72, 78))

    # Upper right: F1 vs conditional IoU.
    scatter = (975, 105, 1865, 610)
    draw.rounded_rectangle(scatter, radius=12, fill=(250, 251, 252),
                           outline=(175, 182, 190), width=2)
    draw.text((1000, 125), "Coverage vs successful-match quality",
              font=font(24, True), fill=(28, 33, 38))
    sx0, sy0, sx1, sy1 = 1050, 540, 1815, 190
    draw.line((sx0, sy0, sx1, sy0), fill=(80, 87, 94), width=2)
    draw.line((sx0, sy0, sx0, sy1), fill=(80, 87, 94), width=2)
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x = sx0 + tick * (sx1 - sx0)
        draw.line((x, sy0, x, sy0 + 6), fill=(80, 87, 94), width=2)
        draw.text((x, sy0 + 12), f"{tick:.2g}", anchor="ma", font=font(14),
                  fill=(70, 77, 84))
    for tick in (0.65, 0.75, 0.85, 0.95):
        y = sy0 - ((tick - 0.65) / 0.30) * (sy0 - sy1)
        draw.line((sx0 - 6, y, sx0, y), fill=(80, 87, 94), width=2)
        draw.text((sx0 - 10, y), f"{tick:.2f}", anchor="rm", font=font(14),
                  fill=(70, 77, 84))
        draw.line((sx0, y, sx1, y), fill=(225, 228, 231), width=1)
    label_offsets = {
        "17DR": (9, -15), "1LXt": (9, 7), "2t7W": (-50, -17),
        "5LpN": (-46, -12), "HxpK": (9, 0), "JeFG": (9, 0),
        "JmbY": (9, -17), "S9hN": (9, -18), "TbHJ": (9, 8),
        "WYY7": (-50, -13), "YFuZ": (9, 8), "e9zR": (9, 0),
        "i5no": (9, -8), "jtcx": (-48, -14), "p5wJ": (9, 9),
        "r47D": (9, -13),
    }
    for row in rows:
        x = sx0 + row["room_f1"] * (sx1 - sx0)
        y = sy0 - np.clip((row["matched_room_iou"] - 0.65) / 0.30, 0, 1) * (sy0 - sy1)
        color = (64, 124, 78) if row["flags"]["geometry_good"] else (169, 112, 15)
        if row["flags"]["level_mismatch_or_extra"]:
            draw.line((x - 7, y - 7, x + 7, y + 7), fill=color, width=4)
            draw.line((x - 7, y + 7, x + 7, y - 7), fill=color, width=4)
        else:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        dx, dy = label_offsets.get(row["scene"][:4], (9, -2))
        draw.text((x + dx, y + dy), row["scene"][:5], font=font(12),
                  fill=(55, 62, 68))
    draw.text(((sx0 + sx1) / 2, 585), "Room F1", anchor="mm", font=font(16),
              fill=(65, 72, 78))
    draw.text((997, 360), "matched\nIoU", anchor="mm", font=font(14),
              fill=(65, 72, 78))

    # Lower right: transparent, non-exclusive diagnostic taxonomy.
    flags_panel = (975, 650, 1865, 985)
    draw.rounded_rectangle(flags_panel, radius=12, fill=(250, 251, 252),
                           outline=(175, 182, 190), width=2)
    draw.text((1000, 670), "Non-exclusive diagnostic flags",
              font=font(24, True), fill=(28, 33, 38))
    labels = [
        ("geometry_good", "Geometry good"),
        ("under_segmented", "Under-segmented"),
        ("level_mismatch_or_extra", "Level mismatch / extra"),
        ("door_or_topology_weak", "Door / topology weak"),
    ]
    for index, (key, label) in enumerate(labels):
        hits = [row["scene"][:5] for row in rows if row["flags"][key]]
        y = 725 + index * 60
        color = (64, 124, 78) if key == "geometry_good" else (128, 55, 49)
        draw.text((1000, y), f"{label}: {len(hits)}/16", font=font(18, True), fill=color)
        draw.text((1285, y + 1), ", ".join(hits) or "—", font=font(15),
                  fill=(70, 77, 84))
    draw.text((WIDTH / 2, 1030),
              "Flags are interpretation rules, not benchmark pass/fail thresholds · "
              "matched IoU is conditional on successful room matches",
              anchor="mm", font=font(17), fill=(110, 78, 16))
    output = f"{OUT}/error_analysis_summary.png"
    image.save(output, format="PNG", optimize=True)
    return output, json_path


def main():
    with open(SCORES_PATH, "rb") as stream:
        scores = pickle.load(stream)
    room_f1 = {scene: prf(*result["A"]["room"])["f1"]
               for scene, result in scores.items()}
    ordered = sorted(room_f1, key=room_f1.get)
    worst = ordered[0]
    best = ordered[-1]
    median_value = float(np.median(list(room_f1.values())))
    representative = min(ordered, key=lambda scene: abs(room_f1[scene] - median_value))
    multi = max(scores, key=lambda scene: len(pickle.load(
        open(f"{BASE}/pred/{scene}.pkl", "rb"))["level_z"]))

    outputs = [
        render(best, "best", scores),
        render(representative, "representative", scores),
        render(worst, "worst", scores),
        render(multi, "multi_level", scores, prefer_upper=True),
    ]
    outputs.extend(render_error_analysis(scores))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
