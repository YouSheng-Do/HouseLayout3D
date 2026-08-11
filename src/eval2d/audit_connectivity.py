"""Create a deterministic 30-case visual audit of derived Tier-C connectivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_derive import Door, Room, derive_access_graph  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


ROOT = "/home/ado/storage/HouseLayout3D"
GT_DIR = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
SPLIT = f"{ROOT}/configs/eval2d/split_v0_1.json"
MANUAL_LABELS = f"{ROOT}/configs/eval2d/connectivity_audit_manual_v0_1.json"
OUT = f"{ROOT}/outputs/eval2d/connectivity_audit_v0_1"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
TARGETS = {"success": 10, "one_outside": 10, "same_room": 9, "overlap": 1}
PALETTE = [
    (169, 201, 229), (246, 186, 150), (174, 218, 166),
    (242, 166, 166), (198, 186, 218), (230, 205, 148),
    (184, 220, 218), (244, 202, 228), (214, 228, 154),
]


def font(size, bold=False):
    path = FONT_BOLD if bold and os.path.exists(FONT_BOLD) else FONT
    return ImageFont.truetype(path, size=size)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_gt(scene, manifest):
    path = os.path.join(GT_DIR, f"{scene}.pkl")
    if sha256(path) != manifest["files_sha256"][f"{scene}.pkl"]:
        raise RuntimeError(f"GT hash mismatch: {scene}")
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _containing(point, rooms):
    p = Point(point)
    return [room["idx"] for room in rooms if Polygon(room["poly"]).contains(p)]


def _march_empty_side(midpoint, normal, hits_a, hits_b, rooms,
                      start_m=0.35, stop_m=1.50, step_m=0.05):
    """Audit-only heuristic: look beyond the empty 0.30 m probe.

    A later room hit is only a thick-gap *candidate*.  No portal/exterior
    annotation exists, so neither branch is treated as connectivity truth.
    """
    if bool(hits_a) == bool(hits_b):
        return None
    direction = -normal if hits_a else normal
    for distance in np.arange(start_m, stop_m + step_m / 2.0, step_m):
        hits = _containing(midpoint + distance * direction, rooms)
        if hits:
            return {
                "class": "far_room_after_gap_candidate",
                "first_hit_distance_m": float(distance),
                "first_hits": hits,
            }
    return {
        "class": "no_included_room_within_1.5m_exterior_candidate",
        "first_hit_distance_m": None,
        "first_hits": [],
    }


def collect_cases(scenes, manifest):
    cases = []
    totals = Counter()
    for scene in scenes:
        gt = load_verified_gt(scene, manifest)
        for level in sorted(gt["lvl_z"]):
            included = [room for room in gt["rooms"]
                        if room["level"] == level and room["in_roomset"]]
            excluded = [room for room in gt["rooms"]
                        if room["level"] == level and not room["in_roomset"]]
            doors = [(index, door) for index, door in enumerate(gt["doors"])
                     if door["level"] == level]
            room_inputs = [Room(room["idx"], room["type"], room["poly"])
                           for room in included]
            door_inputs = [Door(index, door["seg"][0], door["seg"][1])
                           for index, door in doors]
            _, records = derive_access_graph(room_inputs, door_inputs, d=0.30)
            record_by_id = {record["door_id"]: record for record in records}
            support = unary_union([Polygon(room["poly"]) for room in included])
            for door_id, door in doors:
                record = record_by_id[door_id]
                totals[record["outcome"]] += 1
                segment = np.asarray(door["seg"], dtype=float)
                midpoint = segment.mean(axis=0)
                tangent = segment[1] - segment[0]
                length = float(np.linalg.norm(tangent))
                normal = np.array([-tangent[1], tangent[0]]) / max(length, 1e-12)
                probe_a = midpoint + 0.30 * normal
                probe_b = midpoint - 0.30 * normal
                march = _march_empty_side(
                    midpoint, normal, record["probe_a"], record["probe_b"],
                    included) if record["outcome"] == "one_outside" else None
                cases.append({
                    "case_key": f"{scene}:L{level}:D{door_id}",
                    "scene": scene, "level": level, "door_id": door_id,
                    "outcome": record["outcome"],
                    "segment": segment.tolist(), "length_m": length,
                    "midpoint": midpoint.tolist(),
                    "probe_a": probe_a.tolist(), "probe_b": probe_b.tolist(),
                    "hits_a": record["probe_a"], "hits_b": record["probe_b"],
                    "empty_side_march": march,
                    "excluded_hits_a": _containing(probe_a, excluded),
                    "excluded_hits_b": _containing(probe_b, excluded),
                    "midpoint_to_included_union_boundary_m": float(
                        support.boundary.distance(Point(midpoint))),
                    "included_rooms": [{"id": room["idx"],
                                        "type": room["type"],
                                        "poly": np.asarray(room["poly"]).tolist()}
                                       for room in included],
                    "excluded_rooms": [{"id": room["idx"],
                                        "type": room["type"],
                                        "poly": np.asarray(room["poly"]).tolist()}
                                       for room in excluded],
                })
    return cases, totals


def select_cases(cases):
    selected = []
    for outcome, count in TARGETS.items():
        candidates = [case for case in cases if case["outcome"] == outcome]
        candidates.sort(key=lambda case: hashlib.sha256(
            case["case_key"].encode()).hexdigest())
        if len(candidates) < count:
            raise RuntimeError(f"not enough {outcome}: {len(candidates)} < {count}")
        selected.extend(candidates[:count])
    selected.sort(key=lambda case: (
        list(TARGETS).index(case["outcome"]), case["case_key"]))
    for index, case in enumerate(selected, 1):
        case["audit_id"] = f"C{index:02d}"
    return selected


def render_case(draw, panel, case):
    x0, y0, x1, y1 = panel
    draw.rounded_rectangle(panel, radius=10, fill=(250, 251, 252),
                           outline=(170, 178, 186), width=2)
    draw.text((x0 + 14, y0 + 10),
              f"{case['audit_id']} · {case['outcome']} · {case['scene']} L{case['level']} D{case['door_id']}",
              font=font(17, True), fill=(22, 27, 32))
    midpoint = np.asarray(case["midpoint"])
    radius = 3.0
    plot = (x0 + 18, y0 + 48, x1 - 18, y1 - 66)
    scale = min((plot[2] - plot[0]) / (2 * radius),
                (plot[3] - plot[1]) / (2 * radius))

    def transform(point):
        point = np.asarray(point)
        return (int(plot[0] + (point[0] - midpoint[0] + radius) * scale),
                int(plot[3] - (point[1] - midpoint[1] + radius) * scale))

    for index, room in enumerate(case["included_rooms"]):
        polygon = [transform(point) for point in room["poly"]]
        draw.polygon(polygon, fill=PALETTE[index % len(PALETTE)],
                     outline=(65, 72, 78))
        centre = np.asarray(room["poly"]).mean(axis=0)
        if np.all(np.abs(centre - midpoint) <= radius):
            draw.text(transform(centre), f"R{room['id']} {room['type']}",
                      anchor="mm", font=font(13, True), fill=(25, 30, 35))
    for room in case["excluded_rooms"]:
        polygon = [transform(point) for point in room["poly"]]
        draw.polygon(polygon, fill=(220, 220, 220), outline=(100, 100, 100))

    draw.line([transform(point) for point in case["segment"]],
              fill=(225, 25, 55), width=7)
    for point, colour, label in ((case["probe_a"], (20, 90, 220), "A"),
                                 (case["probe_b"], (230, 125, 15), "B")):
        px, py = transform(point)
        draw.ellipse((px-6, py-6, px+6, py+6), fill=colour,
                     outline=(255, 255, 255), width=2)
        draw.text((px + 9, py), label, anchor="lm", font=font(13, True),
                  fill=colour)
    footer = (f"A={case['hits_a']}  B={case['hits_b']}  "
              f"excluded A/B={case['excluded_hits_a']}/{case['excluded_hits_b']}  "
              f"boundary d={case['midpoint_to_included_union_boundary_m']:.2f}m")
    draw.text((x0 + 14, y1 - 42), footer, font=font(13), fill=(45, 52, 60))


def render_pages(cases, out_dir):
    page_paths = []
    width, height = 1800, 1200
    for page_index in range(0, len(cases), 6):
        image = Image.new("RGB", (width, height), (239, 242, 245))
        draw = ImageDraw.Draw(image)
        page = cases[page_index:page_index + 6]
        for local, case in enumerate(page):
            row, column = divmod(local, 3)
            # Render into a separate tile so off-crop polygons are clipped at
            # panel boundaries instead of spilling into neighbouring cases.
            tile = Image.new("RGB", (550, 540), (250, 251, 252))
            render_case(ImageDraw.Draw(tile), (0, 0, 549, 539), case)
            image.paste(tile, (30 + column * 590, 30 + row * 580))
        path = os.path.join(out_dir, f"cases_{page_index + 1:02d}_{page_index + len(page):02d}.png")
        image.save(path, optimize=True)
        page_paths.append(path)
    return page_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--manual-labels", default=MANUAL_LABELS)
    args = parser.parse_args()

    split = load_split(args.split)
    scenes = select_scenes(split, "all", args.checkpoint_name)
    manifest_path = os.path.join(GT_DIR, "candidate_manifest.json")
    with open(manifest_path) as stream:
        gt_manifest = json.load(stream)
    all_cases, totals = collect_cases(scenes, gt_manifest)
    selected = select_cases(all_cases)
    os.makedirs(args.out, exist_ok=True)
    pages = render_pages(selected, args.out)
    payload = {
        "status": "geometry_consistency_audit_not_independent_connectivity_ground_truth",
        "checkpoint_name": args.checkpoint_name,
        "split": os.path.relpath(args.split, ROOT),
        "split_sha256": sha256(args.split),
        "gt_manifest": os.path.relpath(manifest_path, ROOT),
        "gt_manifest_sha256": sha256(manifest_path),
        "probe_distance_m": 0.30,
        "population_outcomes": dict(totals),
        "one_outside_march_outcomes": dict(Counter(
            case["empty_side_march"]["class"] for case in all_cases
            if case["empty_side_march"] is not None)),
        "one_outside_march_contract": (
            "audit-only 0.35m..1.50m in 0.05m steps; candidate heuristic, "
            "not independent exterior/thick-wall truth"),
        "sample_targets": TARGETS,
        "selection": "SHA-256(case_key) within each outcome stratum",
        "cases": selected,
    }
    cases_path = os.path.join(args.out, "auto_cases.json")
    with open(cases_path, "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
        stream.write("\n")

    with open(args.manual_labels) as stream:
        manual = json.load(stream)
    selected_keys = {(case["audit_id"], case["case_key"], case["outcome"])
                     for case in selected}
    manual_keys = {(case["audit_id"], case["case_key"], case["outcome"])
                   for case in manual["labels"]}
    if selected_keys != manual_keys or len(manual["labels"]) != 30:
        raise RuntimeError("manual connectivity labels do not exactly match the selected cases")
    manual["auto_cases_sha256"] = sha256(cases_path)
    manual["manual_source"] = os.path.relpath(args.manual_labels, ROOT)
    manual["manual_source_sha256"] = sha256(args.manual_labels)
    review_path = os.path.join(args.out, "manual_review.json")
    with open(review_path, "w") as stream:
        json.dump(manual, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
        stream.write("\n")
    output_hashes = {os.path.basename(path): sha256(path) for path in pages}
    output_hashes["auto_cases.json"] = sha256(cases_path)
    output_hashes["manual_review.json"] = sha256(review_path)
    with open(os.path.join(args.out, "manifest.json"), "w") as stream:
        json.dump({"outputs": output_hashes,
                   "manual_labels_sha256": sha256(args.manual_labels),
                   "source_sha256": sha256(__file__)}, stream,
                  ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"population={dict(totals)}; selected={dict(Counter(c['outcome'] for c in selected))}")
    print(f"30-case audit -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
