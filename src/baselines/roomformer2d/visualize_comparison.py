#!/usr/bin/env python
"""Render GT / per-floor / per-room predictions in one metric-frame image."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval2d"))
from house_floor_gt import build_scene

from common import load_openings, scene_ids


PANEL = 640
MARGIN = 45
COLORS = [(242, 120, 98), (102, 194, 165), (141, 160, 203),
          (231, 138, 195), (166, 216, 84), (255, 217, 47),
          (229, 196, 148), (179, 179, 179)]


def mapper(bounds):
    lo, hi = bounds
    span = np.maximum(hi - lo, 1e-6)
    scale = (PANEL - 2 * MARGIN) / max(span)

    def convert(points):
        p = np.asarray(points, float)
        x = MARGIN + (p[:, 0] - lo[0]) * scale
        y = PANEL - MARGIN - (p[:, 1] - lo[1]) * scale
        return [tuple(xy) for xy in np.column_stack([x, y])]
    return convert


def panel(title, rooms, doors, windows, bounds, gt=False):
    image = Image.new("RGB", (PANEL, PANEL), (20, 22, 26))
    draw = ImageDraw.Draw(image)
    convert = mapper(bounds)
    for index, room in enumerate(rooms):
        pts = convert(room["poly"])
        color = (190, 190, 190) if gt else COLORS[index % len(COLORS)]
        draw.polygon(pts, fill=tuple(int(c * 0.20) for c in color), outline=color)
        draw.line(pts + [pts[0]], fill=color, width=3)
    for opening in windows:
        draw.line(convert(opening["seg"]), fill=(60, 175, 255), width=6)
    for opening in doors:
        draw.line(convert(opening["seg"]), fill=(255, 75, 65), width=6)
    draw.rectangle((0, 0, PANEL, 34), fill=(5, 7, 10))
    draw.text((12, 10), f"{title}  rooms={len(rooms)} doors={len(doors)} "
              f"windows={len(windows)}", fill="white", font=ImageFont.load_default())
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in scene_ids():
        gt = build_scene(scene)
        gt["windows"] = load_openings(scene, "windows", gt["lvl_z"])
        predictions = {}
        for mode in ("per_floor", "per_room"):
            with (args.root / mode / "pred" / f"{scene}.pkl").open("rb") as stream:
                predictions[mode] = pickle.load(stream)
        for level in sorted(gt["lvl_z"]):
            gt_rooms = [r for r in gt["rooms"] if r["level"] == level
                        and r["in_roomset"] and not r["is_stairs"]]
            if not gt_rooms:
                continue
            all_xy = np.concatenate([np.asarray(r["poly"]) for r in gt_rooms])
            pad = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.04
            bounds = (all_xy.min(0) - pad, all_xy.max(0) + pad)
            panels = []
            for title, data, is_gt in (
                    ("GT", gt, True),
                    ("RoomFormer per-floor", predictions["per_floor"], False),
                    ("RoomFormer per-room", predictions["per_room"], False)):
                panels.append(panel(
                    title,
                    [r for r in data["rooms"] if r["level"] == level],
                    [d for d in data["doors"] if d["level"] == level],
                    [w for w in data.get("windows", []) if w["level"] == level],
                    bounds, gt=is_gt,
                ))
            canvas = Image.new("RGB", (PANEL * 3, PANEL + 32), (5, 7, 10))
            for index, image in enumerate(panels):
                canvas.paste(image, (PANEL * index, 32))
            ImageDraw.Draw(canvas).text(
                (12, 11), f"{scene} / level {level} - red=door, blue=window",
                fill="white", font=ImageFont.load_default())
            canvas.save(args.out / f"{scene}__l{level:02d}.png")
    print(f"Wrote {len(list(args.out.glob('*.png')))} comparison images to {args.out}")


if __name__ == "__main__":
    main()
