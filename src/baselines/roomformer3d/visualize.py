#!/usr/bin/env python
"""Create side-by-side 3D wireframe views of the two lifted modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def equal_axes(ax, points):
    points = np.concatenate(points)
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    radius = max(hi - lo) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(lo[2], hi[2])


def draw_scene(ax, canonical, records, title):
    all_points = []
    for level in canonical["levels"]:
        for room in level["rooms"]:
            q = records[room["source_sample"]]["height_quantiles_m"]
            xy = np.asarray(room["polygon"], dtype=float)
            for z, color in ((q["q05"], "#73c2fb"), (q["q95"], "#ffb347")):
                xyz = np.column_stack([xy, np.full(len(xy), z)])
                closed = np.vstack([xyz, xyz[0]])
                ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color,
                        linewidth=0.8, alpha=0.85)
                all_points.append(xyz)
            step = max(1, len(xy) // 8)
            for i in range(0, len(xy), step):
                ax.plot([xy[i, 0], xy[i, 0]], [xy[i, 1], xy[i, 1]],
                        [q["q05"], q["q95"]], color="#aaaaaa", linewidth=0.5)
        for item, color, width in [
                *[(x, "red", 2.0) for x in level["doors"]],
                *[(x, "dodgerblue", 2.0) for x in level["windows"]]]:
            q = records[item["source_sample"]]["height_quantiles_m"]
            if color == "red":
                z0, z1 = q["q05"], q["q05"] + 2.10
            else:
                h = q["q95"] - q["q05"]
                z0, z1 = q["q05"] + 0.1 * h, q["q05"] + 0.9 * h
            seg = np.asarray(item["segment"])
            rectangle = np.array([
                [seg[0, 0], seg[0, 1], z0], [seg[1, 0], seg[1, 1], z0],
                [seg[1, 0], seg[1, 1], z1], [seg[0, 0], seg[0, 1], z1],
                [seg[0, 0], seg[0, 1], z0],
            ])
            ax.plot(rectangle[:, 0], rectangle[:, 1], rectangle[:, 2],
                    color=color, linewidth=width)
    if all_points:
        equal_axes(ax, all_points)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input_manifest.open() as stream:
        m = json.load(stream)
    records = {r["sample_id"]: r for r in m["records"]}
    fig = plt.figure(figsize=(16, 7))
    for index, mode in enumerate(("per_floor", "per_room"), 1):
        path = args.baseline_root / mode / "canonical" / f"{args.scene}.json"
        with path.open() as stream:
            canonical = json.load(stream)
        draw_scene(fig.add_subplot(1, 2, index, projection="3d"), canonical,
                   records, f"RoomFormer {mode}")
    fig.suptitle(f"{args.scene}: Appendix E.1 3D lifting")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(args.out)


if __name__ == "__main__":
    main()
