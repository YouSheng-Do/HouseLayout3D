"""Integration validation for the 16-scene official MP3D floor GT adapter."""
import glob
import os
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from house_floor_gt import ROOT, build_scene, overlap_stats  # noqa: E402


def main():
    scenes = sorted(os.path.basename(path)[:-6] for path in glob.glob(
        f"{ROOT}/data/mp3d/v1/scans/*/*/house_segmentations/*.house"))
    checks = []
    totals = {"levels": 0, "regions": 0, "rooms": 0, "doors": 0, "stairs": 0}
    overlap_excess = overlap_union = 0.0

    for scene in scenes:
        gt = build_scene(scene)
        ids = [r["idx"] for r in gt["rooms"]]
        checks.append((f"{scene}: unique region IDs", len(ids) == len(set(ids))))
        for room in gt["rooms"]:
            geom = Polygon(room["poly"])
            checks.append((f"{scene}/R{room['idx']}: valid finite CCW",
                           np.isfinite(room["poly"]).all() and geom.is_valid
                           and geom.area > 0 and geom.exterior.is_ccw))
        totals["levels"] += len(gt["lvl_z"])
        totals["regions"] += len(gt["rooms"])
        totals["rooms"] += sum(r["in_roomset"] and not r["is_stairs"] for r in gt["rooms"])
        totals["doors"] += len(gt["doors"])
        totals["stairs"] += sum(r["is_stairs"] for r in gt["rooms"])
        stats = overlap_stats(gt)
        overlap_excess += stats["overlap_excess_m2"]
        overlap_union += stats["union_m2"]

    checks.extend([
        ("16 official scenes", len(scenes) == 16),
        ("32 MP3D levels", totals["levels"] == 32),
        ("354/354 regions have one valid floor", totals["regions"] == 354),
        ("explicit roomset has 325 rooms", totals["rooms"] == 325),
        ("23 MP3D stair regions retained", totals["stairs"] == 23),
        ("292 HouseLayout3D doors retained", totals["doors"] == 292),
        ("direct-floor overlap below 0.2%",
         overlap_union > 0 and overlap_excess / overlap_union < 0.002),
    ])

    failures = [name for name, ok in checks if not ok]
    print(f"validated {len(scenes)} scenes / {totals['regions']} polygons")
    print(f"totals: {totals}")
    print(f"overlap: {100 * overlap_excess / overlap_union:.4f}%")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS {len(checks)}/{len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
