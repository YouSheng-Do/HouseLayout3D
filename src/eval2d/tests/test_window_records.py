"""Pure geometry/provenance contracts for annotated window records."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "stage4"))
from windows import (  # noqa: E402
    LEGACY_WINDOW_RAY_CLASSES, WINDOW_RAY_CLASSES, _record_from_cluster,
)


def main():
    checks = {}
    checks["paper ray classes add outdoor"] = (
        WINDOW_RAY_CLASSES == (*LEGACY_WINDOW_RAY_CLASSES, "outdoor"))

    # Vertical x=0 wall, 1.0m horizontal span and 1.5m height.
    points = np.array([
        [0.0, y, z]
        for y in np.linspace(0.0, 1.0, 5)
        for z in np.linspace(0.5, 2.0, 4)
    ])
    ray_names = ["outdoor" if index % 2 else "window"
                 for index in range(len(points))]
    wall = SimpleNamespace(
        plane=np.array([1.0, 0.0, 0.0, 0.0]), pid=17)
    record = _record_from_cluster(
        points, wall, 3, ray_names, WINDOW_RAY_CLASSES, 2)
    checks["record fits expected width/height"] = (
        abs(record["width_m"] - 1.0) < 1e-9
        and abs(record["height_m"] - 1.5) < 1e-9)
    checks["record exposes finite 2D wall interval"] = (
        record["segment_2d"].shape == (2, 2)
        and np.isfinite(record["segment_2d"]).all()
        and abs(np.linalg.norm(
            record["segment_2d"][1] - record["segment_2d"][0]) - 1.0) < 1e-9)
    checks["record preserves wall identity"] = (
        record["wall_index"] == 3 and record["wall_pid"] == 17)
    checks["record preserves ray-class evidence"] = (
        record["ray_class_counts"] == {"outdoor": 10, "window": 10}
        and record["included_ray_classes"] == list(WINDOW_RAY_CLASSES))
    checks["record provenance is explicit"] = (
        record["provenance"] ==
        "paper_window_raycast_appendix_b_local_clustering")

    too_small = points.copy()
    too_small[:, 1] *= 0.2
    checks["sub-30cm width rejected"] = _record_from_cluster(
        too_small, wall, 3, ray_names, WINDOW_RAY_CLASSES, 2) is None

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
