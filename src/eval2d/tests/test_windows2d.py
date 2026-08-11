"""Regression contracts for released-window projection and strict levels."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_windows2d import official_windows, score_scene  # noqa: E402


def window(idx, level, x):
    return {
        "idx": idx, "level": level,
        "seg": np.array([[x, 0.0], [x, 1.0]], dtype=float),
    }


def main():
    checks = {}
    gt_z = {0: 0.0, 1: 3.0}
    gt = [window("g0", 0, 0.0), window("g1", 1, 5.0)]
    pred = [window("p0", 10, 0.0), window("p1", 11, 5.0)]
    identity = score_scene(gt_z, gt, {10: 0.0, 11: 3.0}, pred)
    checks["identity windows@0.2"] = identity["counts"]["0.2"] == [2, 0, 0]
    checks["identity windows@0.5"] = identity["counts"]["0.5"] == [2, 0, 0]

    extra = pred + [window("extra", 12, 9.0)]
    strict = score_scene(gt_z, gt, {10: 0.0, 11: 3.0, 12: 7.0}, extra)
    checks["unmatched predicted level counts FP"] = (
        strict["counts"]["0.5"] == [2, 1, 0]
        and strict["unmatched_pred_levels"] == [12])

    released = official_windows("2t7WUuJeko7", {0: 0.0})
    checks["released rectangle projection count stable"] = len(released) == 15
    checks["released rectangle projects finite bottom edges"] = all(
        item["seg"].shape == (2, 2) and np.isfinite(item["seg"]).all()
        for item in released)

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
