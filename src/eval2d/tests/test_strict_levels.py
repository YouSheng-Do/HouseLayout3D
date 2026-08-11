"""Regression tests for eval2d_v3 unmatched-level false-positive accounting.

Run CPU-only:
    CUDA_VISIBLE_DEVICES="" python src/eval2d/tests/test_strict_levels.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import score_scene  # noqa: E402


def room(idx, level, x0, y0, x1, y1, room_type="bedroom"):
    return {
        "idx": idx,
        "level": level,
        "poly": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float),
        "type": room_type,
        "in_roomset": True,
        "is_stairs": False,
    }


def door(level, x, y0, y1):
    return {"level": level, "seg": np.array([[x, y0], [x, y1]], float)}


def fixture():
    gt = {
        "scene": "strict-level-fixture",
        "lvl_z": {0: 0.0},
        "rooms": [room(1, 0, 0, 0, 2, 2)],
        "doors": [door(0, 0, 0.5, 1.5)],
    }
    pred = {
        "scene": "strict-level-fixture",
        "level_z": {0: 0.05, 1: 3.0},
        "rooms": [
            room("matched", 0, 0, 0, 2, 2),
            room("extra", 1, 10, 0, 12, 2),
        ],
        "doors": [
            door(0, 0, 0.5, 1.5),
            door(1, 10, 0.5, 1.5),
        ],
        "edges": [],
    }
    return gt, pred


def main():
    gt, pred = fixture()
    strict = score_scene(gt, pred)
    archived = score_scene(gt, pred, count_unmatched_pred_levels=False)

    checks = {
        "strict room FP": strict["A"]["room"] == [1, 1, 0],
        "strict room+type FP": strict["B"]["room+type"] == [1, 1, 0],
        "strict door@0.2 FP": strict["B"]["doors@0.2"] == [1, 1, 0],
        "strict door@0.5 FP": strict["B"]["doors@0.5"] == [1, 1, 0],
        "strict exterior-edge FP": strict["C"]["outside"] == [1, 1, 0],
        "strict all-edge FP": strict["C"]["all"] == [1, 1, 0],
        "diagnostic identifies extra level": strict["level_diagnostics"]["unmatched_pred"] == [1],
        "diagnostic says FP counted": strict["level_diagnostics"]["unmatched_pred_counted_as_fp"] is True,
        "archived v2 room behavior": archived["A"]["room"] == [1, 0, 0],
        "archived v2 door behavior": archived["B"]["doors@0.5"] == [1, 0, 0],
        "archived v2 edge behavior": archived["C"]["all"] == [1, 0, 0],
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    passed = sum(checks.values())
    print(f"\n{passed}/{len(checks)} PASS")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
