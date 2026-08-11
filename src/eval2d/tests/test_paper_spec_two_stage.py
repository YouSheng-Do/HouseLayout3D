"""Synthetic contracts for Appendix-D.3 two-stage bottleneck behavior."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_spec_two_stage import (  # noqa: E402
    _opening_from_border, split_once, two_stage_labels,
)


RES = 0.10


def dumbbell(connector_width_px, small_right=False):
    mask = np.zeros((90, 150), dtype=bool)
    mask[10:70, 10:70] = True
    if small_right:
        mask[30:50, 90:110] = True
        y0 = 40 - connector_width_px // 2
        mask[y0:y0 + connector_width_px, 70:90] = True
    else:
        mask[10:70, 90:150] = True
        y0 = 40 - connector_width_px // 2
        mask[y0:y0 + connector_width_px, 70:90] = True
    return mask


def main():
    checks = {}

    # Two large cells joined by a 2.0 m neck split during the 2.5 m pass
    # and the second pass must never merge them again.
    mask = dumbbell(20, small_right=False)
    coarse = split_once(mask, 2.5, RES)
    final, diag = two_stage_labels(mask, RES)
    checks["2.5m pass splits two large cells across 2.0m neck"] = (
        coarse.n_output_cells == 2)
    checks["1.5m refinement does not re-merge coarse cells"] = (
        int(final.max()) == 2 and diag["coarse_cells"] == 2)

    # The 2 m small cell has no 2.5 m core and is initially absorbed, but a
    # 1.0 m neck separates it at the 1.5 m refinement scale.
    mask = dumbbell(10, small_right=True)
    coarse = split_once(mask, 2.5, RES)
    final, diag = two_stage_labels(mask, RES)
    checks["small 2m cell is not invented during coarse pass"] = (
        coarse.n_output_cells == 1)
    checks["small 2m cell survives 1.5m refinement"] = int(final.max()) == 2

    # A 2.0 m opening is wider than the fine threshold, so the same small
    # region remains part of the large cell.
    mask = dumbbell(20, small_right=True)
    final, _ = two_stage_labels(mask, RES)
    checks["2.0m opening remains one cell at 1.5m scale"] = int(final.max()) == 1

    # Disconnected components smaller than the coarse scale cannot disappear
    # or be merged by nearest-seed assignment.
    disconnected = np.zeros((60, 100), dtype=bool)
    disconnected[10:25, 10:25] = True
    disconnected[30:45, 70:85] = True
    final, _ = two_stage_labels(disconnected, RES)
    checks["disconnected small cells are preserved"] = int(final.max()) == 2

    empty, diag = two_stage_labels(np.zeros((20, 20), dtype=bool), RES)
    checks["empty floor is deterministic empty"] = (
        int(empty.max()) == 0 and diag["final_cells"] == 0)

    border = np.zeros((20, 20), dtype=bool)
    border[5:15, 8] = True
    segment, width, rectangle = _opening_from_border(
        border, np.array([0.0, 0.0]), RES)
    checks["bottleneck stores finite oriented rectangle"] = (
        segment.shape == (2, 2) and rectangle.shape == (4, 2)
        and np.isfinite(rectangle).all())
    checks["bottleneck width follows rectangle major axis"] = (
        abs(width - 0.9) < 1e-9)

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
