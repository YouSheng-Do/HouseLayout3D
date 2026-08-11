"""Regression tests for the 2D stair footprint/link contract."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "src" / "eval2d"
sys.path.insert(0, str(HERE))

from evaluate_stairs2d import score_scene  # noqa: E402
from geometry_v2 import to_json_geometry, to_shapely  # noqa: E402
from stairs2d import (official_stair_entities, owning_level,  # noqa: E402
                      normalize_stage4_link, stair_rect_footprint)


RESULTS = []


def check(name, value):
    passed = bool(value)
    RESULTS.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'} {name}")


rect = np.asarray([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [0.0, 2.0, 2.0], [1.0, 2.0, 2.0],
])
footprint = stair_rect_footprint(rect)
check("endpoint-edge storage reorders to valid perimeter", footprint.is_valid)
check("reordered rectangle area exact", abs(footprint.area - 2.0) < 1e-12)

link = normalize_stage4_link(
    {"rooms": [[2, "upper"], [0, "lower"]]},
    {0: -1.0, 1: 0.2, 2: 2.5})
check("direct D.5 link normalized low to high",
      link["from_level"] == 0 and link["to_level"] == 2)
check("room ordering follows normalized levels",
      link["adjacent_rooms"] == ["lower", "upper"])
check("owning-floor tolerance absorbs annotation noise",
      owning_level(-0.08, {0: 0.0, 1: 3.0}) == 0)
check("owning-floor chooses highest floor below lower extent",
      owning_level(1.2, {0: 0.0, 1: 3.0}) == 0)

geometry = to_json_geometry(footprint)
gt = [{"idx": "gt", "level": 0, "geometry": geometry}]
pred = [{
    "idx": "pred", "level": 0, "geometry": geometry,
    "from_level": 0, "to_level": 1, "adjacent_rooms": [1, 2],
}]
identity = score_scene({0: 0.0}, gt, {0: 0.0}, pred)
check("identity footprint match", identity["counts"] == [1, 0, 0])
check("prediction direct link coverage preserved",
      identity["prediction_link_coverage"]["level_link_coverage"] == 1.0)
check("released-only Stair-link score stays N/A",
      identity["stair_link_metric"] is None)
strict = score_scene({0: 0.0}, [], {0: 0.0, 1: 3.0}, [dict(pred[0], level=1,
                                                            from_level=1)])
check("unmatched predicted level counts FP", strict["counts"] == [0, 1, 0])

stair_root = ROOT / "external" / "houselayout3d" / "data" / "stairs"
gt_root = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
           "mp3d_house_floor_v0_1")
entities = []
for gt_path in sorted(gt_root.glob("*.pkl")):
    with gt_path.open("rb") as stream:
        scene_gt = pickle.load(stream)
    entities.extend(official_stair_entities(
        gt_path.stem, scene_gt["lvl_z"], stair_root))
check("all 34 released stair entities retained", len(entities) == 34)
check("released footprints finite valid positive", all(
    to_shapely(entity["geometry"]).is_valid
    and to_shapely(entity["geometry"]).area > 0
    for entity in entities))
check("released links explicitly unavailable", all(
    entity["link_status"] == "unavailable_in_released_annotation"
    for entity in entities))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} PASS")
raise SystemExit(0 if all(RESULTS) else 1)
