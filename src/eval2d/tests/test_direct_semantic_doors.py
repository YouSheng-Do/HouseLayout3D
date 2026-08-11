"""Synthetic contract tests for direct semantic 2D door candidates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "eval2d"))

from direct_semantic_doors import (assign_levels, extract_candidates,  # noqa: E402
                                   nearest_wall_assignments,
                                   voxel_centers)


def level(level_id, elevation, wall):
    return {
        "id": level_id, "elevation": elevation, "walls": [wall],
        "rooms": [], "doors": [], "windows": [], "stairs": [],
        "graph": {"edges": []}, "footprint": None,
    }


artifact = {"levels": [
    level(0, 0.0, {"id": "W0", "polyline": [[0, 0], [4, 0]],
                   "adjacent_rooms": [1, 2]}),
    level(1, 3.0, {"id": "W1", "polyline": [[0, 0], [4, 0]],
                   "adjacent_rooms": [3]}),
]}

results = []


def check(name, condition):
    passed = bool(condition)
    results.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'} {name}")


raw = np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02],
                  [0.19, 0.01, 0.01]])
voxels = voxel_centers(raw, voxel_m=0.1)
check("voxelization is deterministic and removes duplicates", len(voxels) == 2)

assigned, levels = assign_levels(
    np.asarray([[1, 0, 0.2], [1, 0, 3.2], [1, 0, 7.0]]), artifact)
check("vertical assignment is level-qualified",
      assigned.tolist() == [0, 1, -1] and [row[0] for row in levels] == [0, 1])

segments = [{"p0": np.asarray([0., 0.]), "p1": np.asarray([4., 0.]),
             "direction": np.asarray([1., 0.]), "length": 4.}]
wall_index, along, distance = nearest_wall_assignments(
    np.asarray([[1., .1], [3., .4]]), segments, max_distance=.3)
check("nearest-wall rule rejects distant evidence",
      wall_index.tolist() == [0, -1] and np.isclose(along[0], 1.)
      and np.isclose(distance[1], .4))

# Dense valid door evidence spans 0.8m along W0 and 1.8m vertically.  A second
# label has identical geometry but must not be admitted.
door = []
for x in np.arange(1.0, 1.81, 0.1):
    for z in np.arange(0.1, 2.01, 0.1):
        door.append([x, 0.04, z])
noise = [[3.0 + i * 0.01, 0.03, 0.2] for i in range(20)]
points = np.asarray(door + noise)
labels = ["wall", "door"]
ray_labels = np.asarray([1] * len(door) + [0] * len(noise))
candidates, diagnostics = extract_candidates(
    points, ray_labels, np.ones(len(points), dtype=bool), labels, artifact)
check("one valid semantic door becomes one wall-snapped segment",
      len(candidates) == 1 and candidates[0]["wall_id"] == "W0")
check("candidate width and level follow evidence",
      candidates[0]["level"] == 0
      and 0.6 < candidates[0]["width_m"] < 1.0)
check("candidate preserves support and direct provenance",
      candidates[0]["support_voxels"] >= 100
      and "OneFormer_door_ray" in candidates[0]["provenance"])
check("diagnostics expose raw and filtered counts",
      diagnostics["raw_door_labeled_rays"] == len(door)
      and diagnostics["candidate_count"] == 1)

invalid = np.zeros(len(points), dtype=bool)
empty, empty_diag = extract_candidates(
    points, ray_labels, invalid, labels, artifact)
check("invalid-depth door rays cannot create candidates",
      empty == [] and empty_diag["valid_door_labeled_rays"] == 0)

print(f"\n{sum(results)}/{len(results)} PASS")
raise SystemExit(0 if all(results) else 1)
