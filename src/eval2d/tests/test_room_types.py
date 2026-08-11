"""Unit tests for room-type feature aggregation and pruning metadata."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "src" / "eval2d"
sys.path.insert(0, str(HERE))

from geometry_v2 import to_json_geometry  # noqa: E402
from room_types import (ROOM_TYPES, aggregate_mesh_vertices,  # noqa: E402
                        aggregate_sample_points,
                        annotate_pruning_candidates,
                        classify_room_features)


def room(room_id, x0, x1):
    return {"id": room_id,
            "geometry": to_json_geometry(Polygon(
                [(x0, 0), (x1, 0), (x1, 2), (x0, 2)]))}


def level(level_id, elevation, rooms):
    return {
        "id": level_id, "elevation": elevation, "rooms": rooms,
        "graph": {"edges": []}, "stairs": [],
    }


artifact = {"levels": [
    level(0, 0.0, [room(1, 0, 2), room(2, 2, 4)]),
    level(1, 3.0, [room(1, 0, 2)]),
]}
points = np.asarray([
    [0.5, 0.5, 0.2], [1.0, 1.0, 0.4],
    [2.5, 0.5, 0.2], [3.0, 1.0, 0.4],
    [0.5, 0.5, 3.2], [1.0, 1.0, 3.4],
])
features = np.asarray([
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 0, 0], [0, 1, 0, 0],
    [0, 0, 1, 0], [0, 0, 1, 0],
], dtype=np.float32)

results = []


def check(name, condition):
    passed = bool(condition)
    results.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'} {name}")


sums, counts, records, diagnostics = aggregate_sample_points(
    points, features, artifact)
check("sample points assigned to three level-qualified rooms",
      counts.tolist() == [2, 2, 2])
check("assignment is complete and unambiguous",
      diagnostics["assignment"] == {
          "assigned": 6, "unassigned": 0, "ambiguous_hits": 0})

# k=5 projection is exercised with exact mesh/sample positions.  Counts are
# mesh-vertex counts, independent of how often a feature sample is reused.
vertex_sums, vertex_counts, vertex_records, vertex_diag = \
    aggregate_mesh_vertices(points, features, points, artifact, k=5,
                            chunk_size=2)
check("mesh-vertex path retains per-room vertex counts",
      vertex_counts.tolist() == [2, 2, 2])
check("mesh-vertex path records frozen k", vertex_diag["knn_k"] == 5)
check("variant room order is identical",
      [record["key"] for record in records] ==
      [record["key"] for record in vertex_records])

text = np.zeros((len(ROOM_TYPES), 4), dtype=np.float32)
text[0] = [1, 0, 0, 0]
text[1] = [0, 1, 0, 0]
text[10] = [0, 0, 1, 0]
annotations = classify_room_features(sums, counts, records, text)
annotations = annotate_pruning_candidates(annotations, records, artifact)
check("all 15 cosine scores preserved",
      all(len(value["scores"]) == 15 for value in annotations.values()))
check("top classes follow room feature means",
      [annotations[key]["type"] for key in
       ["L0_R1", "L0_R2", "L1_R1"]] ==
      ["bathroom", "bedroom", "spa/sauna"])
check("paper last-five leaf becomes candidate but is not deleted",
      annotations["L1_R1"]["is_paper_pruning_candidate"] is True
      and len(artifact["levels"][1]["rooms"]) == 1)
check("indoor room is not pruning candidate",
      annotations["L0_R1"]["is_paper_pruning_candidate"] is False)

print(f"\n{sum(results)}/{len(results)} PASS")
raise SystemExit(0 if all(results) else 1)
