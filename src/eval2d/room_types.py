"""CPU-only room-type aggregation over retained OpenSeg features.

Two explicitly named variants share the same canonical rooms and OpenSeg
samples:

* ``sample_point_room_mean`` directly averages backprojected sample features;
* ``mesh_vertex_k5_room_mean`` first projects them to structural mesh vertices
  with the official mesh-segmentation KNN value k=5, then averages vertices.

The latter follows Appendix D.4's stated aggregation path.  Retained OpenSeg
sampling and room/vertex association details remain reproducible local
interpretations because the released Stage-4 implementation is absent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import shapely
from scipy.spatial import cKDTree

from geometry_v2 import to_shapely


ROOM_TYPES = [
    "bathroom", "bedroom", "living room", "garage", "entrance",
    "kitchen", "office", "stairs", "gym", "classroom",
    "spa/sauna", "mirror", "grass/bushes/trees", "driveway",
    "veranda/terrace/balcony",
]
N_OUTDOOR_TAIL = 5
KNN_K = 5
FLOOR_BELOW_TOLERANCE_M = 0.30
LEVEL_VERTICAL_EXTENT_M = 3.50


def read_ply_vertices(path):
    """Memory-map a binary PLY and return an N×3 coordinate array."""
    from plyfile import PlyData

    data = PlyData.read(str(path), mmap=True)["vertex"].data
    vertices = np.column_stack((data["x"], data["y"], data["z"]))
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid mesh vertices {vertices.shape}")
    return vertices


def room_records(artifact):
    records = []
    for level in sorted(artifact["levels"], key=lambda item: int(item["id"])):
        for room in sorted(level["rooms"], key=lambda item: str(item["id"])):
            records.append({
                "key": f"L{level['id']}_R{room['id']}",
                "level": int(level["id"]),
                "room_id": room["id"],
                "geometry": to_shapely(room["geometry"]),
            })
    return records


def assign_positions_to_rooms(positions, artifact):
    """Assign each 3D position to at most one canonical room.

    A vertex/sample belongs to the highest predicted floor at or below z+0.3m
    and must lie no more than 3.5m above that floor.  Within a level, stable
    room-ID order resolves the rare overlapping polygon case.  Ambiguity is
    counted rather than hidden.
    """
    positions = np.asarray(positions, dtype=np.float64)
    records = room_records(artifact)
    levels = sorted(
        ((int(level["id"]), float(level["elevation"]))
         for level in artifact["levels"]),
        key=lambda item: (item[1], item[0]))
    if not levels:
        return np.full(len(positions), -1, dtype=np.int32), records, {
            "assigned": 0, "unassigned": len(positions), "ambiguous_hits": 0}
    elevations = np.asarray([item[1] for item in levels])
    floor_position = np.searchsorted(
        elevations, positions[:, 2] + FLOOR_BELOW_TOLERANCE_M,
        side="right") - 1
    valid = floor_position >= 0
    clipped = np.clip(floor_position, 0, len(levels) - 1)
    valid &= positions[:, 2] <= (
        elevations[clipped] + LEVEL_VERTICAL_EXTENT_M)
    assignment = np.full(len(positions), -1, dtype=np.int32)
    ambiguous = 0
    by_level = defaultdict(list)
    for room_index, record in enumerate(records):
        by_level[record["level"]].append((room_index, record))
    for level_position, (level_id, _) in enumerate(levels):
        indices = np.flatnonzero(valid & (floor_position == level_position))
        if not len(indices):
            continue
        x, y = positions[indices, 0], positions[indices, 1]
        local = np.full(len(indices), -1, dtype=np.int32)
        for room_index, record in by_level[level_id]:
            hits = shapely.intersects_xy(record["geometry"], x, y)
            ambiguous += int(np.count_nonzero(hits & (local >= 0)))
            local[hits & (local < 0)] = room_index
        assignment[indices] = local
    return assignment, records, {
        "assigned": int(np.count_nonzero(assignment >= 0)),
        "unassigned": int(np.count_nonzero(assignment < 0)),
        "ambiguous_hits": ambiguous,
    }


def _accumulate_feature_rows(features, assignment, n_rooms,
                             chunk_size=4096):
    dimension = int(features.shape[1])
    sums = np.zeros((n_rooms, dimension), dtype=np.float64)
    counts = np.zeros(n_rooms, dtype=np.int64)
    indices = np.flatnonzero(assignment >= 0)
    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start:start + chunk_size]
        values = np.asarray(features[chunk_indices], dtype=np.float32)
        labels = assignment[chunk_indices]
        for room_index in np.unique(labels):
            mask = labels == room_index
            sums[room_index] += values[mask].sum(axis=0, dtype=np.float64)
            counts[room_index] += int(mask.sum())
    return sums, counts


def aggregate_sample_points(points, features, artifact):
    assignment, records, assignment_stats = assign_positions_to_rooms(
        points, artifact)
    sums, counts = _accumulate_feature_rows(
        features, assignment, len(records))
    return sums, counts, records, {
        "assignment": assignment_stats,
        "feature_projection": "direct_backprojected_OpenSeg_sample_points",
        "knn_k": None,
    }


def aggregate_mesh_vertices(points, features, mesh_vertices, artifact,
                            k=KNN_K, chunk_size=2048):
    assignment, records, assignment_stats = assign_positions_to_rooms(
        mesh_vertices, artifact)
    if len(points) < k:
        raise ValueError(f"need at least {k} OpenSeg points")
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    dimension = int(features.shape[1])
    sums = np.zeros((len(records), dimension), dtype=np.float64)
    counts = np.zeros(len(records), dtype=np.int64)
    distance_sums = np.zeros(len(records), dtype=np.float64)
    indices = np.flatnonzero(assignment >= 0)
    for start in range(0, len(indices), chunk_size):
        vertex_indices = indices[start:start + chunk_size]
        distances, neighbors = tree.query(
            mesh_vertices[vertex_indices], k=k, workers=1)
        neighbor_features = np.asarray(features[neighbors], dtype=np.float32)
        vertex_features = neighbor_features.mean(axis=1)
        labels = assignment[vertex_indices]
        nearest = distances[:, 0] if distances.ndim == 2 else distances
        for room_index in np.unique(labels):
            mask = labels == room_index
            sums[room_index] += vertex_features[mask].sum(
                axis=0, dtype=np.float64)
            counts[room_index] += int(mask.sum())
            distance_sums[room_index] += float(nearest[mask].sum())
    mean_nearest = [
        float(distance_sums[index] / counts[index])
        if counts[index] else None
        for index in range(len(records))
    ]
    return sums, counts, records, {
        "assignment": assignment_stats,
        "feature_projection": (
            "OpenSeg_samples_to_structural_mesh_vertices_KNN_then_room_mean"),
        "knn_k": int(k),
        "mean_nearest_sample_distance_m_by_room": mean_nearest,
    }


def classify_room_features(sums, counts, records, text_embeddings):
    text = np.asarray(text_embeddings, dtype=np.float32)
    if text.shape != (len(ROOM_TYPES), sums.shape[1]):
        raise ValueError(
            f"text embedding shape {text.shape} incompatible with {sums.shape}")
    annotations = {}
    for index, record in enumerate(records):
        if counts[index] <= 0:
            annotations[record["key"]] = {
                "type": "unknown", "scores": None, "confidence": None,
                "feature_count": 0, "status": "no_assigned_features",
            }
            continue
        average = np.asarray(sums[index], dtype=np.float64)
        average /= np.linalg.norm(average) + 1e-12
        similarities = text @ average.astype(np.float32)
        order = np.argsort(-similarities, kind="stable")
        top = int(order[0])
        annotations[record["key"]] = {
            "type": ROOM_TYPES[top],
            "scores": {
                name: round(float(similarities[class_index]), 6)
                for class_index, name in enumerate(ROOM_TYPES)
            },
            "confidence": round(
                float(similarities[order[0]] - similarities[order[1]]), 6),
            "feature_count": int(counts[index]),
            "status": "classified_argmax_cosine",
        }
    return annotations


def room_degrees(artifact):
    degrees = {
        (int(level["id"]), room["id"]): 0
        for level in artifact["levels"] for room in level["rooms"]
    }
    for level in artifact["levels"]:
        level_id = int(level["id"])
        for edge in level["graph"]["edges"]:
            if edge["kind"] == "stair":
                continue
            for room_id in edge["rooms"]:
                key = (level_id, room_id)
                if key in degrees:
                    degrees[key] += 1
        for stair in level["stairs"]:
            if len(stair["adjacent_rooms"]) != 2 or stair["to_level"] is None:
                continue
            endpoints = [
                (int(stair["from_level"]), stair["adjacent_rooms"][0]),
                (int(stair["to_level"]), stair["adjacent_rooms"][1]),
            ]
            for endpoint in endpoints:
                if endpoint in degrees:
                    degrees[endpoint] += 1
    return degrees


def annotate_pruning_candidates(annotations, records, artifact):
    degrees = room_degrees(artifact)
    outdoor_classes = set(ROOM_TYPES[-N_OUTDOOR_TAIL:])
    for record in records:
        annotation = annotations[record["key"]]
        degree = degrees[(record["level"], record["room_id"])]
        is_outdoor = annotation["type"] in outdoor_classes
        annotation["graph_degree"] = int(degree)
        annotation["is_outdoor_candidate"] = bool(is_outdoor)
        annotation["is_paper_pruning_candidate"] = bool(
            is_outdoor and degree <= 1)
    return annotations
