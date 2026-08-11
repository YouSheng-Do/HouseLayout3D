"""CPU-only 2D door candidates from retained OneFormer ray endpoints.

This is an annotated-best extension, not a paper method.  Door-labeled valid
ray endpoints are voxelized, assigned to predicted levels, snapped to the
nearest retained Stage-3 wall segment, and grouped along that wall.  The
module deliberately has no GT or evaluator dependency.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


VOXEL_M = 0.10
FLOOR_BELOW_TOLERANCE_M = 0.30
LEVEL_VERTICAL_EXTENT_M = 3.20
WALL_DISTANCE_M = 0.30
MAX_MEAN_WALL_DISTANCE_M = 0.15
CLUSTER_GAP_M = 0.35
MIN_VOXELS = 100
MIN_VERTICAL_SPAN_M = 1.40
MIN_WIDTH_M = 0.35
MAX_WIDTH_M = 1.50
DUPLICATE_MIDPOINT_M = 0.35


def voxel_centers(points, voxel_m=VOXEL_M):
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / voxel_m).astype(np.int64)
    unique = np.unique(keys, axis=0)
    return (unique.astype(np.float64) + 0.5) * voxel_m


def assign_levels(points, artifact):
    """Assign points to the highest predicted floor below the point."""
    levels = sorted(
        ((int(level["id"]), float(level["elevation"]))
         for level in artifact["levels"]),
        key=lambda item: (item[1], item[0]))
    assignment = np.full(len(points), -1, dtype=np.int32)
    if not levels or not len(points):
        return assignment, levels
    elevations = np.asarray([item[1] for item in levels])
    position = np.searchsorted(
        elevations, points[:, 2] + FLOOR_BELOW_TOLERANCE_M,
        side="right") - 1
    valid = position >= 0
    clipped = np.clip(position, 0, len(levels) - 1)
    valid &= points[:, 2] <= elevations[clipped] + LEVEL_VERTICAL_EXTENT_M
    assignment[valid] = position[valid]
    return assignment, levels


def wall_segments(level):
    segments = []
    for wall in sorted(level["walls"], key=lambda item: str(item["id"])):
        line = np.asarray(wall["polyline"], dtype=np.float64)
        for segment_index in range(len(line) - 1):
            p0, p1 = line[segment_index:segment_index + 2]
            length = float(np.linalg.norm(p1 - p0))
            if length <= 1e-9:
                continue
            segments.append({
                "wall_id": wall["id"],
                "segment_index": segment_index,
                "p0": p0,
                "p1": p1,
                "direction": (p1 - p0) / length,
                "length": length,
                "adjacent_rooms": list(wall["adjacent_rooms"]),
            })
    return segments


def nearest_wall_assignments(xy, segments, max_distance=WALL_DISTANCE_M):
    """Return segment indices, along-wall metres, and perpendicular distance."""
    n = len(xy)
    best_segment = np.full(n, -1, dtype=np.int32)
    best_along = np.zeros(n, dtype=np.float64)
    best_distance = np.full(n, np.inf, dtype=np.float64)
    for index, segment in enumerate(segments):
        delta = xy - segment["p0"]
        along = np.clip(delta @ segment["direction"], 0, segment["length"])
        closest = (segment["p0"][None, :] +
                   along[:, None] * segment["direction"][None, :])
        distance = np.linalg.norm(xy - closest, axis=1)
        better = distance < best_distance
        best_distance[better] = distance[better]
        best_along[better] = along[better]
        best_segment[better] = index
    best_segment[best_distance > max_distance] = -1
    return best_segment, best_along, best_distance


def split_by_gap(indices, along, gap_m=CLUSTER_GAP_M):
    if not len(indices):
        return []
    order = indices[np.argsort(along[indices], kind="stable")]
    cuts = np.flatnonzero(np.diff(along[order]) > gap_m) + 1
    return [group for group in np.split(order, cuts) if len(group)]


def _candidate(segment, local_group, global_group, along, points, level_id):
    if len(local_group) < MIN_VOXELS:
        return None
    z = points[global_group, 2]
    z_span = float(np.quantile(z, 0.95) - np.quantile(z, 0.05))
    if z_span < MIN_VERTICAL_SPAN_M:
        return None
    lo, hi = np.quantile(along[local_group], [0.05, 0.95])
    width = float(hi - lo)
    if not MIN_WIDTH_M <= width <= MAX_WIDTH_M:
        return None
    p0 = segment["p0"] + lo * segment["direction"]
    p1 = segment["p0"] + hi * segment["direction"]
    return {
        "level": int(level_id),
        "segment": np.asarray([p0, p1], dtype=np.float64),
        "width_m": width,
        "wall_id": segment["wall_id"],
        "wall_segment_index": int(segment["segment_index"]),
        "adjacent_rooms": segment["adjacent_rooms"],
        "support_voxels": int(len(local_group)),
        "vertical_span_m": z_span,
        "mean_wall_distance_m": None,
        "provenance": "OneFormer_door_ray_wall_snap_local_extension",
    }


def deduplicate_candidates(candidates, midpoint_m=DUPLICATE_MIDPOINT_M):
    kept = []
    ordered = sorted(
        candidates,
        key=lambda item: (-item["support_voxels"], item["level"],
                          item["wall_id"], item["wall_segment_index"]))
    for candidate in ordered:
        midpoint = candidate["segment"].mean(axis=0)
        duplicate = any(
            other["level"] == candidate["level"] and
            np.linalg.norm(other["segment"].mean(axis=0) - midpoint) <
            midpoint_m
            for other in kept)
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (
        item["level"], float(item["segment"].mean(axis=0)[0]),
        float(item["segment"].mean(axis=0)[1]), item["wall_id"]))


def extract_candidates(ray_points, ray_labels, ray_valid, labels, artifact):
    """Extract semantic candidates and return them with auditable diagnostics."""
    labels = [str(value) for value in labels]
    if "door" not in labels:
        raise ValueError("semantic label list has no door class")
    ray_points = np.asarray(ray_points)
    ray_labels = np.asarray(ray_labels)
    ray_valid = np.asarray(ray_valid, dtype=bool)
    if not (len(ray_points) == len(ray_labels) == len(ray_valid)):
        raise ValueError("ray point/label/valid lengths differ")
    raw_mask = ray_labels == labels.index("door")
    valid_points = np.asarray(ray_points[raw_mask & ray_valid], dtype=np.float64)
    points = voxel_centers(valid_points)
    level_assignment, ordered_levels = assign_levels(points, artifact)
    artifact_levels = {int(level["id"]): level for level in artifact["levels"]}
    candidates = []
    level_diagnostics = []
    for level_position, (level_id, _) in enumerate(ordered_levels):
        indices = np.flatnonzero(level_assignment == level_position)
        segments = wall_segments(artifact_levels[level_id])
        if len(indices) and segments:
            assigned_segment, along, distance = nearest_wall_assignments(
                points[indices, :2], segments)
        else:
            assigned_segment = np.full(len(indices), -1, dtype=np.int32)
            along = np.zeros(len(indices), dtype=np.float64)
            distance = np.full(len(indices), np.inf, dtype=np.float64)
        level_candidates = []
        for segment_index, segment in enumerate(segments):
            local_indices = np.flatnonzero(assigned_segment == segment_index)
            for local_group in split_by_gap(local_indices, along):
                global_group = indices[local_group]
                candidate = _candidate(
                    segment, local_group, global_group, along, points,
                    level_id)
                if candidate is None:
                    continue
                candidate["mean_wall_distance_m"] = float(
                    distance[local_group].mean())
                if (candidate["mean_wall_distance_m"] >
                        MAX_MEAN_WALL_DISTANCE_M):
                    continue
                level_candidates.append(candidate)
        candidates.extend(level_candidates)
        level_diagnostics.append({
            "level": level_id,
            "door_voxels": int(len(indices)),
            "wall_assigned_voxels": int(np.count_nonzero(
                assigned_segment >= 0)),
            "candidate_count_before_dedup": len(level_candidates),
            "wall_segment_count": len(segments),
        })
    before = len(candidates)
    candidates = deduplicate_candidates(candidates)
    diagnostics = {
        "raw_door_labeled_rays": int(np.count_nonzero(raw_mask)),
        "valid_door_labeled_rays": int(len(valid_points)),
        "door_voxels": int(len(points)),
        "unassigned_level_voxels": int(np.count_nonzero(
            level_assignment < 0)),
        "candidate_count_before_dedup": before,
        "candidate_count": len(candidates),
        "levels": level_diagnostics,
        "contract": {
            "voxel_m": VOXEL_M,
            "wall_distance_m": WALL_DISTANCE_M,
            "max_mean_wall_distance_m": MAX_MEAN_WALL_DISTANCE_M,
            "cluster_gap_m": CLUSTER_GAP_M,
            "min_voxels": MIN_VOXELS,
            "vertical_span_m": [MIN_VERTICAL_SPAN_M, None],
            "width_m": [MIN_WIDTH_M, MAX_WIDTH_M],
            "duplicate_midpoint_m": DUPLICATE_MIDPOINT_M,
        },
    }
    return candidates, diagnostics
