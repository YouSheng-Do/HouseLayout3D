"""Reproducible interpretation of HouseLayout3D Appendix D.3.

The paper specifies two applications of HOV-SG-style morphology-based room
segmentation, first with a 2.5 m bottleneck and then with 1.5 m, but the
released HouseLayout3D Stage-4 implementation is empty and HOV-SG's public
code does not expose these two width parameters.  This module therefore
implements the stated geometry contract explicitly:

1. a bottleneck width ``w`` is a Euclidean erosion radius ``w / 2``;
2. connected eroded cores are room seeds;
3. pixels in each connected free-space component are assigned to the nearest
   seed;
4. the 1.5 m pass is applied independently inside every 2.5 m cell, so it can
   add small cells without merging coarse cells again.

This is ``paper_spec_two_stage``, not an author-code or literal HOV-SG port.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class SplitPass:
    labels: np.ndarray
    cores: np.ndarray
    n_input_components: int
    n_seed_components: int
    n_output_cells: int


def _stable_relabel(labels: np.ndarray) -> np.ndarray:
    """Relabel cells by centroid (y, x), independent of incidental IDs."""
    rows = []
    for old_id in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == old_id)
        if len(xs):
            rows.append((float(ys.mean()), float(xs.mean()), old_id))
    output = np.zeros(labels.shape, dtype=np.int32)
    for new_id, (_, _, old_id) in enumerate(sorted(rows), 1):
        output[labels == old_id] = new_id
    return output


def split_once(free_mask: np.ndarray, bottleneck_m: float,
               resolution_m: float) -> SplitPass:
    """Split free space at bottlenecks narrower than ``bottleneck_m``."""
    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2:
        raise ValueError("free_mask must be 2D")
    if bottleneck_m <= 0 or resolution_m <= 0:
        raise ValueError("bottleneck and resolution must be positive")

    components, n_components = ndimage.label(free, structure=CONNECTIVITY)
    labels = np.zeros(free.shape, dtype=np.int32)
    cores_all = np.zeros(free.shape, dtype=bool)
    next_id = 1
    seed_total = 0

    for component_id in range(1, n_components + 1):
        component = components == component_id
        distance_m = ndimage.distance_transform_edt(component) * resolution_m
        cores = component & (distance_m >= bottleneck_m / 2.0)
        core_labels, n_cores = ndimage.label(cores, structure=CONNECTIVITY)
        cores_all |= cores
        seed_total += int(n_cores)

        if n_cores == 0:
            # A disconnected cell smaller than this pass' scale must survive
            # so the second, narrower pass can still represent it.
            labels[component] = next_id
            next_id += 1
            continue

        _, (nearest_y, nearest_x) = ndimage.distance_transform_edt(
            core_labels == 0, return_indices=True)
        assigned = core_labels[nearest_y, nearest_x]
        for local_id in range(1, n_cores + 1):
            region = component & (assigned == local_id)
            if region.any():
                labels[region] = next_id
                next_id += 1

    labels = _stable_relabel(labels)
    return SplitPass(
        labels=labels,
        cores=cores_all,
        n_input_components=int(n_components),
        n_seed_components=seed_total,
        n_output_cells=int(labels.max()),
    )


def two_stage_labels(free_mask: np.ndarray, resolution_m: float,
                     first_bottleneck_m: float = 2.5,
                     second_bottleneck_m: float = 1.5):
    """Apply the 2.5 m pass, then refine every cell with the 1.5 m pass."""
    if first_bottleneck_m <= second_bottleneck_m:
        raise ValueError("first bottleneck must be wider than second")
    coarse = split_once(free_mask, first_bottleneck_m, resolution_m)
    final = np.zeros(coarse.labels.shape, dtype=np.int32)
    fine_cores = np.zeros(coarse.labels.shape, dtype=bool)
    fine_seed_total = 0
    next_id = 1
    final_to_coarse = {}

    for coarse_id in range(1, coarse.n_output_cells + 1):
        cell = coarse.labels == coarse_id
        fine = split_once(cell, second_bottleneck_m, resolution_m)
        fine_cores |= fine.cores
        fine_seed_total += fine.n_seed_components
        for fine_id in range(1, fine.n_output_cells + 1):
            region = fine.labels == fine_id
            if region.any():
                final[region] = next_id
                final_to_coarse[next_id] = coarse_id
                next_id += 1

    final = _stable_relabel(final)
    # Stable relabeling can change final IDs; recover their unique coarse ID.
    stable_to_coarse = {}
    for final_id in range(1, int(final.max()) + 1):
        coarse_hits = np.unique(coarse.labels[final == final_id])
        coarse_hits = coarse_hits[coarse_hits > 0]
        if len(coarse_hits) != 1:
            raise RuntimeError("fine cell crossed a coarse boundary")
        stable_to_coarse[final_id] = int(coarse_hits[0])

    diagnostics = {
        "first_bottleneck_m": float(first_bottleneck_m),
        "second_bottleneck_m": float(second_bottleneck_m),
        "coarse_cells": coarse.n_output_cells,
        "coarse_seed_components": coarse.n_seed_components,
        "fine_seed_components": int(fine_seed_total),
        "final_cells": int(final.max()),
        "final_to_coarse": stable_to_coarse,
        "coarse_labels": coarse.labels,
        "coarse_cores": coarse.cores,
        "fine_cores": fine_cores,
    }
    return final, diagnostics


def _opening_from_border(border: np.ndarray, origin: np.ndarray,
                         resolution_m: float):
    ys, xs = np.where(border)
    points = (np.c_[xs, ys] * resolution_m + np.asarray(origin)
              + resolution_m / 2.0)
    if not len(points):
        return None
    if len(points) == 1:
        rectangle = np.repeat(points, 4, axis=0)
        return points[[0, 0]], 0.0, rectangle
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    direction = vh[0]
    perpendicular = np.array([-direction[1], direction[0]])
    positions = (points - centre) @ direction
    transverse = (points - centre) @ perpendicular
    segment = np.array([
        centre + positions.min() * direction,
        centre + positions.max() * direction,
    ])
    rectangle = np.array([
        centre + positions.min() * direction + transverse.min() * perpendicular,
        centre + positions.min() * direction + transverse.max() * perpendicular,
        centre + positions.max() * direction + transverse.max() * perpendicular,
        centre + positions.max() * direction + transverse.min() * perpendicular,
    ])
    return segment, float(positions.max() - positions.min()), rectangle


def segment_level(level, wall_buffer_m: float = 0.08,
                  resolution_m: float = 0.05,
                  first_bottleneck_m: float = 2.5,
                  second_bottleneck_m: float = 1.5,
                  door_max_width_m: float = 1.5):
    """Apply paper-spec D.3 to an already identified Stage-4 ``Level``.

    The 0.08 m wall raster adapter is intentionally shared with watershed_v3
    so this A/B isolates D.3 segmentation.  It is not claimed as a paper-
    specified parameter.
    """
    import scene_graph as stage4

    if abs(resolution_m - stage4.RES) > 1e-12:
        raise ValueError("experiment resolution must match Stage-4 raster")
    grid_full, origin = stage4._rasterize(level.floorplan)
    wall_mask = np.zeros(grid_full.shape, dtype=np.uint8)
    for wall in level.walls:
        geometry = stage4.poly_2d(wall).buffer(wall_buffer_m)
        stage4._fill_geom(wall_mask, geometry, origin, 1)
    free = grid_full & ~wall_mask.astype(bool)

    free_labels, diagnostics = two_stage_labels(
        free, resolution_m,
        first_bottleneck_m=first_bottleneck_m,
        second_bottleneck_m=second_bottleneck_m,
    )
    final = free_labels.copy()
    fill_mask = grid_full & (final == 0)
    if fill_mask.any() and (final > 0).any():
        _, (nearest_y, nearest_x) = ndimage.distance_transform_edt(
            final == 0, return_indices=True)
        final[fill_mask] = final[nearest_y, nearest_x][fill_mask]

    level.rooms = []
    for room_id in range(1, int(final.max()) + 1):
        level.rooms.append({
            "id": room_id,
            "mask": final == room_id,
            "coarse_id": diagnostics["final_to_coarse"][room_id],
        })

    level.openings = []
    dilation_structure = np.ones((3, 3), dtype=bool)
    for index, room_a in enumerate(level.rooms):
        free_a = free_labels == room_a["id"]
        for room_b in level.rooms[index + 1:]:
            free_b = free_labels == room_b["id"]
            border = ndimage.binary_dilation(
                free_a, structure=dilation_structure) & free_b
            opening = _opening_from_border(border, origin, resolution_m)
            if opening is None:
                continue
            segment, width, rectangle = opening
            stage = ("2.5m" if room_a["coarse_id"] != room_b["coarse_id"]
                     else "1.5m")
            level.openings.append({
                "rooms": (room_a["id"], room_b["id"]),
                "width": width,
                "seg": segment,
                "oriented_rectangle": rectangle,
                "is_door": bool(width < door_max_width_m),
                "bottleneck_stage": stage,
                "provenance": "paper_spec_morphology_bottleneck_boundary",
            })

    level._grid_origin = origin
    level._grid = grid_full
    level._carved = free
    level._final = final
    level._paper_spec_diagnostics = {
        key: value for key, value in diagnostics.items()
        if not isinstance(value, np.ndarray)
    }
    level._paper_spec_diagnostics.update({
        "wall_buffer_m_shared_ab_adapter": float(wall_buffer_m),
        "resolution_m": float(resolution_m),
        "opening_count": len(level.openings),
        "door_count": sum(opening["is_door"] for opening in level.openings),
    })
    return level
