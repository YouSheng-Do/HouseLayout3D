"""Shared 2D stair projection and link helpers.

Released HouseLayout3D stair annotations are triangle meshes.  They provide
geometry, but no official room or level-link labels.  This module therefore
keeps footprint projection separate from prediction-side link metadata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geometry_v2 import to_json_geometry, to_shapely


STAIR_LEVEL_Z_TOLERANCE_M = 0.15


def stair_rect_footprint(vertices):
    """Return the valid XY footprint of a Stage-4 D.5 stair rectangle.

    D.5 stores the two short endpoint edges as ``(0,1)`` and ``(2,3)``.
    Consequently the perimeter order is ``0,1,3,2`` rather than the raw
    storage order.  Using ``0,1,2,3`` creates a self-intersecting bow-tie.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (4, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"invalid Stage-4 stair rectangle {vertices.shape}")
    geometry = to_shapely(Polygon(vertices[[0, 1, 3, 2], :2]))
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("empty Stage-4 stair footprint")
    return geometry


def mesh_footprint(mesh):
    """Project a released triangle mesh to an exact polygonal XY union."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(vertices) or not len(triangles):
        raise ValueError("released stair mesh is empty")
    polygons = []
    for triangle in triangles:
        polygon = Polygon(vertices[triangle, :2])
        if polygon.is_valid and polygon.area > 1e-10:
            polygons.append(polygon)
    geometry = to_shapely(unary_union(polygons))
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("released stair mesh has no projected area")
    return geometry


def owning_level(z_min, level_z, tolerance_m=STAIR_LEVEL_Z_TOLERANCE_M):
    """Assign a stair entity to the highest floor at/below its lower extent.

    The small tolerance absorbs annotation noise around a floor plane.  This
    is an owning-level rule for 2D footprint evaluation, not a derived
    ``to_level`` label.
    """
    if not level_z:
        raise ValueError("cannot assign stair without floor elevations")
    z_min = float(z_min)
    eligible = [(level, float(z)) for level, z in level_z.items()
                if float(z) <= z_min + float(tolerance_m)]
    if eligible:
        return max(eligible, key=lambda item: (item[1], str(item[0])))[0]
    return min(level_z, key=lambda level: (float(level_z[level]), str(level)))


def official_stair_entities(scene, level_z, stairs_root):
    """Load all released entities for ``scene`` without grouping fragments."""
    import open3d as o3d

    scene_dir = Path(stairs_root) / scene
    entities = []
    for path in sorted(scene_dir.glob("*.ply")) if scene_dir.exists() else []:
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        geometry = mesh_footprint(mesh)
        entities.append({
            "idx": path.stem,
            "level": owning_level(vertices[:, 2].min(), level_z),
            "geometry": to_json_geometry(geometry),
            "z_min": float(vertices[:, 2].min()),
            "z_max": float(vertices[:, 2].max()),
            "source_file": str(path),
            "link_status": "unavailable_in_released_annotation",
        })
    return entities


def normalize_stage4_link(stair, level_z):
    """Normalize a direct D.5 endpoint pair to lower→upper elevation order."""
    rooms = stair.get("rooms")
    if not isinstance(rooms, list) or len(rooms) != 2:
        raise ValueError("Stage-4 stair must have exactly two endpoint rooms")
    endpoints = []
    for endpoint in rooms:
        if not isinstance(endpoint, list) or len(endpoint) != 2:
            raise ValueError(f"invalid Stage-4 stair endpoint {endpoint}")
        level, room = endpoint
        if level not in level_z:
            raise ValueError(f"stair references unknown level {level}")
        endpoints.append((int(level), room, float(level_z[level])))
    endpoints.sort(key=lambda item: (item[2], item[0], str(item[1])))
    return {
        "from_level": endpoints[0][0],
        "to_level": endpoints[1][0],
        "adjacent_rooms": [endpoints[0][1], endpoints[1][1]],
        "elevation_delta_m": endpoints[1][2] - endpoints[0][2],
    }
