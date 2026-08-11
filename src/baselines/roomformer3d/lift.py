#!/usr/bin/env python
"""Lift canonical RoomFormer 2D predictions to 3D HouseLayout3D entities.

Appendix E.1 of HouseLayout3D is followed directly:

* planar floor and ceiling at the lower/upper 5% input-point quantiles;
* doors from floor to 2.10 m above the floor;
* windows span the centred 80% of the wall height.

RoomFormer does not predict stairs, so no stair geometry is fabricated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import triangulate
from shapely.validation import make_valid


GRAY = np.array([191, 191, 191], dtype=np.float64) / 255.0
GREEN = np.array([0, 255, 0], dtype=np.float64) / 255.0
BLUE = np.array([0, 0, 255], dtype=np.float64) / 255.0


def _polygon_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [part for child in geom.geoms for part in _polygon_parts(child)]
    return []


def repair_polygon(points: np.ndarray) -> tuple[Polygon, str]:
    """Return one deterministic valid polygon and a repair status."""
    raw = Polygon(np.asarray(points, dtype=np.float64))
    if raw.is_valid and not raw.is_empty and raw.area > 0:
        return raw, "unchanged"
    parts = [g for g in _polygon_parts(make_valid(raw)) if g.area > 0]
    if not parts:
        raise ValueError("Room polygon cannot be repaired")
    result = max(parts, key=lambda g: g.area)
    status = "make_valid_largest_component"
    # RoomFormer emits one boundary ring per room.  A hole created solely by
    # repairing a self-intersection is not a predicted semantic hole, and a
    # touching hole can crash native constrained triangulators.
    if result.interiors:
        result = Polygon(result.exterior)
        status += "_holes_filled"
    return result, status


def o3d_mesh(vertices: np.ndarray, triangles: np.ndarray,
             color: np.ndarray = GRAY) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    if not np.isfinite(vertices).all() or len(vertices) < 3 or len(triangles) < 1:
        raise ValueError("Invalid mesh geometry")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(vertices), 1)))
    mesh.compute_vertex_normals()
    return mesh


def horizontal_mesh(polygon: Polygon, z: float,
                    color: np.ndarray = GRAY) -> o3d.geometry.TriangleMesh:
    triangles = [triangle for triangle in triangulate(polygon)
                 if polygon.covers(triangle)]
    if not triangles:
        raise ValueError("Polygon triangulation produced no interior faces")
    vertex_index = {}
    vertices2 = []
    faces = []
    for triangle in triangles:
        face = []
        for xy in list(triangle.exterior.coords)[:3]:
            key = (float(xy[0]), float(xy[1]))
            if key not in vertex_index:
                vertex_index[key] = len(vertices2)
                vertices2.append(key)
            face.append(vertex_index[key])
        faces.append(face)
    verts2 = np.asarray(vertices2, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    vertices = np.column_stack([verts2, np.full(len(verts2), float(z))])
    return o3d_mesh(vertices, faces, color)


def quad_mesh(p0: np.ndarray, p1: np.ndarray, z0: float, z1: float,
              color: np.ndarray = GRAY) -> o3d.geometry.TriangleMesh:
    vertices = np.array([
        [p0[0], p0[1], z0], [p1[0], p1[1], z0],
        [p1[0], p1[1], z1], [p0[0], p0[1], z1],
    ], dtype=np.float64)
    return o3d_mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]), color)


def wall_meshes(polygon: Polygon, z0: float, z1: float):
    meshes = []
    rings = [polygon.exterior, *polygon.interiors]
    for ring in rings:
        xy = np.asarray(ring.coords[:-1], dtype=np.float64)
        for index in range(len(xy)):
            p0, p1 = xy[index], xy[(index + 1) % len(xy)]
            if np.linalg.norm(p1 - p0) > 1e-6:
                meshes.append(quad_mesh(p0, p1, z0, z1))
    return meshes


def opening_rectangle(segment, z0: float, z1: float,
                      color: np.ndarray) -> tuple[np.ndarray, o3d.geometry.TriangleMesh]:
    segment = np.asarray(segment, dtype=np.float64)
    mesh = quad_mesh(segment[0], segment[1], z0, z1, color)
    return np.asarray(mesh.vertices), mesh


def merge_meshes(meshes: list[o3d.geometry.TriangleMesh]):
    vertices, triangles, colors = [], [], []
    offset = 0
    for mesh in meshes:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        c = np.asarray(mesh.vertex_colors)
        vertices.append(v)
        triangles.append(f + offset)
        colors.append(c)
        offset += len(v)
    if not vertices:
        return o3d.geometry.TriangleMesh()
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.concatenate(triangles)),
    )
    result.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    result.compute_vertex_normals()
    return result


def height_for(source_sample: str, records: dict) -> tuple[float, float]:
    record = records[source_sample]
    values = record.get("height_quantiles_m")
    if values is None:
        raise ValueError(f"Input manifest lacks height quantiles: {source_sample}")
    floor, ceiling = float(values["q05"]), float(values["q95"])
    if ceiling <= floor:
        raise ValueError(f"Invalid height interval for {source_sample}: {floor}, {ceiling}")
    return floor, ceiling


def lift_scene(canonical: dict, records: dict, out_dir: Path) -> dict:
    entity_dir = out_dir / "entities"
    entity_dir.mkdir(parents=True, exist_ok=False)
    all_meshes = []
    structure_meta = []
    doors_json, windows_json = [], []
    repairs = []

    def save_structure(mesh, kind, room_id, level, sample_id):
        index = len(structure_meta)
        rel = Path("entities") / f"structure_{index:05d}.ply"
        if not o3d.io.write_triangle_mesh(str(out_dir / rel), mesh,
                                          write_ascii=False,
                                          compressed=False,
                                          print_progress=False):
            raise IOError(out_dir / rel)
        structure_meta.append({
            "id": index, "kind": kind, "room_id": room_id,
            "level": level, "source_sample": sample_id, "path": str(rel),
            "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
        })
        all_meshes.append(mesh)

    for level in canonical["levels"]:
        level_id = level["id"]
        for room in level["rooms"]:
            floor, ceiling = height_for(room["source_sample"], records)
            try:
                polygon, status = repair_polygon(np.asarray(room["polygon"]))
            except ValueError as error:
                repairs.append({"room_id": room["id"], "level": level_id,
                                "status": "dropped", "reason": str(error)})
                continue
            repairs.append({"room_id": room["id"], "level": level_id,
                            "status": status})
            save_structure(horizontal_mesh(polygon, floor), "floor", room["id"],
                           level_id, room["source_sample"])
            save_structure(horizontal_mesh(polygon, ceiling), "ceiling", room["id"],
                           level_id, room["source_sample"])
            for wall in wall_meshes(polygon, floor, ceiling):
                save_structure(wall, "wall", room["id"], level_id,
                               room["source_sample"])

        for door in level["doors"]:
            floor, _ = height_for(door["source_sample"], records)
            vertices, mesh = opening_rectangle(
                door["segment"], floor, floor + 2.10, GREEN)
            doors_json.append({
                "vertices": vertices.tolist(), "level": level_id,
                "source_sample": door["source_sample"],
                "type_confidence": door.get("type_confidence"),
            })
            all_meshes.append(mesh)

        for window in level["windows"]:
            floor, ceiling = height_for(window["source_sample"], records)
            height = ceiling - floor
            vertices, mesh = opening_rectangle(
                window["segment"], floor + 0.10 * height,
                floor + 0.90 * height, BLUE)
            windows_json.append({
                "vertices": vertices.tolist(), "level": level_id,
                "source_sample": window["source_sample"],
                "type_confidence": window.get("type_confidence"),
            })
            all_meshes.append(mesh)

    with (out_dir / "doors.json").open("w") as stream:
        json.dump({"doors": doors_json}, stream, indent=2)
    with (out_dir / "windows.json").open("w") as stream:
        json.dump({"windows": windows_json}, stream, indent=2)
    with (out_dir / "stairs.json").open("w") as stream:
        json.dump({"stairs": []}, stream, indent=2)
    combined = merge_meshes(all_meshes)
    if not o3d.io.write_triangle_mesh(str(out_dir / "combined.ply"), combined,
                                      write_ascii=False, compressed=False,
                                      print_progress=False):
        raise IOError(out_dir / "combined.ply")
    return {
        "scene": canonical["scene_id"],
        "structures": len(structure_meta), "doors": len(doors_json),
        "windows": len(windows_json), "stairs": 0,
        "combined_vertices": len(combined.vertices),
        "combined_triangles": len(combined.triangles),
        "structure_entities": structure_meta, "polygon_repairs": repairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True,
                        help="2D baseline root containing <mode>/canonical")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("per_floor", "per_room"),
                        default=["per_floor", "per_room"])
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()

    with args.input_manifest.open() as stream:
        input_manifest = json.load(stream)
    records = {record["sample_id"]: record for record in input_manifest["records"]}
    all_scenes = args.scenes or sorted(
        path.stem for path in (args.canonical_root / "per_floor/canonical").glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    modes_out = {}
    for mode in args.modes:
        mode_started = time.time()
        scene_results = []
        for index, scene in enumerate(all_scenes, 1):
            source = args.canonical_root / mode / "canonical" / f"{scene}.json"
            scene_out = args.out / mode / scene
            if scene_out.exists():
                raise FileExistsError(
                    f"Refusing to mix with an existing 3D scene directory: {scene_out}")
            scene_out.mkdir(parents=True)
            with source.open() as stream:
                canonical = json.load(stream)
            result = lift_scene(canonical, records, scene_out)
            scene_results.append(result)
            with (scene_out / "lift_manifest.json").open("w") as stream:
                json.dump(result, stream, indent=2)
            print(f"{mode} [{index}/{len(all_scenes)}] {scene}: "
                  f"{result['structures']} structures, {result['doors']} doors, "
                  f"{result['windows']} windows", flush=True)
        modes_out[mode] = {
            "elapsed_seconds": time.time() - mode_started,
            "scenes": scene_results,
        }
    manifest = {
        "version": "roomformer3d_lift_v1",
        "source_2d": str(args.canonical_root),
        "input_manifest": str(args.input_manifest),
        "rules": {
            "floor_z": "input surface-point q05",
            "ceiling_z": "input surface-point q95",
            "quantile_interpretation": "lower and upper 5% tails",
            "door": "floor_z to floor_z + 2.10m",
            "window": "centered 80% of (ceiling_z-floor_z)",
            "stairs": "not predicted by RoomFormer; empty",
            "structure_decomposition": "one floor + one ceiling + one wall per room polygon edge",
        },
        "elapsed_seconds": time.time() - started,
        "modes": modes_out,
    }
    with (args.out / "lift_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"DONE 3D lift in {manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
