"""Shared, auditable geometry and Matterport3D I/O for panorama baselines.

The local panorama frame used here is ``(right, forward, up)``.  A VP
alignment matrix maps aligned-frame vectors back to the original stitched
panorama frame.  The Matterport camera pose then maps those vectors into the
native MP3D world frame.  No room geometry is used in either transform.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parents[3]
MP3D_ROOT = ROOT / "data/mp3d/v1/scans"
GT_ROOT = ROOT / "outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
BASE_OUT = ROOT / "outputs/eval2d/baselines/panorama_layout_v0_1"
DATA_OUT = BASE_OUT / "data"

ROOM_EXCLUDE = {"x", "Z"}
ROOM_METRIC_EXCLUDE = {"x", "Z", "s"}
SKYBOX_RE = re.compile(r"^([0-9a-f]+)_skybox([0-5])_sami\.jpg$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_ids() -> List[str]:
    return sorted(path.name for path in MP3D_ROOT.iterdir()
                  if (path / path.name / "house_segmentations" /
                      (path.name + ".house")).exists())


def scene_root(scene: str) -> Path:
    return MP3D_ROOT / scene / scene


def house_path(scene: str) -> Path:
    return scene_root(scene) / "house_segmentations" / (scene + ".house")


def skybox_dir(scene: str) -> Path:
    return scene_root(scene) / "matterport_skybox_images"


def camera_pose_path(scene: str, pano_id: str) -> Path:
    # Matterport's 18 perspective poses are not six skybox-face poses.  EDM
    # (CVPR 2025) established that the 12th one-based pose, pose_1_5 in the
    # released ordering, has the direction of skybox image 1.  PanFusion's
    # released convention identifies image 1 as the left cubemap face.
    return (scene_root(scene) / "matterport_camera_poses" /
            f"{pano_id}_pose_1_5.txt")


def signed_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return 0.5 * float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]))


def parse_house(scene: str) -> dict:
    """Parse regions, panorama centers and ordered floor polygons."""
    header = {}
    regions: Dict[int, dict] = {}
    panoramas: List[dict] = []
    surface_region: Dict[int, int] = {}
    surface_kind: Dict[int, str] = {}
    vertices: Dict[int, list] = defaultdict(list)

    with house_path(scene).open(errors="ignore") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            kind = fields[0]
            if kind == "H":
                header = {
                    "n_images": int(fields[3]),
                    "n_panoramas": int(fields[4]),
                    "n_regions": int(fields[10]),
                    "n_levels": int(fields[12]),
                }
            elif kind == "R":
                regions[int(fields[1])] = {
                    "region_id": int(fields[1]),
                    "level": int(fields[2]),
                    "label": fields[5],
                    "zlo": float(fields[11]),
                    "height": float(fields[15]),
                }
            elif kind == "P":
                panoramas.append({
                    "pano_id": fields[1],
                    "pano_index": int(fields[2]),
                    "region_id_raw": int(fields[3]),
                    "position": np.asarray(fields[5:8], dtype=np.float64),
                })
            elif kind == "S":
                surface_region[int(fields[1])] = int(fields[2])
                surface_kind[int(fields[1])] = fields[4]
            elif kind == "V":
                vertices[int(fields[2])].append(
                    (float(fields[4]), float(fields[5]), float(fields[6])))

    if header.get("n_regions") != len(regions):
        raise ValueError(f"{scene}: region count mismatch")
    if header.get("n_panoramas") != len(panoramas):
        raise ValueError(f"{scene}: panorama count mismatch")

    floor_candidates: Dict[int, list] = defaultdict(list)
    for surface_id, points in vertices.items():
        if surface_kind.get(surface_id) == "F":
            region_id = surface_region[surface_id]
            floor_candidates[region_id].append(
                (surface_id, np.asarray(points, dtype=np.float64)))

    floors = {}
    for region_id, region in sorted(regions.items()):
        candidates = floor_candidates.get(region_id, [])
        if len(candidates) != 1:
            raise ValueError(
                f"{scene}: region {region_id} has {len(candidates)} floor surfaces")
        surface_id, points3 = candidates[0]
        points2 = points3[:, :2]
        polygon = Polygon(points2)
        if (len(points2) < 3 or not np.isfinite(points2).all()
                or not polygon.is_valid or polygon.area <= 0):
            raise ValueError(f"{scene}: invalid floor polygon for region {region_id}")
        floors[region_id] = {
            "surface_id": surface_id,
            "points": points2,
            "z": float(np.median(points3[:, 2])),
            "area": float(polygon.area),
        }
        region["floor_z"] = floors[region_id]["z"]

    level_values: Dict[int, list] = defaultdict(list)
    for region in regions.values():
        level_values[region["level"]].append(region["zlo"])
    level_z = {level: float(np.median(values))
               for level, values in sorted(level_values.items())}

    # Derive one robust camera-z center per floor from non-stair, assigned
    # panoramas, then classify *all* viewpoints by nearest center.  This is a
    # pose-z rule: stair regions in MP3D can span several physical levels, so
    # blindly copying their single region level would be wrong.
    level_camera_values: Dict[int, list] = defaultdict(list)
    for pano in panoramas:
        region = regions.get(pano["region_id_raw"])
        if region is not None and region["label"] not in ROOM_METRIC_EXCLUDE:
            level_camera_values[region["level"]].append(pano["position"][2])
    level_camera_z = {}
    for level in sorted(level_z):
        values = level_camera_values.get(level, [])
        if not values:
            # This fallback should be rare and remains an explicit z rule.
            level_camera_z[level] = level_z[level] + 1.5
        else:
            level_camera_z[level] = float(np.median(values))

    panoramas.sort(key=lambda item: item["pano_index"])
    for pano in panoramas:
        raw_region = pano["region_id_raw"]
        region = regions.get(raw_region)
        assignment_source = "official_house_P"
        if region is None:
            # A few official P records use region=-1.  Assign a room only when
            # the XY center is unambiguously covered by one floor polygon;
            # otherwise preserve it as unassigned instead of inventing GT.
            point = Point(pano["position"][:2])
            containing = [
                region_id for region_id, floor in floors.items()
                if Polygon(floor["points"]).covers(point)
            ]
            if containing:
                raw_region = min(containing, key=lambda key: floors[key]["area"])
                region = regions[raw_region]
                assignment_source = "geometric_exact_for_official_minus1"
            else:
                assignment_source = "unassigned_official_minus1"

        if region is not None:
            pano["region_id"] = raw_region
            pano["region_level"] = region["level"]
            pano["region_label"] = region["label"]
        else:
            pano["region_id"] = None
            pano["region_level"] = None
            pano["region_label"] = None
        pano["level"] = min(
            level_camera_z,
            key=lambda key: abs(level_camera_z[key] - pano["position"][2]))
        if (region is not None
                and region["label"] not in ROOM_METRIC_EXCLUDE):
            floor_z = region["floor_z"]
        else:
            floor_z = level_z[pano["level"]]
        pano["assignment_source"] = assignment_source
        pano["camera_height"] = float(pano["position"][2] - floor_z)

    return {
        "scene": scene,
        "header": header,
        "regions": regions,
        "floors": floors,
        "panoramas": panoramas,
        "level_z": level_z,
        "level_camera_z": level_camera_z,
    }


def skybox_faces(scene: str) -> Dict[str, Dict[int, Path]]:
    grouped: Dict[str, Dict[int, Path]] = defaultdict(dict)
    directory = skybox_dir(scene)
    if not directory.is_dir():
        return {}
    for path in directory.iterdir():
        match = SKYBOX_RE.match(path.name)
        if match:
            grouped[match.group(1)][int(match.group(2))] = path
    return dict(grouped)


def load_pose(scene: str, pano_id: str) -> np.ndarray:
    path = camera_pose_path(scene, pano_id)
    if not path.exists():
        raise FileNotFoundError(path)
    pose = np.loadtxt(str(path), dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{path}: expected finite 4x4 pose, got {pose.shape}")
    rotation = pose[:3, :3]
    ortho_error = float(np.max(np.abs(rotation.T.dot(rotation) - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if ortho_error > 2e-3 or abs(determinant - 1.0) > 2e-3:
        raise ValueError(
            f"{path}: invalid rotation det={determinant:.6f}, err={ortho_error:.6g}")
    return pose


def load_alignment(path: Path) -> np.ndarray:
    with path.open() as stream:
        payload = json.load(stream)
    matrix = np.asarray(payload["aligned_to_original"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: invalid alignment matrix")
    return matrix


def aligned_points_to_world(local_points: np.ndarray, pano: dict,
                            pose: np.ndarray, aligned_to_original: np.ndarray) -> np.ndarray:
    """Map aligned floor rays to the corresponding MP3D world floor plane.

    HorizonNet's VP warp maps directions, not Euclidean points.  We therefore
    transform each predicted ray and intersect it with the known horizontal
    floor plane instead of incorrectly treating the VP matrix as a rigid 3D
    transform.  Metric scale comes only from camera-to-floor height.
    """
    local_points = np.asarray(local_points, dtype=np.float64)
    original_rays = local_points.dot(np.asarray(aligned_to_original).T)
    rotation = pose[:3, :3]
    # pose_1_5 columns are camera (right, down, forward), while its forward
    # direction is the skybox LEFT face.  Thus panorama (right, forward, up)
    # maps to (-camera-forward, camera-right, -camera-down).  This fixed
    # cubemap/pose convention was frozen after the first-scene orientation
    # smoke test and is never optimized per viewpoint.
    original_to_world = np.column_stack(
        [-rotation[:, 2], rotation[:, 0], -rotation[:, 1]])
    world_rays = original_rays.dot(original_to_world.T)
    position = np.asarray(pano["position"], dtype=np.float64)
    floor_z = float(position[2] - pano["camera_height"])
    denominator = world_rays[:, 2]
    if np.any(np.abs(denominator) < 1e-7):
        raise ValueError("predicted floor ray is parallel to world floor")
    scales = (floor_z - position[2]) / denominator
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("predicted floor ray does not intersect floor in front of camera")
    return position[None, :] + scales[:, None] * world_rays


def horizon_uv_to_local_floor(uv: Sequence[Sequence[float]],
                              camera_height: float) -> np.ndarray:
    """HorizonNet alternating ceiling/floor UV -> metric local floor points."""
    uv_array = np.asarray(uv, dtype=np.float64)
    if uv_array.ndim != 2 or uv_array.shape[1] != 2 or len(uv_array) < 6:
        raise ValueError(f"invalid HorizonNet UV array {uv_array.shape}")
    floor_uv = uv_array[1::2]
    lon = (floor_uv[:, 0] - 0.5) * 2.0 * np.pi
    lat = (0.5 - floor_uv[:, 1]) * np.pi
    rays = np.column_stack([
        np.cos(lat) * np.sin(lon),
        np.cos(lat) * np.cos(lon),
        np.sin(lat),
    ])
    if np.any(rays[:, 2] >= -1e-5):
        raise ValueError("HorizonNet floor ray does not point below the camera")
    scale = -float(camera_height) / rays[:, 2]
    return rays * scale[:, None]


def dopnet_xyz_to_local_floor(xyz: np.ndarray, camera_height: float) -> np.ndarray:
    """DOPNet unit-camera-height ``(right, down, forward)`` -> local metric."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise ValueError(f"invalid DOPNet XYZ array {xyz.shape}")
    return np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]]) * float(camera_height)


def clean_polygon(points: np.ndarray) -> Optional[np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None
    if not np.isfinite(points).all():
        return None
    # Remove consecutive duplicates before asking GEOS to repair crossings.
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-7
    points = points[keep]
    if len(points) > 2 and np.linalg.norm(points[0] - points[-1]) <= 1e-7:
        points = points[:-1]
    if len(points) < 3:
        return None
    geometry = Polygon(points)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
    if geometry.geom_type != "Polygon" or geometry.area <= 1e-6:
        return None
    ring = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
    if signed_area(ring) < 0:
        ring = ring[::-1]
    return ring


def containment_diagnostic(parsed: dict, tolerance: float = 0.02) -> dict:
    exact = tolerant = wrong_region = unassigned = 0
    distances = []
    failures = []
    for pano in parsed["panoramas"]:
        if pano["region_id"] is None:
            unassigned += 1
            failures.append({
                "pano_id": pano["pano_id"],
                "region_id": None,
                "distance_m": None,
                "reason": pano["assignment_source"],
            })
            continue
        polygon = Polygon(parsed["floors"][pano["region_id"]]["points"])
        point = Point(pano["position"][:2])
        distance = float(polygon.distance(point))
        distances.append(distance)
        inside = polygon.covers(point)
        near = distance <= tolerance
        exact += int(inside)
        tolerant += int(inside or near)
        if not (inside or near):
            wrong_region += 1
            failures.append({
                "pano_id": pano["pano_id"],
                "region_id": pano["region_id"],
                "distance_m": distance,
            })
    return {
        "n": len(parsed["panoramas"]),
        "inside_exact": exact,
        "inside_or_within_2cm": tolerant,
        "outside_2cm": wrong_region,
        "unassigned": unassigned,
        "max_distance_m": max(distances) if distances else None,
        "failures": failures,
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
