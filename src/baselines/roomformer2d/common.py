"""Shared geometry and I/O helpers for the RoomFormer 2D baseline.

The density-image transform follows RoomFormer's SceneCAD preprocessing.  A
coordinate is rasterised with ``round(normalised_xy * 256)`` and clipped to
``[0, 255]``.  RoomFormer annotations are integer pixels divided by 255, so
the inverse used for predictions is ``min_xy + pixel * max_range / 256``.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path("/home/ado/storage/HouseLayout3D")
MP3D_ROOT = ROOT / "data/mp3d/v1/scans"
OPENING_ROOT = ROOT / "external/houselayout3d/data"
IMAGE_SIZE = 256
EXCLUDED_LABELS = {"x", "Z", "s"}  # outdoor, junk, stairs

ROOMFORMER_TYPE_NAMES = {
    0: "living", 1: "kitchen", 2: "bedroom", 3: "bathroom",
    4: "balcony", 5: "hallway", 6: "dining", 7: "office",
    # Cross-dataset metric names: Structured3D "store room" is mapped to the
    # closest Matterport3D class, closet.  The native ids remain in every raw
    # prediction and run manifest.
    8: "studio", 9: "closet", 10: "outdoor", 11: "laundry",
    12: "office", 13: "basement", 14: "garage", 15: "other",
}
DOOR_ID = 16
WINDOW_ID = 17


def scene_ids() -> List[str]:
    paths = glob.glob(str(MP3D_ROOT / "*/*/house_segmentations/*.house"))
    return sorted(Path(path).stem for path in paths)


def house_path(scene: str) -> Path:
    return MP3D_ROOT / scene / scene / "house_segmentations" / f"{scene}.house"


def region_mesh_path(scene: str, region_id: int) -> Path:
    return (MP3D_ROOT / scene / scene / "region_segmentations" /
            f"region{region_id}.ply")


def parse_regions(scene: str) -> Tuple[List[dict], Dict[int, float]]:
    """Read region id, level, label and a robust level elevation from .house."""
    regions = []
    with house_path(scene).open(errors="ignore") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "R":
                regions.append({
                    "region_id": int(fields[1]),
                    "level": int(fields[2]),
                    "label": fields[5],
                    "zlo": float(fields[11]),
                })
    if not regions:
        raise ValueError(f"No regions found for {scene}")
    level_z = {}
    for level in sorted({item["level"] for item in regions}):
        values = [item["zlo"] for item in regions if item["level"] == level]
        level_z[level] = float(np.median(values))
    return regions, level_z


def eligible_regions(scene: str) -> Tuple[List[dict], Dict[int, float]]:
    regions, level_z = parse_regions(scene)
    return [r for r in regions if r["label"] not in EXCLUDED_LABELS], level_z


def stable_seed(seed: int, *parts: object) -> int:
    raw = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def density_image(points_xyz: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Generate the exact SceneCAD-style RoomFormer density representation."""
    xyz = np.asarray(points_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"Expected nonempty Nx3 points, got {xyz.shape}")
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    max_range = float(np.max(maxs[:2] - mins[:2]))
    if not np.isfinite(max_range) or max_range <= 0:
        raise ValueError(f"Degenerate XY point extent: {max_range}")
    padding = max_range * 0.05
    min_xyz = (maxs + mins) / 2.0 - max_range / 2.0
    min_xyz -= padding
    max_range += 2.0 * padding

    coords = np.rint((xyz[:, :2] - min_xyz[:2]) / max_range * IMAGE_SIZE)
    coords = np.clip(coords, 0, IMAGE_SIZE - 1).astype(np.int32)
    linear = coords[:, 1] * IMAGE_SIZE + coords[:, 0]
    counts = np.bincount(linear, minlength=IMAGE_SIZE * IMAGE_SIZE)
    density = counts.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    peak = float(density.max())
    if peak > 0:
        density /= peak
    transform = {
        "min_x": float(min_xyz[0]),
        "min_y": float(min_xyz[1]),
        "max_range": max_range,
        "image_size": IMAGE_SIZE,
        "raster_rule": "round(norm_xy*256)_clip_0_255",
    }
    return density, transform


def pixels_to_world(pixels: np.ndarray, transform: dict) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    origin = np.array([transform["min_x"], transform["min_y"]])
    return origin + pixels * float(transform["max_range"]) / IMAGE_SIZE


def world_to_annotation_pixels(xy: np.ndarray, transform: dict) -> np.ndarray:
    """SceneCAD's annotation transform (floor, unlike density's round)."""
    xy = np.asarray(xy, dtype=np.float64)
    origin = np.array([transform["min_x"], transform["min_y"]])
    pixels = np.floor((xy - origin) / float(transform["max_range"]) * IMAGE_SIZE)
    return np.clip(pixels, 0, IMAGE_SIZE - 1).astype(np.int32)


def opening_segment(vertices: Iterable[Iterable[float]]) -> np.ndarray:
    """Collapse a vertical opening rectangle to its two distinct XY endpoints."""
    verts = np.asarray(list(vertices), dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] < 3 or len(verts) < 2:
        raise ValueError(f"Malformed opening vertices: {verts.shape}")
    unique_xy = np.unique(np.round(verts[:, :2], decimals=7), axis=0)
    if len(unique_xy) < 2:
        raise ValueError("Opening has fewer than two unique XY endpoints")
    dist2 = ((unique_xy[:, None] - unique_xy[None, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    return unique_xy[[i, j]].astype(np.float64)


def load_openings(scene: str, kind: str, level_z: Dict[int, float]) -> List[dict]:
    if kind not in {"doors", "windows"}:
        raise ValueError(kind)
    path = OPENING_ROOT / kind / f"{scene}.json"
    if not path.exists():
        return []
    with path.open() as stream:
        items = json.load(stream).get(kind, [])
    result = []
    for item in items:
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        level = min(level_z, key=lambda key: abs(level_z[key] - vertices[:, 2].min()))
        result.append({
            "seg": opening_segment(vertices),
            "level": level,
            "z": float(vertices[:, 2].mean()),
        })
    return result


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
