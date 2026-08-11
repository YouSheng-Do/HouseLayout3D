"""Lossless polygon topology helpers for canonical floorplan v0.2.

The v0.1 evaluator accepted only one ``N x 2`` exterior ring.  This module
adds a small GeoJSON-compatible representation that preserves polygon holes
and disconnected MultiPolygon components while remaining backward compatible
with legacy rings.
"""
from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.validation import make_valid


SUPPORTED_TYPES = {"Polygon", "MultiPolygon"}


def _polygonal_only(geometry: BaseGeometry) -> BaseGeometry:
    """Discard non-area remnants produced by deterministic validity repair."""
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type in SUPPORTED_TYPES:
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            polygonal = _polygonal_only(part)
            if polygonal.geom_type == "Polygon" and not polygonal.is_empty:
                polygons.append(polygonal)
            elif polygonal.geom_type == "MultiPolygon":
                polygons.extend(polygonal.geoms)
        return unary_union(polygons) if polygons else Polygon()
    return Polygon()


def to_shapely(value: Any, repair: bool = True) -> BaseGeometry:
    """Convert a legacy ring, v0.2 geometry dict, or Shapely object."""
    legacy_ring = False
    if isinstance(value, BaseGeometry):
        geometry = value
    elif isinstance(value, dict):
        if value.get("type") not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported geometry type: {value.get('type')}")
        geometry = shape(value)
    else:
        legacy_ring = True
        ring = np.asarray(value, dtype=np.float64)
        if ring.ndim != 2 or ring.shape[1:] != (2,) or len(ring) < 3:
            return Polygon()
        if not np.isfinite(ring).all():
            raise ValueError("non-finite polygon coordinates")
        geometry = Polygon(ring)

    if repair and not geometry.is_valid:
        if legacy_ring:
            # Frozen evaluator v1-v3 contract.  Preserve historical scores for
            # legacy single-ring artifacts; topology absent from that source
            # cannot be reconstructed safely after the fact.
            geometry = geometry.buffer(0)
            if geometry.geom_type == "MultiPolygon":
                geometry = max(geometry.geoms, key=lambda part: part.area)
        else:
            geometry = make_valid(geometry)
    geometry = _polygonal_only(geometry)
    if not geometry.is_empty and not np.isfinite(np.asarray(geometry.bounds)).all():
        raise ValueError("non-finite geometry bounds")
    return geometry


def _ring_json(coordinates: Iterable) -> list[list[float]]:
    ring = np.asarray(coordinates, dtype=np.float64)
    if len(ring) and not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    if not np.isfinite(ring).all():
        raise ValueError("non-finite ring coordinates")
    return [[float(point[0]), float(point[1])] for point in ring]


def _polygon_coordinates(polygon: Polygon) -> list[list[list[float]]]:
    polygon = orient(polygon, sign=1.0)
    return [_ring_json(polygon.exterior.coords)] + [
        _ring_json(interior.coords) for interior in polygon.interiors
    ]


def to_json_geometry(value: Any) -> dict:
    """Return deterministic GeoJSON-compatible Polygon/MultiPolygon geometry."""
    geometry = to_shapely(value)
    if geometry.is_empty:
        return {"type": "Polygon", "coordinates": []}
    if geometry.geom_type == "Polygon":
        return {"type": "Polygon",
                "coordinates": _polygon_coordinates(geometry)}
    polygons = sorted(
        geometry.geoms,
        key=lambda polygon: (-polygon.area, polygon.centroid.x,
                             polygon.centroid.y),
    )
    return {"type": "MultiPolygon",
            "coordinates": [_polygon_coordinates(polygon)
                            for polygon in polygons]}


def geometry_rings(value: Any, include_holes: bool = True) -> list[np.ndarray]:
    """Return every boundary ring without its duplicated closing coordinate."""
    geometry = to_shapely(value)
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else list(geometry.geoms) if geometry.geom_type == "MultiPolygon"
                else [])
    rings = []
    for polygon in polygons:
        rings.append(np.asarray(polygon.exterior.coords[:-1], dtype=np.float64))
        if include_holes:
            rings.extend(np.asarray(ring.coords[:-1], dtype=np.float64)
                         for ring in polygon.interiors)
    return rings


def primary_exterior(value: Any) -> np.ndarray:
    """Legacy compatibility view; never use this for area/IoU evaluation."""
    geometry = to_shapely(value)
    if geometry.is_empty:
        return np.empty((0, 2), dtype=np.float64)
    polygon = (geometry if geometry.geom_type == "Polygon"
               else max(geometry.geoms, key=lambda part: part.area))
    return np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)


def topology_counts(value: Any) -> dict[str, int]:
    geometry = to_shapely(value)
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else list(geometry.geoms) if geometry.geom_type == "MultiPolygon"
                else [])
    return {
        "components": len(polygons),
        "holes": sum(len(polygon.interiors) for polygon in polygons),
    }


def _contour_ring(contour: np.ndarray, origin: np.ndarray, resolution: float,
                  rdp_m: float) -> np.ndarray | None:
    if rdp_m > 0:
        contour = cv2.approxPolyDP(
            contour, epsilon=float(rdp_m / resolution), closed=True)
    ring = contour.reshape(-1, 2).astype(np.float64)
    if len(ring) < 3:
        return None
    return ring * resolution + origin + resolution / 2.0


def mask_to_shapely(mask: np.ndarray, origin: np.ndarray, resolution: float,
                    rdp_m: float = 0.10, min_area_m2: float = 0.0) -> BaseGeometry:
    """Hierarchy-aware raster polygonization preserving holes and components.

    Even-depth OpenCV contours are filled components; their immediate odd-depth
    children are holes.  Nested islands become independent polygon components.
    The coordinate convention intentionally matches the legacy extractor's
    pixel-centre convention so a topology upgrade does not add a frame shift.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be 2D")
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return Polygon()
    hierarchy = hierarchy[0]

    depth = []
    for index in range(len(contours)):
        value = 0
        parent = int(hierarchy[index][3])
        while parent >= 0:
            value += 1
            parent = int(hierarchy[parent][3])
        depth.append(value)

    polygons = []
    origin = np.asarray(origin, dtype=np.float64)
    for index, contour in enumerate(contours):
        if depth[index] % 2:
            continue
        exterior = _contour_ring(contour, origin, resolution, rdp_m)
        if exterior is None:
            continue
        holes = []
        child = int(hierarchy[index][2])
        while child >= 0:
            if depth[child] == depth[index] + 1:
                ring = _contour_ring(
                    contours[child], origin, resolution, rdp_m)
                if ring is not None:
                    holes.append(ring)
            child = int(hierarchy[child][0])
        polygon = to_shapely(Polygon(exterior, holes))
        if polygon.area >= min_area_m2:
            if polygon.geom_type == "Polygon":
                polygons.append(polygon)
            elif polygon.geom_type == "MultiPolygon":
                polygons.extend(part for part in polygon.geoms
                                if part.area >= min_area_m2)
    return _polygonal_only(unary_union(polygons) if polygons else Polygon())


def mask_to_json_geometry(mask: np.ndarray, origin: np.ndarray,
                          resolution: float, rdp_m: float = 0.10,
                          min_area_m2: float = 0.0) -> dict | None:
    geometry = mask_to_shapely(
        mask, origin, resolution, rdp_m=rdp_m,
        min_area_m2=min_area_m2)
    return None if geometry.is_empty else to_json_geometry(geometry)
