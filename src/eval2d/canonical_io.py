"""Canonical annotated-floorplan v0.2 build/load.

v0.2 adds lossless Polygon/MultiPolygon topology, explicit source capability,
stable entity IDs, and artifact provenance.  The loader remains compatible
with archived v0.1 JSON files.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_derive import (Door, ONE_OUTSIDE, Room, SUCCESS,
                           derive_access_graph)  # noqa: E402
from geometry_v2 import (primary_exterior, to_json_geometry, to_shapely,
                         topology_counts)  # noqa: E402


SCHEMA_VERSION = "annotated_floorplan_v0.2"
LEGACY_SCHEMA_VERSION = "annotated_floorplan_v0.1"


def _finite_float(value):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite value {value}")
    return result


def _segment_json(segment):
    points = np.asarray(segment, dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError(f"invalid door segment shape={points.shape}")
    return [[float(value) for value in point] for point in points]


def _legacy_ring_json(ring):
    points = np.asarray(ring, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError(f"invalid legacy room ring shape={points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("non-finite legacy room ring")
    return [[float(value) for value in point] for point in points]


def _room_geometry(room):
    return room["geometry"] if "geometry" in room else room["poly"]


def _stable_door_key(door):
    segment = np.asarray(door["seg"], dtype=np.float64)
    midpoint = segment.mean(axis=0)
    endpoints = sorted(tuple(np.round(point, 9)) for point in segment)
    return tuple(np.round(midpoint, 9)) + endpoints[0] + endpoints[1]


def _validate_room_ids(rooms, level):
    ids = [str(room["idx"]) for room in rooms]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate room ID within level {level}: {ids}")


def build_canonical(pred, baseline_id="watershed_v3_pre_report",
                    d_probe=0.30, source_artifact=None,
                    source_sha256=None, method=None,
                    segmentation_rerun=False):
    """Frozen prediction dict -> v0.2 artifact without rerunning segmentation."""
    method = deepcopy(method) if method is not None else {
        "name": "watershed_v3",
        "kind": "local_engineering_variant",
        "source": "stage4a_pre_extrusion",
    }
    method["baseline_id"] = baseline_id

    level_z = pred.get("level_z", {})
    level_ids = sorted(set(level_z)
                       | {room["level"] for room in pred.get("rooms", [])}
                       | {door["level"] for door in pred.get("doors", [])})
    levels_out = []
    source_capabilities = set()

    for level in level_ids:
        rooms = [room for room in pred.get("rooms", [])
                 if room["level"] == level]
        _validate_room_ids(rooms, level)
        doors = sorted([door for door in pred.get("doors", [])
                        if door["level"] == level], key=_stable_door_key)

        rooms_json = []
        access_rooms = []
        for room in sorted(rooms, key=lambda item: str(item["idx"])):
            source_capability = ("structured_polygon_topology" if "geometry" in room
                                 else "legacy_single_exterior_ring")
            source_capabilities.add(source_capability)
            geometry = to_json_geometry(_room_geometry(room))
            polygon = to_shapely(geometry)
            rooms_json.append({
                "id": room["idx"],
                "geometry": geometry,
                "legacy_metric_ring": (
                    _legacy_ring_json(room["poly"])
                    if source_capability == "legacy_single_exterior_ring"
                    else None),
                "metric_geometry_contract": (
                    "legacy raw ring; buffer(0)+largest component for IoU; "
                    "raw ring for Corner/Angle"
                    if source_capability == "legacy_single_exterior_ring"
                    else "structured geometry"),
                "type_raw": room.get("type", "unknown"),
                "type_scores": room.get("type_scores"),
                "type_confidence": room.get("type_confidence"),
                "geometry_area_m2": _finite_float(polygon.area),
                "source_area_m2": (_finite_float(room["area"])
                                   if room.get("area") is not None else None),
                "geometry_topology": topology_counts(geometry),
                "source_geometry_capability": source_capability,
                "provenance": room.get(
                    "geometry_provenance", "watershed_v3_frozen_legacy_ring"),
            })
            access_rooms.append(Room(
                room["idx"], room.get("type", "unknown"), geometry))

        access_doors = []
        doors_json = []
        direct_edges = []
        room_ids = {room["idx"] for room in rooms}
        for index, door in enumerate(doors):
            door_id = f"L{level}_D{index:04d}"
            segment = np.asarray(door["seg"], dtype=np.float64)
            access_doors.append(Door(door_id, segment[0], segment[1]))
            room_a = door.get("room_a")
            room_b = door.get("room_b")
            direct = room_a is not None and room_b is not None
            if direct:
                invalid = {value for value in (room_a, room_b)
                           if value != "OUTSIDE" and value not in room_ids}
                if invalid:
                    raise ValueError(
                        f"door {door.get('idx', door_id)} references unknown "
                        f"rooms on level {level}: {sorted(invalid, key=str)}")
            door_json = {
                "id": door_id,
                "segment": _segment_json(segment),
                "width_m": _finite_float(door.get(
                    "width_m", np.linalg.norm(segment[1] - segment[0]))),
                "room_a": room_a if direct else None,
                "room_b": room_b if direct else None,
                "association_status": (door.get(
                    "association_status", "direct_method_association")
                    if direct else "unresolved"),
                "confidence": door.get("confidence"),
                "provenance": door.get(
                    "provenance", "watershed_v3_frozen_opening_geometry"),
            }
            doors_json.append(door_json)
            if direct:
                kind = ("room-outside-candidate" if "OUTSIDE" in (room_a, room_b)
                        else "room-room")
                direct_edges.append({
                    "door_id": door_id,
                    "rooms": [room_a, room_b],
                    "kind": kind,
                    "status": door_json["association_status"],
                })

        _, records = derive_access_graph(
            access_rooms, access_doors, d=d_probe) if access_doors else ({}, [])
        record_by_id = {record["door_id"]: record for record in records}
        edges = list(direct_edges)
        for door in doors_json:
            if door["association_status"] != "unresolved":
                continue
            record = record_by_id.get(door["id"])
            if record is None:
                continue
            if record["outcome"] == SUCCESS:
                a, b = record["probe_a"][0], record["probe_b"][0]
                door["room_a"], door["room_b"] = a, b
                edges.append({"door_id": door["id"], "rooms": [a, b],
                              "kind": "room-room",
                              "status": "derived_geometry_candidate"})
                door["association_status"] = "derived_geometry_candidate"
            elif record["outcome"] == ONE_OUTSIDE:
                a = (record["probe_a"] or record["probe_b"])[0]
                door["room_a"], door["room_b"] = a, "OUTSIDE"
                edges.append({"door_id": door["id"],
                              "rooms": [a, "OUTSIDE"],
                              "kind": "room-outside-candidate",
                              "status": "derived_unverified_exterior_candidate"})
                door["association_status"] = (
                    "derived_unverified_exterior_candidate")

        levels_out.append({
            "id": int(level),
            "elevation": _finite_float(level_z.get(level, 0.0)),
            "footprint": None,
            "rooms": rooms_json,
            "walls": [],
            "doors": doors_json,
            "windows": [],
            "stairs": [],
            "graph": {
                "nodes": [{"id": room["id"],
                           "type_raw": room["type_raw"]}
                          for room in rooms_json],
                "edges": edges,
                "edge_status": (
                    "direct_method_associations" if direct_edges and
                    len(direct_edges) == len(doors_json)
                    else "mixed_direct_and_derived" if direct_edges
                    else "derived_geometry_candidates"),
                "probe_distance_m": _finite_float(d_probe),
            },
        })

    structured = source_capabilities == {"structured_polygon_topology"}
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": pred["scene"],
        "artifact_provenance": {
            "source_artifact": source_artifact,
            "source_sha256": source_sha256,
            "segmentation_rerun": bool(segmentation_rerun),
        },
        "method": method,
        "coordinate_system": {
            "units": "metres", "frame": "mp3d_native",
            "resolution_m": 0.05,
        },
        "geometry_contract": {
            "representation": "GeoJSON-compatible Polygon/MultiPolygon",
            "rings_closed": True,
            "schema_supports_holes": True,
            "schema_supports_multipolygons": True,
            "source_topology_fully_observable": structured,
            "source_capabilities": sorted(source_capabilities),
        },
        "limitations": {
            "legacy_source_holes_recoverable": structured,
            "room_types_reliable": False,
            "connectivity": "derived_geometry_candidates_not_independent_truth",
            "windows_included": False,
            "stairs_included": False,
            "footprint_included": False,
        },
        "levels": levels_out,
    }


def _load_v1(canonical):
    rooms, doors, edges, level_z = [], [], [], {}
    for level in canonical["levels"]:
        level_id = level["id"]
        level_z[level_id] = level["elevation"]
        for room in level["rooms"]:
            polygon = np.asarray(room["polygon"], dtype=np.float64)
            rooms.append({
                "idx": room["id"], "level": level_id, "poly": polygon,
                "type": room["type_raw"], "area": room.get("area_m2", 0.0),
                "in_roomset": True,
            })
        for door in level["doors"]:
            doors.append({"seg": np.asarray(door["segment"], dtype=np.float64),
                          "level": level_id})
        for edge in level["graph"]["edges"]:
            edges.append({"rooms": tuple(edge["rooms"]),
                          "kind": edge["kind"], "level": level_id})
    return {"scene": canonical["scene_id"], "rooms": rooms,
            "doors": doors, "edges": edges, "level_z": level_z,
            "n_levels": len(canonical["levels"])}


def load_canonical(canonical):
    """v0.1/v0.2 JSON dict -> evaluator prediction dict."""
    version = canonical.get("schema_version")
    if version == LEGACY_SCHEMA_VERSION:
        return _load_v1(canonical)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical schema {version}")

    rooms, doors, edges, level_z = [], [], [], {}
    for level in canonical["levels"]:
        level_id = level["id"]
        level_z[level_id] = _finite_float(level["elevation"])
        for room in level["rooms"]:
            geometry = deepcopy(room["geometry"])
            loaded_room = {
                "idx": room["id"], "level": level_id,
                "geometry": geometry,
                # Compatibility only; metrics must use the structured geometry.
                "poly": primary_exterior(geometry),
                "type": room.get("type_raw", "unknown"),
                "type_scores": room.get("type_scores"),
                "area": room.get("geometry_area_m2", 0.0),
                "in_roomset": True,
            }
            if room.get("legacy_metric_ring") is not None:
                loaded_room["metric_geometry"] = np.asarray(
                    room["legacy_metric_ring"], dtype=np.float64)
                loaded_room["poly"] = loaded_room["metric_geometry"]
            rooms.append(loaded_room)
        for door in level["doors"]:
            doors.append({
                "idx": door["id"], "level": level_id,
                "seg": np.asarray(door["segment"], dtype=np.float64),
                "room_a": door.get("room_a"), "room_b": door.get("room_b"),
            })
        for edge in level["graph"]["edges"]:
            edges.append({"rooms": tuple(edge["rooms"]),
                          "kind": edge["kind"], "level": level_id,
                          "door_id": edge.get("door_id")})
    return {"scene": canonical["scene_id"], "rooms": rooms,
            "doors": doors, "edges": edges, "level_z": level_z,
            "n_levels": len(canonical["levels"]),
            "canonical_schema_version": version}
