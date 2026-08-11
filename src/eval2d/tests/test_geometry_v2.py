"""Regression fixtures for holes, MultiPolygons, and v0.2 round-trip."""
import json
import os
import sys

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from jsonschema import Draft202012Validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from access_derive import Door, Room, derive_access_graph  # noqa: E402
from canonical_io import SCHEMA_VERSION, build_canonical, load_canonical  # noqa: E402
from geometry_v2 import (mask_to_shapely, to_json_geometry, to_shapely,
                         topology_counts)  # noqa: E402
from metrics import score_scene  # noqa: E402


def fixture_geometry():
    outer = [(0, 0), (8, 0), (8, 8), (0, 8)]
    hole = [(2, 2), (2, 6), (6, 6), (6, 2)]
    island = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    detached = Polygon([(10, 0), (12, 0), (12, 2), (10, 2)])
    return MultiPolygon([Polygon(outer, [hole]), island, detached])


def main():
    checks = {}

    mask = np.zeros((40, 55), np.uint8)
    mask[2:32, 2:32] = 1
    mask[10:24, 10:24] = 0
    mask[14:20, 14:20] = 1
    mask[5:16, 40:52] = 1
    raster_geometry = mask_to_shapely(
        mask, np.array([0.0, 0.0]), 0.1, rdp_m=0)
    checks["mask keeps 3 components"] = topology_counts(raster_geometry)["components"] == 3
    checks["mask keeps 1 hole"] = topology_counts(raster_geometry)["holes"] == 1
    checks["mask geometry valid"] = raster_geometry.is_valid

    geometry = to_json_geometry(fixture_geometry())
    round_trip = to_shapely(geometry)
    checks["JSON is MultiPolygon"] = geometry["type"] == "MultiPolygon"
    checks["JSON topology exact"] = topology_counts(round_trip) == {
        "components": 3, "holes": 1}
    checks["JSON area exact"] = abs(round_trip.area - fixture_geometry().area) < 1e-9

    pred = {
        "scene": "topology-fixture", "level_z": {0: 0.0}, "doors": [],
        "rooms": [{"idx": "room", "level": 0, "geometry": geometry,
                   "type": "unknown", "area": round_trip.area}],
    }
    canonical = build_canonical(pred, baseline_id="fixture")
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "configs", "eval2d", "annotated_floorplan_v0_2.schema.json")
    with open(schema_path) as stream:
        Draft202012Validator(json.load(stream)).validate(canonical)
    loaded = load_canonical(canonical)
    checks["canonical schema v0.2"] = canonical["schema_version"] == SCHEMA_VERSION
    checks["canonical validates against frozen JSON schema"] = True
    checks["canonical declares topology observable"] = canonical[
        "geometry_contract"]["source_topology_fully_observable"] is True
    checks["canonical topology exact"] = topology_counts(
        loaded["rooms"][0]["geometry"]) == {"components": 3, "holes": 1}

    gt = {
        "scene": "topology-fixture", "lvl_z": {0: 0.0}, "doors": [],
        "rooms": [{"idx": "gt", "level": 0, "geometry": geometry,
                   "type": "unknown", "in_roomset": True,
                   "is_stairs": False}],
    }
    score = score_scene(gt, loaded)
    checks["structured geometry scores as exact room"] = score["A"]["room"] == [1, 0, 0]
    checks["structured geometry IoU exact"] = abs(score["iou"][0] - 1.0) < 1e-12

    # A probe inside the hole must remain outside the room.  This catches
    # accidental flattening of interior rings in downstream graph derivation.
    room = Room("room", "unknown", geometry)
    door = Door("door", np.array([2.4, 4.0]), np.array([2.6, 4.0]))
    _, records = derive_access_graph([room], [door], d=0.30)
    checks["PIP respects hole"] = records[0]["probe_a"] == [] and records[0]["probe_b"] == []

    # Method-native room adjacency must survive canonicalization.  Falling
    # back to geometry probes here would discard the D.3 bottleneck edge.
    room_a = to_json_geometry(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    room_b = to_json_geometry(Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]))
    direct_pred = {
        "scene": "direct-door-fixture", "level_z": {0: 0.0},
        "rooms": [
            {"idx": "a", "level": 0, "geometry": room_a, "type": "unknown"},
            {"idx": "b", "level": 0, "geometry": room_b, "type": "unknown"},
        ],
        "doors": [{
            "idx": "bottleneck", "level": 0,
            "seg": np.array([[2.0, 0.5], [2.0, 1.5]]),
            "width_m": 1.0, "room_a": "a", "room_b": "b",
            "association_status": "direct_bottleneck_boundary",
            "provenance": "paper_spec_morphology_bottleneck_boundary",
        }],
    }
    direct = build_canonical(direct_pred, baseline_id="direct-fixture")
    direct_door = direct["levels"][0]["doors"][0]
    direct_edge = direct["levels"][0]["graph"]["edges"][0]
    checks["direct door room association preserved"] = (
        direct_door["room_a"] == "a" and direct_door["room_b"] == "b"
        and direct_door["association_status"] == "direct_bottleneck_boundary")
    checks["direct bottleneck graph edge preferred over probe"] = (
        direct_edge["rooms"] == ["a", "b"]
        and direct_edge["status"] == "direct_bottleneck_boundary"
        and direct["levels"][0]["graph"]["edge_status"] ==
        "direct_method_associations")
    invalid_pred = dict(direct_pred)
    invalid_pred["doors"] = [dict(direct_pred["doors"][0], room_b="missing")]
    try:
        build_canonical(invalid_pred, baseline_id="invalid-direct-fixture")
        rejects_dangling = False
    except ValueError:
        rejects_dangling = True
    checks["canonical rejects dangling direct room reference"] = rejects_dangling

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
