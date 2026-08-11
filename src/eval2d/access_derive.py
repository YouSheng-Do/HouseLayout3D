"""
Backbone-agnostic access-graph derivation.

The core function `derive_access_graph` takes generic geometry:
    room_polygons : list of Room(id, type, polygon[N,2])
    door_segments : list of Door(id, p1[2], p2[2])
and returns an access graph plus a per-door outcome record.

It knows nothing about Structured3D, RoomFormer, or CAGE. The parser
(parse_stru3d.py) or any future backbone adapter is responsible for
producing `room_polygons` and `door_segments` in this format, so the
*identical* rule can be run on ground-truth and predicted geometry
(Stage 1 vs Stage 2 of EXPERIMENT_SPEC.md).

This is deliberately the "obvious" geometric rule described in the spec.
It is NOT tuned to maximise success rate.
"""

from collections import namedtuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

Room = namedtuple("Room", ["id", "type", "polygon"])   # polygon: (N,2) float array
Door = namedtuple("Door", ["id", "p1", "p2"])           # p1,p2: (2,) float arrays

# Outcome categories for a single door (mutually exclusive).
SUCCESS = "success"              # two probes -> two distinct rooms
SAME_ROOM = "same_room"          # both probes in the same single room
ONE_OUTSIDE = "one_outside"      # exactly one probe in no room
BOTH_OUTSIDE = "both_outside"    # both probes in no room
OVERLAP = "overlap"              # a probe falls inside >1 room (overlapping polygons)
DEGENERATE = "degenerate"        # door segment has ~zero length; no normal defined

FAILURE_MODES = [SAME_ROOM, ONE_OUTSIDE, BOTH_OUTSIDE, OVERLAP, DEGENERATE]
ALL_OUTCOMES = [SUCCESS] + FAILURE_MODES


def _prepare_rooms(room_polygons, fix_invalid=True):
    """Build shapely polygons once. Returns list of (room, shapely_poly, prepared)."""
    prepared = []
    for room in room_polygons:
        poly = Polygon(np.asarray(room.polygon, dtype=float))
        valid = poly.is_valid
        if not valid and fix_invalid:
            # buffer(0) repairs most self-intersections; only needed for
            # predicted geometry (GT polygons are clean simple polygons).
            poly = poly.buffer(0)
        prepared.append((room, poly, prep(poly), valid))
    return prepared


def _rooms_containing(pt, prepared):
    """Return list of room ids whose polygon strictly contains pt."""
    p = Point(float(pt[0]), float(pt[1]))
    hits = []
    for room, poly, pr, _valid in prepared:
        if pr.contains(p):
            hits.append(room.id)
    return hits


def _normal(p1, p2):
    """Unit normal to segment p1->p2, or None if degenerate."""
    t = np.asarray(p2, float) - np.asarray(p1, float)
    L = float(np.hypot(t[0], t[1]))
    if L < 1e-9:
        return None, L
    n = np.array([-t[1], t[0]]) / L
    return n, L


def derive_access_graph(room_polygons, door_segments, d, fix_invalid=True):
    """
    Derive a room-to-room access graph from geometry using the probe rule.

    For each door:
      * midpoint m of the door segment
      * unit normal n perpendicular to the segment
      * two probes  m + d*n  and  m - d*n
      * point-in-polygon test each probe against all room polygons
      * classify the outcome (see ALL_OUTCOMES)

    Parameters
    ----------
    room_polygons : list[Room]
    door_segments : list[Door]
    d : float
        Probe offset distance (same units as the coordinates; here density-map
        pixels for Structured3D montefloor data).
    fix_invalid : bool
        Repair self-intersecting predicted polygons with buffer(0).

    Returns
    -------
    graph : dict
        {
          "nodes": [{"id", "type"} ...],
          "edges": [{"door_id", "rooms": [a, b]} ...],   # only successful doors
          "d": d,
        }
    records : list[dict]
        One per door: outcome + which rooms each probe hit. Used for the
        failure-mode breakdown.
    """
    prepared = _prepare_rooms(room_polygons, fix_invalid=fix_invalid)

    nodes = [{"id": r.id, "type": r.type} for r in room_polygons]
    edges = []
    records = []

    for door in door_segments:
        p1 = np.asarray(door.p1, float)
        p2 = np.asarray(door.p2, float)
        m = (p1 + p2) / 2.0
        n, L = _normal(p1, p2)

        rec = {"door_id": door.id, "length": L, "midpoint": m.tolist()}

        if n is None:
            rec["outcome"] = DEGENERATE
            rec["probe_a"] = rec["probe_b"] = None
            records.append(rec)
            continue

        a = m + d * n
        b = m - d * n
        hits_a = _rooms_containing(a, prepared)
        hits_b = _rooms_containing(b, prepared)
        rec["probe_a"] = hits_a
        rec["probe_b"] = hits_b

        if len(hits_a) > 1 or len(hits_b) > 1:
            outcome = OVERLAP
        elif len(hits_a) == 0 and len(hits_b) == 0:
            outcome = BOTH_OUTSIDE
        elif len(hits_a) == 0 or len(hits_b) == 0:
            outcome = ONE_OUTSIDE
        elif hits_a[0] == hits_b[0]:
            outcome = SAME_ROOM
        else:
            outcome = SUCCESS
            edges.append({"door_id": door.id, "rooms": sorted([hits_a[0], hits_b[0]])})

        rec["outcome"] = outcome
        records.append(rec)

    graph = {"nodes": nodes, "edges": edges, "d": d}
    return graph, records
