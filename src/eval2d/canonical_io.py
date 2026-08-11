"""canonical annotated-floorplan v0.1 — schema build/load（下游可讀、不依 pipeline runtime）。
只從 frozen pred PKL 建；不重跑 segmentation。誠實規則：frozen 沒有的（windows/stairs/
confidence/holes）用空 list/null 並在 limitations 標記，不捏造；不混入 GT。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_derive import derive_access_graph, Room, Door, SUCCESS, ONE_OUTSIDE

SCHEMA_VERSION = "annotated_floorplan_v0.1"


def _f(x):
    """→ python float，且必須 finite。"""
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"non-finite coordinate {x}")
    return v


def _poly_list(poly):
    return [[_f(p[0]), _f(p[1])] for p in np.asarray(poly)]


def build_canonical(pred, baseline_id="watershed_v3_pre_report", d_probe=0.30):
    """frozen pred dict → canonical schema dict。門的 room_a/room_b 用同一 PIP 規則
    從 frozen 幾何推導（connectivity=derived），非重跑 segmentation。"""
    levels_out = []
    lvl_ids = sorted(set(r["level"] for r in pred["rooms"]) | set(d["level"] for d in pred["doors"]))
    level_z = pred.get("level_z", {})
    for li in lvl_ids:
        rooms_l = [r for r in pred["rooms"] if r["level"] == li]
        doors_l = [d for d in pred["doors"] if d["level"] == li]
        # 逐門推導 room_a/room_b（PIP）
        R = [Room(id=r["idx"], type=r.get("type", "?"), polygon=r["poly"]) for r in rooms_l]
        Dd = [Door(id=k, p1=np.asarray(d["seg"])[0], p2=np.asarray(d["seg"])[1]) for k, d in enumerate(doors_l)]
        door_rooms = {}
        if Dd:
            _, recs = derive_access_graph(R, Dd, d=d_probe)
            for rec in recs:
                k = rec["door_id"]
                if rec["outcome"] == SUCCESS:
                    door_rooms[k] = (rec["probe_a"][0], rec["probe_b"][0])
                elif rec["outcome"] == ONE_OUTSIDE:
                    door_rooms[k] = ((rec["probe_a"] or rec["probe_b"])[0], "OUTSIDE")
                else:
                    door_rooms[k] = (None, None)
        rooms_json = [{"id": r["idx"], "polygon": _poly_list(r["poly"]),
                       "type_raw": r.get("type", "?"), "type_confidence": None,
                       "area_m2": _f(r.get("area", 0.0)),
                       "provenance": "watershed_v3_segment_rooms"} for r in rooms_l]
        doors_json = []
        for k, d in enumerate(doors_l):
            a, b = door_rooms.get(k, (None, None))
            doors_json.append({"id": k, "segment": _poly_list(d["seg"]),
                               "room_a": a, "room_b": b,
                               "provenance": "derived_pip_from_frozen_geometry"})
        # graph（nodes=rooms；edges 含 room-room 與 exterior＝room↔OUTSIDE，兩者都保留）
        nodes = [{"id": r["idx"], "type_raw": r.get("type", "?")} for r in rooms_l]
        edges = []
        for k in sorted(door_rooms):
            a, b = door_rooms[k]
            if b == "OUTSIDE":
                edges.append({"door_id": k, "rooms": [a, "OUTSIDE"], "kind": "exterior"})
            elif b is not None and a is not None:
                edges.append({"door_id": k, "rooms": [a, b], "kind": "room-room"})
        levels_out.append({
            "id": int(li), "elevation": _f(level_z.get(li, 0.0)),
            "rooms": rooms_json, "doors": doors_json,
            "windows": [], "stairs": [],
            "graph": {"nodes": nodes, "edges": edges, "edge_status": "derived"}})
    return {
        "schema_version": SCHEMA_VERSION, "scene_id": pred["scene"],
        "method": {"name": "watershed_v3", "kind": "local_engineering_variant",
                   "source": "stage4a_pre_extrusion", "baseline_id": baseline_id},
        "coordinate_system": {"units": "metres", "frame": "mp3d_native", "resolution_m": 0.05},
        "limitations": {"holes_preserved": False, "multipolygons_preserved": False,
                        "room_types_reliable": False, "connectivity": "derived",
                        "windows_included": False, "stairs_included": False},
        "levels": levels_out}


def load_canonical(cdict):
    """canonical dict → 供 score_scene 的 pred-like dict（rooms/doors/edges/level_z/n_levels）。
    graph edges（room-room＋exterior）一併讀回，格式對齊 frozen pred['edges']。"""
    rooms, doors, edges, level_z = [], [], [], {}
    for lv in cdict["levels"]:
        li = lv["id"]; level_z[li] = lv["elevation"]
        for r in lv["rooms"]:
            rooms.append({"idx": r["id"], "level": li, "poly": np.asarray(r["polygon"], float),
                          "type": r["type_raw"], "area": r.get("area_m2", 0.0), "in_roomset": True})
        for d in lv["doors"]:
            doors.append({"seg": np.asarray(d["segment"], float), "level": li})
        for e in lv["graph"]["edges"]:
            edges.append({"rooms": tuple(e["rooms"]), "kind": e["kind"], "level": li})
    return {"scene": cdict["scene_id"], "rooms": rooms, "doors": doors, "edges": edges,
            "level_z": level_z, "n_levels": len(cdict["levels"])}
