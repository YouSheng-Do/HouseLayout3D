"""2D eval suite — Task 2: 抽「預測」的 2D floorplan（Stage 4a、擠出前）。

依 spec：直接抽 scene_graph 內部的 per-level 2D 房間分割（segment_rooms 的產物），
**不切擠出後的 3D mesh**（否則把 floorplan 品質和擠出器品質混淆）。
房型由 v3 全量的 scene_graph.json（room_classify 產）依 room id 併入。

輸入（v3 全量保留的 Stage-4 輸入）：
  outputs/mp3d/{S}/stage3/fit/fitted_mesh.ply, skeleton/ceiling_wall_floor_mesh.ply,
  stage3/coarse/{cwf_classes,labels}.npy, stage4/scene_graph.json
輸出：outputs/eval2d/pred/{S}.pkl（格式對齊 GT：rooms[poly,type,level] / doors[seg,level] / edges）
"""
import argparse, glob, json, os, pickle, sys
from collections import defaultdict

import numpy as np

ROOT = "/home/ado/storage/HouseLayout3D"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src/stage4")
from access_derive import derive_access_graph, Room, Door, SUCCESS, ONE_OUTSIDE  # noqa: E402
from geometry_v2 import (mask_to_json_geometry, primary_exterior,
                         to_shapely)  # noqa: E402


def mask_to_geometry(mask, origin, res, rdp=0.10):
    """Hierarchy-aware Polygon/MultiPolygon output for canonical v0.2."""
    return mask_to_json_geometry(mask, origin, res, rdp_m=rdp)


def mask_to_poly(mask, origin, res, rdp=0.10):
    """Deprecated single-ring compatibility view; topology lives in geometry."""
    geometry = mask_to_geometry(mask, origin, res, rdp=rdp)
    return None if geometry is None else primary_exterior(geometry)


def extract_scene(scene, derive_d=0.30):
    import scene_graph as SG
    from load_prototype import load_prototype
    O = f"{ROOT}/outputs/mp3d/{scene}"
    proto = load_prototype(f"{O}/stage3/fit/fitted_mesh.ply",
                           f"{O}/skeleton/ceiling_wall_floor_mesh.ply",
                           f"{O}/stage3/coarse/cwf_classes.npy",
                           f"{O}/stage3/coarse/labels.npy")
    levels = SG.identify_levels(proto)
    # 房型：v3 scene_graph.json 依 (level idx, room id)
    types = {}
    sgp = f"{O}/stage4/scene_graph.json"
    if os.path.exists(sgp):
        sg = json.load(open(sgp))
        for li, lv in enumerate(sg.get("levels", [])):
            for r in lv.get("rooms", []):
                types[(li, r["id"])] = r.get("type", "?")

    rooms, doors = [], []
    level_z = {}
    for li, lv in enumerate(levels):
        level_z[li] = float(lv.elevation)
        SG.segment_rooms(lv)
        origin = lv._grid_origin; final = lv._final
        for r in lv.rooms:
            geometry = mask_to_geometry(final == r["id"], origin, SG.RES)
            if geometry is None:
                continue
            poly = primary_exterior(geometry)
            rooms.append({"idx": r["id"], "level": li, "poly": poly,
                          "geometry": geometry,
                          "type": types.get((li, r["id"]), "?"),
                          "area": float(to_shapely(geometry).area),
                          "source_mask_area": float((final == r["id"]).sum() * SG.RES * SG.RES),
                          "geometry_provenance": "hierarchy_aware_mask_polygonization_v0.2",
                          "in_roomset": True})
        for op in lv.openings:
            if op.get("is_door"):
                doors.append({"seg": np.asarray(op["seg"]), "level": li})
    return {"scene": scene, "rooms": rooms, "doors": doors, "n_levels": len(levels),
            "level_z": level_z}


def derive_pred_edges(pred, d=0.30):
    edges = []
    by_lvl = defaultdict(lambda: {"rooms": [], "doors": []})
    for r in pred["rooms"]:
        geometry = r["geometry"] if "geometry" in r else r["poly"]
        by_lvl[r["level"]]["rooms"].append(
            Room(id=r["idx"], type=r["type"], polygon=geometry))
    for k, dr in enumerate(pred["doors"]):
        by_lvl[dr["level"]]["doors"].append(Door(id=k, p1=dr["seg"][0], p2=dr["seg"][1]))
    for lvl, g in by_lvl.items():
        if not g["doors"]:
            continue
        _, records = derive_access_graph(g["rooms"], g["doors"], d=d)
        for rec in records:
            if rec["outcome"] == SUCCESS:
                edges.append({"rooms": (rec["probe_a"][0], rec["probe_b"][0]), "kind": "room-room", "level": lvl})
            elif rec["outcome"] == ONE_OUTSIDE:
                hit = (rec["probe_a"] or rec["probe_b"])[0]
                edges.append({"rooms": (hit, "OUTSIDE"),
                              "kind": "room-outside-candidate", "level": lvl})
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--out", default=f"{ROOT}/outputs/eval2d/pred")
    args = ap.parse_args()
    scenes = args.scenes or sorted(p.split("/outputs/mp3d/")[1].split("/")[0]
        for p in glob.glob(f"{ROOT}/outputs/mp3d/*/stage3/fit/fitted_mesh.ply"))
    os.makedirs(args.out, exist_ok=True)
    print(f"{'scene':>12s} | {'lvl':>3s} {'rooms':>5s} {'doors':>5s} {'edges(rr/ext)':>13s}")
    for S in scenes:
        try:
            pred = extract_scene(S)
        except Exception as e:
            print(f"{S:>12s} | ERROR {type(e).__name__}: {e}"); continue
        pred["edges"] = derive_pred_edges(pred)
        rr = sum(e["kind"] == "room-room" for e in pred["edges"])
        ext = sum(e["kind"] == "room-outside-candidate"
                  for e in pred["edges"])
        print(f"{S:>12s} | {pred['n_levels']:>3d} {len(pred['rooms']):>5d} {len(pred['doors']):>5d} {f'{rr}/{ext}':>13s}")
        with open(f"{args.out}/{S}.pkl", "wb") as f:
            pickle.dump(pred, f)


if __name__ == "__main__":
    main()
