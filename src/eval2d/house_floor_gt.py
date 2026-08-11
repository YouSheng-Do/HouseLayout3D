"""Official Matterport3D ``.house`` floor-polygon GT adapter.

Room geometry is read directly from the single ``S ... F`` surface associated
with every ``R`` region and its ordered ``V`` records.  No mesh projection,
rasterization, morphology, contour extraction, or RDP simplification is used.

This module deliberately writes to a separately named candidate directory.  It
must never overwrite the frozen raster-derived baseline GT.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = "/home/ado/storage/HouseLayout3D"
DEFAULT_OUT = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
ROOM_EXCLUDE = {"x", "Z"}
LABEL_NAME = {
    "a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining",
    "e": "entryway", "f": "familyroom", "g": "garage", "h": "hallway",
    "i": "library", "j": "laundry", "k": "kitchen", "l": "living",
    "m": "meeting", "n": "lounge", "o": "office", "p": "porch",
    "r": "rec", "s": "stairs", "t": "toilet", "u": "utility",
    "v": "tv", "w": "workout", "x": "outdoor", "y": "balcony",
    "z": "other", "B": "bar", "C": "classroom", "D": "dining_booth",
    "S": "spa", "Z": "junk", "-": "none",
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_derive import (  # noqa: E402
    Door, ONE_OUTSIDE, Room, SUCCESS, derive_access_graph,
)


def _signed_area(points):
    p = np.asarray(points, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                              - np.roll(p[:, 0], -1) * p[:, 1]))


def parse_house(path):
    """Parse metadata and the ordered floor-surface polygon for every region."""
    header = {}
    levels = {}
    regions = {}
    surface_region = {}
    surface_kind = {}
    vertices = defaultdict(list)

    with open(path, errors="ignore") as stream:
        for line in stream:
            f = line.split()
            if not f:
                continue
            if f[0] == "H":
                header = {
                    # H name label #images #panoramas #vertices #surfaces
                    #   #segments #objects #categories #regions #portals #levels
                    "n_regions": int(f[10]),
                    "n_portals": int(f[11]),
                    "n_levels": int(f[12]),
                }
            elif f[0] == "L":
                levels[int(f[1])] = {"idx": int(f[1])}
            elif f[0] == "R":
                regions[int(f[1])] = {
                    "idx": int(f[1]), "level": int(f[2]), "label": f[5],
                    "zlo": float(f[11]), "height": float(f[15]),
                }
            elif f[0] == "S":
                surface_region[int(f[1])] = int(f[2])
                surface_kind[int(f[1])] = f[4]
            elif f[0] == "V":
                vertices[int(f[2])].append(
                    (float(f[4]), float(f[5]), float(f[6])))

    floor_by_region = defaultdict(list)
    for surface_id, points in vertices.items():
        if surface_kind.get(surface_id) == "F":
            floor_by_region[surface_region[surface_id]].append(
                (surface_id, np.asarray(points, dtype=np.float64)))

    if header.get("n_regions") != len(regions):
        raise ValueError(
            f"{path}: header says {header.get('n_regions')} regions, parsed {len(regions)}")
    if header.get("n_levels") != len(levels):
        raise ValueError(
            f"{path}: header says {header.get('n_levels')} levels, parsed {len(levels)}")
    if set(floor_by_region) != set(regions):
        missing = sorted(set(regions) - set(floor_by_region))
        extra = sorted(set(floor_by_region) - set(regions))
        raise ValueError(f"{path}: floor/region mismatch missing={missing}, extra={extra}")

    floors = {}
    for region_id in sorted(regions):
        candidates = floor_by_region[region_id]
        if len(candidates) != 1:
            raise ValueError(
                f"{path}: region {region_id} has {len(candidates)} floor surfaces")
        surface_id, points3 = candidates[0]
        points2 = points3[:, :2]
        geom = Polygon(points2)
        if len(points2) < 3 or not np.isfinite(points2).all():
            raise ValueError(f"{path}: region {region_id} has malformed floor vertices")
        if not geom.is_valid or geom.is_empty or geom.area <= 0:
            raise ValueError(f"{path}: region {region_id} has invalid floor polygon")
        if _signed_area(points2) <= 0 or not geom.exterior.is_ccw:
            raise ValueError(f"{path}: region {region_id} floor polygon is not CCW")
        floors[region_id] = {
            "surface_id": surface_id,
            "points2": points2,
            "z": float(np.median(points3[:, 2])),
            "area": float(geom.area),
        }
    return header, levels, regions, floors


def _doors(scene, level_z):
    path = f"{ROOT}/external/houselayout3d/data/doors/{scene}.json"
    result = []
    if not os.path.exists(path):
        return result
    with open(path) as stream:
        payload = json.load(stream)
    for item in payload["doors"]:
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        bottom = vertices[np.argsort(vertices[:, 2])][:2]
        level = min(level_z, key=lambda k: abs(level_z[k] - vertices[:, 2].min()))
        result.append({
            "seg": bottom[:, :2], "normal": item.get("normal"),
            "z": float(vertices[:, 2].mean()), "level": level,
        })
    return result


def derive_edges(gt, probe=0.30):
    edges = []
    outcomes = Counter()
    by_level = defaultdict(lambda: {"rooms": [], "doors": []})
    for room in gt["rooms"]:
        if room["in_roomset"]:
            by_level[room["level"]]["rooms"].append(
                Room(room["idx"], room["type"], room["poly"]))
    for index, door in enumerate(gt["doors"]):
        by_level[door["level"]]["doors"].append(
            Door(index, np.asarray(door["seg"])[0], np.asarray(door["seg"])[1]))

    for level, graph_input in by_level.items():
        if not graph_input["doors"]:
            continue
        _, records = derive_access_graph(
            graph_input["rooms"], graph_input["doors"], d=probe)
        for record in records:
            outcomes[record["outcome"]] += 1
            if record["outcome"] == SUCCESS:
                edges.append({
                    "rooms": (record["probe_a"][0], record["probe_b"][0]),
                    "kind": "room-room", "level": level,
                })
            elif record["outcome"] == ONE_OUTSIDE:
                hit = (record["probe_a"] or record["probe_b"])[0]
                edges.append({
                    "rooms": (hit, "OUTSIDE"), "kind": "exterior", "level": level,
                })
    return edges, outcomes


def build_scene(scene):
    house_path = (f"{ROOT}/data/mp3d/v1/scans/{scene}/{scene}/"
                  f"house_segmentations/{scene}.house")
    header, levels, regions, floors = parse_house(house_path)
    level_values = defaultdict(list)
    for region in regions.values():
        level_values[region["level"]].append(region["zlo"])
    level_z = {level: float(np.median(values))
               for level, values in sorted(level_values.items())}

    rooms = []
    for region_id in sorted(regions):
        region = regions[region_id]
        floor = floors[region_id]
        rooms.append({
            "idx": region_id,
            "level": region["level"],
            "label": region["label"],
            "type": LABEL_NAME.get(region["label"], region["label"]),
            "poly": floor["points2"],
            "area": floor["area"],
            "in_roomset": region["label"] not in ROOM_EXCLUDE,
            "is_stairs": region["label"] == "s",
            "floor_surface_id": floor["surface_id"],
        })

    gt = {
        "scene": scene,
        "hdr": header,
        "lvl_z": level_z,
        "rooms": rooms,
        "doors": _doors(scene, level_z),
        "geometry_provenance": "official_mp3d_house_R_to_S_F_to_ordered_V",
        "roomset_definition": "all MP3D regions except x=outdoor and Z=junk; stairs retained but excluded from Room metric",
        "candidate_version": "mp3d_house_floor_v0_1",
    }
    gt["edges"], gt["_outcomes"] = derive_edges(gt)
    return gt


def overlap_stats(gt):
    excess = union_area = 0.0
    for level in sorted(gt["lvl_z"]):
        geoms = [Polygon(room["poly"]) for room in gt["rooms"]
                 if room["level"] == level and room["in_roomset"]
                 and not room["is_stairs"]]
        if not geoms:
            continue
        union = unary_union(geoms).area
        excess += sum(g.area for g in geoms) - union
        union_area += union
    return {
        "overlap_excess_m2": float(excess),
        "union_m2": float(union_area),
        "overlap_ratio": float(excess / union_area) if union_area else 0.0,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_all(out_dir=DEFAULT_OUT, scenes=None):
    if scenes is None:
        scenes = sorted(os.path.basename(path)[:-6] for path in glob.glob(
            f"{ROOT}/data/mp3d/v1/scans/*/*/house_segmentations/*.house"))
    os.makedirs(out_dir, exist_ok=True)
    per_scene = []
    hashes = {}
    totals = Counter()
    overlap_excess = overlap_union = 0.0

    for scene in scenes:
        gt = build_scene(scene)
        path = f"{out_dir}/{scene}.pkl"
        with open(path, "wb") as stream:
            pickle.dump(gt, stream, protocol=4)
        hashes[os.path.basename(path)] = _sha256(path)
        room_metric = sum(r["in_roomset"] and not r["is_stairs"] for r in gt["rooms"])
        stats = overlap_stats(gt)
        row = {
            "scene": scene, "levels": len(gt["lvl_z"]),
            "regions": len(gt["rooms"]), "room_metric": room_metric,
            "stairs_regions": sum(r["is_stairs"] for r in gt["rooms"]),
            "doors": len(gt["doors"]), **stats,
        }
        per_scene.append(row)
        for key in ("levels", "regions", "room_metric", "stairs_regions", "doors"):
            totals[key] += row[key]
        overlap_excess += stats["overlap_excess_m2"]
        overlap_union += stats["union_m2"]

    manifest = {
        "candidate_version": "mp3d_house_floor_v0_1",
        "status": "validated_candidate_not_silently_promoted",
        "source": "official Matterport3D .house R -> S(label F) -> ordered V",
        "transformations": ["drop z coordinate only"],
        "explicitly_not_used": [
            "regionN.ply projection", "rasterization", "morphology",
            "contour extraction", "RDP simplification",
        ],
        "roomset_definition": "exclude x and Z; keep stairs as nodes but exclude stairs from Room metric",
        "n_scenes": len(scenes),
        "totals": dict(totals),
        "overlap": {
            "overlap_excess_m2": overlap_excess,
            "union_m2": overlap_union,
            "overlap_ratio": overlap_excess / overlap_union if overlap_union else 0.0,
        },
        "paper_count_mismatch_unresolved": {
            "candidate_levels": totals["levels"], "paper_levels": 33,
            "candidate_rooms": totals["room_metric"], "paper_rooms": 317,
        },
        "per_scene": per_scene,
        "files_sha256": hashes,
    }
    with open(f"{out_dir}/candidate_manifest.json", "w") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    manifest = build_all(args.out, args.scenes)
    totals = manifest["totals"]
    print(f"candidate -> {args.out}")
    print(f"{manifest['n_scenes']} scenes | {totals['levels']} levels | "
          f"{totals['regions']} regions | {totals['room_metric']} Room-metric rooms | "
          f"{totals['doors']} doors")
    print(f"overlap ratio {100 * manifest['overlap']['overlap_ratio']:.4f}%")


if __name__ == "__main__":
    main()
