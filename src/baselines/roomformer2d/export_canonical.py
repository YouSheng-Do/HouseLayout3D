#!/usr/bin/env python
"""Export RoomFormer predictions as metric-frame annotated floorplan JSON."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval2d"))
from access_derive import Door, ONE_OUTSIDE, Room, SUCCESS, derive_access_graph

from common import ROOMFORMER_TYPE_NAMES, scene_ids


def points(value):
    return [[float(x), float(y)] for x, y in np.asarray(value, dtype=float)]


def convert(pred: dict, mode: str, checkpoint_sha256: str) -> dict:
    for room in pred["rooms"]:
        room["type"] = ROOMFORMER_TYPE_NAMES[room["type_id"]]
    levels = []
    for level, elevation in sorted(pred["level_z"].items()):
        rooms = [r for r in pred["rooms"] if r["level"] == level]
        doors = [d for d in pred["doors"] if d["level"] == level]
        windows = [w for w in pred.get("windows", []) if w["level"] == level]
        room_objects = [Room(r["idx"], r["type"], r["poly"]) for r in rooms]
        door_objects = [Door(i, np.asarray(d["seg"])[0], np.asarray(d["seg"])[1])
                        for i, d in enumerate(doors)]
        associations = {}
        graph_edges = []
        if door_objects:
            _, records = derive_access_graph(room_objects, door_objects, d=0.30)
            for record in records:
                door_id = record["door_id"]
                if record["outcome"] == SUCCESS:
                    a, b = record["probe_a"][0], record["probe_b"][0]
                    associations[door_id] = [a, b]
                    graph_edges.append({"door_id": door_id, "rooms": [a, b],
                                        "kind": "room-room"})
                elif record["outcome"] == ONE_OUTSIDE:
                    a = (record["probe_a"] or record["probe_b"])[0]
                    associations[door_id] = [a, "OUTSIDE"]
                    graph_edges.append({"door_id": door_id,
                                        "rooms": [a, "OUTSIDE"],
                                        "kind": "exterior"})
        levels.append({
            "id": int(level), "elevation": float(elevation),
            "rooms": [{
                "id": int(r["idx"]), "polygon": points(r["poly"]),
                "type_raw": r["type"], "type_id": int(r["type_id"]),
                "type_confidence": float(r["type_confidence"]),
                "corner_confidence": float(r["corner_confidence"]),
                "source_sample": r["sample_id"],
            } for r in rooms],
            "doors": [{
                "id": i, "segment": points(d["seg"]),
                "room_a": associations.get(i, [None, None])[0],
                "room_b": associations.get(i, [None, None])[1],
                "type_confidence": float(d["type_confidence"]),
                "corner_confidence": float(d["corner_confidence"]),
                "source_sample": d["sample_id"],
            } for i, d in enumerate(doors)],
            "windows": [{
                "id": i, "segment": points(w["seg"]),
                "type_confidence": float(w["type_confidence"]),
                "corner_confidence": float(w["corner_confidence"]),
                "source_sample": w["sample_id"],
            } for i, w in enumerate(windows)],
            "stairs": [],
            "graph": {
                "nodes": [{"id": int(r["idx"]), "type_raw": r["type"]}
                          for r in rooms],
                "edges": graph_edges, "edge_status": "derived_from_predicted_doors",
            },
        })
    return {
        "schema_version": "annotated_floorplan_v0.1",
        "scene_id": pred["scene"],
        "method": {
            "name": "RoomFormer", "mode": mode,
            "checkpoint": "roomformer_stru3d_semantic_rich.pth",
            "checkpoint_sha256": checkpoint_sha256,
        },
        "coordinate_system": {"units": "metres", "frame": "mp3d_native"},
        "limitations": {
            "prediction_is_2d": True, "holes_preserved": False,
            "stairs_included": False, "connectivity": "derived",
        },
        "levels": levels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for mode in ("per_floor", "per_room"):
        mode_root = args.root / mode
        with (mode_root / "run_manifest.json").open() as stream:
            run_manifest = json.load(stream)
        out = mode_root / "canonical"
        out.mkdir(exist_ok=True)
        for scene in scene_ids():
            with (mode_root / "pred" / f"{scene}.pkl").open("rb") as stream:
                pred = pickle.load(stream)
            payload = convert(pred, mode, run_manifest["checkpoint_sha256"])
            with (out / f"{scene}.json").open("w") as stream:
                json.dump(payload, stream, indent=2)
        print(f"{mode}: wrote 16 canonical scenes to {out}")


if __name__ == "__main__":
    main()
