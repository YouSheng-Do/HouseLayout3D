#!/usr/bin/env python
"""Sample Matterport3D region meshes and create RoomFormer density inputs.

Run this with ``envs/hl3d/bin/python``; that environment supplies trimesh.
The paper says surface points were sampled but does not publish the sample
count.  The defaults below are therefore explicit reproduction assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from common import (density_image, eligible_regions, region_mesh_path,
                    scene_ids, stable_seed)


def sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def save_density(out_root: Path, mode: str, sample_id: str,
                 points: np.ndarray, metadata: dict) -> dict:
    density, transform = density_image(points)
    q05, q95 = np.quantile(np.asarray(points)[:, 2], [0.05, 0.95])
    rel = Path("inputs") / mode / f"{sample_id}.npy"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, density)
    return {
        **metadata,
        "sample_id": sample_id,
        "mode": mode,
        "input": str(rel),
        "n_points": int(len(points)),
        "height_quantiles_m": {"q05": float(q05), "q95": float(q95)},
        "transform": transform,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--room-samples", type=int, default=200_000)
    parser.add_argument("--floor-samples-per-m2", type=float, default=1000.0)
    parser.add_argument("--floor-min-per-region", type=int, default=20_000)
    parser.add_argument("--floor-max-per-region", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = args.scenes or scene_ids()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()

    for scene_index, scene in enumerate(selected, 1):
        regions, level_z = eligible_regions(scene)
        floor_points = defaultdict(list)
        floor_region_ids = defaultdict(list)
        scene_started = time.time()
        for region_index, region in enumerate(regions, 1):
            region_id = region["region_id"]
            mesh_path = region_mesh_path(scene, region_id)
            mesh = trimesh.load(mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError(f"Not a triangle mesh: {mesh_path}")
            area = float(mesh.area)
            floor_count = int(np.clip(
                math.ceil(area * args.floor_samples_per_m2),
                args.floor_min_per_region,
                args.floor_max_per_region,
            ))
            sample_count = max(args.room_samples, floor_count)
            points = sample_mesh(
                mesh, sample_count,
                stable_seed(args.seed, scene, region_id, "surface"),
            )
            room_points = points[:args.room_samples]
            floor_points[region["level"]].append(points[:floor_count])
            floor_region_ids[region["level"]].append(region_id)
            records.append(save_density(
                args.out, "per_room", f"{scene}__r{region_id:03d}", room_points,
                {
                    "scene": scene, "level": region["level"],
                    "level_z": level_z[region["level"]],
                    "region_ids": [region_id], "mp3d_label": region["label"],
                    "mesh_surface_area_m2": area,
                },
            ))
            del mesh, points, room_points
            if region_index % 10 == 0 or region_index == len(regions):
                print(f"[{scene_index}/{len(selected)}] {scene}: "
                      f"regions {region_index}/{len(regions)}", flush=True)

        for level in sorted(floor_points):
            points = np.concatenate(floor_points[level], axis=0)
            records.append(save_density(
                args.out, "per_floor", f"{scene}__l{level:02d}", points,
                {
                    "scene": scene, "level": level, "level_z": level_z[level],
                    "region_ids": floor_region_ids[level],
                    "mp3d_label": None, "mesh_surface_area_m2": None,
                },
            ))
        print(f"[{scene_index}/{len(selected)}] {scene}: complete in "
              f"{time.time() - scene_started:.1f}s", flush=True)

    records.sort(key=lambda item: (item["mode"], item["scene"], item["level"],
                                   item["sample_id"]))
    manifest = {
        "version": "roomformer2d_inputs_v1",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "scene_count": len(selected),
        "sampling": {
            "method": "trimesh.sample.sample_surface area-weighted triangle sampling",
            "seed": args.seed,
            "per_room_points": args.room_samples,
            "per_floor_points_per_mesh_m2": args.floor_samples_per_m2,
            "per_floor_min_points_per_region": args.floor_min_per_region,
            "per_floor_max_points_per_region": args.floor_max_per_region,
            "paper_disclosure": "HouseLayout3D specifies surface sampling but not point count",
        },
        "selection": "MP3D regions excluding x=outdoor, Z=junk, s=stairs",
        "records": records,
    }
    with (args.out / "input_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    counts = {mode: sum(r["mode"] == mode for r in records)
              for mode in ("per_floor", "per_room")}
    print(f"DONE: {counts}, elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
