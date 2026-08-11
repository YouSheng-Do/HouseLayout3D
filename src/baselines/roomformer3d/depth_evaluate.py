#!/usr/bin/env python
"""Shared-ray Table-2 layout-depth evaluation for both RoomFormer modes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval"))
from depth_metrics import _make_scene, _z_factor
from gt_loader import ALL_SCENES, load_scene_gt


MODES = ("per_floor", "per_room")
TAUS_CM = (5.0, 10.0)


def tensor_scene(mesh):
    return _make_scene(mesh)


def evaluate_scene_modes(scene: str, pred_root: Path, stride: int,
                         progress_every: int = 100) -> dict:
    gt = load_scene_gt(scene, load_poses=True)
    scenes = {"gt": tensor_scene(gt.structures_mesh)}
    for mode in MODES:
        mesh = o3d.io.read_triangle_mesh(
            str(pred_root / mode / scene / "combined.ply"))
        if len(mesh.triangles) == 0:
            raise ValueError(f"Empty prediction mesh: {mode}/{scene}")
        scenes[mode] = tensor_scene(mesh)

    frames = gt.poses["frames"][::stride]
    zfac_cache = {}
    hits = {mode: {tau: 0 for tau in TAUS_CM} for mode in MODES}
    valid_total = 0
    started = time.time()
    for index, frame in enumerate(frames, 1):
        h, w = frame["h"], frame["w"]
        fx, fy, cx, cy = (frame["fl_x"], frame["fl_y"],
                          frame["cx"], frame["cy"])
        key = (h, w, fx, fy, cx, cy)
        if key not in zfac_cache:
            zfac_cache[key] = _z_factor(h, w, fx, fy, cx, cy)
        zfac = zfac_cache[key]
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w = c2w.copy()
        c2w[0:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)
        rays = scenes["gt"].create_rays_pinhole(
            o3d.core.Tensor(intrinsic), o3d.core.Tensor(w2c), w, h)
        gt_t = scenes["gt"].cast_rays(rays)["t_hit"].numpy()
        gt_z = gt_t * zfac
        valid = np.isfinite(gt_z)
        n_valid = int(valid.sum())
        if n_valid:
            valid_total += n_valid
            for mode in MODES:
                pred_t = scenes[mode].cast_rays(rays)["t_hit"].numpy()
                pred_z = pred_t * zfac
                diff = np.abs(np.where(np.isfinite(pred_z), pred_z, 1e9)
                              - gt_z)[valid]
                for tau in TAUS_CM:
                    hits[mode][tau] += int((diff <= tau / 100.0).sum())
        if progress_every and (index % progress_every == 0 or index == len(frames)):
            print(f"{scene}: {index}/{len(frames)} frames, "
                  f"{time.time() - started:.1f}s", flush=True)
    scores = {
        mode: {f"delta_{int(tau)}": 100.0 * hits[mode][tau] / max(valid_total, 1)
               for tau in TAUS_CM}
        for mode in MODES
    }
    return {
        "scene": scene, "stride": stride, "frames": len(frames),
        "valid_gt_pixels": valid_total, "scores": scores,
        "elapsed_seconds": time.time() - started,
    }


def aggregate(per_scene: dict, stride: int) -> dict:
    output = {
        "version": "roomformer3d_depth_v1", "stride": stride,
        "scene_count": len(per_scene), "per_scene": per_scene, "modes": {},
    }
    paper = {
        "per_floor": {"delta_5": (24.9, 11.5), "delta_10": (32.9, 14.9)},
        "per_room": {"delta_5": (37.3, 10.4), "delta_10": (44.8, 10.7)},
    }
    for mode in MODES:
        output["modes"][mode] = {}
        for key in ("delta_5", "delta_10"):
            values = np.asarray([per_scene[s]["scores"][mode][key]
                                 for s in per_scene])
            output["modes"][mode][key] = {
                "mean": float(values.mean()), "std": float(values.std()),
                "paper_mean": paper[mode][key][0],
                "paper_std": paper[mode][key][1],
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    scenes = args.scenes or sorted(ALL_SCENES)
    out_dir = args.pred_root / "depth"
    out_dir.mkdir(exist_ok=True)
    results = {}
    for index, scene in enumerate(scenes, 1):
        path = out_dir / f"{scene}.json"
        if args.resume and path.exists():
            with path.open() as stream:
                result = json.load(stream)
            if result.get("stride") != args.stride:
                raise ValueError(f"Stride mismatch in resumed file: {path}")
            print(f"[{index}/{len(scenes)}] resume {scene}", flush=True)
        else:
            print(f"[{index}/{len(scenes)}] depth {scene}", flush=True)
            result = evaluate_scene_modes(
                scene, args.pred_root, args.stride, args.progress_every)
            with path.open("w") as stream:
                json.dump(result, stream, indent=2)
        results[scene] = result
    summary = aggregate(results, args.stride)
    with (args.pred_root / "depth_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["modes"], indent=2))


if __name__ == "__main__":
    main()
