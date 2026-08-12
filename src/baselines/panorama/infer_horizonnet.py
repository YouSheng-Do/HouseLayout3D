#!/usr/bin/env python3
"""Run the official pretrained HorizonNet on the prepared MP3D panoramas."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from common import (
    BASE_OUT,
    DATA_OUT,
    aligned_points_to_world,
    clean_polygon,
    horizon_uv_to_local_floor,
    json_ready,
    load_alignment,
    load_pose,
    parse_house,
    scene_ids,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
HORIZON_ROOT = ROOT / "external/HorizonNet"
DEFAULT_CHECKPOINT = HORIZON_ROOT / "ckpt/resnet50_rnn__mp3d.pth"
OUTPUT_ROOT = BASE_OUT / "predictions/horizonnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-cuboid", action="store_true")
    return parser.parse_args()


def result_payload(scene: str, pano: dict, uv: np.ndarray, z0: float,
                   z1: float, local: np.ndarray, world: np.ndarray,
                   elapsed: float, checkpoint_hash: str) -> dict:
    return {
        "status": "ok",
        "model": "HorizonNet",
        "model_source": "official sunset1995/HorizonNet",
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "scene": scene,
        "pano_id": pano["pano_id"],
        "level": pano["level"],
        "region_id": pano["region_id"],
        "region_label": pano["region_label"],
        "camera_height_m": pano["camera_height"],
        "pose_orientation": (
            "pose_1_5 forward = skybox left; panorama right/forward/up = "
            "-camera-forward/camera-right/-camera-down"),
        "uv": uv,
        "z0": z0,
        "z1": z1,
        "polygon_local_xy_m": local[:, :2],
        "polygon_world_xy_m": world,
        "inference_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if str(HORIZON_ROOT) not in sys.path:
        sys.path.insert(0, str(HORIZON_ROOT))
    from inference import inference as official_inference
    from misc.utils import load_trained_model
    from model import HorizonNet

    device = torch.device(args.device)
    net = load_trained_model(HorizonNet, str(args.checkpoint)).to(device)
    net.eval()
    checkpoint_hash = sha256_file(args.checkpoint)
    scenes = args.scenes or scene_ids()
    tasks = []
    parsed_by_scene = {}
    for scene in scenes:
        parsed = parse_house(scene)
        parsed_by_scene[scene] = parsed
        for pano in parsed["panoramas"]:
            tasks.append((scene, pano))
    if args.limit is not None:
        tasks = tasks[:args.limit]

    completed = skipped = failed = 0
    durations = []
    with torch.no_grad():
        for index, (scene, pano) in enumerate(tasks, 1):
            output = OUTPUT_ROOT / scene / f"{pano['pano_id']}.json"
            if output.exists() and not args.force:
                skipped += 1
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            image_path = DATA_OUT / "aligned" / scene / f"{pano['pano_id']}.png"
            align_path = DATA_OUT / "aligned" / scene / f"{pano['pano_id']}.align.json"
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                if image.size != (1024, 512):
                    image = image.resize((1024, 512), Image.BICUBIC)
                array = np.asarray(image).transpose(2, 0, 1).copy()
                tensor = torch.from_numpy(array[None] / 255.0).float()
                uv, z0, z1, _ = official_inference(
                    net, tensor, device, force_cuboid=args.force_cuboid)
                local = horizon_uv_to_local_floor(uv, pano["camera_height"])
                pose = load_pose(scene, pano["pano_id"])
                aligned_to_original = load_alignment(align_path)
                world3 = aligned_points_to_world(
                    local, pano, pose, aligned_to_original)
                world = clean_polygon(world3[:, :2])
                if world is None:
                    raise ValueError("post-transform polygon is empty or invalid")
                elapsed = time.perf_counter() - started
                payload = result_payload(
                    scene, pano, uv, float(z0), float(z1), local, world,
                    elapsed, checkpoint_hash)
                completed += 1
                durations.append(elapsed)
            except Exception as error:  # Preserve auditable per-view failures.
                payload = {
                    "status": "failed",
                    "model": "HorizonNet",
                    "checkpoint_sha256": checkpoint_hash,
                    "scene": scene,
                    "pano_id": pano["pano_id"],
                    "level": pano["level"],
                    "region_id": pano["region_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failed += 1
            with output.open("w") as stream:
                json.dump(json_ready(payload), stream, indent=2)
            if index == 1 or index % 25 == 0 or index == len(tasks):
                mean = float(np.mean(durations)) if durations else float("nan")
                print(f"[{index}/{len(tasks)}] ok={completed} failed={failed} "
                      f"skipped={skipped} mean={mean:.3f}s", flush=True)

    summary = {
        "model": "HorizonNet",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "tasks": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "mean_inference_seconds": float(np.mean(durations)) if durations else None,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "run_summary.json").open("w") as stream:
        json.dump(json_ready(summary), stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
