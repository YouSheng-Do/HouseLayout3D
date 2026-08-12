#!/usr/bin/env python3
"""Run the official pretrained DOPNet MP3D checkpoint without retraining."""
from __future__ import annotations

import argparse
import json
import logging
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
    dopnet_xyz_to_local_floor,
    json_ready,
    load_alignment,
    load_pose,
    parse_house,
    scene_ids,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
DOPNET_ROOT = ROOT / "external/DOPNet"
DEFAULT_CHECKPOINT = (
    DOPNET_ROOT
    / "checkpoints_release/My_Layout_Net/mp3d/model_best_mp3d.pkl"
)
OUTPUT_ROOT = BASE_OUT / "predictions/dopnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    if str(DOPNET_ROOT) not in sys.path:
        sys.path.insert(0, str(DOPNET_ROOT))
    from models.my_layout_net import My_Layout_Net

    model = My_Layout_Net(ckpt_dir=None, backbone="resnet34")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: missing="
            f"{incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    from postprocessing.post_process import post_process
    from utils.misc import tensor2np

    checkpoint_hash = sha256_file(args.checkpoint)
    scenes = args.scenes or scene_ids()
    tasks = []
    for scene in scenes:
        for pano in parse_house(scene)["panoramas"]:
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
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array.transpose(2, 0, 1)[None]).to(device)
                prediction = model(tensor)
                processed = post_process(
                    tensor2np(prediction["depth"]), type_name="manhattan")[0]
                local = dopnet_xyz_to_local_floor(
                    processed, pano["camera_height"])
                pose = load_pose(scene, pano["pano_id"])
                aligned_to_original = load_alignment(align_path)
                world3 = aligned_points_to_world(
                    local, pano, pose, aligned_to_original)
                world = clean_polygon(world3[:, :2])
                if world is None:
                    raise ValueError("post-transform polygon is empty or invalid")
                elapsed = time.perf_counter() - started
                payload = {
                    "status": "ok",
                    "model": "DOPNet",
                    "model_source": "official zhijieshen-bjtu/DOPNet",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "scene": scene,
                    "pano_id": pano["pano_id"],
                    "level": pano["level"],
                    "region_id": pano["region_id"],
                    "region_label": pano["region_label"],
                    "camera_height_m": pano["camera_height"],
                    "pose_orientation": (
                        "pose_1_5 forward = skybox left; panorama "
                        "right/forward/up = -camera-forward/camera-right/"
                        "-camera-down"),
                    "ceiling_ratio": float(
                        tensor2np(prediction["ratio"])[0, 0]),
                    "polygon_unit_camera_xyz": processed,
                    "polygon_local_xy_m": local[:, :2],
                    "polygon_world_xy_m": world,
                    "inference_seconds": elapsed,
                }
                completed += 1
                durations.append(elapsed)
            except Exception as error:  # Preserve auditable per-view failures.
                payload = {
                    "status": "failed",
                    "model": "DOPNet",
                    "checkpoint_sha256": checkpoint_hash,
                    "scene": scene,
                    "pano_id": pano["pano_id"],
                    "level": pano["level"],
                    "region_id": pano["region_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failed += 1
                logging.exception("DOPNet failed on %s/%s", scene, pano["pano_id"])
            with output.open("w") as stream:
                json.dump(json_ready(payload), stream, indent=2)
            if index == 1 or index % 25 == 0 or index == len(tasks):
                mean = float(np.mean(durations)) if durations else float("nan")
                print(f"[{index}/{len(tasks)}] ok={completed} failed={failed} "
                      f"skipped={skipped} mean={mean:.3f}s", flush=True)

    summary = {
        "model": "DOPNet",
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
