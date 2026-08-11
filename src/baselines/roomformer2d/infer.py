#!/usr/bin/env python
"""Run the official semantic-rich RoomFormer checkpoint on custom densities."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from common import (DOOR_ID, ROOMFORMER_TYPE_NAMES, WINDOW_ID, pixels_to_world,
                    sha256_file)


def model_args(device: str) -> argparse.Namespace:
    return argparse.Namespace(
        backbone="resnet50", lr_backbone=0, dilation=False,
        position_embedding="sine", position_embedding_scale=2 * np.pi,
        num_feature_levels=4, enc_layers=6, dec_layers=6,
        dim_feedforward=1024, hidden_dim=256, dropout=0.1, nheads=8,
        num_queries=2800, num_polys=70, dec_n_points=4, enc_n_points=4,
        query_pos_type="sine", with_poly_refine=True, masked_attn=False,
        semantic_classes=19, aux_loss=True, device=device,
    )


def postprocess(outputs: dict, batch_records: list) -> list:
    """Official RoomFormer thresholds plus metric-frame coordinate inversion."""
    pred_logits = outputs["pred_logits"]
    pred_coords = outputs["pred_coords"]
    fg_mask = torch.sigmoid(pred_logits) > 0.5
    class_prob = torch.nn.functional.softmax(outputs["pred_room_logits"], -1)
    class_score, class_label = class_prob[..., :-1].max(-1)
    results = []
    for batch_i, record in enumerate(batch_records):
        rooms, doors, windows, raw = [], [], [], []
        for poly_i in range(fg_mask.shape[1]):
            selected = fg_mask[batch_i, poly_i]
            valid = pred_coords[batch_i, poly_i][selected]
            if len(valid) == 0:
                continue
            pixels = np.around((valid * 255).detach().cpu().numpy()).astype(np.int32)
            label = int(class_label[batch_i, poly_i].item())
            score = float(class_score[batch_i, poly_i].item())
            corner_probs = torch.sigmoid(pred_logits[batch_i, poly_i][selected])
            corner_score = float(corner_probs.mean().item())
            keep = False
            kind = "discarded"
            if label not in (DOOR_ID, WINDOW_ID):
                if len(pixels) >= 4 and Polygon(pixels).area >= 100:
                    keep, kind = True, "room"
            elif len(pixels) == 2:
                keep = True
                kind = "door" if label == DOOR_ID else "window"
            raw.append({
                "query": poly_i, "kind": kind, "kept": keep,
                "type_id": label, "type_score": score,
                "corner_score_mean": corner_score, "pixels": pixels.tolist(),
            })
            if not keep:
                continue
            world = pixels_to_world(pixels, record["transform"])
            base = {
                "level": record["level"], "sample_id": record["sample_id"],
                "type_id": label, "type_confidence": score,
                "corner_confidence": corner_score,
            }
            if kind == "room":
                rooms.append({**base, "poly": world,
                              "type": ROOMFORMER_TYPE_NAMES[label]})
            elif kind == "door":
                doors.append({**base, "seg": world})
            else:
                windows.append({**base, "seg": world})
        results.append({"rooms": rooms, "doors": doors, "windows": windows,
                        "raw": raw})
    return results


def render_overlay(density: np.ndarray, result: dict, path: Path) -> None:
    base = np.uint8(np.clip(np.sqrt(density), 0, 1) * 210)
    image = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = [(255, 80, 80), (80, 220, 100), (80, 160, 255),
               (255, 190, 60), (210, 100, 255), (70, 230, 220)]
    for item in result["raw"]:
        if not item["kept"]:
            continue
        pts = [tuple(point) for point in item["pixels"]]
        if item["kind"] == "room":
            color = palette[item["type_id"] % len(palette)]
            draw.line(pts + [pts[0]], fill=color, width=2)
        else:
            color = (255, 40, 40) if item["kind"] == "door" else (40, 170, 255)
            draw.line(pts, fill=color, width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((768, 768), resample=Image.Resampling.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roomformer-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("per_floor", "per_room"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.roomformer_root))
    from models import build_model

    with args.manifest.open() as stream:
        input_manifest = json.load(stream)
    records = [r for r in input_manifest["records"] if r["mode"] == args.mode]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError(f"No {args.mode} records")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(model_args(args.device), train=False).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [k for k in unexpected
                  if not (k.endswith("total_params") or k.endswith("total_ops"))]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch missing={missing}, unexpected={unexpected}")

    args.out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(lambda: {"rooms": [], "doors": [], "windows": [],
                                   "level_z": {}})
    started = time.time()
    sample_stats = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            arrays = [np.load(args.input_root / r["input"]).astype(np.float32)
                      for r in batch_records]
            samples = [torch.from_numpy(x[None]).to(device) for x in arrays]
            batch_started = time.time()
            outputs = model(samples)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - batch_started
            processed = postprocess(outputs, batch_records)
            for record, density, result in zip(batch_records, arrays, processed):
                scene = record["scene"]
                target = grouped[scene]
                target["level_z"][record["level"]] = record["level_z"]
                for key in ("rooms", "doors", "windows"):
                    for item in result[key]:
                        item["idx"] = f"{record['sample_id']}:{key[0]}{len(target[key])}"
                        target[key].append(item)
                raw_path = args.out / "raw" / f"{record['sample_id']}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("w") as stream:
                    json.dump({"sample": record, "predictions": result["raw"]},
                              stream, indent=2)
                if args.save_overlays:
                    render_overlay(density, result,
                                   args.out / "overlays" / f"{record['sample_id']}.png")
                sample_stats.append({
                    "sample_id": record["sample_id"],
                    "rooms": len(result["rooms"]), "doors": len(result["doors"]),
                    "windows": len(result["windows"]),
                })
            done = min(start + len(batch_records), len(records))
            print(f"{args.mode}: {done}/{len(records)} samples, "
                  f"batch={elapsed:.3f}s", flush=True)

    pred_dir = args.out / "pred"
    pred_dir.mkdir(exist_ok=True)
    for scene, pred in grouped.items():
        for key in ("rooms", "doors", "windows"):
            for idx, item in enumerate(pred[key]):
                item["idx"] = idx
        pred["scene"] = scene
        pred["n_levels"] = len(pred["level_z"])
        with (pred_dir / f"{scene}.pkl").open("wb") as stream:
            pickle.dump(pred, stream, protocol=4)

    run_manifest = {
        "version": "roomformer2d_inference_v1", "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_load": {"missing": [], "unexpected": []},
        "model": {"num_queries": 2800, "num_polys": 70,
                  "semantic_classes": 19},
        "postprocess": {
            "corner_sigmoid_threshold": 0.5,
            "room_min_corners": 4, "room_min_pixel_area": 100,
            "door_window_exact_corners": 2,
            "coordinate_inverse": "world=min_xy+round(pred_norm*255)*max_range/256",
        },
        "sample_count": len(records), "scene_count": len(grouped),
        "elapsed_seconds": time.time() - started,
        "elapsed_scope": "inference loop + postprocess + artifact writes; excludes process/model/checkpoint startup",
        "samples": sample_stats,
    }
    with (args.out / "run_manifest.json").open("w") as stream:
        json.dump(run_manifest, stream, indent=2)
    print(f"DONE {args.mode}: {len(records)} inputs, {len(grouped)} scenes, "
          f"{run_manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
