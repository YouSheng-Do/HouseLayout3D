"""Inventory, stitch, align, and verify MP3D panoramas for layout baselines."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import py360convert
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

from common import (BASE_OUT, DATA_OUT, ROOT, ROOM_EXCLUDE, ROOM_METRIC_EXCLUDE,
                    camera_pose_path, containment_diagnostic, json_ready,
                    load_pose, parse_house, scene_ids, sha256_file,
                    skybox_faces)


HORIZONNET_ROOT = ROOT / "external/HorizonNet"


def stitch_one(task: Tuple[str, str, dict, str, bool]) -> dict:
    scene, pano_id, face_strings, output_string, force = task
    output = Path(output_string)
    if output.exists() and not force:
        with Image.open(output) as image:
            return {"scene": scene, "pano_id": pano_id,
                    "output": str(output), "size": list(image.size),
                    "skipped": True}

    faces = {index: Path(path) for index, path in face_strings.items()}
    images = {}
    keys = ["U", "L", "F", "R", "B", "D"]
    for index, key in enumerate(keys):
        images[key] = np.asarray(Image.open(faces[index]).convert("RGB"))
    # Exact released MP3D convention used by PanFusion's public stitcher.
    images["R"] = np.flip(images["R"], 1)
    images["B"] = np.flip(images["B"], 1)
    images["U"] = np.rot90(np.flip(images["U"], 0), 1)
    images["D"] = np.rot90(images["D"], 1)
    # py360convert 0.1.0's cube_dict2h has a one-line upstream NameError;
    # materialise its documented F,R,B,L,U,D horizon representation directly.
    horizon = np.concatenate(
        [images[key] for key in ["F", "R", "B", "L", "U", "D"]], axis=1)
    panorama = py360convert.c2e(
        horizon, 512, 1024, mode="bilinear", cube_format="horizon")
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panorama, mode="RGB").save(output, compress_level=2)

    seam_mae = float(np.mean(np.abs(
        panorama[:, 0].astype(np.float32) - panorama[:, -1].astype(np.float32))))
    neighbor_mae = float(np.mean(np.abs(
        panorama[:, 1:].astype(np.float32) - panorama[:, :-1].astype(np.float32))))
    return {
        "scene": scene, "pano_id": pano_id, "output": str(output),
        "size": [1024, 512], "seam_mae": seam_mae,
        "neighbor_mae": neighbor_mae,
        "seam_ratio": seam_mae / max(neighbor_mae, 1e-9),
        "skipped": False,
    }


def align_one(task: Tuple[str, str, str, str, bool]) -> dict:
    scene, pano_id, input_string, output_string, force = task
    output = Path(output_string)
    matrix_path = output.with_suffix(".align.json")
    vp_path = output.with_suffix(".vp.txt")
    if output.exists() and matrix_path.exists() and not force:
        return {"scene": scene, "pano_id": pano_id, "output": str(output),
                "alignment": str(matrix_path), "skipped": True}

    # Import in the worker so the old pylsd/OpenCV stack is not initialized
    # before multiprocessing forks.
    sys.path.insert(0, str(HORIZONNET_ROOT))
    from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama

    image = np.asarray(Image.open(input_string).convert("RGB"))
    _, vp, _, _, _, score, angle = panoEdgeDetection(
        image, qError=0.7, refineIter=3)
    vp_argument = np.asarray(vp[2::-1], dtype=np.float64)
    aligned = rotatePanorama(image.astype(np.float64) / 255.0, vp_argument)
    aligned = np.clip(aligned * 255.0, 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(aligned, mode="RGB").save(output, compress_level=2)
    np.savetxt(str(vp_path), np.asarray(vp, dtype=np.float64), fmt="%.9g")

    aligned_to_original = vp_argument.T
    payload = {
        "scene": scene,
        "pano_id": pano_id,
        "aligned_to_original": aligned_to_original.tolist(),
        "vp_raw": np.asarray(vp).tolist(),
        "orthogonality_max_error": float(np.max(np.abs(
            aligned_to_original.T.dot(aligned_to_original) - np.eye(3)))),
        "determinant": float(np.linalg.det(aligned_to_original)),
        "vp_score": json_ready(score),
        "vp_angle": json_ready(angle),
        "implementation": "HorizonNet misc/pano_lsd_align.py",
    }
    with matrix_path.open("w") as stream:
        json.dump(payload, stream, indent=2)
    return {"scene": scene, "pano_id": pano_id, "output": str(output),
            "alignment": str(matrix_path), "skipped": False,
            "determinant": payload["determinant"],
            "orthogonality_max_error": payload["orthogonality_max_error"]}


def plot_pose_overlay(parsed: dict, output: Path) -> None:
    levels = sorted(parsed["level_z"])
    figure, axes = plt.subplots(1, len(levels), squeeze=False,
                                figsize=(7 * len(levels), 7))
    for column, level in enumerate(levels):
        axis = axes[0, column]
        for region_id, region in parsed["regions"].items():
            if region["level"] != level:
                continue
            points = parsed["floors"][region_id]["points"]
            color = "#dddddd" if region["label"] in ROOM_EXCLUDE else "#d9eaf7"
            axis.fill(points[:, 0], points[:, 1], color=color,
                      edgecolor="#555555", linewidth=0.8)
            center = points.mean(axis=0)
            axis.text(center[0], center[1], str(region_id), fontsize=6,
                      ha="center", va="center", color="#444444")
        panos = [item for item in parsed["panoramas"] if item["level"] == level]
        if panos:
            xy = np.asarray([item["position"][:2] for item in panos])
            axis.scatter(xy[:, 0], xy[:, 1], s=10, color="#c62828",
                         marker="o", label=f"viewpoints ({len(panos)})")
        axis.set_title(f"{parsed['scene']} · level {level}")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best", fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def build_inventory(selected_scenes: List[str]) -> Tuple[dict, List[dict]]:
    rows = []
    all_panos = []
    totals = Counter()
    for scene in selected_scenes:
        parsed = parse_house(scene)
        grouped = skybox_faces(scene)
        expected = {item["pano_id"] for item in parsed["panoramas"]}
        present = set(grouped)
        incomplete = {pano_id: sorted(set(range(6)) - set(faces))
                      for pano_id, faces in grouped.items()
                      if set(faces) != set(range(6))}
        unknown = sorted(present - expected)
        missing = sorted(expected - present)
        if missing or unknown or incomplete:
            raise RuntimeError(
                f"{scene}: skybox mismatch missing={missing}, unknown={unknown}, "
                f"incomplete={incomplete}")

        dimensions = Counter()
        formats = Counter()
        for faces in grouped.values():
            for path in faces.values():
                with Image.open(path) as image:
                    dimensions[str(image.size)] += 1
                    formats[str(image.format)] += 1
        containment = containment_diagnostic(parsed)
        pose_position_deltas = []
        pose_det = []
        pose_ortho = []
        for pano in parsed["panoramas"]:
            pose = load_pose(scene, pano["pano_id"])
            pose_position_deltas.append(float(np.linalg.norm(
                pose[:3, 3] - pano["position"])))
            pose_det.append(float(np.linalg.det(pose[:3, :3])))
            pose_ortho.append(float(np.max(np.abs(
                pose[:3, :3].T.dot(pose[:3, :3]) - np.eye(3)))))

        viewpoints_by_level = Counter(
            item["level"] for item in parsed["panoramas"])
        nonstair_level_disagreements = sum(
            item["region_level"] is not None
            and item["region_label"] not in ROOM_METRIC_EXCLUDE
            and item["region_level"] != item["level"]
            for item in parsed["panoramas"])
        region_views = Counter(
            item["region_id"] for item in parsed["panoramas"]
            if item["region_id"] is not None)
        metric_rooms = [region_id for region_id, region in parsed["regions"].items()
                        if region["label"] not in ROOM_METRIC_EXCLUDE]
        indoor_regions = [region_id for region_id, region in parsed["regions"].items()
                          if region["label"] not in ROOM_EXCLUDE]
        covered_metric = sum(region_views[region_id] > 0 for region_id in metric_rooms)
        covered_indoor = sum(region_views[region_id] > 0 for region_id in indoor_regions)
        row = {
            "scene": scene,
            "viewpoints": len(parsed["panoramas"]),
            "viewpoints_by_level": dict(sorted(viewpoints_by_level.items())),
            "level_camera_z_centers": parsed["level_camera_z"],
            "nonstair_region_vs_z_level_disagreements": nonstair_level_disagreements,
            "levels": len(parsed["level_z"]),
            "metric_rooms": len(metric_rooms),
            "covered_metric_rooms": covered_metric,
            "metric_room_coverage": covered_metric / len(metric_rooms),
            "indoor_regions_including_stairs": len(indoor_regions),
            "covered_indoor_regions": covered_indoor,
            "indoor_region_coverage": covered_indoor / len(indoor_regions),
            "skybox_files": 6 * len(grouped),
            "skybox_dimensions": dict(dimensions),
            "skybox_formats": dict(formats),
            "containment": containment,
            "official_minus1_viewpoints": sum(
                item["region_id_raw"] == -1 for item in parsed["panoramas"]),
            "unassigned_viewpoints": sum(
                item["region_id"] is None for item in parsed["panoramas"]),
            "pose_vs_house_position_max_delta_m": max(pose_position_deltas),
            "pose_rotation_det_range": [min(pose_det), max(pose_det)],
            "pose_rotation_max_orthogonality_error": max(pose_ortho),
            "camera_height_range_m": [
                min(item["camera_height"] for item in parsed["panoramas"]),
                max(item["camera_height"] for item in parsed["panoramas"]),
            ],
        }
        rows.append(row)
        for key in ("viewpoints", "metric_rooms", "covered_metric_rooms",
                    "indoor_regions_including_stairs", "covered_indoor_regions",
                    "skybox_files"):
            totals[key] += row[key]
        all_panos.extend({"scene": scene, **item} for item in parsed["panoramas"])
        plot_pose_overlay(parsed, DATA_OUT / "pose_overlays" / f"{scene}.png")

    manifest = {
        "baseline_id": "panorama_layout_v0_1",
        "scenes": len(selected_scenes),
        "totals": dict(totals),
        "metric_room_coverage": (
            totals["covered_metric_rooms"] / totals["metric_rooms"]),
        "indoor_region_coverage": (
            totals["covered_indoor_regions"] /
            totals["indoor_regions_including_stairs"]),
        "skybox_contract": {
            "faces_per_viewpoint": 6,
            "input_convention": "MP3D U,L,F,R,B,D",
            "output_projection": "equirectangular",
            "output_size": [1024, 512],
        },
        "pose_contract": {
            "position": "official .house P record",
            "orientation": (
                "official matterport_camera_poses pose_1_5; its forward is "
                "skybox image1/left, following EDM CVPR 2025; columns are "
                "camera right, down, forward"),
            "world_frame": "MP3D native, empirically checked against .house floors",
        },
        "per_scene": rows,
    }
    return manifest, all_panos


def execute(tasks: list, function, workers: int, label: str) -> list:
    if workers <= 1:
        return [function(task) for task in tqdm(tasks, desc=label)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(function, tasks), total=len(tasks), desc=label))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = sorted(args.scenes or scene_ids())
    if len(selected) != 16 and args.scenes is None:
        raise RuntimeError(f"expected 16 scenes, found {len(selected)}")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    manifest, panos = build_inventory(selected)

    if args.stitch:
        tasks = []
        for pano in panos:
            faces = skybox_faces(pano["scene"])[pano["pano_id"]]
            output = DATA_OUT / "stitched" / pano["scene"] / (pano["pano_id"] + ".png")
            tasks.append((pano["scene"], pano["pano_id"],
                          {key: str(value) for key, value in faces.items()},
                          str(output), args.force))
        stitch_results = execute(tasks, stitch_one, args.workers, "stitch")
        manifest["stitch"] = {
            "completed": len(stitch_results),
            "new": sum(not item["skipped"] for item in stitch_results),
            "seam_ratio_mean": float(np.mean([
                item["seam_ratio"] for item in stitch_results
                if "seam_ratio" in item])) if any(
                    "seam_ratio" in item for item in stitch_results) else None,
            "results": stitch_results,
        }

    if args.align:
        tasks = []
        for pano in panos:
            source = DATA_OUT / "stitched" / pano["scene"] / (pano["pano_id"] + ".png")
            if not source.exists():
                raise FileNotFoundError(f"stitch before alignment: {source}")
            output = DATA_OUT / "aligned" / pano["scene"] / (pano["pano_id"] + ".png")
            tasks.append((pano["scene"], pano["pano_id"], str(source),
                          str(output), args.force))
        align_results = execute(tasks, align_one, args.workers, "VP align")
        manifest["alignment"] = {
            "completed": len(align_results),
            "new": sum(not item["skipped"] for item in align_results),
            "results": align_results,
        }

    manifest_path = DATA_OUT / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(json_ready(manifest), stream, indent=2)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"per_scene", "stitch", "alignment"}}, indent=2))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
