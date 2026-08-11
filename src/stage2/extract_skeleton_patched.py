"""官方 extract_skeleton.py 的最小修改版（原檔：docs/supplementary/.../multi-floor-3d-code/extract_skeleton.py）。

修改點（其餘邏輯與官方一致）：
  [P1] --seg-dir：分割 PNG 改為 <seg_dir>/<image_stem>.png（官方寫死作者機器路徑）
  [P2] --depth-scale：深度 PNG 尺度參數化（MP3D=0.00025，MuSHRoom iPhone=0.001）
  [P3] file_path / depth_file_path 若為相對路徑，以 poses 檔所在目錄解析
  [P4] superpoint 分割入口改 import 我們的 src/stage2/preprocess_point_cloud（官方 zip 缺該檔）
  [P5] geometry_util 直接從官方 skeleton_extraction/ 目錄 import（保持單一來源）
"""
from collections import Counter
import os
from pathlib import Path
import sys

import PIL
from tqdm import tqdm
from typing import List, Tuple
import open3d as o3d
import numpy as np
import torch
import argparse
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = "/home/ado/storage/HouseLayout3D/docs/supplementary/supplementary/multi-floor-3d-code/skeleton_extraction"
sys.path.insert(0, _HERE)
sys.path.insert(0, _OFFICIAL)
from geometry_util import get_colored_points_from_depth          # [P5] 官方版
from preprocess_point_cloud import segment_point_cloud_superpoints  # [P4] 我們的實作


def parse_args():
    p = argparse.ArgumentParser(description="Annotate a mesh with OneFormer and aggregate labels.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--poses-file", required=True)
    p.add_argument("--mesh-file", required=True)
    p.add_argument("--seg-dir", required=True, help="[P1] OneFormer 簡化標籤 PNG 目錄")
    p.add_argument("--labels-file", default=None, help="預設 <seg-dir>/labels.txt")
    p.add_argument("--depth-scale", type=float, default=0.00025, help="[P2] 深度 PNG × 此值 = 公尺")
    p.add_argument("--samples-per-frame", type=int, default=3000)
    p.add_argument("--n-neighbors-for-aggregation", type=int, default=5)
    p.add_argument("--keep-classes", type=str,
                   default="ceiling,wall,floor,cabinet,door,curtain,window_blind,stairs")
    p.add_argument("--object-classes", type=str, default=None)
    p.add_argument("--window-classes", type=str, default="window")
    return p.parse_args()


def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)


def run_oneformer_for_each_image(poses, samples_per_frame, labels, seg_dir, poses_root, depth_scale):
    ray_origins, points, segmentations, depth_is_valid = [], [], [], []

    for image_idx, pose in enumerate(tqdm(poses["frames"], desc="Processing frames")):
        def _resolve(path):  # [P3]
            return path if os.path.isabs(path) else os.path.normpath(os.path.join(poses_root, path))

        file_path = _resolve(pose["file_path"])
        fl_x, fl_y = pose["fl_x"], pose["fl_y"]
        cx, cy = pose["cx"], pose["cy"]
        H, W = pose["h"], pose["w"]
        transform_matrix = pose["transform_matrix"]

        depth_path = _resolve(pose["depth_file_path"])
        depth_gt = PIL.Image.open(depth_path)
        depth_gt = (np.array(depth_gt) * depth_scale).astype(np.float32)  # [P2]
        assert depth_gt.shape == (H, W), f"{depth_gt.shape} != {(H, W)}"

        image = np.array(PIL.Image.open(file_path))

        stem = os.path.splitext(os.path.basename(file_path))[0]           # [P1]
        pred = PIL.Image.open(os.path.join(seg_dir, stem + ".png"))
        assert np.max(pred) < len(labels)
        if not pred.size == (W, H):
            pred = pred.resize((W, H), PIL.Image.NEAREST)
        pred = np.array(pred)
        assert pred.shape == (H, W) and pred.dtype == np.uint8

        c2w = np.array(transform_matrix).astype(np.float32)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]]).astype(np.float32)
        c2w[0:3, 1:3] *= -1

        depth_map = torch.from_numpy(depth_gt).float().cuda()
        c2w = torch.from_numpy(c2w).cuda()
        image = torch.from_numpy(image).float().cuda()

        valid_mask = torch.ones(H, W).bool().cuda()
        indices = pick_indices_at_random(valid_mask, samples_per_frame)

        valid_depth = (depth_map > 0).view(-1)
        assert not depth_map.isnan().any()
        assert depth_map.isfinite().all()

        depth_map_inpainted = depth_map.clone()
        depth_map_inpainted[depth_map <= 0] = 0.5

        xyzs, rgbs, _ = get_colored_points_from_depth(
            depths=depth_map_inpainted, rgbs=image, features={},
            fx=fl_x, fy=fl_y, cx=cx, cy=cy, img_size=(W, H), c2w=c2w, mask=indices)

        seg = torch.from_numpy(pred).to(indices.device).view(-1)[indices]
        valid_depth = valid_depth.view(-1)[indices]
        assert len(seg) == len(xyzs)

        points.append(xyzs.detach().cpu().numpy())
        ray_origins.append(c2w[:3, 3].unsqueeze(0).repeat(len(xyzs), 1).detach().cpu().numpy())
        segmentations.append(seg.detach().cpu().numpy())
        depth_is_valid.append(valid_depth.detach().cpu().numpy())

    return (np.concatenate(ray_origins), np.concatenate(points),
            np.concatenate(segmentations), np.concatenate(depth_is_valid))


def compute_per_point_feature_knn(vertices, original_points, original_point_features, k=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    per_point_features = [[None for _ in range(len(vertices))] for _ in range(len(original_point_features))]
    for vertex_ix, v in enumerate(tqdm(vertices, desc="KNN aggregation")):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, k)
        for i, f in enumerate(original_point_features):
            per_point_features[i][vertex_ix] = np.mean(f[idx], axis=0)
    return [np.array(p) for p in per_point_features]


def save_filtered(mesh, vertex_class_probabilities, class_names, output_dir,
                  keep_classes, object_classes=None, window_classes=("window",)):
    if len(vertex_class_probabilities.shape) > 1:
        vertex_class_labels = vertex_class_probabilities.argmax(axis=1)
    else:
        vertex_class_labels = vertex_class_probabilities
        vertex_class_probabilities = np.arange(len(class_names))[None, :] == vertex_class_labels[:, None]

    os.makedirs(output_dir, exist_ok=True)
    class_names = np.array(class_names)
    per_vertex_classes_text = class_names[vertex_class_labels]
    ceiling_wall_floor_mask = np.isin(per_vertex_classes_text, keep_classes)

    ceiling_wall_floor_mesh = o3d.geometry.TriangleMesh(mesh)
    ceiling_wall_floor_mesh.remove_vertices_by_mask(~ceiling_wall_floor_mask)
    ceiling_wall_floor_mesh_classes = vertex_class_probabilities[ceiling_wall_floor_mask]
    assert len(ceiling_wall_floor_mesh_classes) == len(ceiling_wall_floor_mesh.vertices)
    o3d.io.write_triangle_mesh(f"{output_dir}/ceiling_wall_floor_mesh.ply", ceiling_wall_floor_mesh)
    np.save(f"{output_dir}/ceiling_wall_floor_mesh_classes.npy",
            ceiling_wall_floor_mesh_classes.astype(np.float16))
    print(f"Classes found: {Counter(per_vertex_classes_text)}")
    print(f"Classes in ceiling_wall_floor_mesh: {Counter(per_vertex_classes_text[ceiling_wall_floor_mask])}")

    window_mask = np.isin(per_vertex_classes_text, window_classes)
    object_mask = ~ceiling_wall_floor_mask & (~window_mask) if object_classes is None \
        else np.isin(per_vertex_classes_text, object_classes)
    objects_mesh = o3d.geometry.TriangleMesh(mesh)
    objects_mesh.remove_vertices_by_mask(~object_mask)
    objects_mesh_classes = vertex_class_probabilities[object_mask]
    assert len(objects_mesh_classes) == len(objects_mesh.vertices)
    o3d.io.write_triangle_mesh(f"{output_dir}/objects_mesh.ply", objects_mesh)
    np.save(f"{output_dir}/objects_mesh_classes.npy", objects_mesh_classes)
    print(f"Classes in objects_mesh: {Counter(per_vertex_classes_text[object_mask])}")

    stair_mask = np.isin(per_vertex_classes_text, ["stairs", "stair"])
    stairs_mesh = o3d.geometry.TriangleMesh(mesh)
    stairs_mesh.remove_vertices_by_mask(~stair_mask)
    o3d.io.write_triangle_mesh(f"{output_dir}/stair_mesh.ply", stairs_mesh)
    print(f"Saved all outputs to {output_dir}")


def extract_skeleton(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {output_dir}")

    mesh = o3d.io.read_triangle_mesh(args.mesh_file)

    with open(args.poses_file, "r") as f:
        poses = json.load(f)
    poses_root = os.path.dirname(os.path.abspath(args.poses_file))  # [P3]

    labels_file = args.labels_file or os.path.join(args.seg_dir, "labels.txt")
    with open(labels_file, "r") as f:
        labels = [l.strip() for l in f if l.strip()]

    ray_origins, points, segmentations, depth_is_valid = run_oneformer_for_each_image(
        poses=poses, samples_per_frame=args.samples_per_frame, labels=labels,
        seg_dir=args.seg_dir, poses_root=poses_root, depth_scale=args.depth_scale)

    colors = np.random.rand(len(labels), 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[segmentations])
    o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    mesh_vertices = np.asarray(mesh.vertices)
    del mesh

    print("Computing aggregated segmentations")
    segmentations_one_hot = np.eye(len(labels)).astype(np.float16)[segmentations]
    aggregated_segmentations, = compute_per_point_feature_knn(
        vertices=mesh_vertices,
        original_points=points[depth_is_valid],
        original_point_features=[segmentations_one_hot[depth_is_valid]],
        k=args.n_neighbors_for_aggregation)

    aggregated_vertex_hard_labels = np.argmax(aggregated_segmentations, axis=-1)
    print("Segmenting mesh")
    segment_point_cloud_superpoints(                       # [P4]
        input_pcd=str(output_dir / "mesh.ply"),
        output_directory=str(output_dir / "spt"))

    aggregated_segmentations_torch = torch.from_numpy(aggregated_segmentations).float().cuda()
    print("Aggregating features")
    for level in tqdm(range(1, 4), desc="Aggregating features"):
        mesh_segmentation = torch.from_numpy(
            np.load(output_dir / "spt" / f"level_{level}_segmentation.npy").astype(np.int32)).long().cuda()
        assert len(mesh_segmentation) == len(mesh_vertices)
        n_segments = int(mesh_segmentation.max()) + 1
        agg = torch.zeros((n_segments, len(labels))).cuda()
        for i in range(n_segments):
            mask = mesh_segmentation == i
            assert mask.sum() > 0
            probabilities = aggregated_segmentations_torch[mask].mean(axis=0)
            agg[i] = probabilities / (probabilities.sum() + 1e-6)
        per_segment_hard_assignments = torch.argmax(agg, dim=-1).cpu().numpy()
        np.save(output_dir / "spt" / f"level_{level}_segment_hard_assignments_simplified.npy",
                per_segment_hard_assignments)
        np.save(output_dir / "spt" / f"level_{level}_segment_probabilities_simplified.npy",
                agg.cpu().numpy().astype(np.float16))

    mesh = o3d.io.read_triangle_mesh(str(output_dir / "mesh.ply"))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        colors[per_segment_hard_assignments[mesh_segmentation.cpu().numpy()]])
    o3d.io.write_triangle_mesh(str(output_dir / "spt" / "mesh_class_colored.ply"), mesh)
    np.save(output_dir / "vertex_probabilities.npy", aggregated_segmentations.astype(np.float16))
    np.save(output_dir / "vertex_hard_assignments.npy", aggregated_vertex_hard_labels.astype(np.uint16))
    np.save(output_dir / "simplified_segmentation_labels.npy", np.array(labels))

    print("Statistics of hard labels")
    print(Counter(np.array(labels)[per_segment_hard_assignments]))

    np.save(output_dir / "full_ray_origins.npy", ray_origins.astype(np.float16))
    np.save(output_dir / "full_ray_dests.npy", points.astype(np.float16))
    np.save(output_dir / "ray_is_valid.npy", depth_is_valid.astype(bool))
    np.save(output_dir / "hard_labels_simplified_segmentations.npy",
            segmentations.astype(np.uint8 if len(labels) < 256 else np.uint16))

    save_filtered(
        mesh, aggregated_segmentations, labels, output_dir,
        keep_classes=args.keep_classes.split(","),
        object_classes=args.object_classes.split(",") if args.object_classes is not None else None,
        window_classes=args.window_classes.split(","))
    print("Done!, wrote to", output_dir)


if __name__ == "__main__":
    extract_skeleton(parse_args())
