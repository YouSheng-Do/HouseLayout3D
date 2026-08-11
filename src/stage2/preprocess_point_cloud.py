"""Stage 2 缺件 (b)：superpoint 分割入口——補上 supplementary 缺的
`superpoint_transformer.scripts.preprocess_point_cloud.segment_point_cloud_superpoints`。

實作：upstream SPT 的標準前處理管線（configs/datamodule/semantic/{default,scannet}.yaml 的
pre_transform 序列，取到 CutPursuitPartition 為止），參數用 ScanNet 室內預設
（voxel 0.02、knn 45、reg [0.01,0.1,0.5]…）。# TODO: tune with MP3D

輸出（對齊官方 extract_skeleton.py 的讀取介面）：
  {output_directory}/level_{1,2,3}_segmentation.npy —— 每個「全解析度頂點」的 superpoint id
"""
import os
import sys

import numpy as np
import open3d as o3d
import torch

SPT_ROOT = "/home/ado/storage/HouseLayout3D/external/superpoint_transformer"
if SPT_ROOT not in sys.path:
    sys.path.insert(0, SPT_ROOT)

from src.data import Data  # noqa: E402
from src.transforms import (  # noqa: E402
    AddKeysTo, AdjacencyGraph, ConnectIsolated, CutPursuitPartition, DataTo,
    GridSampling3D, GroundElevation, KNN, PointFeatures, SaveNodeIndex)

# ScanNet 室內預設（Robert et al. 前處理）  # TODO: tune with MP3D
VOXEL = 0.02
KNN_K, KNN_R = 45, 2
KNN_STEP, KNN_MIN_SEARCH = -1, 25
PCP_REG = [0.01, 0.1, 0.5]
PCP_SPATIAL_W = [0.1, 0.1, 0.1]
PCP_CUTOFF = [10, 10, 10]
PCP_K_ADJ, PCP_W_ADJ, PCP_ITER = 10, 1, 15
GROUND_THRESHOLD, GROUND_XY_GRID, GROUND_SCALE = 1.5, 1, 4.0
FEATURE_KEYS = ["linearity", "planarity", "scattering", "verticality"]
PARTITION_X = ["rgb", "linearity", "planarity", "scattering", "verticality", "elevation"]


def _load_points(input_pcd: str):
    mesh = o3d.io.read_triangle_mesh(input_pcd)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pos = np.asarray(mesh.vertices, dtype=np.float32)
        rgb = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
    else:
        pcd = o3d.io.read_point_cloud(input_pcd)
        pos = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    assert len(pos) > 0, f"讀不到點: {input_pcd}"
    return pos, rgb


def _full_to_voxel(nag, n_full: int) -> torch.Tensor:
    """nag[0].sub（Cluster/CSR：每個 voxel 含哪些全解析度點）→ 每個全解析度點的 voxel id。"""
    sub = nag[0].sub
    pointers = sub.pointers.cpu()
    points = sub.points.cpu()
    sizes = pointers[1:] - pointers[:-1]
    voxel_of_point = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    out = torch.full((n_full,), -1, dtype=torch.long)
    out[points] = voxel_of_point
    assert (out >= 0).all(), "有全解析度點沒被指派 voxel"
    return out


def segment_point_cloud_superpoints(input_pcd: str, output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)
    pos, rgb = _load_points(input_pcd)
    n_full = len(pos)
    print(f"[spt-partition] {input_pcd}: {n_full} 點, rgb={'有' if rgb is not None else '無'}")

    kwargs = dict(pos=torch.from_numpy(pos))
    if rgb is not None and len(rgb) == n_full:
        kwargs["rgb"] = torch.from_numpy(rgb)
    data = Data(**kwargs)

    partition_x = [k for k in PARTITION_X if k != "rgb" or "rgb" in kwargs]

    tfs = [
        DataTo("cuda"),
        SaveNodeIndex(key="sub"),
        GridSampling3D(size=VOXEL, inplace=True),
        KNN(k=KNN_K, r_max=KNN_R, verbose=False),
        PointFeatures(keys=FEATURE_KEYS, k_min=1, k_step=KNN_STEP, k_min_search=KNN_MIN_SEARCH),
        GroundElevation(z_threshold=GROUND_THRESHOLD, xy_grid=GROUND_XY_GRID, scale=GROUND_SCALE),
        AdjacencyGraph(k=PCP_K_ADJ, w=PCP_W_ADJ),
        ConnectIsolated(k=1),
        AddKeysTo(keys=partition_x, to="x", delete_after=False),
        CutPursuitPartition(
            regularization=PCP_REG, spatial_weight=PCP_SPATIAL_W, k_adjacency=PCP_K_ADJ,
            cutoff=PCP_CUTOFF, iterations=PCP_ITER, edge_reduce="mean",
            parallel=True, verbose=False),
    ]
    out = data
    for t in tfs:
        out = t(out)
    nag = out  # CutPursuitPartition 之後是 NAG

    full2vox = _full_to_voxel(nag, n_full)

    # voxel → 各層 superpoint id（super_index 逐層鏈接）
    per_level_voxel = {}
    chain = None
    for level in range(1, nag.num_levels):
        si = nag[level - 1].super_index.cpu()
        chain = si if chain is None else si[chain]
        per_level_voxel[level] = chain

    for level in range(1, nag.num_levels):
        seg_full = per_level_voxel[level][full2vox].numpy().astype(np.int64)
        np.save(os.path.join(output_directory, f"level_{level}_segmentation.npy"), seg_full)
        n_seg = int(seg_full.max()) + 1
        print(f"[spt-partition] level {level}: {n_seg} superpoints "
              f"(每 superpoint 平均 {n_full / max(n_seg,1):.0f} 點)")

    # 視覺化：level 2 上色 ply
    rng = np.random.RandomState(0)
    lvl = 2 if 2 in per_level_voxel else max(per_level_voxel)
    seg = per_level_voxel[lvl][full2vox].numpy()
    colors = rng.rand(seg.max() + 1, 3)[seg]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(output_directory, f"superpoints_level{lvl}_viz.ply"), pcd)
    print(f"[spt-partition] 視覺化: {output_directory}/superpoints_level{lvl}_viz.ply")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pcd", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    segment_point_cloud_superpoints(args.input_pcd, args.output_dir)
