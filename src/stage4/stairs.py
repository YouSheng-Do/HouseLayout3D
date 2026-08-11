"""Stage 4 / D.5 樓梯偵測：stair mesh → connected components → 水平投影 OBB 矩形 →
短邊中點高度（由鄰近 3D 頂點插值）→ 以 D_pp 指派到最近房間 → >50cm 或同房 → 拒絕。
接受者成為 scene graph 的 stair 邊（連接兩房/兩層），矩形 R 為其幾何。
"""
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

REJECT_DIST = 0.5   # D.5：邊中點到房的 D_pp > 50cm → 拒絕
MIN_RECT_AREA = 0.3 # 矩形最小面積 m²（幾何雜訊過濾；附錄未給值）# TODO: tune with MP3D
MIN_RISE = 0.4      # 兩短邊最小高差（樓梯必須爬升）# TODO: tune with MP3D


@dataclass
class StairCandidate:
    rect3d: np.ndarray                 # (4,3) 傾斜矩形（短邊已抬到各自高度）
    end_midpoints: np.ndarray          # (2,3) 兩短邊中點
    rooms: Optional[Tuple] = None      # 指派結果 ((level,room), (level,room))
    rejected: str = ""


def _components(mesh: o3d.geometry.TriangleMesh) -> List[np.ndarray]:
    tc, _, _ = mesh.cluster_connected_triangles()
    tc = np.asarray(tc)
    T = np.asarray(mesh.triangles)
    comps = []
    for c in range(tc.max() + 1 if len(tc) else 0):
        vids = np.unique(T[tc == c])
        if len(vids) >= 3:  # 幾何過濾（面積/爬升）在 OBB 之後做
            comps.append(np.asarray(mesh.vertices)[vids])
    # 鄰近合併：flight/landing 常是分離的 mesh 片（GT 逐矩形標註、pipeline 分割破碎）
    # → 元件間最近點 < merge_dist 視為同一座樓梯（single-linkage）
    merge_dist = 0.4  # TODO: tune with MP3D
    from scipy.spatial import cKDTree
    n = len(comps)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    trees = [cKDTree(c) for c in comps]
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            d, _ = trees[i].query(comps[j], k=1)
            if d.min() < merge_dist:
                parent[find(j)] = find(i)
    merged = defaultdict(list)
    for i in range(n):
        merged[find(i)].append(comps[i])
    return [np.concatenate(g) for g in merged.values()]


def _obb_rect(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """水平投影的最小面積矩形 → 4 角 (4,2)＋短邊對 ((0,1),(2,3)) 保證：回傳角序使
    corners[0]-corners[1] 與 corners[2]-corners[3] 是兩條短邊。"""
    import cv2
    xy = pts[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(xy)
    box = cv2.boxPoints(rect)  # 4x2, 順序繞圈
    e01 = np.linalg.norm(box[1] - box[0])
    e12 = np.linalg.norm(box[2] - box[1])
    if e01 <= e12:  # 0-1 是短邊 → 短邊對 = (0,1),(3,2)
        corners = np.array([box[0], box[1], box[3], box[2]])
    else:           # 1-2 是短邊 → 短邊對 = (1,2),(0,3)
        corners = np.array([box[1], box[2], box[0], box[3]])
    return corners  # 短邊A=(c0,c1), 短邊B=(c2,c3)


def _edge_height(pts: np.ndarray, mid_xy: np.ndarray, r: float = 0.5) -> float:
    d = np.linalg.norm(pts[:, :2] - mid_xy, axis=1)
    sel = pts[d < r]
    while len(sel) == 0 and r < 3.0:
        r *= 1.6
        d = np.linalg.norm(pts[:, :2] - mid_xy, axis=1)
        sel = pts[d < r]
    return float(np.median(sel[:, 2])) if len(sel) else float(np.median(pts[:, 2]))


def detect_stairs(stair_mesh: o3d.geometry.TriangleMesh, levels) -> List[StairCandidate]:
    """levels: scene_graph.Level list（需已 segment_rooms，有 _grid_origin/_final/elevation）。"""
    from scipy import ndimage
    out = []
    for pts in _components(stair_mesh):
        corners2d = _obb_rect(pts)
        zA = _edge_height(pts, corners2d[:2].mean(axis=0))
        zB = _edge_height(pts, corners2d[2:].mean(axis=0))
        wA = np.linalg.norm(corners2d[1] - corners2d[0])
        wL = np.linalg.norm(corners2d[2] - corners2d[0])
        if wA * wL < MIN_RECT_AREA or abs(zA - zB) < MIN_RISE:
            continue  # 幾何過濾：太小或沒有爬升 → 非樓梯
        rect3d = np.array([[*corners2d[0], zA], [*corners2d[1], zA],
                           [*corners2d[2], zB], [*corners2d[3], zB]])
        midA = rect3d[:2].mean(axis=0)
        midB = rect3d[2:].mean(axis=0)
        cand = StairCandidate(rect3d=rect3d, end_midpoints=np.array([midA, midB]))

        # 指派：對每層每房計算 D_pp(mid, 房的 floor 多邊形@elevation)（2D 距離＋高度差合成）
        def assign(mid):
            best = (None, np.inf)
            for lv in levels:
                dz = mid[2] - lv.elevation
                if dz < -1.0 or dz > 3.5:  # 樓梯端點應在該層地板附近或上方
                    continue
                from scene_graph import RES
                final = lv._final
                H, W = final.shape
                j, i = ((mid[:2] - lv._grid_origin) / RES).astype(int)
                # 距離變換到各房：找最近的室內格
                inroom = final > 0
                if not inroom.any():
                    continue
                dist, (iy, ix) = ndimage.distance_transform_edt(~inroom, return_indices=True)
                if 0 <= i < H and 0 <= j < W:
                    d2d = dist[i, j] * RES
                    rid = final[iy[i, j], ix[i, j]]
                else:
                    continue
                d = float(np.hypot(d2d, max(0.0, abs(dz) - 0.3)))
                if d < best[1]:
                    best = ((lv.idx, int(rid)), d)
            return best

        (ra, da), (rb, db) = assign(midA), assign(midB)
        if ra is None or rb is None or da > REJECT_DIST or db > REJECT_DIST:
            cand.rejected = f"D_pp 過遠 (A={da if ra else 'NA'}, B={db if rb else 'NA'})"
        elif ra == rb:
            cand.rejected = "兩端同房"
        else:
            cand.rooms = (ra, rb)
        out.append(cand)
    return out
