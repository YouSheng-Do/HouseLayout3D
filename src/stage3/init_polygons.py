"""Stage 3 缺件：polygon 初始化（Appendix C.1, Algorithm 1）——官方 zip 未附的
`polygon_info.json` ＋ `clean_edge_mesh.ply` 生產者。

Algorithm 1（循序 RANSAC）：
  while 存在未指派頂點數 > K 的 superpoint cluster:
    S* ← 未指派頂點最多的 cluster
    以 RANSAC 對 S* 擬合平面 → 全 mesh 未指派頂點中取 inliers
    取「與 S* 重疊最大」的 inlier 連通元件 C → 指派給新平面
    以 C 的三角形邊界抽多邊形（外輪廓＋洞）加入 P

polygon_info schema（反讀自 fit_prototype.py / from_polygon_info）：
  { "<id>": { contours: [[outer...],[hole...]...],  # 索引 clean_edge_mesh 頂點
              plane_eq: [a,b,c,d], color: [r,g,b], class: str,
              vertices: [inlier ix...], shared_edges: {} } }

# TODO: tune with MP3D —— K、RANSAC 門檻、superpoint level 皆為未定參數（附錄未給值）
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

KEEP_CLASSES = ["ceiling", "wall", "floor", "cabinet", "door", "curtain", "window_blind", "stairs"]


def fit_plane_svd(pts: np.ndarray):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n / np.linalg.norm(n), -float(n @ c)


def ransac_plane(pts: np.ndarray, thresh: float, iters: int, rng) -> tuple:
    best_n, best_d, best_cnt = None, None, -1
    n_pts = len(pts)
    for _ in range(iters):
        ix = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[ix]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    inl = np.abs(pts @ best_n + best_d) < thresh
    if inl.sum() >= 3:
        best_n, best_d = fit_plane_svd(pts[inl])
    return best_n, best_d


def boundary_loops(tris: np.ndarray):
    """三角形集合 → 邊界迴圈（每條邊界邊只屬一個三角形）。回傳頂點索引迴圈 list。"""
    from collections import defaultdict
    edge_cnt = Counter()
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            edge_cnt[(min(e), max(e))] += 1
    b_adj = defaultdict(list)
    for (a, b), cnt in edge_cnt.items():
        if cnt == 1:
            b_adj[a].append(b)
            b_adj[b].append(a)
    visited_e = set()
    loops = []
    for start in list(b_adj):
        for nxt in b_adj[start]:
            if (min(start, nxt), max(start, nxt)) in visited_e:
                continue
            loop = [start]
            prev, cur = start, nxt
            visited_e.add((min(start, nxt), max(start, nxt)))
            guard = 0
            while cur != start and guard < 200000:
                loop.append(cur)
                cands = [v for v in b_adj[cur]
                         if v != prev and (min(cur, v), max(cur, v)) not in visited_e]
                if not cands:
                    break
                prev, cur = cur, cands[0]
                visited_e.add((min(prev, cur), max(prev, cur)))
                guard += 1
            if cur == start and len(loop) >= 3:
                loops.append(loop)
    return loops


def simplify_loop(verts: np.ndarray, loop, n: np.ndarray, tol: float):
    """把跟隨 mesh 三角化的鋸齒邊界迴圈用 RDP 化簡成乾淨直邊多邊形（在多邊形平面 2D 上做），
    回傳保留的頂點索引 list。tol=0 時不化簡。"""
    loop = list(map(int, loop))
    if tol <= 0 or len(loop) < 4:
        return loop
    from rdp import rdp as _rdp
    basis_a = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(basis_a) < 1e-6:
        basis_a = np.cross(n, [0, 1.0, 0])
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(n, basis_a)
    p = verts[loop]
    uv = np.c_[p @ basis_a, p @ basis_b]
    keep = _rdp(uv, epsilon=tol, return_mask=True)   # 閉環：頭尾同點附近皆保留
    idx = [loop[i] for i in range(len(loop)) if keep[i]]
    return idx if len(idx) >= 3 else loop


def loop_area(verts: np.ndarray, loop, n: np.ndarray) -> float:
    """迴圈在平面上的投影面積（shoelace）。"""
    basis_a = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(basis_a) < 1e-6:
        basis_a = np.cross(n, [0, 1.0, 0])
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(n, basis_a)
    p = verts[loop]
    u, v = p @ basis_a, p @ basis_b
    return 0.5 * abs(float(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", required=True, help="extract_skeleton 的 output dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--level", type=int, default=3, help="superpoint level  # TODO: tune with MP3D")
    ap.add_argument("--min-unassigned", type=int, default=-1,
                    help="Algorithm 1 的 K；-1=自適應 max(150, n_verts//2000)。"
                         "絕對值在不同 mesh 密度間不可移植（K=2000 曾使 MP3D 各棟 25–99% 未指派）")
    ap.add_argument("--ransac-thresh", type=float, default=0.02, help="inlier 門檻 m  # TODO: tune")
    ap.add_argument("--ransac-iters", type=int, default=200)
    ap.add_argument("--min-component-verts", type=int, default=30)
    ap.add_argument("--min-hole-area", type=float, default=0.02, help="洞輪廓最小面積 m²")
    ap.add_argument("--contour-rdp", type=float, default=0.0,
                    help="邊界 RDP 化簡容差 m（實驗：0.03 對 17DRP 無助益且傷 coverage，預設關閉）")
    ap.add_argument("--max-polygons", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    os.makedirs(args.out_dir, exist_ok=True)
    s2 = args.stage2_dir

    labels = np.array([str(x) for x in np.load(f"{s2}/simplified_segmentation_labels.npy")])
    probs = np.load(f"{s2}/vertex_probabilities.npy")
    hard_full = probs.argmax(axis=1)
    keep_mask = np.isin(labels[hard_full], KEEP_CLASSES)

    skel = o3d.io.read_triangle_mesh(f"{s2}/ceiling_wall_floor_mesh.ply")
    V = np.asarray(skel.vertices)
    T = np.asarray(skel.triangles)
    full_ids = np.where(keep_mask)[0]
    assert len(full_ids) == len(V), f"skeleton 對應失敗: {len(full_ids)} vs {len(V)}"

    seg_full = np.load(f"{s2}/spt/level_{args.level}_segmentation.npy")
    clusters = seg_full[full_ids].astype(np.int64)
    vclass = labels[hard_full[full_ids]]
    print(f"[init] skeleton {len(V)} 頂點, {len(T)} 三角形, level-{args.level} clusters "
          f"{len(np.unique(clusters))} 個")

    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)

    K = args.min_unassigned if args.min_unassigned > 0 else max(150, len(V) // 2000)
    print(f"[init] 停止閾值 K = {K}（{'自適應' if args.min_unassigned <= 0 else '指定'}）")

    unassigned = np.ones(len(V), dtype=bool)
    polygon_info = {}
    t0 = time.time()
    n_cluster = int(clusters.max()) + 1
    it = 0
    while len(polygon_info) < args.max_polygons:
        it += 1
        cnt = np.bincount(clusters[unassigned], minlength=n_cluster)
        c_star = int(cnt.argmax())
        if cnt[c_star] <= K:
            break
        cand = np.where(unassigned & (clusters == c_star))[0]
        sub = cand if len(cand) <= 20000 else cand[rng.choice(len(cand), 20000, replace=False)]
        n, d = ransac_plane(V[sub], args.ransac_thresh, args.ransac_iters, rng)

        inl_mask = unassigned & (np.abs(V @ n + d) < args.ransac_thresh)
        e_ok = inl_mask[edges[:, 0]] & inl_mask[edges[:, 1]]
        e = edges[e_ok]
        if len(e) == 0:
            unassigned[cand] = False
            continue
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
        n_comp, comp = connected_components(g, directed=False)
        comp_of_inl = comp[inl_mask]
        overlap = Counter(comp[cand[np.abs(V[cand] @ n + d) < args.ransac_thresh]])
        if not overlap:
            unassigned[cand] = False
            continue
        best_comp = overlap.most_common(1)[0][0]
        C = np.where(inl_mask & (comp == best_comp))[0]
        if len(C) < args.min_component_verts:
            unassigned[cand[np.abs(V[cand] @ n + d) < args.ransac_thresh]] = False
            continue

        in_C = np.zeros(len(V), dtype=bool)
        in_C[C] = True
        tri_mask = in_C[T[:, 0]] & in_C[T[:, 1]] & in_C[T[:, 2]]
        tris_C = T[tri_mask]
        if len(tris_C) == 0:
            unassigned[C] = False
            continue
        loops = boundary_loops(tris_C)
        if not loops:
            unassigned[C] = False
            continue
        areas = [loop_area(V, l, n) for l in loops]
        order = np.argsort(areas)[::-1]
        contours = [simplify_loop(V, loops[order[0]], n, args.contour_rdp)]
        for oi in order[1:]:
            if areas[oi] >= args.min_hole_area:
                contours.append(simplify_loop(V, loops[oi], n, args.contour_rdp))

        # 法向對齊表面（以 C 的平均頂點法向定向）
        skel.compute_vertex_normals()
        vn = np.asarray(skel.vertex_normals)[C].mean(axis=0)
        if float(vn @ n) < 0:
            n, d = -n, -d

        cls = Counter(vclass[C]).most_common(1)[0][0]
        polygon_info[str(len(polygon_info))] = {
            "contours": contours,
            "plane_eq": [float(x) for x in (*n, d)],
            "color": [float(x) for x in rng.rand(3)],
            "class": str(cls),
            "vertices": list(map(int, C[:: max(1, len(C) // 2000)])),
            "shared_edges": {},
        }
        unassigned[C] = False
        if it % 10 == 0 or len(polygon_info) <= 3:
            print(f"  iter {it}: polygons={len(polygon_info)}, cluster {c_star} ({cnt[c_star]} pts), "
                  f"C={len(C)}, class={cls}, 未指派 {unassigned.sum()/len(V)*100:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(f"{args.out_dir}/polygon_info.json", "w") as f:
        json.dump(polygon_info, f)
    o3d.io.write_triangle_mesh(f"{args.out_dir}/clean_edge_mesh.ply", skel)

    cls_stat = Counter(v["class"] for v in polygon_info.values())
    print(f"[init] 完成: {len(polygon_info)} polygons, 類別 {dict(cls_stat)}, "
          f"未指派比例 {unassigned.sum()/len(V)*100:.1f}%, 耗時 {time.time()-t0:.0f}s")

    # 視覺化：多邊形輪廓 3D 線段 ply
    ls_pts, ls_lines, ls_cols = [], [], []
    for pid, poly in polygon_info.items():
        col = poly["color"]
        for contour in poly["contours"]:
            base = len(ls_pts)
            ls_pts.extend(V[contour].tolist())
            m = len(contour)
            ls_lines.extend([[base + i, base + (i + 1) % m] for i in range(m)])
            ls_cols.extend([col] * m)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(ls_pts)),
                              o3d.utility.Vector2iVector(np.array(ls_lines)))
    ls.colors = o3d.utility.Vector3dVector(np.array(ls_cols))
    o3d.io.write_line_set(f"{args.out_dir}/polygon_contours_viz.ply", ls)
    print(f"[init] 視覺化: {args.out_dir}/polygon_contours_viz.ply")


if __name__ == "__main__":
    main()
