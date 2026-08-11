"""Stage 4 核心（Appendix D 規格自寫）：樓層識別 → 2D floorplan → 房間分割（HOV-SG 兩段式）
→ 房間 extrusion（CDT＋向上射線指派天花板）→ 門框 → 輸出 combined.ply（灰/綠/藍同官方配色）。

規格出處：Appendix D.1–D.6；參數：50cm 樓層合併、ceiling→floor ≥1m、牆 z 區間 [0,2.5]m、
bottleneck 2.5m→1.5m、door<1.5m、門框高 2.10m、每房 ≤30 天花板候選。
"""
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon as ShPolygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_prototype import Prototype, ProtoPolygon, load_prototype  # noqa: E402

RES = 0.05           # floorplan 格點解析度 m
LEVEL_MERGE = 0.5    # D.1 樓層高度合併門檻
CEIL_MIN_ABOVE = 1.0 # D.2 ceiling 需高出 floor 至少 1m
WALL_Z_BAND = 2.5    # D.2 牆的 z 區間
BOTTLENECK_1 = 2.5   # D.3 第一輪（舊 Voronoi 法用；保留參照）
BOTTLENECK_2 = 1.5   # D.3 第二輪
ROOM_CORE_MIN = 0.50 # watershed 房間核心：距牆 > 此值才算房間內部  # TODO: tune with MP3D
DOOR_MAX_W = 1.5     # door vs opening（舊；保留給 extrude classify 參照）
DOOR_W_MIN = 0.55    # 門寬下界（GT 門中位 ~0.78m；濾過度分割碎縫）  # TODO: tune with MP3D
DOOR_W_MAX = 1.30    # 門寬上界（濾寬拱門/合併房邊界）              # TODO: tune with MP3D
DOOR_H = 2.10        # D.6 門框高
MAX_CEILINGS = 30
# 結構牆高度門檻（2t7W 假牆診斷 2026-07-14：桌沿/櫃體矮直立面被擠出放大成全高牆）
# 真牆需「頂到天花帶」且跨度佔層高一定比例；門楣（2.1~天花）通過、家具（<2.2m 且不及頂）擋下。
WALL_TOP_MARGIN = 0.5    # z1 ≥ ceil_ref - margin 才算及頂  # TODO: tune with MP3D
WALL_MIN_SPAN_FRAC = 0.35  # (z1-z0) ≥ frac*(ceil_ref-elev)   # TODO: tune with MP3D

GRAY = (191 / 255, 191 / 255, 191 / 255)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def poly_2d(p: ProtoPolygon) -> ShPolygon:
    """polygon 的 BEV 精確足跡：三角形投影的聯集（支援凹形；凸殼會吃掉缺角）。"""
    if p.tris3d is not None and len(p.tris3d) > 0:
        tris = []
        for t in p.tris3d:
            xy = t[:, :2]
            try:
                s = ShPolygon(xy)
                if s.area > 1e-8:
                    tris.append(s.buffer(1e-6))
            except Exception:
                continue
        if tris:
            u = unary_union(tris)
            if isinstance(u, MultiPolygon):
                u = max(u.geoms, key=lambda g: g.area)
            return u
    from scipy.spatial import ConvexHull
    xy = p.verts[:, :2]
    if len(xy) < 3:
        return ShPolygon()
    try:
        h = ConvexHull(xy)
        return ShPolygon(xy[h.vertices])
    except Exception:
        return ShPolygon()


@dataclass
class Level:
    idx: int
    floors: List[ProtoPolygon]
    elevation: float
    ceilings: List[ProtoPolygon] = field(default_factory=list)
    walls: List[ProtoPolygon] = field(default_factory=list)
    floorplan: ShPolygon = None
    rooms: List[dict] = field(default_factory=list)      # {mask, poly2d, id}
    openings: List[dict] = field(default_factory=list)   # {rooms:(a,b), width, seg(2,2), is_door}


def identify_levels(proto: Prototype, min_floor_area: float = 5.0,
                    max_level_span: float = 0.8) -> List[Level]:
    floors_all = proto.by_class("floor")
    if not floors_all:
        # fallback：語意投票全判 ceiling 的 corner case（p5wJjkQkbXX）——
        # 依高度分群，把每群「最低的水平面」重判為 floor
        horiz = [p for p in proto.polygons if p.cls in ("ceiling", "slanted")
                 and abs(p.plane[2]) > 0.94]
        if horiz:
            print("  ⚠ 無 floor polygon → fallback：以各高度群最低水平面充當 floor")
            zs = sorted(set(round(p.elevation, 1) for p in horiz))
            groups = []
            for z in zs:
                if groups and z - groups[-1][-1] <= 1.0:
                    groups[-1].append(z)
                else:
                    groups.append([z])
            floor_zs = {g[0] for g in groups}
            for p in horiz:
                if round(p.elevation, 1) in floor_zs and p.area >= 2.0:
                    p.cls = "floor"
            floors_all = proto.by_class("floor")
    if not floors_all:
        raise RuntimeError("無 floor polygon（fallback 亦失敗）")
    # 樓層圖只用大面積 floor（樓梯平台/踏步不該定義樓層）；小 floor 之後歸入最近層
    floors = [f for f in floors_all if f.area >= min_floor_area] or floors_all
    n = len(floors)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    span_lo = {i: floors[i].elevation for i in range(n)}
    span_hi = {i: floors[i].elevation for i in range(n)}
    order = sorted(((abs(floors[i].elevation - floors[j].elevation), i, j)
                    for i in range(n) for j in range(i + 1, n)))
    for d, i, j in order:
        if d > LEVEL_MERGE:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        lo, hi = min(span_lo[ri], span_lo[rj]), max(span_hi[ri], span_hi[rj])
        if hi - lo <= max_level_span:   # 防樓梯平台鏈式塌縮
            parent[rj] = ri
            span_lo[ri], span_hi[ri] = lo, hi
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(floors[i])
    levels = [Level(idx=k, floors=g, elevation=float(np.mean([f.elevation for f in g])))
              for k, g in enumerate(sorted(groups.values(), key=lambda g: np.mean([f.elevation for f in g])))]
    # 小 floor（平台等）併入最近層的 floorplan
    for f in floors_all:
        if f.area < min_floor_area:
            best = min(levels, key=lambda lv: abs(lv.elevation - f.elevation))
            if abs(best.elevation - f.elevation) <= LEVEL_MERGE:
                best.floors.append(f)

    # D.2：ceiling → 最近的 next-lower floor（中心下方 ≥1m）
    for c in proto.by_class("ceiling") + proto.by_class("slanted"):
        if c.cls == "slanted":
            continue
        cz = c.elevation
        cands = [lv for lv in levels if cz - lv.elevation >= CEIL_MIN_ABOVE]
        if cands:
            best = max(cands, key=lambda lv: lv.elevation)
            best.ceilings.append(c)

    # 每層 floorplan：floors ∪ ceilings 的 BEV 聯集
    # 2t7W 診斷（2026-07-14）：floors∪ceilings 蓋不滿真足跡時，缺口斜邊界被 extrude
    # 當「外牆」立成穿房假牆、缺口區無天花/地板 → front/nohit 主源之一。
    # → 補牆足跡條帶 ＋ 形態學閉合 ＋ 填內孔，把 floorplan 撐到真外牆。
    for lv in levels:
        shapes = [poly_2d(p) for p in lv.floors + lv.ceilings]
        u = unary_union([s for s in shapes if s.area > 0.05])
        if isinstance(u, MultiPolygon):
            u = max(u.geoms, key=lambda g: g.area)
        lv.floorplan = u
    # （已回退）floorplan 撐大實驗：填洞讓房間融穿「init 無資料區」的真牆位置，
    # behind 5→29%、Δ5 −15 —— 缺口區沒有牆可刻，Stage 4 撐 floorplan 只是把
    # 「front 假牆」換成「behind 穿透」。根治需 init 覆蓋，非 Stage 4 可救。

    # 牆：BEV 交 floorplan 且 z 交 [elev, elev+2.5]
    # ＋高度門檻（共面群組制）：fit 常把一面真牆切成上下疊的碎片，逐片檢查會誤殺——
    # 先按 (2D 線方向, 線偏移) 聚類，取群組 z 聯集過門檻；矮孤兒（桌沿/櫃體）才擋下。
    for lv in levels:
        ceil_zs = [c.elevation for c in lv.ceilings]
        lv._ceil_ref = float(np.median(ceil_zs)) if ceil_zs else lv.elevation + WALL_Z_BAND
    walls_all = proto.by_class("wall")
    grp = list(range(len(walls_all)))          # union-find

    def find(i):
        while grp[i] != i:
            grp[i] = grp[grp[i]]
            i = grp[i]
        return i

    lines = []
    for w in walls_all:
        n2 = w.plane[:2] / max(np.linalg.norm(w.plane[:2]), 1e-9)
        c2 = w.verts[:, :2].mean(0)
        lines.append((n2, float(n2 @ c2)))
    for i in range(len(walls_all)):
        for j in range(i + 1, len(walls_all)):
            ni, oi = lines[i]
            nj, oj = lines[j]
            dot = abs(float(ni @ nj))
            if dot < 0.966:                     # 夾角 >15° 非同牆
                continue
            off = abs(oi - (oj if float(ni @ nj) > 0 else -oj))
            if off > 0.15:                      # 線偏移 >15cm 非同牆
                continue
            # BEV 需相近（同一直線上遠端不同牆段也算同牆組，對 z 聯集無害）
            di = np.linalg.norm(walls_all[i].verts[:, :2].mean(0) - walls_all[j].verts[:, :2].mean(0))
            if di > 6.0:
                continue
            grp[find(i)] = find(j)
    gz = {}                                     # 群組 z 聯集
    for i, w in enumerate(walls_all):
        g = find(i)
        z0, z1 = float(w.verts[:, 2].min()), float(w.verts[:, 2].max())
        if g in gz:
            gz[g] = (min(gz[g][0], z0), max(gz[g][1], z1))
        else:
            gz[g] = (z0, z1)
    for i, w in enumerate(walls_all):
        z0, z1 = w.verts[:, 2].min(), w.verts[:, 2].max()
        gz0, gz1 = gz[find(i)]
        for lv in levels:
            if z1 < lv.elevation or z0 > lv.elevation + WALL_Z_BAND:
                continue
            H = max(lv._ceil_ref - lv.elevation, 1.0)
            if gz1 < lv._ceil_ref - WALL_TOP_MARGIN or (gz1 - gz0) < WALL_MIN_SPAN_FRAC * H:
                continue  # 整組皆不及頂/跨度不足 → 家具/矮件，非結構牆
            wp = poly_2d(w)
            if wp.buffer(0.05).intersects(lv.floorplan):
                lv.walls.append(w)
    return levels


def _fill_geom(mask: np.ndarray, geom, origin: np.ndarray, value: int):
    """用 cv2.fillPoly 把 shapely (Multi)Polygon 填進柵格（外環填 value、內環回填反值）。
    取代 shapely.vectorized.contains——大棟複雜聯集會讓 GEOS 段錯誤且吃記憶體。"""
    import cv2
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    for g in geoms:
        if g.is_empty or g.area < 1e-9:
            continue
        ext = ((np.asarray(g.exterior.coords)[:, :2] - origin) / RES).astype(np.int32)
        cv2.fillPoly(mask, [ext], value)
        for hole in g.interiors:
            h = ((np.asarray(hole.coords)[:, :2] - origin) / RES).astype(np.int32)
            cv2.fillPoly(mask, [h], 1 - value)


def _rasterize(fp: ShPolygon) -> Tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = fp.bounds
    origin = np.array([minx - 5 * RES, miny - 5 * RES])
    W = int((maxx - origin[0]) / RES) + 10
    H = int((maxy - origin[1]) / RES) + 10
    mask = np.zeros((H, W), dtype=np.uint8)
    _fill_geom(mask, fp, origin, 1)
    return mask.astype(bool), origin


def _watershed_rooms(grid_full: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """牆屏障 watershed（HOV-SG 真實作法；2026-08-04 取代舊 Voronoi 回填）。
    距牆距離 > ROOM_CORE_MIN 的連通塊＝房間核心種子；於「牆影像」上 cv2.watershed 淹水，
    分水嶺脊線落在牆上、門洞（無牆）處兩房自然相接 → 房間邊界貼真牆。回傳 label（0=非室內）。
    對照舊 _split_rooms：那是侵蝕種子＋最近種子回填＝Voronoi 中線，邊界不貼牆（診斷根因）。"""
    import cv2
    barrier = wall_mask.astype(bool) | (~grid_full)
    free = grid_full & ~wall_mask.astype(bool)
    dist = ndimage.distance_transform_edt(free)               # 像素距離（到最近牆/室外）
    cores = free & (dist * RES > ROOM_CORE_MIN)               # 房間內部（濾門洞/窄縫）
    markers, n = ndimage.label(cores)
    if n == 0:
        out = np.zeros(grid_full.shape, np.int32); out[grid_full] = 1
        return out
    bg = n + 1
    markers = markers.astype(np.int32)
    markers[~grid_full] = bg                                  # 室外＝背景種子（外牆成界）
    img = np.where(barrier, 255, 0).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)  # 脊線沿牆；界＝ -1
    lab = markers.copy()
    lab[(lab == bg) | (lab < 0)] = 0                          # 去背景＋分水嶺界線
    unknown = grid_full & (lab <= 0)                          # 牆帶/界線/殘洞 → 最近房
    if unknown.any() and (lab > 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(lab <= 0, return_indices=True)
        lab[unknown] = lab[iy, ix][unknown]
    lab[~grid_full] = 0
    return lab


def _split_rooms(grid: np.ndarray, bottleneck_m: float) -> np.ndarray:
    """[舊法，保留參照] Voronoi 回填：腐蝕 r=bottleneck/2 → 種子 CC → 最近種子回填。"""
    r = max(1, int(round(bottleneck_m / 2 / RES)))
    eroded = ndimage.binary_erosion(grid, structure=np.ones((3, 3)), iterations=r)
    seeds, n = ndimage.label(eroded)
    if n <= 1:
        out = np.zeros_like(grid, dtype=np.int32)
        out[grid] = 1
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    out = seeds[iy, ix]
    out[~grid] = 0
    return out


def segment_rooms(lv: Level, wall_buffer: float = 0.08):
    grid_full, origin = _rasterize(lv.floorplan)
    # D.3「using the level's walls」：刻蝕柵格【只用於分割】（門洞留空 → 腐蝕即可分房）；
    # 分割與 openings 算完後回填到完整 floorplan（牆帶歸最近房），房殼才會落在真牆線上。
    H, W = grid_full.shape
    wall_mask = np.zeros((H, W), dtype=np.uint8)
    for w in lv.walls:  # 逐牆填，避免一次 union 巨型幾何
        wp = poly_2d(w).buffer(wall_buffer)
        _fill_geom(wall_mask, wp, origin, 1)
    grid = grid_full & ~wall_mask.astype(bool)
    # 房間分割：牆屏障 watershed（取代舊 Voronoi 回填 _split_rooms；根因見診斷）
    final = _watershed_rooms(grid_full, wall_mask)
    lv.rooms = []
    for rid in range(1, final.max() + 1):
        mask = final == rid
        if mask.sum() * RES * RES < 1.0:   # <1 m² 併入鄰居
            final[mask] = 0
            continue
        lv.rooms.append({"id": rid, "mask": mask})
    # 併掉的孔洞回填到最近房
    if (final == 0).any() and lv.rooms:
        _, (iy, ix) = ndimage.distance_transform_edt(final == 0, return_indices=True)
        fill = final[iy, ix]
        final[(final == 0) & grid] = fill[(final == 0) & grid]
        for r in lv.rooms:
            r["mask"] = final == r["id"]

    # openings：相鄰房間的邊界帶
    # 門幾何（2026-08-04 修）：seg 舊版＝邊界帶 AABB 對角，L 形帶會歪。改 PCA 主軸取兩端
    # → 門段沿真實門洞方向、寬度＝主軸延伸量。is_door 舊版只有上界 span<1.5（把 0.16/0.35m
    # 過度分割碎縫也當門）→ 改門寬帶 [DOOR_W_MIN, DOOR_W_MAX]，濾掉碎縫與寬拱門。
    # 註：recall 天花板受房間分割限制（16 棟僅 38 個 opening 落門寬帶 vs 292 GT 門），
    #     本修主要回收 precision；真正拉 recall 需改善房間分割（與擠出器同源）。
    for i, ra in enumerate(lv.rooms):
        for rb in lv.rooms[i + 1:]:
            border = ndimage.binary_dilation(ra["mask"]) & rb["mask"]
            if border.sum() == 0:
                continue
            ys, xs = np.where(border)
            pts = np.c_[xs, ys] * RES + origin + RES / 2
            if len(pts) >= 2:
                c = pts.mean(0)
                _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
                d = vt[0]
                t = (pts - c) @ d
                p0, p1 = c + t.min() * d, c + t.max() * d
                width = float(t.max() - t.min())
            else:
                p0 = p1 = pts[0]; width = 0.0
            lv.openings.append({
                "rooms": (ra["id"], rb["id"]), "width": width,
                "seg": np.array([p0, p1]),
                "is_door": bool(DOOR_W_MIN <= width <= DOOR_W_MAX)})
    # 回填：房間標籤擴張到完整 floorplan（牆帶＋殘洞歸最近房）→ 供 extrusion / 樓梯 D_pp 用
    if lv.rooms:
        fill_mask = grid_full & (final == 0)
        if fill_mask.any():
            _, (iy, ix) = ndimage.distance_transform_edt(final == 0, return_indices=True)
            final = final.copy()
            final[fill_mask] = final[iy, ix][fill_mask]
        for r in lv.rooms:
            r["mask"] = final == r["id"]
    lv._grid_origin = origin
    lv._grid = grid_full
    lv._carved = grid          # 分割用刻蝕柵格（牆帶=False）：extrusion 判「真牆 vs 門洞/開口」
    lv._final = final
    return lv


def _wall_runs(cnt: np.ndarray, z_of: list) -> List[Tuple[int, int]]:
    """房輪廓的連續共線邊（同高度）合併成牆段 → [(起點索引, 終點索引(不含)), ...]。"""
    n = len(cnt)
    runs = []
    i = 0
    while i < n:
        j = i + 1
        d0 = cnt[(i + 1) % n] - cnt[i]
        d0 = d0 / (np.linalg.norm(d0) + 1e-12)
        while j < n:
            dj = cnt[(j + 1) % n] - cnt[j]
            dj = dj / (np.linalg.norm(dj) + 1e-12)
            if abs(float(d0 @ dj)) < 0.999 or abs(z_of[j % n] - z_of[i]) > 0.02:
                break
            j += 1
        runs.append((i, j))
        i = j
    return runs


def extrude_level(lv: Level, out_tris: list, out_cols: list, entities: dict = None):
    """D 節 extrusion（簡化版 CDT：per-room 光柵輪廓 → marching squares 多邊形 → triangle CDT）。"""
    import triangle as tr
    ceilings = sorted(lv.ceilings, key=lambda c: -c.area)[:MAX_CEILINGS]
    floor_z = lv.elevation

    def plane_z(plane, x, y):
        a, b, c, d = plane
        if abs(c) < 1e-6:
            return None
        return (-d - a * x - b * y) / c

    for room in lv.rooms:
        mask = room["mask"]
        # 輪廓：光柵 → 多邊形（cv2 findContours 於 uint8）
        import cv2
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt_px = max(cnts, key=cv2.contourArea)
        cnt_px = cv2.approxPolyDP(cnt_px, epsilon=0.05 / RES, closed=True)  # RDP 去光柵鋸齒
        cnt = cnt_px.reshape(-1, 2).astype(np.float64) * RES + lv._grid_origin + RES / 2
        if len(cnt) < 3:
            continue
        # RDP 可能產生自交輪廓 → Shewchuk Triangle 對無效 PSLG 會 C 端段錯誤（jtcx 崩因）。
        # 三角化前以 shapely 驗證，無效則 make_valid 並取最大有效多邊形外環。
        try:
            shp = ShPolygon(cnt)
            if not shp.is_valid:
                from shapely.validation import make_valid
                fixed = make_valid(shp)
                geoms = list(fixed.geoms) if hasattr(fixed, "geoms") else [fixed]
                polys = [g for g in geoms if g.geom_type == "Polygon" and g.area > 0.5]
                if not polys:
                    continue
                cnt = np.asarray(max(polys, key=lambda g: g.area).exterior.coords)[:-1, :2]
            if len(cnt) < 3 or abs(ShPolygon(cnt).area) < 0.5:
                continue
        except Exception:
            continue
        seg = {"vertices": cnt,
               "segments": np.array([[i, (i + 1) % len(cnt)] for i in range(len(cnt))])}
        t = tr.triangulate(seg, "p")
        if "triangles" not in t:
            continue
        v2 = t["vertices"]; tris2 = t["triangles"]

        # 指派天花板：三角形中心向上（取含中心 BEV 的最低-可用 ceiling；無 → 最大 ceiling）
        centers = v2[tris2].mean(axis=1)
        ceil_polys = [(c, poly_2d(c)) for c in ceilings]
        from shapely.geometry import Point
        floor_ent = []                    # 本房地板實體三角形
        ceil_ent = defaultdict(list)      # 天花板實體：per (room, ceiling pid)
        for tri, ctr in zip(tris2, centers):
            zc = None; chosen = None
            for c, cp in ceil_polys:
                if cp.contains(Point(ctr)):
                    z = plane_z(c.plane, ctr[0], ctr[1])
                    if z is not None and z > floor_z + 0.5 and (zc is None or z < zc):
                        zc, chosen = z, c
            if chosen is None and ceil_polys:
                chosen = ceil_polys[0][0]
            p3f = [[*v2[i], floor_z] for i in tri]
            p3c = [[*v2[i], plane_z(chosen.plane, v2[i][0], v2[i][1]) if chosen else floor_z + 2.5]
                   for i in tri]
            out_tris.append(p3f); out_cols.append(GRAY)
            out_tris.append(p3c[::-1]); out_cols.append(GRAY)
            floor_ent.append(p3f)
            ceil_ent[chosen.pid if chosen else -1].append(p3c[::-1])
        if entities is not None:
            if floor_ent:
                entities["structures"].append(np.array(floor_ent))
            for tris_c in ceil_ent.values():
                entities["structures"].append(np.array(tris_c))

        # 天花板不連續垂直帶（官方 D 節「vertical rectangles along discontinuous edges」）：
        # CDT 內共享邊兩側三角形若指派到不同天花板高度（>3cm）→ 補垂直矩形
        edge_map = {}
        tri_zfun = []
        for tri, ctr in zip(tris2, centers):
            zc = None; chosen = None
            for c, cp in ceil_polys:
                if cp.contains(Point(ctr)):
                    z = plane_z(c.plane, ctr[0], ctr[1])
                    if z is not None and z > floor_z + 0.5 and (zc is None or z < zc):
                        zc, chosen = z, c
            if chosen is None and ceil_polys:
                chosen = ceil_polys[0][0]
            tri_zfun.append(chosen.plane if chosen else None)
        for t_i, tri in enumerate(tris2):
            for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(e), max(e))
                edge_map.setdefault(key, []).append(t_i)
        for (va, vb), tlist in edge_map.items():
            if len(tlist) != 2:
                continue
            pa, pb = tri_zfun[tlist[0]], tri_zfun[tlist[1]]
            if pa is None or pb is None or (pa == pb).all():
                continue
            A2d, B2d = v2[va], v2[vb]
            zA1, zB1 = plane_z(pa, *A2d), plane_z(pa, *B2d)
            zA2, zB2 = plane_z(pb, *A2d), plane_z(pb, *B2d)
            if None in (zA1, zB1, zA2, zB2):
                continue
            if max(abs(zA1 - zA2), abs(zB1 - zB2)) < 0.03:
                continue
            A = [*A2d, zA1]; B = [*B2d, zB1]; A2v = [*A2d, zA2]; B2v = [*B2d, zB2]
            out_tris.append([A, B, B2v]); out_cols.append(GRAY)
            out_tris.append([A, B2v, A2v]); out_cols.append(GRAY)
            if entities is not None:
                entities["structures"].append(np.array([[A, B, B2v], [A, B2v, A2v]]))

        # 牆：房輪廓邊 extrude；沿邊逐樣本分類（D.6）：
        #   exterior→整面牆；interior 真牆（刻蝕帶）→整面牆；door→僅 2.1m 以上補牆；opening→不加牆
        carved = lv._carved
        gfull = lv._grid
        Hg, Wg = gfull.shape
        org = lv._grid_origin

        def cell(pt):
            j, i = int((pt[0] - org[0]) / RES), int((pt[1] - org[1]) / RES)
            return i, j

        def classify(mid, nrm):
            probe_out = mid + nrm * (2.5 * RES)
            probe_in = mid - nrm * (2.5 * RES)
            im, jm = cell(mid)
            i2, j2 = cell(probe_out)
            i3, j3 = cell(probe_in)
            mid_carved = (0 <= im < Hg and 0 <= jm < Wg) and not carved[im, jm] and gfull[im, jm]
            out_carved = (0 <= i2 < Hg and 0 <= j2 < Wg) and not carved[i2, j2] and gfull[i2, j2]
            in_carved = (0 <= i3 < Hg and 0 <= j3 < Wg) and not carved[i3, j3] and gfull[i3, j3]
            io, jo = cell(probe_out)
            if not (0 <= io < Hg and 0 <= jo < Wg) or not gfull[io, jo]:
                # 外邊界：有牆證據（刻蝕帶）→ 外牆；無 → floorplan 缺口（init 無資料區），
                # 不立牆也不填洞——立牆=front 假牆、填洞=behind 穿透（2t7W 兩試皆證偽）。
                return "wall" if (mid_carved or out_carved or in_carved) else "open"
            if mid_carved or out_carved:
                return "wall"          # 內牆（刻蝕帶）
            # 門洞必須通向另一個「房間」；label 0 / 外側 視為牆，不得開洞。
            # 2t7W 驗屍（2026-07-15）：探針落回**自己房間**＝這段輪廓是假邊界
            # （RDP 弦切進自己 mask / 回填鋸齒帶），舊版歸「wall」→ 沿假邊界立全高
            # 雙面牆（front 失敗主源）。真牆由 CARVE/EXT 分支把關 → 同房一律 open。
            other = lv._final[i2, j2] if (0 <= i2 < Hg and 0 <= j2 < Wg) else 0
            # own-room 假邊界三政策實測（2t7W，同 fit 確定性對照，2026-07-15）：
            #   全立牆 Δ5=51.8/F1=0.118 > 貼牆線 50.1/0.083 > 全 open 48.3/0.063。
            # 假邊界多沿 init 缺資料的真牆位置 → 亂立牆「碰巧近似正確」，
            # 拆牆只是 front→nohit 的類別搬移且丟 F1。維持立牆；根治=init 覆蓋（Stage 3）。
            return "door" if (other > 0 and other != room["id"]) else "wall"

        for i in range(len(cnt)):
            a, b = cnt[i], cnt[(i + 1) % len(cnt)]
            L = np.linalg.norm(b - a)
            if L < 1e-6:
                continue
            mid_all = (a + b) / 2
            zc = None
            for c, cp in ceil_polys:
                if cp.buffer(0.2).contains(Point(mid_all)):
                    z = plane_z(c.plane, mid_all[0], mid_all[1])
                    if z and z > floor_z + 0.5 and (zc is None or z < zc):
                        zc = z
            if zc is None:
                zc = floor_z + (np.median([c.elevation for c in ceilings]) - floor_z if ceilings else 2.5)
            d = (b - a) / L
            nrm = np.array([d[1], -d[0]])
            n_s = max(1, int(L / RES))
            ts = (np.arange(n_s) + 0.5) / n_s
            cls_s = [classify(a + t * L * d, nrm) for t in ts]
            # 連續同類 run → 出幾何
            s0 = 0
            for k in range(1, n_s + 1):
                if k < n_s and cls_s[k] == cls_s[s0]:
                    continue
                p0 = a + (s0 / n_s) * L * d
                p1 = a + (k / n_s) * L * d
                seg_cls = cls_s[s0]
                z_lo = floor_z if seg_cls == "wall" else floor_z + DOOR_H
                if seg_cls != "open" and zc > z_lo + 0.02 and np.linalg.norm(p1 - p0) > 0.03:
                    A = [*p0, z_lo]; B = [*p1, z_lo]; A2 = [*p0, zc]; B2 = [*p1, zc]
                    out_tris.append([A, B, B2]); out_cols.append(GRAY)
                    out_tris.append([A, B2, A2]); out_cols.append(GRAY)
                    if entities is not None:
                        entities["structures"].append(np.array([[A, B, B2], [A, B2, A2]]))
                if seg_cls != "wall" and np.linalg.norm(p1 - p0) > 0.03:
                    # 跨房界天花板高差帶：鄰房天花板與本房不同高 → 補垂直帶（否則開放通道上方漏洞）
                    midp = (p0 + p1) / 2 + nrm * (3 * RES)
                    zc_out = None
                    for c, cp in ceil_polys:
                        if cp.buffer(0.2).contains(Point(midp)):
                            z = plane_z(c.plane, midp[0], midp[1])
                            if z and z > floor_z + 0.5 and (zc_out is None or z < zc_out):
                                zc_out = z
                    if zc_out is not None and abs(zc_out - zc) > 0.03:
                        zl, zh = sorted((zc, zc_out))
                        A = [*p0, zl]; B = [*p1, zl]; A2 = [*p0, zh]; B2 = [*p1, zh]
                        out_tris.append([A, B, B2]); out_cols.append(GRAY)
                        out_tris.append([A, B2, A2]); out_cols.append(GRAY)
                        if entities is not None:
                            entities["structures"].append(np.array([[A, B, B2], [A, B2, A2]]))
                s0 = k

    # 門框（綠）：opening 的 seg → 垂直門框 2.10m（幾何為單面帶；實體=門矩形 4 角）
    for op in lv.openings:
        if not op["is_door"]:
            continue
        (x0, y0), (x1, y1) = op["seg"]
        z0, z1 = lv.elevation, lv.elevation + DOOR_H
        A = [x0, y0, z0]; B = [x1, y1, z0]; A2 = [x0, y0, z1]; B2 = [x1, y1, z1]
        out_tris.append([A, B, B2]); out_cols.append(GREEN)
        out_tris.append([A, B2, A2]); out_cols.append(GREEN)
        if entities is not None:
            entities["doors"].append(np.array([A, B, B2, A2]))


def build_scene_graph(proto: Prototype, out_dir: str, stage2_dir: str = None,
                      stair_mesh_path: str = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    levels = identify_levels(proto)
    tris, cols = [], []
    entities = {"structures": [], "doors": [], "windows": [], "stairs": []}
    graph = {"levels": []}
    for lv in levels:
        segment_rooms(lv)
        extrude_level(lv, tris, cols, entities)
        graph["levels"].append({
            "idx": lv.idx, "elevation": lv.elevation,
            "n_floors": len(lv.floors), "n_ceilings": len(lv.ceilings),
            "n_walls": len(lv.walls),
            "rooms": [{"id": r["id"], "area_m2": float(r["mask"].sum() * RES * RES)} for r in lv.rooms],
            "openings": [{"rooms": o["rooms"], "width": o["width"], "is_door": o["is_door"]}
                          for o in lv.openings]})

    # 窗偵測（藍）
    graph["windows"] = []
    if stage2_dir is not None:
        from windows import detect_windows
        rects = detect_windows(proto, stage2_dir)
        for r in rects:
            tris.append([r[0].tolist(), r[1].tolist(), r[2].tolist()]); cols.append(BLUE)
            tris.append([r[0].tolist(), r[2].tolist(), r[3].tolist()]); cols.append(BLUE)
            graph["windows"].append({"corners": r.tolist()})
            entities["windows"].append(r)
        print(f"[stage4] windows: {len(rects)}")

    # 樓梯（D.5）
    graph["stairs"] = []
    if stair_mesh_path is not None and os.path.exists(stair_mesh_path):
        from stairs import detect_stairs
        sm = o3d.io.read_triangle_mesh(stair_mesh_path)
        if len(sm.triangles) > 0:
            for cand in detect_stairs(sm, levels):
                if cand.rooms is None:
                    continue
                graph["stairs"].append({"rooms": cand.rooms, "rect": cand.rect3d.tolist()})
                entities["stairs"].append(cand.rect3d)
                r = cand.rect3d
                tris.append([r[0].tolist(), r[1].tolist(), r[2].tolist()]); cols.append(GRAY)
                tris.append([r[0].tolist(), r[2].tolist(), r[3].tolist()]); cols.append(GRAY)
        print(f"[stage4] stairs kept: {len(graph['stairs'])}")

    # 原生實體匯出（鏡射 HF GT 格式；eval_scene 直接可吃）
    ent_dir = os.path.join(out_dir, "entities")
    os.makedirs(ent_dir, exist_ok=True)
    import glob as _glob
    for f in _glob.glob(os.path.join(ent_dir, "structure_*.ply")):
        os.remove(f)
    for i, tri_arr in enumerate(entities["structures"]):
        v = tri_arr.reshape(-1, 3)
        fidx = np.arange(len(v)).reshape(-1, 3)
        em = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                       o3d.utility.Vector3iVector(fidx))
        em = em.merge_close_vertices(1e-6)
        o3d.io.write_triangle_mesh(os.path.join(ent_dir, f"structure_{i:04d}.ply"), em)
    for name in ("doors", "windows", "stairs"):
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({name: [{"vertices": np.asarray(r).tolist()} for r in entities[name]]}, f)
    print(f"[stage4] entities: {len(entities['structures'])} structures, "
          f"{len(entities['doors'])} doors, {len(entities['windows'])} windows, "
          f"{len(entities['stairs'])} stairs → {ent_dir}")

    # 輸出 combined ply
    verts = np.array(tris, dtype=np.float64).reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    vcols = np.repeat(np.array(cols, dtype=np.float64), 3, axis=0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                  o3d.utility.Vector3iVector(faces))
    m.vertex_colors = o3d.utility.Vector3dVector(vcols)
    m = m.merge_close_vertices(1e-4)
    o3d.io.write_triangle_mesh(f"{out_dir}/combined.ply", m)
    with open(f"{out_dir}/scene_graph.json", "w") as f:
        json.dump(graph, f, indent=1)
    print(f"[stage4] levels={len(levels)}; " + "; ".join(
        f"L{lv.idx}: rooms={len(lv.rooms)}, doors={sum(o['is_door'] for o in lv.openings)}, "
        f"openings={len(lv.openings)}, walls={len(lv.walls)}, ceilings={len(lv.ceilings)}"
        for lv in levels))
    print(f"[stage4] combined.ply: {len(np.asarray(m.vertices))} verts, {len(faces)} tris → {out_dir}")
    return graph


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage2-dir", default=None)
    ap.add_argument("--stair-mesh", default=None)
    a = ap.parse_args()
    proto = load_prototype(a.fitted, a.skeleton, a.probs, a.labels)
    build_scene_graph(proto, a.out_dir, stage2_dir=a.stage2_dir, stair_mesh_path=a.stair_mesh)
