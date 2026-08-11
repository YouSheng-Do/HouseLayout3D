"""Audit the frozen 2D GT overlap without changing the evaluated baseline.

The frozen GT was derived from projected ``regionN.ply`` vertices, followed by
5 cm raster closing/fill and 10 cm RDP.  MP3D ``.house`` files also contain the
original floor-surface polygons (S ... F + V records).  This script compares
the two representations and writes a read-only diagnostic figure/JSON/Markdown.
"""
import glob
import json
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
OUT = f"{BASE}/eval2d_v2_hungarian"
FIG = f"{OUT}/gt_overlap_audit.png"
JSON_OUT = f"{OUT}/gt_overlap_audit.json"
MD_OUT = f"{ROOT}/docs/gt_overlap_audit.md"
RES = 0.05


def configure_font():
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
    # The bundled TTC registers itself as the JP face in Matplotlib even though
    # it contains the Traditional Chinese glyphs we need.  Put that registered
    # face first so report figures do not fall back to missing-glyph boxes.
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def safe_polygon(points):
    g = Polygon(np.asarray(points, float))
    if not g.is_valid:
        g = g.buffer(0)
    return g


def read_house_floor_polygons(scene):
    """Return region_id -> original MP3D floor-surface polygon."""
    path = (f"{ROOT}/data/mp3d/v1/scans/{scene}/{scene}/"
            f"house_segmentations/{scene}.house")
    surface_region = {}
    surface_kind = {}
    vertices = defaultdict(list)
    for line in open(path, errors="ignore"):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "S":
            surface_region[int(fields[1])] = int(fields[2])
            surface_kind[int(fields[1])] = fields[4]
        elif fields[0] == "V":
            vertices[int(fields[2])].append((float(fields[4]), float(fields[5])))
    result = {}
    for surface_id, points in vertices.items():
        if surface_kind.get(surface_id) != "F" or len(points) < 3:
            continue
        geom = safe_polygon(points)
        if not geom.is_empty and geom.area > 0:
            result[surface_region[surface_id]] = geom
    return result


def overlap_excess(geometries):
    """Return sum(area)-union(area), union(area)."""
    geometries = [g for g in geometries if not g.is_empty and g.area > 0]
    if not geometries:
        return 0.0, 0.0
    union_area = unary_union(geometries).area
    return float(sum(g.area for g in geometries) - union_area), float(union_area)


def audit_scene(path):
    scene = os.path.basename(path)[:-4]
    gt = pickle.load(open(path, "rb"))
    house = read_house_floor_polygons(scene)
    raw_overlap = raw_union = current_overlap = current_union = 0.0
    house_overlap = house_union = 0.0
    representation_ious = []

    for level in sorted(gt["lvl_z"]):
        rooms = [r for r in gt["rooms"] if r["level"] == level
                 and r.get("in_roomset") and not r.get("is_stairs")]
        occs = [np.asarray(r["_occ"], bool) for r in rooms if "_occ" in r]
        if occs:
            counts = np.sum(occs, axis=0)
            raw_overlap += float(np.maximum(counts - 1, 0).sum() * RES * RES)
            raw_union += float((counts > 0).sum() * RES * RES)

        current_geoms = [safe_polygon(r["poly"]) for r in rooms]
        direct_geoms = [house[r["idx"]] for r in rooms if r["idx"] in house]
        overlap, union = overlap_excess(current_geoms)
        current_overlap += overlap
        current_union += union
        overlap, union = overlap_excess(direct_geoms)
        house_overlap += overlap
        house_union += union

        for room, current in zip(rooms, current_geoms):
            direct = house.get(room["idx"])
            if direct is None:
                continue
            union = current.union(direct).area
            if union > 0:
                representation_ious.append(current.intersection(direct).area / union)

    return {
        "scene": scene,
        "raw_raster_overlap_excess_m2": raw_overlap,
        "raw_raster_union_m2": raw_union,
        "raw_raster_overlap_ratio": raw_overlap / raw_union if raw_union else 0.0,
        "current_polygon_overlap_excess_m2": current_overlap,
        "current_polygon_union_m2": current_union,
        "current_polygon_overlap_ratio": current_overlap / current_union if current_union else 0.0,
        "house_floor_overlap_excess_m2": house_overlap,
        "house_floor_union_m2": house_union,
        "house_floor_overlap_ratio": house_overlap / house_union if house_union else 0.0,
        "current_vs_house_mean_iou": float(np.mean(representation_ious)) if representation_ious else None,
        "n_rooms_compared": len(representation_ious),
    }


def ratio(total_overlap, total_union):
    return total_overlap / total_union if total_union else 0.0


def main():
    configure_font()
    rows = [audit_scene(path) for path in sorted(glob.glob(f"{BASE}/gt/*.pkl"))]
    totals = {
        "raw_raster_overlap_excess_m2": sum(r["raw_raster_overlap_excess_m2"] for r in rows),
        "raw_raster_union_m2": sum(r["raw_raster_union_m2"] for r in rows),
        "current_polygon_overlap_excess_m2": sum(r["current_polygon_overlap_excess_m2"] for r in rows),
        "current_polygon_union_m2": sum(r["current_polygon_union_m2"] for r in rows),
        "house_floor_overlap_excess_m2": sum(r["house_floor_overlap_excess_m2"] for r in rows),
        "house_floor_union_m2": sum(r["house_floor_union_m2"] for r in rows),
        "n_rooms_compared": sum(r["n_rooms_compared"] for r in rows),
    }
    totals["raw_raster_overlap_ratio"] = ratio(
        totals["raw_raster_overlap_excess_m2"], totals["raw_raster_union_m2"])
    totals["current_polygon_overlap_ratio"] = ratio(
        totals["current_polygon_overlap_excess_m2"], totals["current_polygon_union_m2"])
    totals["house_floor_overlap_ratio"] = ratio(
        totals["house_floor_overlap_excess_m2"], totals["house_floor_union_m2"])
    total_rooms = sum(r["n_rooms_compared"] for r in rows)
    totals["current_vs_house_mean_iou"] = (
        sum(r["current_vs_house_mean_iou"] * r["n_rooms_compared"] for r in rows
            if r["current_vs_house_mean_iou"] is not None) / max(total_rooms, 1))

    payload = {
        "status": "diagnostic_only_frozen_gt_unchanged",
        "frozen_baseline": "watershed_v3_pre_report",
        "current_gt_method": "regionN.ply all-vertex XY projection -> 0.05m raster closing/fill -> external contour -> 0.10m RDP",
        "candidate_reference": "MP3D .house S(type=F) floor-surface V vertices",
        "summary": totals,
        "per_scene": rows,
        "interpretation": (
            "Most visible overlap is introduced by the current derived-GT polygonization, "
            "not by the original .house floor polygons. Do not overwrite the frozen baseline; "
            "build and validate a separately named house-floor GT candidate."
        ),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ordered = sorted(rows, key=lambda r: r["current_polygon_overlap_ratio"])
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.2, 1]})
    current_pct = [100 * r["current_polygon_overlap_ratio"] for r in ordered]
    house_pct = [100 * r["house_floor_overlap_ratio"] for r in ordered]
    axes[0].barh(y + 0.18, current_pct, height=0.34, color="#c46b3c", label="目前 frozen GT polygon")
    axes[0].barh(y - 0.18, house_pct, height=0.34, color="#3f7a6f", label="直接 .house floor polygon")
    axes[0].set_yticks(y, [r["scene"] for r in ordered], fontsize=8)
    axes[0].set_xlabel("房間重疊 excess / union (%)")
    axes[0].set_title("16 scenes：目前 GT overlap 明顯高於原始 floor polygons")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(loc="lower right")

    ious = [r["current_vs_house_mean_iou"] for r in ordered]
    axes[1].barh(y, ious, color="#668db3")
    axes[1].axvline(totals["current_vs_house_mean_iou"], color="#a4392f", ls="--", lw=1.5,
                    label=f"全體 mean = {totals['current_vs_house_mean_iou']:.3f}")
    axes[1].set_yticks(y, [r["scene"] for r in ordered], fontsize=8)
    axes[1].set_xlim(0.75, 1.0)
    axes[1].set_xlabel("目前 GT polygon vs .house floor polygon mean IoU")
    axes[1].set_title("同一 room ID 的兩種 GT 表示並不完全相同")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(loc="lower right")

    fig.suptitle("GT geometry quality audit（diagnostic only；未改 frozen baseline）", fontsize=15)
    fig.text(0.5, 0.025,
             f"總體 overlap：目前 polygon {100*totals['current_polygon_overlap_ratio']:.2f}% · "
             f"原始 .house floor {100*totals['house_floor_overlap_ratio']:.2f}% · "
             f"raw raster {100*totals['raw_raster_overlap_ratio']:.2f}%｜"
             "結論：overlap 主要來自 XY raster closing/fill 與 contour 衍生流程",
             ha="center", fontsize=10, color="#4c5661")
    fig.tight_layout(rect=[0, 0.055, 1, 0.95])
    fig.savefig(FIG, dpi=120, facecolor="white")
    plt.close(fig)

    md = f"""# GT Floorplan Overlap Audit

> 這是 legacy raster GT 的診斷紀錄；原 frozen baseline 至今未被覆寫。
>
> **2026-08-11 後續已完成：**另建並驗證 `mp3d_house_floor_v0_1`，再以相同 frozen predictions
> 做嚴格 evaluator v3 A/B。目前建議報告 official `.house` floor GT 結果：Room F1 **0.602**、
> matched-room conditional mean IoU **0.785**。詳見
> `outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1/eval_frozen_watershed_v3/RESULTS.md`。

## GT 怎麼來的

目前 `watershed_v3_pre_report` 的 2D room GT 使用兩種資料來源：

- room identity、level、room type：MP3D `.house` 的 `R` records；
- room geometry：每個 MP3D `region_segmentations/regionN.ply` 的**全部 3D mesh vertices**投影到 XY，
  以 0.05 m raster 做 binary closing、fill holes，再取最大 external contour，最後用 0.10 m RDP；
- doors：HouseLayout3D 手工 `doors/{{scene}}.json`；
- connectivity：以 room polygons＋door segments 做 PIP probes 推導，並非 annotated portals。

## 為什麼會 overlap

投影 `regionN.ply` 的所有 vertices 會把 floor、ceiling、walls 的 XY 支撐一起壓到平面；後續 raster
closing/fill 又會把邊界擴成有寬度的 occupied bands。相鄰 rooms 因此可能同時包含牆附近的 cells。
這不是理想的互斥 CAD tiling。

## 16-scene 實測

| 表示 | overlap excess | union area | overlap ratio |
|---|---:|---:|---:|
| raw 5 cm raster masks | {totals['raw_raster_overlap_excess_m2']:.2f} m² | {totals['raw_raster_union_m2']:.1f} m² | **{100*totals['raw_raster_overlap_ratio']:.2f}%** |
| 目前 frozen GT polygons | {totals['current_polygon_overlap_excess_m2']:.2f} m² | {totals['current_polygon_union_m2']:.1f} m² | **{100*totals['current_polygon_overlap_ratio']:.2f}%** |
| 直接 MP3D `.house` floor polygons | {totals['house_floor_overlap_excess_m2']:.2f} m² | {totals['house_floor_union_m2']:.1f} m² | **{100*totals['house_floor_overlap_ratio']:.2f}%** |

目前 frozen GT 與 `.house` floor polygon 對同一批 {totals['n_rooms_compared']} rooms 的 mean IoU 為
**{totals['current_vs_house_mean_iou']:.3f}**。因此大部分可見 overlap 是目前 GT 衍生方式引入，而不是
原始 floor polygons 本來就大量互相覆蓋。

## 對 legacy v2 報告的影響

- Room F1 0.613／matched-room IoU 0.793 仍是 frozen baseline 下可重現的數字，但 GT geometry
  應繼續標 **PRELIMINARY / raster-derived**。
- Corner/Angle 最容易受此問題影響；Room IoU 與 PIP connectivity 也可能受邊界 bands 影響。
- 不應今晚直接替換 GT，因為那會改變 evaluation definition 與全部數字。

## 當時規劃的下一步（現已完成）

建立獨立命名的 `mp3d_house_floor_gt_candidate`：直接讀 `.house` 的 `S(type=F)`＋`V` vertices，
驗證 polygon validity、room count、level assignment、door alignment 與 connectivity，再用同一 frozen
predictions 做 current-GT vs candidate-GT A/B。此工作已於 2026-08-11 完成；candidate 與 legacy baseline
仍各自保留，未互相覆寫。

視覺化：`outputs/eval2d/baselines/watershed_v3_pre_report/eval2d_v2_hungarian/gt_overlap_audit.png`
"""
    with open(MD_OUT, "w") as f:
        f.write(md)

    print(f"GT overlap audit -> {FIG}")
    print(f"current polygon overlap {100*totals['current_polygon_overlap_ratio']:.3f}% | "
          f"house floor {100*totals['house_floor_overlap_ratio']:.3f}% | "
          f"mean representation IoU {totals['current_vs_house_mean_iou']:.3f}")


if __name__ == "__main__":
    main()
