# GT Floorplan Overlap Audit

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
- doors：HouseLayout3D 手工 `doors/{scene}.json`；
- connectivity：以 room polygons＋door segments 做 PIP probes 推導，並非 annotated portals。

## 為什麼會 overlap

投影 `regionN.ply` 的所有 vertices 會把 floor、ceiling、walls 的 XY 支撐一起壓到平面；後續 raster
closing/fill 又會把邊界擴成有寬度的 occupied bands。相鄰 rooms 因此可能同時包含牆附近的 cells。
這不是理想的互斥 CAD tiling。

## 16-scene 實測

| 表示 | overlap excess | union area | overlap ratio |
|---|---:|---:|---:|
| raw 5 cm raster masks | 119.52 m² | 4978.3 m² | **2.40%** |
| 目前 frozen GT polygons | 70.42 m² | 4879.0 m² | **1.44%** |
| 直接 MP3D `.house` floor polygons | 6.27 m² | 4769.3 m² | **0.13%** |

目前 frozen GT 與 `.house` floor polygon 對同一批 324 rooms 的 mean IoU 為
**0.882**。因此大部分可見 overlap 是目前 GT 衍生方式引入，而不是
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
