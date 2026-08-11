# MULTIFLOOR3D 參數清單 v2

> 取代 v1。來源升級：正文＋**Appendix_A（p.21–25）＋官方 supplementary code**（`polygon_fitting_config.py` 等）。
> 標記：✅=官方值（config/appendix/code）｜🔶=官方材料缺、需自定（附建議）｜🔧=第三方預設起步
> 原則不變：全部集中進 config；🔶/🔧 標 `# TODO: tune with MP3D`。

## Stage 2 — Skeleton Extraction（extract_skeleton.py 為準）

| 參數 | 狀態 | 值 |
|---|---|---|
| 每幀取樣 rays | ✅ code 預設 **3000**（論文寫 M=5000，不一致） | 預設 3000，config 可切 5000 |
| 標籤→mesh 頂點 | ✅ | per-vertex **KNN k=5** 對反投影點雲取 one-hot 平均（非論文寫的 nearest-vertex 投票） |
| MP3D 深度尺度 | ✅ | PNG × **0.00025** m；c2w `[0:3,1:3] *= -1` |
| COCO→四類映射 | ✅ Appendix Table 5 | Wall: wall-brick/stone/tile/wood/other-merged；Ceiling: ceiling-merged；Floor: floor-wood/floor-other-merged/**rug-merged**；Surfaces: cabinet-merged/**door-stuff**/curtain/window-blind；Windows: window-other；Mirrors: mirror-stuff；Outdoor: gravel/tree-merged/sky-other-merged/pavement-merged/grass-merged/dirt-merged；Stairs: stairs；Objects: 其餘 |
| skeleton 保留類 | ✅ | ceiling,wall,floor,surface,door,stairs（window 另存；object=其餘） |
| superpoint 層級 | ✅ 介面 | level 1–3，各存 per-segment hard/probabilities；`segment_point_cloud_superpoints` **本體缺→自寫（upstream SPT）** 🔶 |
| OneFormer checkpoint | 🔶 | 官方未載明；建議 HF `shi-labs/oneformer_coco_swin_large`（COCO panoptic 確定，backbone 未知）；semantic map 輸出 uint8 PNG＋labels.txt 自產；之後對照 predictions-ours 驗證 |

## Stage 3 — Prototype Fitting（MatterportConfig，全 ✅）

| 參數 | 值 |
|---|---|
| iterations / warmup | 4000 / 50 |
| lr_start / lr_normal / plane_lr_downscale | 1e-3 / 5e-4 / 50 |
| τ_merge（vertex_merge_thresh，兼 RDP tol） | **0.02** |
| τ_inter（ray_intersection_margin） | **0.1** |
| L_connect 忽略門檻（magnetism_threshold） | **0.1** |
| max_dist | 0.2 |
| L_empty 實作 | ray-tracing（chamfer/point-triangle 關閉）；max_ray_intersects_per_m2 = **50000** |
| 簡化排程 simplification_steps | [50,100,150,225,300,…,6000]（31 個節點） |
| 平面合併排程 plane_merge_steps | [90,200,275,…,5000]（26 節點）；max_plane_merge_dist **0.2**、max_plane_merge_angle **30°**、projection_baseline_error 0.1、normal_baseline_error 0.3 |
| 物件投影補地板 | step **600** 執行 |
| 牆延伸 | do_extend_walls_to_floor/ceiling=True；max_wall_angle_for_extension **5°** |
| 正則 | regularization_strength **1e-6**；edge/vertex/face magnetism on；non_watertight on |
| 其他 | up=Z、multi_floor=True、recompute_normals=True、recompute_interval 10、delete_polygons_not_aligned_with_semantics=**False**（MP3D） |
| 初始化 | Algorithm 1：循序 RANSAC（選未指派頂點最多的 cluster→擬平面→全域 inlier→取與該 cluster 重疊最大的連通元件→邊界抽多邊形）；**K（最小未指派頂點數）在 fit_prototype.py 內找** 🔶 |
| 頂點共享 | 每頂點 ≤3 平面約束、存取時投影到交線/交點；近平行多邊形不合併頂點 |
| pytorch3d | 可選（try/except fallback；config 不用 chamfer）→ 先不裝 |

## Stage 4 — Scene Graph（**code 全缺**，依 Appendix D 規格自寫；規格 ✅、少數執行細節 🔶）

| 項目 | 規格 |
|---|---|
| 樓層識別（D.1） | floor polygon 建圖，高度差 ≤ **50cm** 加邊；連通元件＝樓層；取平均 elevation |
| 樓層 floorplan（D.2） | ceiling→「中心下方 ≥1m 的最近 next-lower floor」；floorplan＝該層 floors∪ceilings；walls＝BEV 交 floorplan 且垂直交 [0,2.5]m |
| 房間分割（D.3） | HOV-SG 形態學法**跑兩次**：bottleneck **2.5m** → **1.5m**；邊寬 <1.5m＝door，否則 opening |
| 房型＋剪枝（D.4） | OpenSeg 每頂點特徵→房平均→CLIP 15 類分類；**葉節點屬最後五類（室外類）者剪除**（Table 4 增益來源之一）；OpenSeg 依賴重 🔶（初版策略待定） |
| 樓梯（D.5） | stair mesh CC→水平投影 OBB→短邊中點高度插值→D_pp 指派房間；距離 >**50cm** 或兩端同房→拒絕 |
| 門（D.6） | HOV-SG 分割邊界→oriented 2D bbox→固定高 **2.10m** 門框（四矩形面）；房殼在門 bbox 上方才補牆 |
| 樓梯 extrude（D.6） | 2D 先從房間減去樓梯區→樓梯照房間 extrude→四角高度調到銜接樓層→樓梯與房間之間不加牆 |
| 窗偵測（正文 4.4） | window(+window-blind,curtain,−mirror) 像素射線交牆→LOF 濾群外→按牆分→DBSCAN→群 ≥**k=10** 擬合軸對齊矩形→高寬 >**30cm**；LOF/DBSCAN 超參 🔶（sklearn 預設起步） |
| 天花板候選上限 | 每房 **30**（正文） |

## 評估（腳本全缺，自寫；🔶 用 predictions-ours 校準）

- d_E（矩形類，Hungarian 角點）＋F1@0.5；d_H（Hausdorff，牆/地板/天花板）；Δ5/Δ10（cm）。
- Avg F1 門檻集合 🔶（SceneScript 慣例）→ **用 `predictions-ours/` 16 場景對 HF GT 重算，調到能復現 Table 2 的 0.40/0.38 等值**，即可反推門檻集合並驗證 eval 正確性（不需 MP3D）。
- Baseline 適配規格（Appendix E）已記錄：floor 高度 5% 分位、door 2.10m、window 80% 牆高置中、SceneScript 半稠密取樣 top-5% 梯度範數＋隨機點。

## 仍開放的問題（會在對應階段問）

1. OneFormer 確切 checkpoint／推論解析度（先假設 coco_swin_large，用官方預測反驗）。
2. superpoint 分割參數（voxel、regularization、3 levels 對應）——upstream SPT 預設起步。
3. Algorithm 1 的 K 與 RANSAC 內距門檻——待精讀 fit_prototype.py（1840 行）確認是否已含。
4. LOF/DBSCAN 窗偵測超參。
5. ~~Avg F1 門檻集合~~ → **工作假設已定（2026-07-07 校準）：τ∈{0.05,…,1.0} 步長 0.05 的平均**（doors/windows 對官方 Avg 最吻合；見 `eval_calibration_report.md`）。
