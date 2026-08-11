# HouseLayout3D 論文一致性稽核

> 稽核日期：2026-08-10  
> 對象：目前 `HouseLayout3D` workspace 內的 pipeline、evaluator、文件與實驗紀錄  
> 用途：提供人類協作者與 Claude Code 一份穩定的 paper/code alignment 基準  
> 性質：read-only audit；本文件記錄已觀察到的狀態，不代表相關偏離已修正

## 手機版摘要

目前程式不是完全忠實的 HouseLayout3D reproduction，而是「作者 Stage 2/3 程式＋本地補完與改良 Stage 4」的 reimplementation。  
Table 2 的 MP3D 輸入路徑正確；Stage 2 核心與 Stage 3 optimizer 最接近作者實作。  
最大偏離集中在 Stage 4：房間分割、extrusion、doors、stairs、outdoor pruning 與 scene graph 內容。  
Evaluator 另有一個高優先 correctness bug：34 個 stairs GT 中有 5 個非四邊形被靜默略過。  
在修正 evaluator 與 Stage 4 前，結果應稱為 reimplementation result，不宜宣稱 strict paper reproduction。

---

## 1. 結論與正確定位

### 1.1 核心結論

目前 code 與論文之間存在實質偏離，而且不只是數值尚未追上論文：

- 有些是論文未公開實作細節，本地必須自行補完。
- 有些是作者公開 code 與論文文字本身不一致。
- 有些是為了穩定性或品質加入的本地 heuristic。
- 有些則直接改變論文所描述的演算法、輸出或 evaluator 定義。

因此，現階段最準確的專案定位是：

> 以 HouseLayout3D 為核心、整合作者 Stage 2/3 code，並自行重建 Stage 4 的 research reimplementation。

在目前狀態下，不應使用下列說法：

- 「完全按照論文實作」
- 「與作者 evaluator 完全相同」
- 「已完整重現 Table 2」
- 「stairs 已超越論文」

### 1.2 Claude Code 後續工作的判定規則

處理任何 paper alignment 工作時，必須把變更分成以下四類，不得混寫：

1. **Paper-faithful**：論文或 Appendix 有明確規格，實作遵守該規格。
2. **Official-code-faithful**：遵守作者公開 code，但可能和論文文字不同。
3. **Local engineering variant**：為穩定性、效果或缺失補完而加入的本地方法。
4. **Unverified assumption**：論文與作者 code 都沒有足夠資訊，暫用工作假設。

若要改動第 1 類方法成第 3 類，應先明確告知使用者，不得以「修 bug」名義靜默改變研究方法。

---

## 2. 稽核資料與證據層級

本次交叉檢查的資料包括：

- `docs/arxiv_2512.02450_full.pdf`：論文全文。
- `docs/supplementary/supplementary/Appendix_A.pdf`：方法與實驗 Appendix。
- 作者 supplementary 的 `extract_skeleton.py`、`fit_prototype.py`、`mesh_fitting_3D/`。
- 作者 supplementary 中 Stage 4 與 OneFormer 相關目錄。
- 本地 `src/stage1/`～`src/stage4/`。
- `src/eval/` 與 calibration/aggregation scripts。
- 所有主要 pipeline、batch、retry scripts。
- `PROGRESS.md`、scorecard、parameter inventory、debug 與 audit 文件。
- 外部 HOV-SG 的房間分割來源。
- 現存的 MP3D 16-scene 實驗紀錄與 aggregate results。

未逐一審讀模型權重、每張 RGB/depth image 或全部第三方 vendor code；這些大型二進位／資料資產不影響本文對 pipeline method alignment 的判定。

### 2.1 發生衝突時的使用方式

不同來源回答不同問題，不應武斷地用一個來源蓋過另一個：

- 判斷「論文聲稱的方法」：以論文與 Appendix 為準。
- 判斷「作者釋出的實作」：以 supplementary code 為準。
- 判斷「目前實際會執行什麼」：以本地 code 與 runner 為準。
- 判斷「現在實驗跑到哪裡」：以 `PROGRESS.md` 最新日期段落與輸出 artifact 交叉驗證。

論文與作者 code 衝突時，報告中必須明寫衝突，不得直接宣稱其中一方就是唯一 ground truth。

---

## 3. 一致性總覽

| 區塊 | 一致性 | 判定 |
|---|---:|---|
| Table 2 MP3D input path | 高 | 正確從 MP3D poses、RGB-D、mesh 開始，略過 Stage 1 |
| Stage 1，一般場景重建 | 低～中 | smoke test 可跑，但未證明 Metric3D depth 已接入 DN-Splatter 正式流程 |
| Stage 2 semantic skeleton | 中～高 | 核心接近作者 code；paper/code 有 sampling 與 KNN 方向差異 |
| Stage 3 polygon optimization | 高 | 直接使用作者 `fit_prototype.py` 與 Matterport config |
| Stage 3 polygon initialization | 低～中 | 作者入口缺失，本地以大量 heuristic 近似重建 |
| Stage 4 level/room/graph | 低～中 | 作者檔案為空，本地重建；數個步驟直接不同於 Appendix |
| Doors | 中 | 有 adjacency recall 上限，door width rule 已改變 |
| Windows | 中 | 高階流程接近，但漏掉 Appendix 指定的 `outdoor` rays |
| Stairs | 低 | 偵測部分存在，論文要求的 subtraction/extrusion 未完整實作 |
| F1/depth evaluation | 中～高 | dE/dH/depth 主體可信；stairs loader 與完整 Table 2 尚有問題 |
| Scene graph schema | 低 | 目前是摘要 JSON，不是論文描述的完整 room/entity graph |
| 文件一致性 | 中～低 | 多份文件已有 stale 或被後續實驗推翻的敘述 |

---

## 4. Stage 1：benchmark 路徑正確，完整方法尚未重現

### 4.1 論文規格

完整 Stage 1 為：

1. unposed RGB camera localization；
2. Metric3D monocular depth；
3. DN-Splatter 產生 triangle mesh/depth。

但論文的 HouseLayout3D Table 2 benchmark 直接使用 MP3D camera poses、RGB-D 與 mesh，所以該實驗本來就略過 DN-Splatter Stage 1。

### 4.2 目前狀態

- Table 2 runner 從 MP3D assets 開始：與論文 benchmark protocol 一致。
- coffee-room Stage 1 smoke 使用 `depth_mode=sensor`。
- 實際 config 中 `load_normals=false`，normal losses 關閉。
- Metric3D 曾獨立 smoke-test，但沒有足夠證據顯示 Metric3D output 已接到目前 DN-Splatter training path。
- Table 3 的 50 個 ScanNet++ scenes 未正式重現。

### 4.3 判定

- 對 Table 2：此處不是主要偏離來源。
- 對完整 unposed RGB pipeline：只能稱為 smoke-tested partial implementation。

---

## 5. Stage 2：接近作者 code，但作者 code 與論文文字有衝突

### 5.1 已對齊的核心

- OneFormer semantic segmentation。
- 語意映射至四個 coarse categories。
- RGB-D backprojection。
- mesh semantic propagation。
- superpoint majority labeling 與 skeleton extraction。

### 5.2 Paper/code conflict：每張圖 sampling 數

- 論文：每張影像隨機抽 `M=5000` pixels。
- 作者 `extract_skeleton.py`：預設 3000。
- 本地 patched version：跟隨作者 code，使用 3000。

因此目前是 **official-code-faithful，但不是 paper-text-faithful**。

### 5.3 Paper/code conflict：mesh semantic propagation 方向

- 論文文字：每個 backprojected point 投票給 nearest mesh vertex。
- 作者與本地 code：對每個 mesh vertex 找 `k=5` 鄰近 backprojected points 並平均。

此差異應保留在報告中，不應被描述成純 adapter 細節。

### 5.4 無法驗證的 SPT preprocessing

作者未提供完整 SPT preprocessing 入口。本地使用 upstream/ScanNet 類似設定，例如：

- voxel size 0.02 m；
- KNN 45；
- 本地 cut-pursuit 參數。

這些屬於 **unverified assumptions**，不是已驗證的作者參數。

### 5.5 隨機性

目前 sampling 沒有 pipeline-wide 固定 seed。既有記錄顯示相同 config 可有約 ±6 個百分點 Δ5 變異，因此：

- 不應用單次、單棟結果宣稱小幅修改有效。
- A/B 至少需要固定 seed，或以多次／多場景統計支持。

---

## 6. Stage 3：作者 optimizer 忠實，initializer 與 adapter 是主要缺口

### 6.1 高一致性部分

目前 runner 直接呼叫作者 `fit_prototype.py --scene-type matterport`。Loss、optimization schedule、merge 邏輯與 Matterport config 都來自作者 code，是專案中最可信的官方實作核心之一。

### 6.2 Polygon initializer 是本地重建

作者沒有釋出完整 initializer，因此 `src/stage3/init_polygons.py` 使用本地方法，包含：

- adaptive `K=max(150, n_vertices//2000)`；
- RANSAC threshold 0.02 m；
- 200 iterations；
- 每次最多 20,000 points；
- minimum component 30 vertices；
- minimum hole area 0.02 m²；
- pipeline 最多 600 polygons；
- fit failure 後解除候選歸屬等處理。

論文沒有給出這些參數，且過去固定 `K=2000` 曾造成嚴重 skeleton coverage 問題，因此 initializer 不是中性的 I/O adapter，而是結果高度敏感的 reimplementation component。

### 6.3 Stage 3 → Stage 4 identity/class adapter

作者 optimizer state 包含 polygon identity/class，但目前 Stage 4 沒有直接延續該 state；本地會從 fitted mesh vertex color 重建 polygon、重新 fit plane，並以幾何與 Stage 2 semantic 重新判類。

可能影響：

- optimizer merge 後的 polygon identity；
- class assignment；
- Stage 4 看到的 floor/wall/ceiling/stair 候選集合。

### 6.4 文件中的 MatterportConfig 數值錯誤

`docs/param_inventory_v2.md` 記載：

- `max_plane_merge_dist=0.2`
- `projection_baseline_error=0.1`

作者實際 `MatterportConfig` 是：

- `max_plane_merge_dist=0.06`
- `projection_baseline_error=0.03`

目前 runtime 使用作者 config，故程式是對的；錯誤位於文件及部分早期 debug 敘事。

---

## 7. Stage 4：最主要的結構性偏離

作者 supplementary 中 `create_scene_graph.py`、`floor_extrusion.py`、`free_space_test.py` 均為 0-byte 空檔。因此 Stage 4 本來就只能依 Appendix 重建，但目前有多項實作已不同於 Appendix。

### 7.1 D.1 Level identification

論文：floor polygons 為 nodes，高度差不超過 0.5 m 即連 edge，connected components 形成 levels。

本地另加入：

- minimum floor area 5 m²；
- 小 floor 先排除，再掛回最近 level；
- `max_level_span=0.8m` 防止 transitive chain collapse；
- 無 floor 時把最低 horizontal planes fallback 成 floor。

這些是有理由的 safeguards，但應標記為 local engineering variants。

### 7.2 D.2 Ceiling 與 wall assignment

論文把 ceiling 指派給 ceiling center 下方至少 1 m、空間上最接近的 next-lower floor polygon。

本地主要依 elevation 選較低 level，沒有完整實作 ceiling-center-to-floor-polygon 的空間最近關係。

Wall assignment 額外加入：

- coplanar grouping；
- top margin 0.5 m；
- vertical span 至少層高 35%；
- 15°、15 cm、6 m 等 grouping threshold；
- 5 cm buffer。

這些 threshold 並非論文明訂，部分曾依有限場景調整，需防止 dataset-specific overfitting。

### 7.3 D.3 Room segmentation：直接不同於論文

論文要求套用 HOV-SG morphology segmentation 兩次：

1. bottleneck width 2.5 m；
2. bottleneck width 1.5 m。

目前 code 雖保留 `BOTTLENECK_1=2.5`、`BOTTLENECK_2=1.5` 常數，但目前實際執行的是單次 wall-barrier watershed：

- `distance-to-wall > ROOM_CORE_MIN=0.5m` 形成 room cores；
- 在 wall image 上做 watershed；
- 丟棄小於 1 m² 的 rooms；
- nearest assignment 補洞。

這個方法對本地結果有明顯改善，但它是 **local method variant**，不是論文的 two-stage 2.5 m/1.5 m procedure。

### 7.4 Door rule 改變

論文：bottleneck width `<1.5m` 為 door，否則為 opening。

本地：

- shared room boundary 以 PCA 擬合；
- door width 必須位於 `[0.55m, 1.30m]`。

此 rule 已造成 17DRP、WYY 的 doors 被濾除。此外，adjacency-only detection 先天無法完整處理：

- 同一 predicted room 內的 GT door；
- room segmentation 未切開的 door；
- 部分對戶外 door。

既有 GT-region ablation 顯示，292 個 GT doors 中只有約 58% 位於 GT-region boundary，故目前方法存在結構性 recall ceiling。

### 7.5 Room extrusion 是 simplified implementation

論文 constrained Delaunay triangulation 應包含：

- room boundary；
- ceiling candidate edges；
- ceiling planes 兩兩交線的投影。

接著應對 triangle center 向上 ray casting；未命中者透過相鄰 triangle graph 取得 lowest reachable ceiling。

目前本地：

- CDT 主要只有 simplified raster room contour；
- 沒加入 ceiling edges；
- 沒加入 ceiling-plane intersection lines；
- ceiling assignment 主要使用 BEV `contains(center)`；
- fallback 使用 largest-area ceiling；
- 沒有實作 lowest-reachable-ceiling graph propagation。

此外，contour handling 主要取最大 contour，interior holes 可能被填掉。

### 7.6 Door geometry 不同

論文 doorframe 使用固定 2.10 m 高度的四個 rectangular faces。本地 combined mesh 中 door 主要是單一 rectangular sheet（兩個 triangles），不是相同幾何表示。

### 7.7 Stairs handling 未完成

論文要求：

- stair mesh connected components；
- horizontal oriented bounding rectangle；
- 短邊中點高度插值；
- 對應相鄰 rooms/levels；
- 從 room 2D geometry 扣除 stairs；
- extrusion stair volume；
- 四角高度對齊 levels；
- room/stair 共邊不建牆；
- optional steps。

目前本地較接近完成 component/rectangle detection 與 graph edge 推導，但沒有完整完成 room subtraction、stair volume extrusion、四角高度調整、共邊牆處理與 steps。

### 7.8 D.4 OpenSeg/CLIP 與 outdoor pruning

論文：OpenSeg features 應像 Stage 2 一樣投影至 mesh vertices，再對每個 room 平均 mesh vertex features，最後以 CLIP 分 15 類並剪掉 outdoor leaf rooms。

本地：

- 每 8 幀抽一幀；
- 每幀最多 2,000 pixels；
- backproject 為 3D samples；
- 依 BEV room 平均 samples，而不是先形成 per-mesh-vertex features。

更重要的是執行順序：

1. `scene_graph.py` 先輸出 `combined.ply` 與 entities；
2. `room_classify.py` 才修改 `scene_graph.json`；
3. pruning 沒有回頭刪除 combined geometry 或 structure entities。

因此目前 outdoor pruning 只改 JSON，不會影響 Table 2 F1 或 depth；這不同於論文 Table 4 中 pruning 對 layout geometry 的作用。

### 7.9 Windows 漏掉 `outdoor`

Appendix B 指定 window detection 使用：

- `window`
- `window_blind`
- `curtain`
- outdoor classes

本地 `WINDOW_RAY_CLASSES` 只有前三者，漏掉 `outdoor`。

其餘 ray-to-layout-wall、outlier filtering、per-wall split、DBSCAN、rectangle fitting 與 minimum size 的高階流程接近論文，但 DBSCAN/LOF/ray slack 參數仍屬無法從作者 code 驗證的本地假設。

---

## 8. Scene graph schema 不完整

論文 scene graph 應包含：

- room nodes；
- door/stair edges；
- 每個 room 關聯的一個 floor；
- associated walls、ceilings、windows。

目前 JSON 的 room 主要只有：

- `id`
- `area_m2`
- `type`

Openings 記錄 rooms/width/door flag，但缺少完整 geometry association；windows 與 stairs 多為全域列表，也沒有完整掛回 room/entity hierarchy。

目前輸出應稱為 **layout summary graph**，而不是已完整實現論文 scene graph schema。

---

## 9. Evaluator 稽核

### 9.1 高優先 correctness bug：非矩形 stairs GT 被丟棄 — **FIXED AND TESTED（2026-08-11）**

> **狀態：FIXED AND TESTED（2026-08-11，Claude Code）。** 修正屬 **paper-faithful fix**（恢復論文對非矩形實體用 d_H 的規格）。
> 舊 evaluator 凍存 `src/eval/eval_v1_buggy.py`；新版 `src/eval/eval_scene.py`(v2_nonrect) 載入全 34 GT、
> d_E/d_H 混用單一 Hungarian（`metrics.pairwise_stairs`）。regression `src/eval/compare_stairs_eval.py`
> 斷言 n_gt=34、非矩形場景無漏，✅ 通過。
> **同一組凍結 predictions 重算：Stairs F1@0.5 0.473 → 0.411（論文 0.42；改為略低於、非達標）。**
> 逐場景老/新見 compare 腳本輸出；最誇張 r47D 舊版假 1.000（GT 被丟→both-empty=perfect）實為 miss 0.0。
> evaluator comparability 影響：stairs 現與論文可比（全 34 GT）；depth/dE/dH 主體不變。

（以下為原始 audit 記錄，保留供追溯）



目前 stairs GT 共 34 個：

- 29 個是四頂點 rectangle；
- 5 個分別有 7、8、8、12、14 個頂點。

`src/eval/eval_scene.py` 會略過非四頂點 stairs。論文對非矩形 entities 應使用 generalized Hausdorff distance `d_H`，而不是忽略。

直接後果：

- 目前 stairs F1 只評到 29/34 GT；
- 報告的 0.473 不能直接與論文 0.42 比較；
- 在修正前不得宣稱 stairs 已達標或超越論文。

### 9.2 Avg F1 thresholds 尚未完全驗證

目前使用 `0.05, 0.10, ..., 1.00` 作為 Avg F1 thresholds。論文未明確列出這組 thresholds。

以作者 predictions calibration：

- doors Avg 約 0.462，論文 0.44；
- windows Avg 約 0.417，論文 0.38。

兩者接近但不相同，故這組 thresholds 應標為 working assumption。

### 9.3 Table 2 尚未完整重現

目前 aggregate 主報表涵蓋：

- F1@0.5；
- Δ5；
- Δ10。

尚未完整提供：

- Avg F1；
- 論文定義的 `#Vertices`。

目前 triangulated `combined.ply` vertex count 也不能直接當成論文的 layout boundary vertex count。

### 9.4 Depth evaluator 相對可信

作者 prediction calibration 約得到：

- Δ5：61.9，論文 61.1；
- Δ10：75.8，論文 76.3。

差異很小，顯示 depth protocol 大致可信。但 identity tests 只能證明內部一致，不能單獨證明與作者私有 evaluator 完全相同。

### 9.5 Structure entity granularity 可能不同

部分輸出中，本地 predicted structure entities 數量明顯高於 GT，例如曾觀察到約 2 倍的差距。這顯示本地 extrusion/export 的 entity segmentation granularity 可能與作者 evaluator 預期不同，會直接影響 Hungarian matching 與 entity F1。

---

## 10. 目前 16-scene 結果應如何解讀

最新記錄的 watershed v3 aggregate：

| 指標 | v2 | v3 | 論文 |
|---|---:|---:|---:|
| Structures F1@0.5 | 0.162 | 0.217 | 0.40 |
| Doors F1@0.5 | 0.023 | 0.210 | 0.55 |
| Windows F1@0.5 | 0.284 | 0.271 | 0.43 |
| Stairs F1@0.5 | 0.411 | 0.411 | 0.42 |
| Δ5 | 40.0 | 45.2 | 61.1 |
| Δ10 | 51.5 | 57.9 | 76.3 |

> **Stairs 0.411（舊 buggy：0.473，丟 5 個非矩形 GT）**：evaluator 修正後（見 §9.1 FIXED AND TESTED）
> 同一組 predictions 重算為 0.411，略低於論文 0.42、**非達標**；舊 0.473 已作廢，勿再引用。
> 此表其餘數字未受該 evaluator bug 影響。

可支持的結論：

- watershed variant 對 structures 有 14/16 scenes 改善；
- doors 有 13/16 scenes 改善；
- 新機制具有跨場景的普遍正向訊號。

不能支持的結論：

- 「只差一點 tuning 就完全重現論文」；
- 「目前 Stage 4 與論文相同」；
- 「stairs 已超過論文」。

其中 stairs 受 GT 漏評影響；其他指標則同時受到 Stage 3 initialization、Stage 4 method differences、entity granularity 與本地 thresholds 影響。

---

## 11. 文件漂移與目前閱讀規則

### 11.1 已知 stale/inaccurate 內容

- `PROGRESS.md` header 的 last-updated 日期早於其最新紀錄。
- `docs/reproduction_scorecard.md` 仍有「等待 MP3D」與過度樂觀的完成度敘述，已被後續正式 run 推翻。
- `PROGRESS.md` 稱目前方法為 HOV-SG two-stage segmentation，但 runtime 是單次 wall-barrier watershed。
- `docs/param_inventory_v2.md` 的 Matterport merge config 數值錯誤。
- 部分文件說 OpenSeg outdoor pruning path 已完成，但目前只更新 JSON，沒有更新 evaluated geometry。
- `docs/debug_mp3d_quality.md` 是研究過程紀錄，含後來被推翻的中間 root-cause；不可把單一舊段落當最終結論。

### 11.2 後續閱讀優先序

1. 論文／Appendix：paper specification。
2. 作者 supplementary code：official implementation behavior。
3. 本文件：截至 2026-08-10 的 alignment 結論與已知差距。
4. 本地 source：目前 runtime truth。
5. `PROGRESS.md` 最新日期段落：實驗進度與歷史。
6. 其他 debug/scorecard 文件：背景資料，不是獨立真相來源。

若本文件與之後修改過的 code 不一致，應更新本文件並附：

- 修改日期；
- 對應 commit/diff（若可用）；
- 變更屬於 paper-faithful fix 或 local variant；
- 對 evaluator comparability 的影響。

---

## 12. 建議修正優先序

### P0：先恢復結果可解釋性

1. 修正 stairs evaluator，讓全部 34 個 GT 都進入 matching。
2. 補 regression test，至少包含非四邊形 stair polygon。
3. 正式報表標記 method variant、evaluator version、seed 與 artifact path。
4. 不再用目前 stairs 0.473 宣稱已達 paper target。

### P1：建立 paper-faithful Stage 4 基線

1. 保留目前 watershed 為明確命名的 local variant。
2. 另外實作 Appendix 的 2.5 m → 1.5 m two-stage room segmentation。
3. 以相同 Stage 3 input、固定 seed、公平 evaluator 做兩條路徑 A/B。
4. 將 door paper rule `<1.5m` 與本地 `[0.55m,1.30m]` 分開，不要共用無標記的輸出。

### P2：補完整 geometry pipeline

1. CDT constraints 加入 ceiling edges 與 pairwise ceiling-plane intersection projections。
2. 實作 upward ray assignment 與 lowest-reachable-ceiling propagation。
3. doorframe 改為論文的四面表示。
4. 完成 stair room subtraction、volume extrusion、level corner adjustment 與 shared-boundary wall handling。
5. 決定並測試 room contour holes 的正確處理。

### P3：修正 semantic graph 路徑

1. windows 加入 outdoor rays。
2. OpenSeg features 依論文形成 per-mesh-vertex features。
3. 在 geometry/entities 產生前完成 outdoor leaf pruning，或 pruning 後重建全部輸出。
4. scene graph 補齊 room → floor/walls/ceilings/windows association 與 door/stair edges。

### P4：收斂不可驗證部分與完整報表

1. 確認 exact OneFormer checkpoint/inference resolution。
2. 追查或系統性驗證 SPT preprocessing parameters。
3. 將 Stage 3 initializer 所有 heuristic 集中配置並做 sensitivity study。
4. 補 Avg F1 與 paper-compatible `#Vertices`。
5. 若要重現完整論文，再執行 ScanNet++ Table 3、ablations 與 baseline adaptations。

---

## 13. 對外與內部建議用語

### 可使用

> 我們整合了 HouseLayout3D 作者公開的 Stage 2/3 程式，並依 Appendix 自行重建 Stage 4。目前結果驗證了若干方法機制，但 Stage 4 與 evaluator 仍有已知差距，因此屬 research reimplementation，而不是完全同協定的 strict reproduction。

### 暫不應使用

> 我們已完整重現 HouseLayout3D。

> 我們的 stairs 已超越論文。

> 目前所有 Stage 4 步驟都與 Appendix 相同。

---

## 14. 快速索引

- 論文：`docs/arxiv_2512.02450_full.pdf`
- Appendix：`docs/supplementary/supplementary/Appendix_A.pdf`
- 作者 Stage 2/3：`docs/supplementary/supplementary/multi-floor-3d-code/`
- 本地 Stage 3 initializer：`src/stage3/init_polygons.py`
- 本地 Stage 4：`src/stage4/scene_graph.py`
- Windows：`src/stage4/windows.py`
- Room classification/pruning：`src/stage4/room_classify.py`
- Evaluator：`src/eval/eval_scene.py`
- Aggregate report：`src/eval/aggregate_results.py`
- 最新實驗歷史：`PROGRESS.md`
- Annotated 2D floorplan 執行計畫：`docs/annotated_floorplan_roadmap.md`

---

## 15. 維護要求

Claude Code 或其他協作者若修正本文列出的項目，應同步更新本文件；至少將該項標示為：

- `OPEN`
- `PARTIALLY FIXED`
- `FIXED AND TESTED`
- `INTENTIONAL VARIANT`

只有在有測試、artifact 或 calibration evidence 時，才可使用 `FIXED AND TESTED`。
