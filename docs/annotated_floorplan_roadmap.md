# Annotated Floorplan 重現與開發 Roadmap

> 建立日期：2026-08-10  
> 專案目標：由 HouseLayout3D pipeline 產生可供後續研究使用的 **annotated 2D floorplan**  
> 方法原則：盡量遵守論文／Appendix，同時把本地改良明確隔離為可比較的 variant  
> 相關稽核：`docs/paper_alignment_audit.md`  
> 本文件狀態：執行規劃；尚未開始的項目不代表已承諾完成日期

## 手機版摘要

研究目標已收斂為 annotated 2D floorplan，因此不用完成論文昂貴的 3D CDT/extrusion、3D doorframe、stair volume 與 depth evaluation。  
必須保留並改善的是 Stage 2 semantic skeleton、Stage 3 polygon fitting、D.1 levels、D.2 footprint、D.3 rooms，以及 doors/windows/stairs/room types/topology。  
目前 2D baseline：Room F1 0.61、matched IoU 0.79，但 Room+type 0.15、Doors@0.5m 0.245、edge-all 0.12，精確 corners/angles 也偏低。  
建議先建立可重現的 canonical 2D artifact，再做 paper-spec room segmentation 與 annotation layers；Stage 3 initializer 放在便宜的 Stage 4 改良到頂後處理。  
沿用目前 Stage 2/3 的 annotated-floorplan MVP 約 3–5 週；連 Stage 2/3 都做高忠實對齊，實際約 6–10 週。

---

## 1. 研究目標與範圍

### 1.1 目標產物

每棟建築、每個 level 輸出一份 metric-aligned annotated floorplan，至少包含：

- level ID、elevation、coordinate frame 與單位；
- outer footprint、interior holes 與可能的 disconnected components；
- room polygons 與 stable room IDs；
- wall segments／boundaries；
- door/opening segments、width、相鄰 rooms、interior/exterior 狀態；
- window segments 與所屬 wall/room；
- stair footprints、up/down levels 與相鄰 rooms；
- room type、class scores/confidence；
- room adjacency/access graph；
- 每個 annotation 的 method provenance 與 confidence；
- pipeline/config/model/seed version，讓後續研究可以追溯。

建議主要 machine-readable 格式為 versioned JSON 或 GeoJSON；另輸出 SVG/PNG 作人工檢查。下游研究不得依賴 `scene_graph.py` 的暫存 Python objects 或由 3D mesh 重新切片。

### 1.2 兩個方法輸出，不混在一起

為同時兼顧論文重現與後續研究品質，保留兩條路徑：

1. **`paper_spec_2d`**
   - 依論文／Appendix 明文規格實作。
   - paper/code 衝突處明確記錄選擇。
   - 未公開參數標記為 assumption，不以 benchmark GT 靜默調分。

2. **`annotated_best_2d`**
   - 可使用 current watershed、direct door detection、adaptive thresholds 等本地改良。
   - 每項改良必須命名、做 A/B 並記錄，不可稱為 paper-faithful。

下游研究原則上使用品質較好的 `annotated_best_2d`，論文方法分析與可比性使用 `paper_spec_2d`。

### 1.3 不在主要範圍內

只為 3D layout 而存在的工作不列入 annotated-floorplan 主線：

- constrained Delaunay 3D room extrusion；
- ceiling triangle ray assignment 與 propagation；
- vertical ceiling-discontinuity rectangles；
- 2.10 m 高、四面組成的 3D doorframe；
- stair volume extrusion、steps 與 3D shared-wall handling；
- `combined.ply` 作為主要研究產物；
- Δ5/Δ10 depth evaluation；
- 論文 3D `#Vertices`；
- ScanNet++ Table 3、完整 3D ablations 與 baseline adaptations。

這些項目未來若研究問題需要，可以另立支線；目前不應阻擋 2D 產物。

---

## 2. Floorplan 在 pipeline 中的位置

```text
MP3D RGB-D / poses / metric mesh
        │
        ├─ Stage 1：MP3D benchmark 略過
        ▼
Stage 2：OneFormer → semantic skeleton / superpoints
        │
        ▼
Stage 3：polygon initialization → official polygon optimizer
        │
        ▼
D.1：level identification
        │
        ▼
D.2：floors ∪ suitable ceilings → base 2D footprint
        │
        ▼
D.3：room segmentation → room-aware floorplan
        │
        ├─ doors/openings
        ├─ windows
        ├─ stairs / inter-level links
        ├─ room type
        └─ access graph
        │
        ▼
Canonical annotated 2D artifact  ← 本研究的終點
        │
        └─ 3D extrusion（本研究主線略過）
```

Stage 2/3 雖然處理 3D 資料，仍是 floorplan 的必要上游；不能因最終只要 2D 就直接刪除。

---

## 3. 截至目前的基線與已完成事項

### 3.1 已完成

- `[FIXED AND TESTED]` 3D stairs evaluator 已載入全部 34 GT：舊 0.473 修正為 0.411。
- `[DONE, PRELIMINARY]` `src/eval2d/` 已可建立 GT/pred、計算 Tier A/B/C 並產生報告。
- `[DONE]` 預測 2D floorplan 從 Stage 4a、extrusion 前直接抽取，不從 3D mesh 切片。
- `[DONE]` 16 scenes 的 Stage 4 inputs 已保留，可 Stage-4-only 重跑。
- `[DONE]` current watershed v3 已有 16-scene baseline。

### 3.2 初版 2D baseline

| 指標 | 目前結果 | 判讀 |
|---|---:|---|
| Room F1 @ IoU>0.5 | 0.61 | 多數房間可對上，但漏房／錯切仍明顯 |
| Mean Room IoU（matched） | 0.79 | 粗粒度 room extent 尚可 |
| Corner F1 @0.1m | 約 0.18 | 不足以支撐精確邊界屬性 |
| Angle F1 @0.1m | 約 0.09 | 邊界角度品質不足 |
| Room+type F1 | 0.15 | 目前 CLIP room typing 不可作可靠 annotation |
| Doors F1 @0.5m | 0.245 | 門的 recall/定位仍不足 |
| Access edge-all F1 | 0.12 | 尚不足以支撐可靠導航 graph |

此 baseline 表示：目前 floorplan 可支撐部分粗粒度 room-level attributes，但還不能當成精確、語意完整、導航可靠的 annotated floorplan。

### 3.3 目前 2D evaluator 仍屬 preliminary

已知待硬化項目：

- connectivity GT 是從人工作圖＋door geometry **推導**，不是 MP3D portal annotation；16 棟 portal count 實為 0。
- on-GT 內門推導準確率約 78.9%，外門約 25%，Tier C 尚不能視為高可信 ground truth。
- room matching 需確認／改為 deterministic Hungarian，而不是受排序影響的 greedy matching。
- GT/pred polygonization 都可能受 raster resolution 與 RDP tolerance 影響。
- current prediction extractor 會重新呼叫 `identify_levels/segment_rooms`，沒有讀取凍結的 canonical artifact；code 一改，歷史 prediction 可能跟著變。
- windows、inter-level stairs、完整 Room++ 尚未形成等同 Tier A/B 的可靠評估閉環。
- room-type ontology 與 MP3D labels 的 crosswalk 尚未充分定義。

---

## 4. 實驗與資料治理原則

### 4.1 先凍結 dev/held-out split

16 scenes 已被多次觀察，無法聲稱完全 untouched test set；但從現在起仍應降低進一步 overfitting：

- 依 building size、level count、room count 分層建立固定 dev/held-out split。
- 所有 threshold/heuristic 只在 dev scenes 調整。
- held-out 只在 phase checkpoint 執行，不做逐次 debug。
- paper-spec 路徑不以成績反推明文參數。
- split、scene IDs 與理由寫進 config。

建議先採 8 dev / 8 held-out；若後續研究需要 cross-validation，再改為 grouped folds。

### 4.2 每次輸出必須記錄

- method：`paper_spec_2d` 或 `annotated_best_2d`；
- scene ID；
- source artifact hashes/paths；
- random seed；
- config snapshot；
- model/checkpoint IDs；
- code version或檔案 hash；
- coordinate transform、units、resolution；
- timestamp；
- known assumptions。

### 4.3 不以單棟小幅改善下結論

Stage 2 sampling 曾觀察到約 ±6 Δ5 等級的 run variance。所有 upstream A/B 必須：

- 固定 seed；或
- 使用多 seed；且
- 同時報 dev distribution 與 held-out checkpoint。

---

## 5. Phase 0 — 文件與 evaluator 收斂

### 目標

確保後續每一項變更都能被可信地量測，並清掉仍會誤導協作者的舊宣稱。

### 工作項目

1. 更新仍顯示 stairs 0.473「達標」的 HTML/report；歷史 log 保留但加 correction banner。
2. 將 `paper_alignment_audit.md` 中 stairs 項維持 `FIXED AND TESTED`，並連到 compare regression。
3. 將 2D room matching 改為 deterministic one-to-one Hungarian IoU matching。
4. 為 polygon holes、multipolygons、空 predictions、跨 level matching 建 regression fixtures。
5. 人工檢查約 30 個 derived connectivity cases，分 interior/exterior/厚牆 miss 記錄。
6. 重新定義 Tier C 信心：在 GT derivation 改善前，以 exploratory/lower-confidence 報告，不宣稱 annotated truth。
7. 補 windows 與 inter-level stair evaluation spec；可先不實作完整 metric，但 schema 先固定。

### 產物

- evaluator version `eval2d_v2`；
- regression tests；
- connectivity validation table；
- 修正過的 current reports。

### 時程

- 修改與測試：**2–4 個工作天**。
- 執行：2D 16-scene evaluator 通常為 **數分鐘到 30 分鐘**。

### Checkpoint 0

- 相同凍結 artifact 重跑兩次得到完全相同結果。
- room matching 不受輸入順序影響。
- holes/multipolygons 不被靜默丟棄。
- Tier C 的限制與人工 agreement rate 明確寫出。

---

## 6. Phase 1 — Canonical annotated-floorplan artifact

### 目標

把 Stage 4a 的 2D 結果變成正式 pipeline output，而不是 evaluator 臨時重新計算。

### 工作項目

1. 在 extrusion 前寫出 versioned 2D JSON/GeoJSON。
2. 保存 vector floorplan outer boundary、holes 與 multipolygons。
3. raster room labels polygonize 時使用 hierarchy-aware contours，不能只取最大 `RETR_EXTERNAL` contour。
4. RDP tolerance 成為 config，並保存 unsimplified/simplified coordinates 或 provenance。
5. room IDs 在分類、pruning、door/window/stair linking 後仍保持穩定。
6. 所有 entity 使用同一 coordinate frame、metres 與 level index。
7. 產生 SVG/PNG quicklook，顯示 rooms、IDs、doors、windows、stairs、types、graph edges。
8. `src/eval2d/extract_pred.py` 改讀 canonical artifact，不再重新執行 segmentation。

### 建議 schema

```text
building
  metadata: method/config/seed/frame/units
  levels[]
    id, elevation
    footprint: outer rings + holes
    rooms[]
      id, polygon, type, type_scores, confidence, provenance
    walls[]
      id, segment/polyline, adjacent_rooms
    doors[]
      id, segment, width, room_a, room_b|OUTSIDE, confidence, provenance
    windows[]
      id, segment, wall_id, room_id, confidence, provenance
    stairs[]
      id, polygon, from_level, to_level, adjacent_rooms, confidence
    graph
      nodes, edges, edge_kind, derived/direct flags
```

### 時程

- 修改與測試：**2–4 個工作天**。
- 單棟 smoke：數分鐘。
- 16-scene Stage-4-only export：**約 1–2 小時**。

### Checkpoint 1

- artifact round-trip 無幾何／ID 損失。
- evaluator 不依賴 runtime Stage 4 functions。
- 同一 artifact 永遠得到相同 evaluation。
- 下游研究可只讀此 schema，不需 import pipeline code。

---

## 7. Phase 2 — Paper-spec base footprint 與 rooms

### 7.1 D.1 level identification

`paper_spec_2d`：

- floor polygons 為 graph nodes；
- elevation difference ≤0.5 m 建 edge；
- connected components 形成 levels；
- level elevation 取平均。

目前的 `min_floor_area=5m²`、`max_level_span=0.8m`、fallback-floor reclassification 留在 `annotated_best_2d` 或以明確 flags 關閉。

估計：**0.5–1 天**。

### 7.2 D.2 floor/ceiling/wall assignment

需要：

- ceiling 指派給其 center 下方至少 1 m、空間上最近的 next-lower floor polygon；
- floorplan 為該 level floors＋assigned ceilings 的 projected union；
- holes 與 disconnected components 不丟失；
- wall selection 遵守 BEV intersection 與 `[elevation, elevation+2.5m]` vertical interval；
- current wall grouping/top/span heuristics 僅留在 best variant。

估計：**1–2 天**。

### 7.3 D.3 paper-spec two-stage room segmentation

建立獨立路徑：

1. HOV-SG-style segmentation，bottleneck width 2.5 m；
2. 對結果再以 1.5 m segmentation；
3. 保存實際 bottleneck boundaries；
4. 不用 `ROOM_CORE_MIN=0.5` 冒充論文兩階段規格。

同時保留 current `watershed_v3`，並避免稱為 HOV-SG literal port。

建議比較三個清楚名稱：

- `paper_spec_two_stage`；
- `watershed_v3`；
- `hovsg_code_port`（只有真的逐項移植原始 HOV-SG 時才使用）。

估計：**2–4 天**。

### 評估

- Room F1/IoU；
- room-count error；
- Corner/Angle sweeps；
- holes/level count；
- dev/held-out distribution；
- 不以 structures/depth 當主判準。

### Phase 2 時程

- 修改與測試：**4–7 個工作天**。
- 2–4 次 Stage-4-only batch：**約 3–8 小時機器時間**。

### Checkpoint 2

- 有可重跑的 paper-spec 與 watershed A/B。
- 兩條方法都輸出相同 schema。
- room geometry 指標與失敗分布完成，不只報平均。
- 明確決定 downstream 暫用哪一條；選擇品質較好者不改變其 provenance。

---

## 8. Phase 3 — Annotation layers

### 8.1 Doors 與 openings

### Paper-spec

- 使用 two-stage segmentation 的 bottleneck boundaries；
- oriented 2D bounding rectangle；
- width `<1.5m` 標為 door，否則 opening；
- 輸出 2D segment、width、相鄰 rooms；
- 不需要產生 3D doorframe。

### Best variant

current adjacency-only 方法即使用 GT regions 也只能覆蓋約 58% GT doors。為 annotated floorplan 品質，可另外實作：

- OneFormer `door` semantic 的直接 3D/2D projection；
- 與 wall/bottleneck candidates fusion；
- room↔room、room↔OUTSIDE 分開；
- confidence 與 source=`bottleneck|semantic|fused`。

Direct semantic detection 是研究擴充，不得寫成 paper method。

估計：

- paper 2D door rule：**1–2 天**；
- direct/fused variant：**2–4 天**。

### 8.2 Windows

需要：

- rays 使用 `window`、`window_blind`、`curtain`、`outdoor`；
- ray-layout-wall intersection；
- per-wall outlier removal/clustering；
- cluster ≥10、rectangle dimensions >0.3 m 的論文條件；
- annotated 2D 只保存 wall interval/segment、wall/room association；
- 不需要 3D window rectangle height。

另為 2D evaluator 加 windows@0.2/0.5m。

估計：**1–2 天**。

### 8.3 Stairs 與 multi-level topology

annotated 2D 只需：

- stair connected components；
- horizontal oriented rectangle／footprint；
- from/to level；
- adjacent rooms；
- up/down direction或 elevation difference；
- inter-level graph edge。

可跳過 room volume subtraction、3D stair extrusion、corner-height geometry 與 steps。

current nearby-component merge、minimum area/rise 等未公開設定必須標記為 best-variant assumptions。

估計：**1–3 天**。

### 8.4 Room types

目前 Room+type F1 0.15，是 annotated floorplan 的主要 blocker。

Paper-spec 應：

- OpenSeg features 投影到 mesh vertices；
- 依 room 所含 mesh vertices 平均 features；
- CLIP 對論文 15 classes 分類；
- 保存完整 class scores，不只 top-1；
- outdoor leaf classification/pruning 在 2D artifact 產生前完成，或標記而不直接刪除。

為避免錯分類刪掉空間，downstream 初期建議：

- 保留 room polygon；
- `is_outdoor_candidate=true`；
- 由下游或 confidence threshold 決定是否排除。

評估前需建立 MP3D room labels → paper 15 classes 的明確 ontology crosswalk；無對應類別標為 `other/unknown`，不可為提高分數任意合併。

估計：**3–6 天**。

### 8.5 Access graph

需要：

- door/opening 直接產生的 edges 優先於事後 probes；
- interior/exterior 分開；
- stairs 形成 inter-level edges；
- 每條 edge 標記 direct/derived 與 confidence；
- derived connectivity 不冒充人工 annotation。

由於 current GT derivation accuracy 不夠高，若 topology 是研究核心，應建立一個人工驗證／修正的小型 gold subset。

估計：**2–4 天工程＋2–4 小時人工檢查**。

### Phase 3 時程

- 只做 paper-spec annotations：**6–11 個工作天**。
- 同時做 direct-door/best variants：**9–17 個工作天**。
- 3–6 次 Stage-4/OpenSeg batches：**約 5–12 小時機器時間**。

### Checkpoint 3

- schema 中 rooms/doors/windows/stairs/types/edges 全部有值或明確 `unknown`。
- Room+type、Doors、Windows 與 topology 有獨立指標。
- annotations 全部可追溯來源。
- 下游研究不需要 3D extrusion 即可消費產物。

---

## 9. Phase 4 — 上游 Stage 3 改善

Phase 2/3 先利用保留的 Stage 4 inputs 快速迭代。若 Room F1、corner/angle 或 footprint coverage 到頂，再投入 Stage 3。

### 9.1 延續 optimizer polygon identity/class

目前 Stage 4 從 fitted mesh vertex colors 重建 polygons 並重新分類。應改為：

- 直接讀作者 optimizer final state/checkpoint；
- 延續 polygon IDs、planes、classes、merge results；
- fitted mesh 僅作 geometry，不作主要 identity source。

估計：**2–4 天**；需 1–2 次完整 batch，約 **10–25 小時**。

### 9.2 Polygon initializer

盡可能依 Appendix Algorithm 1 重做／收斂：

- RANSAC 與 candidate scheduling；
- connected components；
- shared vertices；
- contour/hole preservation；
- polygon coverage diagnostics；
- simplified object mesh A/B；
- 本地 K/min-area/max-polygons 等集中 config 並標記 assumptions。

作者沒有釋出 initializer，故只能稱 spec-faithful reimplementation。

估計：**4–8 天**；2–4 次完整 batch，約 **20–50 小時**。

### 9.3 Stage 3 的 2D 主判準

Stage 3 A/B 應優先觀察：

- base footprint coverage/IoU；
- room Corner/Angle；
- wall-to-GT-boundary distance；
- holes/disconnected components；
- downstream doors/topology；
- polygon count/complexity。

3D structures/depth 可作診斷，但不是 annotated-floorplan 的通過門檻。

### Phase 4 時程

- 修改與測試：**6–12 個工作天**。
- 完整 pipeline 機器時間：**30–70 小時累積**，可夜間執行。

---

## 10. Phase 5 — Stage 2 對齊與可重現性

### 10.1 固定 sampling

- scene-derived deterministic seed；
- Python/NumPy/PyTorch 全部記錄；
- sampled pixels 或 indices 可選擇保存，讓同一 scene 可精確重播。

估計：**0.5–1 天**。

### 10.2 Exact OneFormer 設定

確認或清楚記錄：

- checkpoint/backbone；
- inference resolution；
- preprocessing/normalization；
- resize policy；
- COCO mapping。

作者 wrapper 為空，因此無法驗證的部分標為 assumptions。

估計：**1–3 天**。

### 10.3 SPT preprocessing

- 集中 voxel/KNN/cut-pursuit parameters；
- 保存 superpoint statistics；
- 對 coverage 與 2D boundary quality 做 sensitivity study；
- upstream defaults 不稱為作者參數。

估計：**2–4 天**。

### 10.4 Paper/code conflict 的雙路徑

作者 code 與論文文字衝突：

- paper 5000 samples vs official code 3000；
- paper point→nearest vertex vs code vertex→5NN points。

建議：

- `official_code`：3000＋5NN，作主要 reported-code reproduction；
- `paper_text`：5000＋nearest-vertex，作 spec A/B；
- 不靜默選一個並稱唯一 ground truth。

估計：**1–2 天**，另需約 2 次完整 batch。

### Phase 5 時程

- 修改與測試：**4.5–10 個工作天**。
- cold/full batches：**20–40 小時以上**，視 OneFormer 是否重跑。

---

## 11. Phase 6 — 最終驗證與研究交付

### 工作項目

1. 凍結 `paper_spec_2d` 與 `annotated_best_2d` configs。
2. dev 不再調整，執行 held-out checkpoint。
3. 產生 per-scene、macro、micro 與 distribution reports。
4. best/median/worst 與 multi-level scenes 人工檢查。
5. 對 artifact schema 做 round-trip、coordinate、ID、topology tests。
6. 建立 downstream research smoke test，確認能讀取 rooms/types/openings/graph。
7. 寫清楚：
   - 哪些是 paper-spec；
   - 哪些是 official-code behavior；
   - 哪些是 local variants；
   - 哪些 GT/edges 是 derived；
   - 哪些 annotations confidence 不足。

### 時程

- 修改、報告與人工檢查：**2–4 個工作天**。
- 最終 Stage-4-only batch：**1–2 小時**。
- 若包含 upstream final rerun：**9–18 小時**。

### 完成定義

只有同時滿足以下條件才稱 annotated-floorplan pipeline 可供研究：

- canonical artifact schema 凍結並 versioned；
- fixed seed/config 可重現；
- room geometry、types、doors、windows、stairs、topology 都有明確 coverage；
- `unknown` 與低 confidence 不被強制猜測；
- paper-spec 與 local variants 可獨立執行／評估；
- evaluator limitations 寫入報告；
- downstream smoke test 通過；
- 沒有依賴 3D extrusion 才能讀取 2D annotation。

---

## 12. 建議時程總表

| Phase | 工作 | 人工作業 | 機器執行 | 優先級 |
|---|---|---:|---:|---:|
| 0 | 文件與 2D evaluator 硬化 | 2–4 天 | <0.5 小時 | P0 |
| 1 | Canonical 2D artifact/schema | 2–4 天 | 1–2 小時 | P0 |
| 2 | Paper-spec D.1–D.3 | 4–7 天 | 3–8 小時 | P1 |
| 3A | Paper-spec annotations | 6–11 天 | 5–12 小時 | P1 |
| 3B | Direct-door/best variants | 額外 3–6 天 | 2–5 小時 | P1/P2 |
| 4 | Stage 3 identity＋initializer | 6–12 天 | 30–70 小時 | P2 |
| 5 | Stage 2 exactness＋A/B | 4.5–10 天 | 20–40+ 小時 | P2 |
| 6 | 最終驗證與交付 | 2–4 天 | 1–18 小時 | P0 at release |

### Scope A：Annotated-floorplan MVP

包含 Phase 0、1、2、3A，沿用目前 Stage 2/3：

- **約 14–26 個工作天**；
- AI-assisted、問題較少時約 **3–5 週**；
- 反覆執行約 **10–25 小時**，多數是可夜間／Stage-4-only 的工作。

### Scope B：研究用 best variant

在 MVP 上加入 direct/fused doors、較可靠 topology 與 best-method A/B：

- **約 18–32 個工作天**；
- 實際約 **4–6 週**。

### Scope C：高忠實上游＋完整 annotated floorplan

再加入 Stage 3、Stage 2 alignment：

- **約 28–50 個工作天**；
- 樂觀 **5–6 週**，實際 **6–10 週**；
- 需要數次 9–18 小時完整 batch。

估時不含 GPU 排隊、第三方環境故障或需要重新取得資料的時間。

---

## 13. 建議的第一個執行區段

先只做以下內容，不一次展開全部 Phase：

1. 硬化 eval2d matching／polygon fixtures。
2. 凍結 dev/held-out split。
3. 建 canonical artifact 與 schema。
4. 實作 `paper_spec_two_stage`，與 `watershed_v3` 做相同輸入 A/B。
5. 實作 paper 2D door rule與 `outdoor` window rays。
6. 先不改 Stage 2/3，也不做 direct-door extension。

預估 **7–12 個工作天**。完成後依 2D Room/Corner/Door 指標決定：

- 若 Stage 4 改善足夠，進入 annotation layers；
- 若 boundary/coverage 仍是主瓶頸，再提前投入 Stage 3 initializer；
- 若 room geometry 足夠但 doors/types 低，集中做 direct doors 與 per-mesh-vertex OpenSeg。

這個順序能先使用保留的 Stage 4 inputs 低成本驗證，不會在 method direction 尚未確認前反覆跑完整 Stage 2/3。

---

## 14. 狀態標記

後續維護本文件時，所有項目使用：

- `OPEN`
- `IN PROGRESS`
- `PARTIALLY FIXED`
- `FIXED AND TESTED`
- `INTENTIONAL VARIANT`
- `DEFERRED / OUT OF SCOPE`

只有具備 regression test、artifact 或固定 evaluator A/B evidence 時，才可標成 `FIXED AND TESTED`。
