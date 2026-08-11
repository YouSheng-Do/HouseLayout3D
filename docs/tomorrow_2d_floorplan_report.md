# 明日 2D Annotated Floorplan 報告提綱（v3／official GT 修正版）

> 建議定位：**16-scene preliminary research reimplementation baseline**。  
> 核心句：coarse room geometry 有初步訊號，但 precise boundaries、doors、connectivity、room types 尚不足；不能稱為 strict paper reproduction 或足以直接支撐 downstream research。

## 30 秒摘要

我們凍結了 16 scenes 的 Stage 4a pre-extrusion predictions，沒有重跑 GPU pipeline。本輪修正 multi-floor evaluator：未匹配 predicted levels 上的 rooms、doors、edges 現在都列為 false positives；並將 room GT 改為直接解析 Matterport3D 官方 `.house` floor polygons，不再由完整 region mesh rasterize 重建。

在 current recommended protocol（`eval2d_v3_strict_levels`＋`mp3d_house_floor_v0_1`）下，Room F1 為 **0.602**；成功 matched 的 193 rooms conditional mean IoU 為 **0.785**。Door F1@0.5m **0.243**、connectivity edge F1 **0.131**、Room+type F1 **0.144**。結論仍是：coarse room geometry 有初步訊號，但 annotated floorplan 尚未足夠。

## 建議投影片順序

### 1. 研究目標

- 目標不是只畫 footprint，而是得到可承載後續 attributes 的 annotated floorplan。
- 今日問題：目前輸出在哪些層次有訊號、哪些層次仍不足？
- `watershed_v3` 是 local engineering variant，不是 strict paper reproduction。

### 2. 評估設定與本輪修正

- 16 scenes、frozen predictions、Stage 4a pre-extrusion；本輪 CPU-only。
- evaluator：`eval2d_v3_strict_levels`，Hungarian room matching。
- v3 修正：unmatched predicted levels 上所有 rooms／doors／derived edges 都算 FP。
- GT：官方 MP3D `.house` 的 `R → S(label=F) → ordered V`；doors 來自 HouseLayout3D。
- connectivity 仍由 room polygons＋door segments 的 PIP probes 推導，不是 annotated graph。

### 3. Headline metrics

| 項目 | Current 結果 | 可說的結論 |
|---|---:|---|
| Room F1 @IoU>0.5 | **0.602** | coarse room geometry 有初步訊號 |
| matched-room mean IoU | **0.785**（n=193） | conditional metric，只描述成功配對 subset |
| Corner F1 @0.1m | **0.195** | precise boundary 仍不足 |
| Door F1 @0.5m | **0.243** | doors 仍不足 |
| connectivity edge F1 | **0.131** | graph 不足，且 GT connectivity 是 derived |
| Room+type F1 | **0.144** | room type 不可靠 |
| Stairs F1 @0.5 | **0.411** | 3D evaluator 修正後略低於 paper 0.42；舊 0.473 作廢 |

建議講法：

> 「在嚴格樓層計分與官方 MP3D floor polygons 下，Room F1 約 0.60；找到的 rooms 形狀通常合理，但 room recall、doors、connectivity 與 room type 尚不足。」

### 4. 為什麼 headline 從 0.613 變成 0.602？

這不是模型退步，而是 evaluation correctness 修正：

| Protocol | Room F1 | 解釋 |
|---|---:|---|
| legacy v2＋raster GT | 0.613 | extra predicted levels 漏算 FP |
| strict v3＋相同 raster GT | 0.597 | 新增 17 extra-level rooms 為 FP |
| strict v3＋official GT、共同 324 IDs | 0.603 | 只換官方 polygons；多匹配 2 rooms |
| strict v3＋official 325-room set | **0.602** | 加回舊 area gate 漏掉的有效 closet |

GT overlap 同時由 **1.443% 降到 0.132%**。因此 0.602 是目前較可信、較容易辯護的 headline。

### 5. 為什麼 F1 約 0.60，但 matched IoU 約 0.79？

Room F1 評估有多少 rooms 被成功找回；matched mean IoU 只平均成功配對的 rooms。

> F1 低代表有漏房、合房或額外房；0.785 代表「成功找到時常畫得像」，不代表所有 rooms 都找得到。

Worst scene `HxpKQynjfin` 是清楚例子：GT 8 rooms、PRED 3 rooms，只成功配到 1 room，因此 Room F1 **0.182**；該唯一 matched room IoU 卻約 **0.881**。

### 6. Qualitative results

請使用新生成的 current 圖，不要再用 `eval2d_v2_hungarian/visualizations` 當現行數字來源。

- best `e9zR4mvMWw7`：Room F1 **0.818**。
- representative `i5noydFURQK`：Room F1 **0.621**，接近中位約 0.635。
- worst `HxpKQynjfin`：Room F1 **0.182**，明顯 under-segmentation。
- multi-level `1LXtFkjw3qL`：Room F1 **0.338**；GT 3 levels、PRED 5 levels，額外 PRED L1/L3 已列入 FP。

### 7. GT validation 與仍未解決的問題

- 16/16 scenes、354/354 regions，各有一個有效 CCW floor polygon。
- 32 MP3D levels、354 regions、325 evaluated rooms、23 stair regions、292 doors。
- Candidate overlap 0.132%；16 個 PKL hashes 全通過。
- 論文報告 317 rooms／33 levels，但公開 release 沒有 exact subset manifest。
- 因此我們宣稱「透明、可重現的官方 MP3D roomset」，不宣稱精確等同論文 subset。

### 8. Error analysis 與下一步

Current diagnostics（flags 可重疊）：

- geometry good：7/16；
- under-segmented：6/16；
- level mismatch／extra：2/16；
- doors／topology weak：12/16。

接下來方法改善優先序：

1. hole-aware／multipolygon-aware prediction polygonization；
2. paper-spec two-stage 與 watershed_v3 的 frozen-input A/B；
3. direct door rule 與 outdoor window rays；
4. room type 最後處理。

## 明天逐圖 Checklist

Current 圖目錄：

`outputs/eval2d/baselines/watershed_v3_pre_report/eval2d_v3_strict_levels/visualizations/`

### Best

檔案：`best_e9zR4mvMWw7.png`

- 圖為 GT L2↔PRED L3；scene aggregate Room F1 0.818。
- PRED 還多一個 unmatched level L1；即使 best scene 也不是完美 multi-level reconstruction。
- 同色不代表 matching；左右配色獨立。

### Representative

檔案：`representative_i5noydFURQK.png`

- Scene Room F1 0.621，接近中位品質。
- 圖中 shown level pair F1 0.700，但 scene aggregate 較低；不要混淆單層與整棟。
- 可指出 over-segmentation 與 room-type confusion。

### Worst

檔案：`worst_HxpKQynjfin.png`

- 必講 GT 8 vs PRED 3，主要 failure 是 under-segmentation。
- shown pair／scene Room F1 0.182，但唯一 matched room IoU 0.881。
- 用來解釋 F1 與 conditional IoU 為何可同時低／高。

### Multi-level

檔案：`multi_level_1LXtFkjw3qL.png`

- 顯示 GT L1↔PRED L2；shown pair F1 0.375，scene aggregate F1 0.338。
- GT 3 levels、PRED 5 levels；unmatched PRED levels=[1,3]。
- 額外兩層共 16 rooms 現已列為 FP；這張圖是 level over-detection failure case。
- GT 側已使用官方 `.house` polygons並排除 stairs。

### Error analysis summary

檔案：`error_analysis_summary.png`

- 這張圖已由 current `eval2d_v3_strict_levels`＋official GT scores 重生，不是 archived v2 圖。
- 左圖先講 16-scene Room F1 分布；右上用 scatter 解釋 coverage 與 conditional IoU；右下是可重疊的 diagnostic flags。
- 旗標是 error taxonomy，不是 benchmark pass/fail threshold；不要把 `geometry good 7/16` 說成 7 棟「達標」。

## 不可宣稱

- 我們已完整 reproduce 論文。
- 0.785 代表所有 rooms 都很準。
- floorplan 已足夠掛 downstream attributes。
- stairs 已達到或超過 paper target（目前 3D stairs F1 為 0.411，低於 paper 0.42）。
- 325 rooms／32 levels 就是論文 317／33 的 exact subset。
- watershed 是論文或 HOV-SG 方法的逐字移植。

## 可安全宣稱

> Frozen 16-scene preliminary baseline 在 strict-level evaluator 與官方 MP3D floor polygons 下，Room F1 0.602；193 個成功 matched rooms 的 conditional mean IoU 0.785。Coarse room geometry 有初步訊號，但 precise boundaries、doors、connectivity 與 room types 尚不足。
