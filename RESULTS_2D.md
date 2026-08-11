# RESULTS_2D — 2D Floorplan Evaluation（watershed_v3，PRELIMINARY）

> **Current recommended protocol**：`eval2d_v3_strict_levels`＋`mp3d_house_floor_v0_1`  
> predictions＝frozen `watershed_v3_pre_report`；CPU-only re-evaluation，未重跑 pipeline。  
> `watershed_v3` 是 local engineering variant，非 strict paper reproduction。

## 本輪 correctness 修正

1. v2 未處罰 unmatched predicted levels；v3 將其 rooms、doors、derived edges 全列為 FP。
2. room GT geometry 改為直接解析官方 Matterport3D `.house`：`R → S(label=F) → ordered V`。
3. 不再從 `regionN.ply` 全 vertices 做 XY raster closing/fill、external contour 與 RDP。
4. 舊 frozen GT/scores 完整保留；新 GT 位於獨立 candidate 目錄，沒有覆寫舊 baseline。

## Current headline：strict v3＋official `.house` GT

| 指標 | 結果 | 備註 |
|---|---:|---|
| **Room F1 @IoU>0.5** | **0.602** | P 0.611 / R 0.594；TP/FP/FN=193/123/132 |
| matched-room mean IoU | **0.785**（n=193） | conditional mean，只看成功 matched rooms |
| Corner F1 @0.1/0.2/0.3m | **0.195 / 0.342 / 0.436** | GT 已改為直接官方 polygons |
| Angle F1 @0.1m | **0.112** | — |
| Room+type F1 | **0.144** | CLIP room type 不可靠 |
| Doors F1 @0.2/0.5m | **0.190 / 0.243** | extra-level doors 已計 FP |
| edge room-room / outside / all | **0.161 / 0.025 / 0.131** | connectivity 仍是 derived，權重下調 |

分布：**14/16** scenes Room F1≥0.5；中位約 **0.635**；best `e9zR4mvMWw7` **0.818**；worst `HxpKQynjfin` **0.182**。

## 可識別 A/B

| variant | GT rooms | overlap | Room P/R/F1 | matched IoU (n) |
|---|---:|---:|---:|---:|
| legacy v2＋raster GT | 324 | 1.443% | 0.639/0.590/**0.613** | 0.793 (191) |
| strict v3＋同 raster GT | 324 | 1.443% | 0.604/0.590/**0.597** | 0.793 (191) |
| strict v3＋official polygons、共同 324 IDs | 324 | 0.131% | 0.611/0.596/**0.603** | 0.785 (193) |
| strict v3＋official polygons、明確 325-room set | 325 | 0.132% | 0.611/0.594/**0.602** | 0.785 (193) |

- v2→v3 單獨顯示 extra-level 漏罰造成的差異：17 rooms 新增為 FP，Room F1 0.613→0.597。
- raster324→house-common324 固定 evaluator 與 denominator，只更換 geometry：Room F1 0.597→0.603，matched rooms 191→193。
- common324→official325 只加入舊 `<0.5 m²` raster area gate 漏掉的一個有效 closet：Room F1 0.603→0.602。
- GT overlap 1.443%→0.132%，證實直接 `.house` polygons 更適合作為 2D room GT。

## GT validation

- 16/16 scenes；354/354 MP3D regions 各有且僅有一個 floor surface。
- 354/354 polygons：finite、valid、positive-area、counter-clockwise。
- 32 MP3D levels、354 regions、325 evaluated non-stair/non-junk rooms、23 stair regions、292 HouseLayout3D doors。
- Candidate 16 PKL SHA-256 全部驗證通過。
- 論文的 **317 rooms／33 levels** 仍沒有公開 exact subset manifest；目前 325／32 是透明且可重現的 MP3D roomset，**不宣稱等同論文 subset**。

## Multi-floor correction

`1LXtFkjw3qL` 有 GT 3 levels、PRED 5 levels。PRED L1/L3 共 16 rooms 未匹配任何 GT level；v2 漏算，v3 已列為 FP。該 scene Room F1：

- 舊圖／v2 raster GT：0.408；
- v3 raster GT：0.308；
- v3 official GT：0.338。

舊 `multi_level_1LXtFkjw3qL.png` 可繼續作 qualitative failure example，但圖片內的 0.41 是 archived v2 數字，不能當 current score。

## 結論

> Frozen 16-scene baseline 在 strict-level evaluator 與官方 MP3D floor polygons 下，Room F1 **0.602**；成功 matched 的 193 rooms conditional mean IoU **0.785**。Coarse room geometry 有初步訊號，但 precise boundaries、doors、connectivity、room types 尚不足。

完整 A/B：[RESULTS.md](outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1/eval_frozen_watershed_v3/RESULTS.md)  
Candidate manifest：[candidate_manifest.json](outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1/candidate_manifest.json)

Current 視覺化與 error analysis：`outputs/eval2d/baselines/watershed_v3_pre_report/eval2d_v3_strict_levels/visualizations/`
