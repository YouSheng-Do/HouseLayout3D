# paper_spec_two_stage vs watershed_v3 — Same-input A/B v0.1

日期：2026-08-12

Checkpoint：`phase2_paper_two_stage_ab_v0_1`
完整 generated report：
`outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/comparison/RESULTS.md`

## 結論

在相同 Stage-4 inputs、predicted levels、wall raster adapter、polygonizer 與
evaluator 下，`paper_spec_two_stage` 沒有優於 `watershed_v3_rerun`：

| split / method | Room P | Room R | Room F1 | matched-only IoU (n) |
|---|---:|---:|---:|---:|
| dev / paper_spec | 0.384 | 0.653 | 0.483 | 0.785 (109) |
| dev / watershed | 0.593 | 0.611 | 0.602 | 0.770 (102) |
| held-out / paper_spec | 0.354 | 0.620 | 0.451 | 0.776 (98) |
| held-out / watershed | 0.625 | 0.570 | 0.596 | 0.793 (90) |
| all / paper_spec | **0.369** | **0.637** | **0.467** | **0.781 (207)** |
| all / watershed | **0.608** | **0.591** | **0.599** | **0.781 (192)** |

matched-only IoU 幾乎相同；差距主要來自 paper interpretation 的
over-segmentation。每個 evaluated level 的 signed room-count error 是 `+6.74`
vs `-0.26`，MAE 是 `7.20` vs `2.26`。paired scenes 中 paper 較好 2/16、
較差 14/16。

決策：研究用 annotated floorplan downstream 暫留 `watershed_v3`；
`paper_spec_two_stage` 保留為 paper-alignment / ablation，不混寫 provenance。

## Faithfulness boundary

[HouseLayout3D Appendix D.3](https://openreview.net/pdf/e20998737506e658d49d8d9d073931ac459638c7.pdf)
明載 HOV-SG-style morphology segmentation 依序使用 2.5 m、1.5 m bottleneck，
並以 `<1.5 m` 判為 door。公開資料沒有給足 morphology internals，HouseLayout3D
Stage-4 author implementation 亦未提供；官方
[HOV-SG repository](https://github.com/hovsg/HOV-SG) 公開的是
distance-transform / watershed pipeline，沒有直接暴露這組 2.5/1.5 m 參數。

本實作把 bottleneck width `w` 定義成 Euclidean erosion radius `w/2`，以
eroded connected components 為 seeds、nearest-seed 回填，並在每個 2.5 m cell
內做 1.5 m refinement。因此只能稱 **reproducible paper-spec interpretation**，
不能稱 author-code reproduction 或 literal HOV-SG port。

## 公平比較與 frozen headline

歷史 frozen watershed headline 是 Room F1 `0.602`；本 A/B 的 fresh same-input
rerun 是 `0.599`。差一個 TP，只發生在 `WYY7iVyf5p8`：歷史 source 只有單一
exterior ring，新 rerun 以 structured v0.2 保存一個 2-component MultiPolygon，
使一個 match 跨過 IoU=0.5 threshold。A/B 採 fresh rerun，因為兩法才共享同一個
hierarchy-aware polygonizer。

## Evidence 與驗證

- generator 不讀 GT 或 evaluator scores；CPU-only，未跑 Stage 2/3、extrusion、GPU。
- 每棟 prototype load 一次、`identify_levels` 一次，再 deep-copy 給兩方法。
- 32 canonical artifacts 通過 v0.2 schema；兩法 predicted levels/elevations 一致。
- synthetic D.3 contract：7/7 PASS。
- formal artifact invariants：10/10 PASS。
- evaluator regression：matching 12/12、strict level 11/11、geometry 13/13、
  split 8/8、legacy canonical round-trip/equivalence 全 PASS。

Generated artifacts：

- `outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/paper_spec_two_stage/`
- `outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/watershed_v3_rerun/`
- `outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/shared_input_manifests/`
- `outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/comparison/`

主要圖：`paired_room_f1.png`、`room_count_comparison.png`、`paired_cases.png`。

## 不可宣稱

- 這不是作者 hidden Stage-4 code 的 reproduction。
- 這不能證明論文方法本身較差，只能描述此明示規格 interpretation 在目前輸入上的結果。
- matched IoU 不是所有 GT rooms 的平均；它只涵蓋 IoU≥0.5 的 matched rooms。
- Room types 在兩法皆設為 `unknown`；door / connectivity 只作診斷。
- `held_out` 不是 untouched test，因 16 scenes 在 split freeze 前都曾被看過。
