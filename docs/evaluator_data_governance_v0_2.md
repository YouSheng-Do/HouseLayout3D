# Evaluator 與資料治理封版紀錄 v0.2

日期：2026-08-12

狀態：**Phase 0 evaluator/data-governance checkpoint complete**
範圍：CPU-only；未重跑 Stage 2/3/4、未使用 GPU、未修改 `watershed_v3` 方法。

## 結論

本輪已把 2D evaluation 從「會隨 code 重算 prediction 的 preliminary script」收斂成：

1. 固定的 8-scene dev／8-scene prospective held-out policy；
2. versioned `annotated_floorplan_v0.2` JSON Schema；
3. 支援 Polygon holes／MultiPolygons 的 geometry contract；
4. 只讀 frozen canonical prediction artifacts 的 evaluator；
5. 對 30 個 connectivity cases 有逐案、可追溯的視覺稽核；
6. 明確將 Tier C 降為 **exploratory/lower-confidence**，不再宣稱「79% 可信 GT」。

正式全 16-scene checkpoint 的核心結果仍是：Room F1 **0.6022**、成功 matched rooms
的 conditional mean IoU **0.7850**（n=193）。這輪改的是可信度、可追溯性與未來 artifact
能力，沒有改方法或把分數做高。

## 1. Split policy

權威檔案：`configs/eval2d/split_v0_1.json`

| partition | scenes | levels | GT Room-metric rooms | 使用規則 |
|---|---:|---:|---:|---|
| dev | 8 | 15 | 167 | routine development／threshold 或 heuristic 變更 |
| held_out | 8 | 17 | 158 | 只可在具名 phase checkpoint 執行 |
| total | 16 | 32 | 325 | 正式 checkpoint 才能彙總 |

選法只使用 official MP3D level count、official325 room count 與 scene ID，不讀 prediction
score 或 qualitative result。配對相近 complexity 後，以 SHA-256(scene ID) 決定 dev／held-out。

重要限制：這不是 untouched test set。16 scenes 在 split 凍結前都曾被看過，因此只能稱
**prospective held-out**。`split_io.select_scenes()` 會阻擋沒有 `checkpoint_name` 的
held-out／all 執行。

## 2. Canonical artifact v0.2

Schema：`configs/eval2d/annotated_floorplan_v0_2.schema.json`

Builder／loader：`src/eval2d/canonical_io.py`
Frozen collection：`outputs/eval2d/canonical/watershed_v3_pre_report_v0_2/`

固定內容包括：

- scene／method／source SHA-256 provenance；
- metre-space coordinate frame 與 level elevation；
- rooms、walls、doors、windows、stairs 與 graph 的 versioned fields；
- Polygon／MultiPolygon、interior rings 與 topology counts；
- stable room IDs、stable sorted door IDs；
- capability flags，用來區分「未產生」與「真的空 prediction」；
- v0.1 backward-compatible loader。

16 個 frozen prediction PKL 經 hash 驗證後輸出為 16 個 JSON；每一份都通過 Draft 2020-12
JSON Schema validation。collection manifest 固定 source／builder／split／artifact hashes。

### 歷史資料的誠實限制

v0.2 **有能力**保存 holes 與 MultiPolygons，但 frozen v0.1 prediction source 的 316 rooms
全都只含單一 exterior ring：

| source capability | rooms |
|---|---:|
| `legacy_single_exterior_ring` | 316 |
| `structured_polygon_topology` | 0 |

因此不能從 frozen source 回復已經被丟掉的洞或 disconnected components。v0.2 同時保存：

- valid structured geometry，供新下游使用；
- exact `legacy_metric_ring`，供歷史 IoU／Corner／Angle 完整重現。

未來由 hierarchy-aware raster polygonization 產生的新 artifact 才能真正保存完整 topology；
不以 schema capability 冒充歷史資料已補回。

## 3. Artifact-only evaluator

入口：`src/eval2d/evaluate_canonical.py`
Evaluator：`eval2d_v3_strict_levels`

評估 scene 時只讀：

- canonical prediction JSON；
- frozen official MP3D floor GT PKL；
- frozen split、GT manifest 與 expected-score artifact。

不 import Stage 4、segmentation 或 `extract_pred`。16/16 scenes 的 Tier A/B/C counts 皆與
frozen source exact 相等；matched IoU 以 atol `1e-12` 驗證。

### 正式 checkpoint：`phase0_canonical_v0_2`

| partition | Room P | Room R | Room F1 | matched conditional IoU | matched n |
|---|---:|---:|---:|---:|---:|
| dev | 0.5930 | 0.6108 | 0.6018 | 0.7749 | 102 |
| prospective held-out | 0.6319 | 0.5759 | 0.6026 | 0.7963 | 91 |
| all | 0.6108 | 0.5938 | **0.6022** | **0.7850** | 193 |

正式輸出：

- `.../eval2d_v3_strict_levels_official325_dev/`
- `.../eval2d_v3_strict_levels_official325_all/`

dev 與 held-out Room F1 接近，只能解讀為這次 frozen checkpoint 沒有明顯 split drift；
不能把 prospective held-out 包裝成從未看過的 test performance。

## 4. Hole／MultiPolygon regression

`geometry_v2.mask_to_shapely()` 使用 hierarchy-aware `RETR_TREE` polygonization；不再只拿
最大 `RETR_EXTERNAL` contour。固定 fixture 同時含：

- 3 個 disconnected components；
- 1 個 interior hole；
- nested island；
- point-in-polygon probe 落在 hole 時保持 outside；
- structured canonical round-trip 與 exact Room IoU。

這些 regression 證明新路徑不會靜默 flatten topology。legacy raw ring 仍使用原本的
`buffer(0)`／largest-component IoU contract，以免 evaluator hardening 偷改歷史分數。

## 5. Tier C connectivity audit

自動 cases：`outputs/eval2d/connectivity_audit_v0_1/auto_cases.json`

人工標註 source：`configs/eval2d/connectivity_audit_manual_v0_1.json`
contact sheets／review／manifest：`outputs/eval2d/connectivity_audit_v0_1/`

### 全 292 doors 的 deterministic probe outcomes

| outcome | count | 比例 | 現在的解讀 |
|---|---:|---:|---|
| `success` | 173 | 59.25% | room↔room candidate |
| `one_outside` | 81 | 27.74% | 65 exterior candidates＋16 far-room/thick-gap candidates；皆非獨立 truth |
| `same_room` | 37 | 12.67% | region granularity 無法表達門兩側差異 |
| `overlap` | 1 | 0.34% | overlapping GT regions，無唯一 adjacency |

Audit-only outward march 由 empty side 的 0.35 m 起，每 0.05 m 掃到 1.50 m：65/81 沒再
碰到 included room，列為 exterior candidate；16/81 之後碰到 room，列為 far-room／
thick-gap candidate。這個 march 不改 frozen GT／score，也不把 candidate 升格成 truth。

目前 derivation 仍將全部 `success` 與 `one_outside` 建成 edges，共 254/292（86.99%）；
其餘 38 doors 不會進 derived graph。這是 coverage 定義，不是 86.99% accuracy。v0.2
canonical 已把後者命名為 `room-outside-candidate`，不再寫成 verified exterior。

### 固定 30-case stratified visual audit

Sampling 是各 outcome 內依 SHA-256(case key) 決定，非挑好看的例子。

| cases | auto outcome | manual interpretation | 結果 |
|---|---|---|---|
| C01, C03–C10 | success | distinct-region room↔room candidates | 9/9 geometry-consistent |
| C02 | success | distinct regions，但 released polygons 在門附近有較寬 gap | geometry-consistent with caveat |
| C11–C16、C18–C20 | one_outside | union-boundary、1.5 m 內沒有其他 included room | 9 exterior candidates；未獨立驗證 |
| C17 | one_outside | 0.30 m 為空，但 0.40 m 碰到 room 2 | 1 far-room／thick-gap candidate |
| C21–C29 | same_room | door/probes 位於同一 released GT region | 9/9 GT granularity limitation |
| C30 | overlap | probe 同時落入兩個 released GT regions | ambiguous |

這次人工 review 看的是 probe 與**同一組 GT geometry**是否一致，不是另外找一份 portal
annotation 作 blind correctness labeling。因此不能報「20/20 正確」、不能把 Structured3D
的 98.2% 移植成 HouseLayout3D accuracy，也不能再使用舊的「GT-side 79% 可信」。

目前 `edge_all F1=0.131` 仍可作 geometry/topology diagnostic，但不是主要 benchmark
claim。若後續研究以 navigation graph 為核心，需另建 independent gold subset，優先人工
標註 81 個 exterior candidates 與 37 個 same-region doors。

## 6. Windows／stairs 的資料契約

v0.2 schema 已固定 `windows[]` 與 `stairs[]`。Phase-0 base artifacts 明確標
`windows_included=false`、`stairs_included=false`，其中空 list 是 **N/A**，不是零分。
後續具名 checkpoint 則各自把 capability 設為 true：window rays v0.1 保存 637/709
candidates 並以 endpoint evaluator 評分；stairs2d v0.1 保存 12 個 existing D.5
predictions。capability=true 後的空 list 才是實際空預測，必須計 FN。

未來 protocol 已固定在 `docs/eval_2d_metric.md`：Windows 使用 metre-space segment endpoint
distance＋Hungarian；Stair 分 footprint IoU 與 inter-level link correctness。既有 3D stairs
F1 0.411 不可當作 2D stairs 分數。

Released stair GT 的 34 個 mesh 沒有 `from_level/to_level` 或 adjacent-room 欄位，且同一
物理樓梯常拆成多個 flight/landing entity。因此 released-only checkpoint 可正式量測
footprint，link correctness 必須是 N/A；prediction-side direct link 只能報 coverage，不能用
mesh z 推導一份假 GT link。

正式 stairs2d v0.1：all-16 footprint P/R/F1 0.417/0.147/0.217（12 pred、34 GT、
TP/FP/FN 5/7/29）；12/12 prediction links references 完整，但不報 accuracy。完整紀錄見
`docs/stairs2d_v0_1.md`。

## 7. Definition of Done

- [x] split fixed、hashable，held-out access 有 checkpoint guard；
- [x] canonical schema versioned，16/16 JSON schema validation；
- [x] same artifact → exact same A/B/C counts 與 IoUs；
- [x] Hungarian permutation／tie regression；
- [x] empty prediction／cross-level／unmatched-level regression；
- [x] holes／MultiPolygons／nested-island／PIP-hole regression；
- [x] v0.1 backward compatibility 與 16-scene coordinate/count/score round-trip；
- [x] 30-case connectivity review、manual labels、contact sheets 與 hashes；
- [x] Tier C claim boundary、Windows/Stair schema 與 metric contract 寫入文件；
- [x] no Stage 2/3/4 rerun、no GPU。

## 8. Phase-0 本輪不包含（歷史 scope）

- 當時不改善 `watershed_v3` 的 room geometry；
- 當時不做 `paper_spec_two_stage` A/B；
- 不回復 frozen legacy source 已丟掉的 holes/components；
- 當時不實作或計算 2D Windows／Stair metrics；
- 不把 canonical writer 接進一次新的完整 pipeline run；
- 當時不調 threshold、不用 held-out debugging。

以上是 Phase-0 governance checkpoint 的範圍，不代表後續仍未完成。其後已用獨立具名
checkpoints 完成 `paper_spec_two_stage` A/B、Windows endpoint metric 與 Stair-footprint
metric，以及 Room-type retained-feature aggregation A/B；它們不回寫成 Phase-0 的一部分。
Room-type checkpoint 使用 explicit conservative crosswalk，paper last-five pruning 只記
candidate 而不刪 room，完整說明見 `docs/room_types_ab_v0_1.md`。任何新的
threshold／heuristic 開發仍先只看 dev。
