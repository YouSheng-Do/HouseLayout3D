# One-shot Task — 報告前 2D Floorplan CPU-only Hardening

> 任務性質：一次性、限時、完成後停止  
> 時間上限：4–6 小時  
> 目標：為明日報告凍結並硬化目前 watershed v3 的 2D baseline  
> 授權範圍：evaluator、tests、artifact exporter、CPU-only evaluation、visualization、current-facing 文件  
> 禁止範圍：GPU、Stage 2/3/4 方法修改、完整 pipeline 重跑、paper two-stage 實作  
> 執行者：Claude Code  
> 任務完成後：停下來回報，不自動進入 roadmap 下一個 Phase

## 手機摘要

今天不是開始 7–12 天的 paper-spec 方法開發，而是把既有 watershed v3 結果變成明天可 defend 的凍結 baseline。  
必做：凍結 inputs/predictions、room/door Hungarian matching、regression tests、canonical JSON v0.1、同一 predictions 重算、四種視覺化與文件校正。  
Hole-aware polygonization 只可作獨立 stretch candidate，不能覆蓋明天 baseline；paper two-stage、outdoor window rays、direct doors、room-type 重寫全部延後。  
明天的核心結論仍是：粗粒度 room geometry 有訊號，精確邊界、types、doors、topology 尚不足。

---

## 1. 執行前必讀

依序完整閱讀：

1. `CLAUDE.md`
2. `docs/paper_alignment_audit.md`
3. `docs/annotated_floorplan_roadmap.md`
4. `docs/eval_2d_metric.md`
5. `PROGRESS.md` 最新三個日期段落

不得只讀本文件而忽略上述 provenance、資料與回報規則。

---

## 2. 對 Claude Code 回饋的正式判定

### 2.1 策略轉向：確認，且必須明說

目前主線已由：

> 完成 HouseLayout3D 全部 3D layout／Table 2–3 reproduction

轉為：

> 以 HouseLayout3D pipeline 為基礎，產生研究可用的 annotated 2D floorplan。

這是使用者研究目標驅動的正式 pivot，不是「3D reproduction 已完成」。因此：

- 不得宣稱完整重現 HouseLayout3D。
- 3D CDT/extrusion/depth/Table 3 被移出目前主線，並不代表它們已完成或不重要。
- `paper_spec_2d` 用來維持方法可比性。
- `annotated_best_2d` 用於下游研究品質。
- 兩者永遠分開命名、輸出與報告。

### 2.2 Paper-spec 可能較差：可作假設，不可先下結論

同意 `paper_spec_two_stage` 可能比 current watershed v3 差，但目前只能稱為待驗證 hypothesis。

不得在未完成 A/B 前寫：

> 論文方法在此資料上較差。

即使 A/B 後 paper-spec implementation 分數較低，正確結論也只能先是：

> 我們依 Appendix 規格重建的 two-stage implementation，在此 evaluator／資料上低於 watershed v3。

原因：

- HouseLayout3D Stage 4 code 是 0-byte，無官方 runtime 可核對。
- Appendix 的 two-stage 描述與 HOV-SG 原始 code 之間有實作解讀空間。
- current 早期 Voronoi/bottleneck 失敗不等同已實作忠實的 paper two-stage。
- 不能把 spec reimplementation 的失敗直接外推成原論文方法的侷限。

今天不實作也不執行此 A/B。

### 2.3 Room types：同意延後並設 go/no-go

Room+type F1 目前約 0.15，確實是 annotated floorplan blocker，但投資不確定性高。

Roadmap 順序維持：

1. room geometry／canonical artifact；
2. doors/windows/stairs；
3. room type feasibility；
4. 完整 OpenSeg per-mesh-vertex 投資需另過 checkpoint。

未來 room type 先做 1–2 天 feasibility：ontology crosswalk、feature/class-score inspection、少量 scene error analysis。若沒有可行訊號，再決定是否投入完整 3–6 天，不因「論文有寫」就無條件做到底。

今天不修改 room classification。

---

## 3. 今日唯一 Checkpoint

完成後應得到：

> 一組不可覆寫的 watershed v3 predictions，加上 deterministic evaluator v2、16 份 canonical annotated-floorplan v0.1 JSON、四類視覺化與一份可供明天報告的誠實能力結論。

今天不要求提高 Room F1。今天的成功判準是：

- 結果可追溯；
- evaluator 不依輸入排序；
- 同 artifact 永遠同分數；
- output 可被下游程式讀取；
- 報告不再混淆 paper-faithful 與 local variant。

---

## 4. 共通執行限制

### 4.1 禁止 GPU

- 所有執行設 `CUDA_VISIBLE_DEVICES=""` 或使用等價 CPU-only 保護。
- 不啟動 OneFormer、OpenSeg、DN-Splatter、Stage 2 skeleton、Stage 3 fit。
- 不執行 `scripts/run_pipeline_scene.sh` 或 `scripts/run_all_mp3d.sh`。
- 若某工具意外要求 GPU，立即停止該步驟，不得繞過限制。

### 4.2 不修改方法輸出

今日不得修改：

- `src/stage2/`
- `src/stage3/`
- `src/stage4/`
- `ROOM_CORE_MIN`
- door width rule
- windows rays
- room classification
- level/floorplan/room segmentation algorithm

### 4.3 不破壞既有資料

- 不刪除既有 outputs/logs。
- 不覆寫 frozen baseline。
- 先凍結，再改 evaluator。
- 產物使用新 versioned 路徑。
- 歷史報表／log 可保留舊數字，但 current-facing artifact 必須有 correction。

---

## 5. Task A — 凍結目前 baseline（第一優先）

### 5.1 建立目錄

建立：

```text
outputs/eval2d/baselines/watershed_v3_pre_report/
  gt/
  pred/
  scores_v1_greedy.pkl
  RESULTS_2D_v1_greedy.md
  visualizations_v1/
  manifest.json
```

複製目前 16-scene GT/pred PKL、scores、報告與既有圖。若複製 GT 太大，可用 immutable references＋SHA-256，但 16 個 GT 檔案都必須列入 manifest。

不得覆寫這個目錄；若已存在，先驗證內容／hash，不得直接重建。

### 5.2 保存 evaluator v1

workspace 沒有可依賴的 git history，因此修改前保存：

```text
src/eval2d/metrics_v1_greedy.py
```

內容必須是目前產出初版報告的 evaluator。檔頭註明：

- archived for reproducibility；
- room/door matching 是 greedy；
- 不作 current evaluator import target。

### 5.3 Manifest

`manifest.json` 至少包括：

- `baseline_id = watershed_v3_pre_report`
- `method = watershed_v3`
- `method_kind = local_engineering_variant`
- `stage = stage4a_pre_extrusion`
- `evaluator_version = eval2d_v1_greedy`
- 16 個 scene IDs
- 每個 input/output file path＋SHA-256
- creation timestamp
- coordinate units/metres
- raster resolution 0.05 m
- RDP tolerance 0.10 m
- room IoU threshold 0.5
- corner thresholds 0.1/0.2/0.3 m
- door thresholds 0.2/0.5 m
- GT room-set definition
- connectivity=`derived_not_annotated`
- known limitations：
  - prediction polygonization 只保留最大 external contour；
  - holes/multipolygons 尚未保留；
  - room types 低可信；
  - windows/stairs 尚未進 canonical 2D output；
  - topology GT derivation 信度有限。

### Task A 驗收

- scene IDs 恰為 16 棟；
- pred/GT count 都正確；
- manifest hashes 可重新驗證；
- baseline 目錄不會被後續命令當 output target。

---

## 6. Task B — Evaluator v2：Hungarian matching

修改 `src/eval2d/metrics.py`，版本標記為 `eval2d_v2_hungarian`。

### 6.1 Rooms

- 建完整 pairwise IoU matrix。
- 使用 deterministic one-to-one Hungarian assignment。
- 有效 match 必須 `IoU > 0.5`，依現有 spec 保持 strict greater-than。
- invalid pairs 使用 threshold-aware cost／dummy handling，不能因強迫 assignment 造成有效 pair 被低 IoU pair 排擠。
- 結果不得受 pred/GT list order 影響。
- 回傳 mapping 與 IoU 格式若修改，需同步所有 callers/tests。

### 6.2 Doors

- 保留目前 midpoint L2 distance metric，不在今天改 metric 定義。
- matching 改成 threshold-aware Hungarian。
- @0.2 m、@0.5 m 都必須 permutation invariant。
- 同一 pred/GT 不可重複配對。

### 6.3 Regression tests

新增：

```text
src/eval2d/tests/test_matching.py
```

至少測：

1. pred rooms 順序反轉，matching/PRF 不變；
2. GT rooms 順序反轉，matching/PRF 不變；
3. adversarial case：greedy 次優、Hungarian 得到最佳合法 matching；
4. IoU 低於、等於、高於 0.5；
5. empty GT；
6. empty pred；
7. both empty；
8. door pred/GT permutation invariance；
9. door threshold boundary；
10. concave valid polygon；
11. invalid polygon 有明確、可測 policy，不可 silent drop 或 nondeterministic repair。

測試必須可在 CPU geometry environment 執行。

### Task B 驗收

- tests 全 PASS；
- 同一 fixture 多次執行一致；
- 不 import archived `metrics_v1_greedy.py` 作 current evaluator。

---

## 7. Task C — 同一 frozen predictions 重算

### 7.1 輸入限制

只使用：

```text
outputs/eval2d/baselines/watershed_v3_pre_report/gt/
outputs/eval2d/baselines/watershed_v3_pre_report/pred/
```

不得：

- 呼叫 `identify_levels`；
- 呼叫 `segment_rooms`；
- 重新執行 `extract_pred.py`；
- 讀取會隨 source code 變動而重算 prediction 的路徑。

### 7.2 輸出

寫至新的 versioned 目錄，例如：

```text
outputs/eval2d/baselines/watershed_v3_pre_report/eval2d_v2_hungarian/
  scores.pkl
  comparison_v1_v2.json
  RESULTS_2D.md
```

對照至少包括：

- Room P/R/F1；
- mean matched IoU；
- Corner/Angle thresholds；
- Room+type；
- Doors@0.2/0.5；
- Tier C diagnostics；
- per-scene changes；
- micro/macro aggregation 定義與數值。

若 v1→v2 沒有數值變化，也要明寫 evaluator hardening 後結果一致；不可為顯示「有做事」而調整 threshold。

### Task C 驗收

- 16 scenes 全部成功；
- predictions hashes 與 Task A 完全相同；
- 數字變化唯一來源是 matching evaluator；
- comparison machine-readable 且有人類摘要。

---

## 8. Task D — Canonical annotated-floorplan v0.1

### 8.1 目的

將 frozen PKL 轉成下游可讀、versioned、不可依賴 pipeline runtime 的 JSON。今天不追求最終 schema，也不補造 frozen PKL 中不存在的資訊。

新增例如：

```text
src/eval2d/export_canonical.py
src/eval2d/canonical_io.py
src/eval2d/tests/test_canonical_roundtrip.py
```

輸入只可使用 frozen pred PKL；不得重新執行 segmentation。

### 8.2 輸出位置

```text
outputs/eval2d/canonical/watershed_v3_pre_report/<scene>.json
```

必須恰有 16 份 scene JSON，再加一份 collection manifest。

### 8.3 Minimum schema

```json
{
  "schema_version": "annotated_floorplan_v0.1",
  "scene_id": "...",
  "method": {
    "name": "watershed_v3",
    "kind": "local_engineering_variant",
    "source": "stage4a_pre_extrusion",
    "baseline_id": "watershed_v3_pre_report"
  },
  "coordinate_system": {
    "units": "metres",
    "frame": "mp3d_native",
    "resolution_m": 0.05
  },
  "limitations": {
    "holes_preserved": false,
    "multipolygons_preserved": false,
    "room_types_reliable": false,
    "connectivity": "derived",
    "windows_included": false,
    "stairs_included": false
  },
  "levels": [
    {
      "id": 0,
      "elevation": 0.0,
      "rooms": [],
      "doors": [],
      "windows": [],
      "stairs": [],
      "graph": {
        "nodes": [],
        "edges": [],
        "edge_status": "derived"
      }
    }
  ]
}
```

Rooms 至少包含：

- stable ID；
- Polygon geometry；
- raw predicted type；
- `type_confidence=null`（目前沒有可信 calibration 時）；
- provenance。

Doors 至少包含：

- ID；
- 2D segment；
- room A／room B 或 OUTSIDE（若 frozen data 可得）；
- provenance。

### 8.4 誠實輸出規則

- 不可捏造 windows、stairs、confidence、holes 或 room associations。
- frozen data 沒有就用空 list／null，並在 limitations 標記。
- JSON 不可含 NumPy scalar/array objects。
- 所有 coordinates 必須 finite 且為 metres。
- 不把 GT data 混入 prediction artifact。
- 不用 GT 修補 predicted geometry。

### 8.5 Canonical loader 與 score equivalence

Evaluator 增加讀 canonical JSON 的路徑，或提供 adapter 將 canonical JSON 轉成相同 scoring structures。

必須測：

- frozen PKL → JSON → loader 後 room/door/edge counts 相同；
- coordinates 在指定 tolerance 內相同；
- 用 PKL 與 canonical JSON 評估，v2 scores 完全相同或只在明確浮點 tolerance 內；
- scene order 不影響 aggregate。

### Task D 驗收

- 16 JSON＋collection manifest；
- round-trip/count tests PASS；
- PKL vs canonical score equivalence PASS；
- 沒有 import／執行 Stage 4 segmentation。

---

## 9. Task E — 報告視覺化與 current report

使用 frozen predictions／canonical JSON 產生四組視覺化：

1. best scene；
2. median scene；
3. worst scene；
4. 一個 multi-level scene。

每組至少顯示：

- GT rooms；
- predicted rooms；
- room IDs；
- raw predicted room types；
- doors；
- derived access edges；
- level ID；
- metric scale；
- scene ID；
- Room F1／mean matched IoU。

Caption／圖內必須標：

- `preliminary annotated-floorplan baseline`；
- `watershed_v3 local engineering variant`；
- connectivity derived；
- room types currently unreliable。

更新 report generator，再產生：

- `RESULTS_2D.md`
- `docs/eval2d_report.html`

不要只手改 generated HTML。

明日 headline 應以 v2 hardened evaluator 的實際重算結果為準，不預先硬編 0.61/0.79；若數字改變，報告 v1→v2 原因。

---

## 10. Task F — 文件校正

### 10.1 Stairs current-facing overclaim

3D stairs evaluator 已修正並得到 0.411；今天不重寫 evaluator，只清 current-facing artifacts。

搜尋：

- `stairs 達標`
- `stairs 超過論文`
- `0.473`
- `strict reproduction`
- current watershed 被稱為 HOV-SG two-stage

規則：

- 歷史 log／舊實驗段落保留。
- current-facing HTML/report 加 correction 或重新產生。
- current stairs：0.411，略低於 paper 0.42，非達標。
- current project：research reimplementation／annotated-floorplan pivot。

`docs/reproduction_review.html` 若有 generator，改 generator 後重建；若找不到 generator，不得假裝 HTML 是可重現 source，至少加醒目 correction banner 並在回報註明。

### 10.2 PROGRESS

只追加本輪 checkpoint，包含：

- frozen baseline ID；
- evaluator v2 change；
- v1→v2 指標；
- canonical JSON count；
- tests；
- 明天可／不可宣稱；
- 未進行任何 GPU／pipeline method rerun。

不重寫歷史段落。

---

## 11. Stretch Task — Hole-aware polygonization（非必做）

Claude 建議 hole-aware polygonization，方向正確，但它會改變 prediction geometry，因此和「明天使用 frozen baseline」不能混在同一結果中。

只有 Task A–F 全部完成且至少還有 60–90 分鐘時，才可開始。

### 允許內容

- 新增獨立 `polygonize_mask_hierarchy` utility；
- 使用 `RETR_CCOMP`／`RETR_TREE` 或等價方法保留 holes；
- 支援 MultiPolygon／disconnected components；
- Shapely validity repair policy 明確；
- 加 synthetic donut、two-island、concave、tiny-hole tests；
- 不修改 `src/stage4/`；
- 不覆寫 frozen pred／canonical v0.1。

### 輸出命名

只能命名為 candidate，例如：

```text
annotated_floorplan_v0.2_hole_candidate
```

### 禁止

- 不把 candidate 數字放進明天 headline；
- 不為提高 Corner/IoU 調 RDP；
- 不重跑 Stage 4 methods；
- 不把 candidate 標 `FIXED AND TESTED`，除非 round-trip、geometry tests、16-scene CPU extraction 與視覺檢查全部完成。

若時間不足，記為下一輪第一項，不做半成品整合。

---

## 12. 今日明確禁止事項

本輪不得做：

- `paper_spec_two_stage`；
- 2.5 m／1.5 m room segmentation；
- `hovsg_code_port`；
- 修改 watershed／`ROOM_CORE_MIN`；
- paper door rule；
- direct/fused door detection；
- outdoor window rays；
- OpenSeg per-mesh-vertex；
- room-type ontology／classifier 重寫；
- Stage 3 initializer/state adapter；
- Stage 2 sampling/OneFormer/SPT；
- 3D extrusion；
- GPU/full pipeline run；
- threshold tuning；
- 刪除、覆寫既有 outputs。

Roadmap §13 的 7–12 天區段要在明日報告後另行啟動；今天不是該區段的縮時版。

---

## 13. Definition of Done

只有以下全部滿足才算今日 checkpoint 完成：

1. frozen baseline directory 建立且 hash 可驗證；
2. 恰有 16 GT／pred scenes；
3. `metrics_v1_greedy.py` 已保存；
4. room/door Hungarian matching 完成；
5. permutation、threshold、adversarial、empty regression tests PASS；
6. 使用完全相同 frozen predictions 完成 v1→v2 重算；
7. comparison 包含 per-scene 與 aggregate；
8. canonical v0.1 JSON 恰有 16 份；
9. canonical round-trip/count/finite-coordinate tests PASS；
10. PKL vs canonical scores equivalent；
11. best/median/worst/multi-level 四類視覺化完成；
12. current `RESULTS_2D.md`／HTML 使用 v2 結果並標 preliminary；
13. current-facing stairs/reproduction overclaims 已校正；
14. `PROGRESS.md` 追加 checkpoint；
15. 無 GPU command、無 Stage 2/3/4 method 修改、無 full pipeline rerun；
16. 完成後停止，不開始 paper two-stage 或下一 Phase。

若 4–6 小時內無法全部完成，優先順序：

1. Task A baseline freeze；
2. Task B evaluator＋tests；
3. Task C frozen re-evaluation；
4. Task D canonical artifact；
5. Task E report／四圖；
6. Task F 全面文件清理；
7. Stretch 一律最後。

至少 Task A–C 完成才可把新 evaluator 數字用於明日報告。若 Task D 未完成，明天只能展示 preliminary evaluation，不能宣稱已有 canonical downstream artifact。

---

## 14. 完成後回報格式

先給 3–6 行手機摘要，再依序列：

1. 是否在 4–6 小時範圍完成；
2. 修改／新增檔案；
3. tests 與 commands（確認 CPU-only）；
4. frozen baseline path＋hash verification；
5. v1 greedy → v2 Hungarian 指標差異；
6. 16-scene success/failure；
7. canonical JSON counts 與 round-trip result；
8. 四組圖路徑；
9. current reports 路徑；
10. 今天沒有做的項目；
11. 明天可以宣稱什麼；
12. 明天不可以宣稱什麼；
13. 下一個建議 checkpoint，但不要自行開始。

若遇到 evaluator 定義、schema 或歷史 artifact 無法判定的問題，先採可逆且明確標記的保守方案；若會改變研究定義，停下來詢問使用者。
