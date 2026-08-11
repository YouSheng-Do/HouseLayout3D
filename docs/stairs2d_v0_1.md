# Stairs 2D footprint checkpoint v0.1

日期：2026-08-12

Checkpoint：`phase3_stairs2d_v0_1`

完整 generated report：
`outputs/eval2d/experiments/stairs2d_v0_1/all/annotated_best/report/RESULTS.md`

## 結論

existing Stage-4 D.5 outputs 轉成 12 個 canonical stair predictions；released GT 的
34 個 stair mesh entities 全數以 exact XY triangle union 投影，沒有丟棄非矩形或先做
fragment grouping。strict level alignment＋IoU>0.5 Hungarian 結果：

| split | Pred | GT | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| dev | 5 | 15 | 0.600 | 0.200 | 0.300 |
| held-out | 7 | 19 | 0.286 | 0.105 | 0.154 |
| all | **12** | **34** | **0.417** | **0.147** | **0.217** |

all TP/FP/FN = 5/7/29。這個 0.217 同時受到真 miss 與 representation mismatch
影響：D.5 用 0.4 m nearby-component merge 合併整座樓梯，released GT 常把同一物理
樓梯拆成 flights/landings。即使 12 個 predictions 全命中，34-entity recall 上限也只有
0.353；另有 5 個含 GT stairs 的 scenes 完全沒有 prediction。

因此不可把 0.217 直接翻譯成「annotated floorplan 的樓梯只有 21.7% 可用」。若要評估
physical staircase，需先建立獨立人工 grouping/link gold subset；不能事後沿用 prediction
的 0.4 m merge rule 合併 GT 來提高分數。

## Canonical links

12/12 predictions 都保存 direct D.5 `from_level/to_level` 與 ordered adjacent rooms，
reference/graph checks 全通過；e9zR 有 1 個 link 跨非相鄰 predicted level index，已列入
audit。這只能宣稱 prediction link coverage 100%，不能宣稱 link accuracy。

正式 Stair-link metric 為 **N/A**。released meshes 沒有官方 level/room relation，且本
checkpoint 不從 z 猜 GT `to_level`。footprint owning floor 使用固定規則
`highest official floor <= min(mesh_z)+0.15m`，這只是 2D entity placement。

## Contract 與驗證

- Stage-4 rectangle raw order 是 endpoint edges `01/23`；canonical footprint 固定轉成
  perimeter `0-1-3-2`，避免 self-intersecting bow-tie。
- 0.4 m merge、0.3 m² minimum area、0.4 m minimum rise、0.5 m room assignment
  rejection 都標為 local best-variant assumptions。
- 2D Stair F1 0.217 與舊 3D Stairs F1 0.411 是不同 metric，不可互比或混寫。
- stair geometry/link contract 13/13、formal artifacts 15/15 PASS。
- 全程 CPU-only，只轉換既有 Stage-4 JSON；未重跑 detection、Stage 2/3、pipeline、
  extrusion 或 GPU。

Artifacts：

- `outputs/eval2d/experiments/stairs2d_v0_1/all/annotated_best/`
- `outputs/eval2d/experiments/stairs2d_v0_1/all/annotated_best/eval2d_stairs_v0_1_footprint_iou_hungarian_all/`
- `outputs/eval2d/experiments/stairs2d_v0_1/all/annotated_best/report/`
