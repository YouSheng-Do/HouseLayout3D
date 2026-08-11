# Direct semantic doors：watershed vs semantic vs fused A/B

日期：2026-08-12

Checkpoint：`phase3_direct_doors_ab_v0_1`

完整 generated report：
`outputs/eval2d/experiments/direct_doors_ab_v0_1/all/comparison/RESULTS.md`

## 結論

使用 retained OneFormer `door` ray endpoints 做 wall-snapped candidates，與 frozen
watershed doors 融合後，Doors@0.5 在 dev 與 held-out 都穩定改善：

| split / variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| dev / watershed | 0.403 | 0.182 | 0.251 |
| dev / semantic only | 0.397 | 0.182 | 0.250 |
| dev / fused | 0.362 | 0.307 | **0.332** |
| held-out / watershed | 0.439 | 0.161 | 0.236 |
| held-out / semantic only | 0.404 | 0.135 | 0.203 |
| held-out / fused | 0.392 | 0.258 | **0.311** |
| all / watershed | 0.420 | 0.171 | 0.243 |
| all / semantic only | 0.400 | 0.158 | 0.226 |
| all / fused | **0.376** | **0.281** | **0.322** |

all-16 control TP/FP/FN=50/69/242，fused=82/136/210。99 個 novel semantic
candidates 帶來 +32 TP、+67 FP；paired scenes 是 11 better／4 worse／1 tie。
semantic-only 沒有優於 control，因此兩個 evidence source 是互補，不應互相取代。

決策：annotated-best doors 改採 `fused_union`；paper branch 仍保留 D.3 two-stage
bottleneck doors。本 fusion 是 local research extension，不是 paper method。

## Frozen contract 與 data separation

在存取本 checkpoint held-out 前，只用 dev 固定以下 geometry rules：

- valid OneFormer door-labeled ray endpoints；
- 0.10 m voxel；point-to-wall ≤0.30 m；cluster mean wall distance ≤0.15 m；
- along-wall gap 0.35 m；support ≥100 voxels；vertical span ≥1.4 m；
- candidate width 0.35–1.5 m；semantic/base midpoint <0.5 m 去重。

正式 checkpoint 才一次性執行 held-out/all；之後沒有依 held-out 調 threshold。16 scenes
共產生 115 semantic candidates，94 個能由 predicted room polygons 得到 derived room/outside
association；fusion 加入 99 個、輸出 218 doors。48 artifacts 的 schema/hash/lineage/
references checks 14/14 PASS。

generator 只讀 frozen canonical artifacts 與 retained
`full_ray_dests.npy`、OneFormer hard labels、valid-depth masks；不讀 GT/score。全程
CPU-only，未重跑 OneFormer、Stage 2/3/4、pipeline、extrusion 或 GPU。

## Topology 與 visual audit

使用同一個 derived-edge diagnostic 時，edge_all F1 由 0.125→0.184。這只是方向性
signal：GT/pred edge associations 都由 room polygons＋door probes 推導，不是 independent
topology ground truth，不能稱正式 accuracy。

四個 reporting cases 的 overlay 顯示 candidates 通常沿 predicted wall，且能補到多個
GT door centers；5LpN、JeFG 等案例也有遠離 GT 的 false positives。這和 aggregate
precision 0.376 一致，故只能宣稱「比 control 改善」，不能說 doors 已可靠。

目前 metric 是 strict-level segment-midpoint Hungarian，沒有評 width/orientation；研究若
需要通行寬度或門向，需新增 endpoint/width metric 與人工抽查，不能沿用 0.322 當品質證明。

## Artifacts

- `outputs/eval2d/experiments/direct_doors_ab_v0_1/all/watershed_control/`
- `outputs/eval2d/experiments/direct_doors_ab_v0_1/all/direct_semantic_only/`
- `outputs/eval2d/experiments/direct_doors_ab_v0_1/all/fused_union/`
- `outputs/eval2d/experiments/direct_doors_ab_v0_1/all/comparison/`
