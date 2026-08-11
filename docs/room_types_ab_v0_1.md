# Room types：direct samples vs mesh-vertex k=5 A/B

日期：2026-08-12

Checkpoint：`phase3_room_types_ab_v0_1`

完整 generated report：
`outputs/eval2d/experiments/room_types_ab_v0_1/all/comparison/RESULTS.md`

## 結論

以相同的 316 個 frozen `watershed_v3` rooms、retained OpenSeg samples、15 組 CLIP
text embeddings 與 cosine argmax 做 same-base A/B：

| split / aggregation | Room+type F1 | conditional Top-1 | Top-3 | matched+mappable n |
|---|---:|---:|---:|---:|
| dev / direct samples | 0.189 | 0.438 | 0.671 | 73 |
| dev / mesh vertices k=5 | 0.153 | 0.356 | 0.589 | 73 |
| held-out / direct samples | 0.185 | 0.424 | 0.652 | 66 |
| held-out / mesh vertices k=5 | 0.179 | 0.409 | 0.606 | 66 |
| all / direct samples | **0.187** | **0.432** | **0.662** | 139 |
| all / mesh vertices k=5 | **0.165** | **0.381** | **0.597** | 139 |

兩法 room geometry 完全相同，Room F1 都是 0.599（TP/FP/FN=192/124/133）。
paired scenes 中 vertex 較好 1/16、較差 5/16、平手 10/16；paper-spec aggregation
沒有改善品質。因此：

- annotated-best 暫採 `sample_point_room_mean`；
- paper branch 保留 `mesh_vertex_k5_room_mean` 作 alignment/ablation；
- 兩者皆標 `room_types_reliable=false`；
- 不依 outdoor class 或低 confidence 結果刪除 room geometry。

## Metric 與 ontology boundary

GT 325 rooms 中只有 221（68.0%）可用保守 crosswalk 對應到 paper 15 classes；
hallway、dining、closet、laundry、utility、bar、rec 等沒有直接對應者保留為 unmappable，
不為提高分數任意合併。

- end-to-end Room+type F1：strict level alignment、room IoU>0.5 Hungarian；geometry
  miss、type 錯誤與 unmappable ontology 都會降低分數。
- conditional Top-1/Top-3：只在 139 個 geometry-matched 且 GT crosswalk 非 null 的
  rooms 上計算；這是 classifier signal，不是完整 annotated-floorplan 成功率。

舊 Room+type F1 0.144 使用不同 artifact/evaluator ontology contract，不能直接用
0.187−0.144 宣稱改善。本 checkpoint 唯一嚴格 A/B 是兩個 aggregation 彼此比較。

## Paper alignment boundary

Appendix D.4 指定：OpenSeg pixel-aligned features 依 mesh segmentation 流程投影到 mesh
vertices，對每個 room 所含 vertices 平均，再以 CLIP 分成 15 classes。本 checkpoint 的
`mesh_vertex_k5_room_mean` 使用 retained samples → structural mesh vertices k=5 KNN →
canonical room mean，並保存完整 15 cosine scores。

作者 Stage-4 implementation 未釋出，因此 retained sampling 與 vertex-to-room association
仍是 reproducible local interpretation，不稱 author-code reproduction。direct sample 路徑
也不是 archived Stage-4 BEV classifier 的逐字重跑，而是同 base/feature 的明確 control。

paper last-five outdoor leaf rule只記錄 candidate：direct sample 45 rooms、vertex 41 rooms；
全部保留。direct sample predicted types 集中在 bedroom 126/316、entrance 107/316，
median Top-1/Top-2 cosine margin 只有 0.0137，顯示目前 classification confidence 很低。

## 驗證與 artifacts

- 16 scenes × 2 variants canonical v0.2 schema/hash/lineage checks 17/17 PASS。
- 316/316 rooms、geometry、doors、windows、stairs、graph 均未因 type augmentation 改變。
- classified rooms 保存完整 15-class finite scores；direct sample 有 1 個 no-feature
  `unknown`，mesh-vertex 316/316 有 features。
- generator diagnostics 明確記 `gt_accessed=false`。
- 全程 CPU-only；未重跑 OpenSeg/CLIP inference、Stage 2/3/4、room segmentation、
  pipeline 或 GPU。

Artifacts：

- `outputs/eval2d/experiments/room_types_ab_v0_1/all/sample_point_room_mean/`
- `outputs/eval2d/experiments/room_types_ab_v0_1/all/mesh_vertex_k5_room_mean/`
- `outputs/eval2d/experiments/room_types_ab_v0_1/all/comparison/`
