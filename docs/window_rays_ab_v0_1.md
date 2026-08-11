# Window rays：legacy three classes vs paper + outdoor

日期：2026-08-12

Checkpoint：`phase3_window_rays_ab_v0_1`

完整 generated report：
`outputs/eval2d/experiments/window_rays_ab_v0_1/all/comparison/RESULTS.md`

## 結論

依 Appendix class list 加入 `outdoor` rays 後，recall 不變，但 false positives
增加，因此 2D Windows F1 下降：

| split / variant | Pred | Precision | Recall | F1@0.5 |
|---|---:|---:|---:|---:|
| dev / legacy three classes | 326 | 0.242 | 0.420 | 0.307 |
| dev / +outdoor | 366 | 0.216 | 0.420 | 0.285 |
| held-out / legacy three classes | 311 | 0.289 | 0.471 | 0.359 |
| held-out / +outdoor | 343 | 0.262 | 0.471 | 0.337 |
| all / legacy three classes | **637** | **0.265** | **0.446** | **0.333** |
| all / +outdoor | **709** | **0.238** | **0.446** | **0.311** |

加入 outdoor 後增加 0 TP、72 FP、FN 不變。paired scenes 中 paper class set
較好 1/16、較差 13/16、平手 2/16；dev 與 held-out 方向一致。

這是 paper alignment 與 annotated-best 分開管理的案例：

- `paper_plus_outdoor` 保留為 paper-spec branch；
- research best branch 暫留 `legacy_three_classes`；
- 若後續替 outdoor rays 加 geometry/semantic gating，必須標為 local extension。

## Annotation 與 evaluator contract

- ray classes control：`window`、`window_blind`、`curtain`。
- paper classes：上述三類加 `outdoor`。
- 每棟 Stage-3 prototype 只 load 一次；兩組使用相同 ray/input hashes。
- window canonical record 保存 2D wall interval、direct Stage-3 wall PID、
  nearest-room candidate、confidence 與 ray-class evidence sidecar。
- GT 使用 released `external/houselayout3d/data/windows/*.json` rectangle 的
  bottom edge 投影成 2D segment。
- evaluator 使用 strict level alignment 與 orientation-invariant segment
  endpoint-max Hungarian matching，threshold 為 0.2/0.5 m。中心位置相同但長度或
  方向錯誤的 segment 不算命中。
- 生成器不讀 GT/score，且沒有 Stage 2/3 rerun、room segmentation、extrusion 或 GPU。

## Faithfulness boundary

加入 `outdoor` 符合論文 window-ray class list；但 DBSCAN、LOF、ray-length slack、
nearest-room association 都仍是本地 assumptions。此 2D Windows metric 也是專案新增
的 diagnostic，不是論文 Table 2 的 3D rectangle metric。

目前只能宣稱：「在現有 local clustering 下，outdoor rays 新增的 false positives
多於 true positives。」不能宣稱 outdoor rays 本身沒有價值，也不能把三類 control
稱作 paper-faithful。

## 驗證與 artifacts

- window record geometry/provenance：7/7 PASS。
- released rectangle projection與 strict-level identity：5/5 PASS。
- formal 32-artifact schema/hash/reference checkpoint：10/10 PASS。

舊 `eval2d_windows_v0_1_midpoint_hungarian` 的 all F1 0.386／0.368 因違反
預註冊的 endpoint contract，已撤回且不得引用；本文件只報 v0.2 正式數字。

Artifacts：

- `outputs/eval2d/experiments/window_rays_ab_v0_1/all/legacy_three_classes/`
- `outputs/eval2d/experiments/window_rays_ab_v0_1/all/paper_plus_outdoor/`
- `outputs/eval2d/experiments/window_rays_ab_v0_1/all/comparison/`
