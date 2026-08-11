# HouseLayout3D Research Reimplementation

這個 repository 以 HouseLayout3D／MULTIFLOOR3D 為核心，整合作者公開的
Stage 2/3 code interface，並自行重建與研究 Stage 4、3D evaluation 與 annotated
2D floorplan evaluation。

目前專案應定位為 **research reimplementation**，不是 strict paper reproduction。
作者釋出的 Stage 4 runtime code 不完整，因此本地方法包含明確標記的 engineering
variants。

## Current status

- 2D floorplan current protocol：`eval2d_v3_strict_levels`，使用官方 Matterport3D
  `.house` floor polygons 作為 room geometry GT。
- Frozen 16-scene watershed baseline：Room F1 0.602；成功 matched rooms 的
  conditional mean IoU 0.785。
- 3D watershed v3：Structures F1@0.5 0.217、Doors 0.210、Windows 0.271、
  corrected Stairs 0.411；Δ5/Δ10 為 45.2/57.9。
- 詳細限制、provenance 與不能宣稱的內容均記錄於下列文件。

## Key documents

- [`RESULTS_2D.md`](RESULTS_2D.md)：current 2D headline、GT/evaluator A/B。
- [`docs/tomorrow_2d_floorplan_report.md`](docs/tomorrow_2d_floorplan_report.md)：
  2D 報告提綱與視覺化 checklist。
- [`docs/paper_alignment_audit.md`](docs/paper_alignment_audit.md)：paper/code alignment audit。
- [`docs/annotated_floorplan_roadmap.md`](docs/annotated_floorplan_roadmap.md)：
  annotated floorplan roadmap。
- [`docs/eval_calibration_report.md`](docs/eval_calibration_report.md)：
  以作者 predictions 校準 3D evaluator。
- [`PROGRESS.md`](PROGRESS.md)：歷次實驗與修正紀錄。

## Repository scope

本 repository 追蹤 source code、scripts、setup scripts、研究文件與小型報告。
MP3D data、conda environments、model checkpoints、downloaded third-party repositories、
cache、logs 與 generated outputs 不納入 Git；它們必須依各自授權與文件另行準備。

主要程式位於：

- `src/stage2/`、`src/stage3/`、`src/stage4/`
- `src/eval/`：3D evaluator
- `src/eval2d/`：2D evaluator、official `.house` GT adapter 與報告工具
- `src/baselines/`：RoomFormer 2D/3D adapters
- `scripts/`：pipeline runners

請勿把本 repository 的 preliminary 結果描述成已完整重現論文或已足以作為
downstream annotated floorplan ground truth。
