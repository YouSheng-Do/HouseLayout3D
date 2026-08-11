# 官方 repo 盤點（A 組 checklist 第 1 項證據）

Clone 自 `github.com/HouseLayout3D/houselayout3d` @ `429d15b`（2026-07-06），共 12 commits。

## 有什麼

| 檔案 | 行數 | 內容 |
|---|---|---|
| `README.md` | 32 | 安裝（conda + pip + git-lfs）與視覺化教學 |
| `main.py` | 10 | 用 HF `datasets` 載入所有 config（doors/poses/stairs/structures/windows） |
| `visualize.py` | 80 | pyviz3d 渲染標註（doors/windows 折線、stairs/structures mesh），預設場景 `1LXtFkjw3qL` |
| `requirements.txt` | 3 行 4 套件 | `datasets`、`pyviz3d`、`typer`、`numpy`（皆未鎖版本） |

## 沒有什麼（全部確認不存在）

- ❌ 四階段 pipeline 程式碼（mesh reconstruction / skeleton / prototype fitting / scene graph）
- ❌ 評估腳本（d_E、d_H、Δτ 皆無）
- ❌ config、conda yml、Dockerfile、任何 torch/nerfstudio/COLMAP 依賴
- ❌ pretrained weights（方法 training-free，無權重可釋出）
- ❌ MP3D 前處理／COLMAP 設定等 issue #2 詢問的細節

→ GitHub 端確認無 pipeline code。（**後續發現**：NeurIPS supplementary zip 內含官方 Stage 2/3 程式碼與 16 場景官方預測——見 `supplementary_inventory.md`；策略已改為「官方 code 為主幹」，本檔僅記錄 GitHub repo 現況。）

## 資料集（HuggingFace `houselayout3d/HouseLayout3D`，MIT，已下載 2026-07-06）

- 71 MB、4,785 個檔案、**16 個場景**（MP3D building id 命名，如 `1LXtFkjw3qL`）
- `doors/{scene}.json`：`{"doors": [{vertices: 4×3, normal: 3}, ...]}`（如 1LXtFkjw3qL 有 24 個門）
- `windows/{scene}.json`：同 doors 格式（vertices 4 角）
- `poses/{scene}.json`：nerfstudio 格式（`camera_model: "OPENCV"`、`frames[]`、`depth_unit_scale_factor`）— 深度評估用
- `stairs/{scene}/*.ply`：34 個樓梯 mesh
- `structures/{scene}.obj`＋`structures/layouts_split_by_entity/{scene}/*.ply`：整體 layout mesh 與逐實體（wall/floor/ceiling）切分（共 ~4,700 檔）
- ⚠️ 只有標註＋位姿；**MP3D 原始 RGB-D／紋理 mesh 需另行取得**（EULA 審核中）

官方 `visualize.py` 已實測可跑：`scene_1LXtFkjw3qL/`（393 個實體、2.3 MB）輸出正常。
