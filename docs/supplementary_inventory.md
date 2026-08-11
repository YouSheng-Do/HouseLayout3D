# Supplementary 內容盤點（2026-07-06，來源：OpenReview zip，使用者手動下載）

位置：`docs/supplementary/supplementary/`，427 檔、88 MB。

## 有什麼

| 項目 | 內容 | 對應 |
|---|---|---|
| `Appendix_A.pdf` | 論文附錄 5 頁（p.21–25）：**Table 5 COCO→四類完整映射**、Algorithm 1 RANSAC 初始化、pytorch3d 多邊形實作/頂點共享、Stage 4 全演算法規格（D.1–D.6）、baseline 適配（E） | 消掉大部分未定參數 |
| `multi-floor-3d-code/extract_skeleton.py`（486 行） | **Stage 2 主體**：讀 poses json＋mesh＋預先算好的 OneFormer 標籤 PNG → 反投影（每幀 3000 rays、MP3D depth ×0.00025）→ per-vertex KNN(k=5) 標籤聚合 → superpoint 聚合（level 1–3）→ 輸出 skeleton/objects/stairs mesh＋ray origins/dests | 可直接改用 |
| `multi-floor-3d-code/fit_prototype.py`（1840 行）＋`mesh_fitting_3D/`（~5.5k 行） | **Stage 3 完整實作**：可微多邊形集合、CGAL CDT、RANSAC 初始化、合併/簡化、ray-tracing L_empty；`polygon_fitting_config.py` 含全部超參（含 MatterportConfig） | 可直接改用 |
| `predictions-ours/{16 scenes}/combined.ply` | **官方最終預測 layout**（Open3D ply，帶頂點色） | eval 開發的參考答案；可先驗證評估腳本再跑 pipeline |
| `fitting_visualization.gif` | Stage 3 優化過程視覺化 | 對照用 |
| vendored `OneFormer/`（原版 SHI-Labs repo）與 `superpoint_transformer/`（閹割版） | 環境參考 | 見缺件 |

## 缺什麼（zip 內 0 bytes 或不存在）

| 缺件 | 影響 | 對策 |
|---|---|---|
| `create_scene_graph.py`、`scene_graph_parsing/floor_extrusion.py`、`free_space_test.py`（皆 0 bytes） | **Stage 4 無程式碼** | 依 Appendix D.1–D.6 規格自行實作（規格非常具體） |
| `OneFormer/house_layout_inference.py`（0 bytes）＋`labels.txt` 未附 | Stage 2 的 2D 分割輸入（uint8 標籤 PNG＋類別表）要自產 | 用已跑通的 HF OneFormer＋Table 5 映射自寫 wrapper；checkpoint 型號未載明（假設 COCO panoptic swin-large，之後對照官方預測驗證） |
| vendored SPT 無 `scripts/preprocess_point_cloud.py`（`segment_point_cloud_superpoints` 本體）；src/ 缺 transforms/data/utils | superpoint 分割入口缺 | 用 upstream SPT 重寫該函式（輸出 level_{1,2,3}_segmentation.npy 介面，依 extract_skeleton.py 的用法） |
| 評估腳本（d_E/d_H/Δτ） | 無法直接算 F1 | 自寫；**用 predictions-ours + HF GT 標註先驗證 eval 正確性（對照 Table 2）**（✅ 2026-07-07 完成，見 eval_calibration_report.md） |
| **polygon 初始化（Algorithm 1）**：無任何檔案產出 fit_prototype 必需的 `polygon_info.json` 與 `clean_edge_mesh.ply`（2026-07-07 確認） | Stage 2→3 的橋斷了 | 依 Appendix C.1（循序 RANSAC）自寫；schema 從 `from_polygon_info()` 反讀：每 polygon {contours(頂點索引環,含洞), plane_eq, color, class, shared_edges, vertices} |
| Stage 1（mesh reconstruction）程式 | 本來就走 DN-Splatter 外部流程／MP3D mesh | 不變 |

## 其他關鍵事實

- README 確認 benchmark 輸入＝`matterport/v1/scans/{scene}/poisson_meshes/{scene}_10.ply`＋OpenNeRF 式 nerfstudio `transforms.json`（含 per-frame `depth_file_path`）→ issue #2 的輸入格式之謎解掉。
- 16 場景清單寫死在 extract_skeleton.py（＝HF 資料集的 16 場景）。
- 官方環境：`oneformer`（detectron2 版）＋`layout-estimation`；root `install.sh` 打包混亂（引用不存在的 requirements.txt），layout env 依賴需從 import 推：torch、open3d（tensor raycasting）、**CGAL SWIG bindings**、shapely、rdp、sklearn、cv2、matplotlib；pytorch3d **必需**（大多處有 fallback，但 `get_vertex_knn_indices`（differentiable_3D_polygon_stuctures.py:1450）硬 import；用官方 wheel `py310_cu118_pyt210` 的 0.7.5 配 torch 2.1.2 實測可用）。
- 論文寫 M=5000，code 預設 `--samples-per-frame 3000` → 以 config 參數化，跑量化時再對。
- Appendix D.4：房型分類/室外剪枝用 **OpenSeg＋CLIP**（正文未明說的額外依賴）。
