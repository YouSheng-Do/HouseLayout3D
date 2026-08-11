# MULTIFLOOR3D 參數清單 v1（**已棄用**，見 `param_inventory_v2.md`）

> ⚠️ 2026-07-06：supplementary 到手後本表被 v2 取代（Stage 3 參數已有官方值）。保留僅供審計對照。



> 來源：論文正文（12 頁版，arXiv:2512.02450 v1 ＝ 專案頁 PDF ＝ 本地 PDF，皆無附錄）。
> 附錄（OpenReview supplementary zip）被人機驗證擋住，**需要使用者手動下載**（見文末）。
> 標記：✅=正文有明確值｜📎=附錄待查｜🔧=第三方工具預設值起步
> 之後寫 pipeline 時，本表所有項目集中進 `configs/default.yaml`；📎/🔧 項標 `# TODO: tune with MP3D`。

## 0. 影響架構的事實（非參數，但決定實作）

| 事實 | 出處 |
|---|---|
| Table 2（HouseLayout3D benchmark）輸入＝**MP3D 官方 camera poses + RGB-D + mesh**（Stage 1 DN-Splatter 被繞過） | §5 Results on HouseLayout3D |
| ScanNet++ 實驗才用 DN-Splatter mesh 當所有方法的輸入 | §5 Results on Scannet++ |
| baselines（RoomFormer/SceneScript）靠 MP3D GT floor/room segmentation 切 per-floor/per-room；MULTIFLOOR3D 不用此特權資訊 | §5 |
| Ablation 的 mesh→polygon 轉換：QSlim 簡化（Table 4 顯示目標 ~2000 頂點）＋貪婪合併法向差 <20° 的相鄰三角形 | §5 Analysis |

## 1. Stage 1 — Mesh Reconstruction

| 參數 | 狀態 | 值／建議 |
|---|---|---|
| 深度模型 | ✅ 型號／📎 變體 | Metric3D；變體（vit_small/large/giant2）附錄待查，建議 **vit_large (v2)** |
| DN-Splatter 訓練設定（iters、depth/normal loss 權重與開關） | 📎 | 建議先用 repo 預設＋mono depth 監督 |
| Poisson 重建參數（depth 等） | 📎 | 建議 dn-splatter 預設流程 |
| COLMAP 設定 | 🔧 | 標準 SfM 預設；僅替代場景／自拍影片需要 |

## 2. Stage 2 — Layout Skeleton Extraction

| 參數 | 狀態 | 值／建議 |
|---|---|---|
| 每張影像反投影取樣像素數 M | ✅ | **M = 5000**（隨機取樣，含預測類別，投到最近 mesh vertex 累積票） |
| OneFormer checkpoint | 📎 | 正文僅說輸出類別映射自 COCO [36]；建議 `shi-labs/oneformer_coco_swin_large` |
| COCO →四類映射表 | 📎 | 附錄待查；查無則我草擬映射表供確認（注意：structural 含 closets 等大型家具；windows+mirrors=幾何不準表面；stairs 獨立） |
| 反投影「最近 vertex」距離上限 | 📎 | 正文未提是否有 cutoff；建議加上限（如 10 cm）防止穿牆票 |
| Superpoint 分群參數（voxel、k-NN、cut-pursuit 正則強度） | 📎→🔧 | 依 Robert et al. [37] 前處理；先用 SPT repo 預設 |

## 3. Stage 3 — Layout Prototype Fitting

| 參數 | 狀態 | 值／建議 |
|---|---|---|
| Loss 總式 | ✅ | L = L_geom + L_connect + L_simple；L_geom = L_prox + L_empty（式 1–5） |
| Loss 相對權重 | 📎 | 正文式子無權重（可能等權）；附錄待查 |
| τ_inter（L_empty 忽略離 polygon 邊界過遠的交點） | 📎 | 附錄待查 |
| L_connect 忽略門檻（D_pp 過大不計） | 📎 | 附錄待查 |
| L 線段集合的取樣規模（每影格幾條／總數） | 📎 | 附錄待查 |
| polygon 初始化（每 superpoint 擬合 1+ 平面的方法、RANSAC 與否） | 📎 | 「detailed in the supplementary」 |
| 頂點共面約束與共享頂點實作 | 📎 | 「detailed in the supplementary」 |
| optimizer / learning rate / iteration 數 / vertex merging 週期 | 📎 | 附錄待查；建議 Adam |
| τ_merge（頂點成對合併距離；**同值**用於 RDP tolerance） | 📎 | 附錄待查 |
| 相近同法向 polygon 合併門檻（法向角、D_pp 距離）＋「L_prox 不得增加太多」容忍度 | 📎 | 附錄待查 |
| τ_extend（牆/天花板延伸：每 cm² 線段交會數低於此值才延伸） | 📎 | 附錄待查 |
| 補地板洞 | ✅ 邏輯 | object mesh 三角形投影到「質心在其下方的最近 floor polygon」平面，取聯集重算 floor polygon |

## 4. Stage 4 — Scene Graph

| 參數 | 狀態 | 值／建議 |
|---|---|---|
| 樓層識別：相近高度 floor polygon 合併門檻 | 📎 | 附錄待查；建議 0.3–0.5 m 起 |
| 2D floorplan：floor 與「suitable」ceiling polygon 合併準則 | 📎 | 附錄待查（正文：天花板較少被遮擋、更可靠） |
| 房間分割演算法 | ✅ 來源／🔧 參數 | HOV-SG [22] 的 room segmentation 子模組；參數用其預設起步 |
| door 判準 | ✅ | opening 寬度 **< 1.5 m** → door，否則 opening |
| room type 判定（kitchen, office…） | 📎 | 正文僅說「each room is associated with a room type」；方法附錄待查 |
| 樓梯：stair mesh connected components 分群 | ✅ 邏輯／📎 連通半徑 | CC 的鄰接判準（距離門檻）附錄待查 |
| Room extrusion | ✅ 邏輯 | CDT（floorplan 邊界＋ceiling 候選邊＋ceiling 平面兩兩交線投影）→ 三角形中心向上射線指派 ceiling → 未命中者指派到「未指派三角形圖」中可達的最低 ceiling → extrude；牆邊補軸對齊矩形；ceiling 不連續邊補垂直矩形 |
| 每房 ceiling 候選上限 | ✅ | **30** 個最大 ceiling |
| extrude 後 door/stair 加回細節 | 📎 | 「provided in the appendix」 |
| 窗偵測：LOF 離群過濾參數 | 📎→🔧 | 附錄待查；先用 sklearn 預設 |
| 窗偵測：DBSCAN eps / min_samples | 📎 | 附錄待查 |
| 窗偵測：最小群頂點數 k | ✅ | **k = 10**（達標才擬合軸對齊矩形） |
| 窗判準：矩形高與寬 | ✅ | 皆 **> 30 cm** 才輸出為窗 |

## 5. 評估指標

| 參數 | 狀態 | 值／建議 |
|---|---|---|
| d_E（doors/windows 等矩形實體） | ✅ | 同類兩矩形對應角點最大距離；角點對應用 Hungarian 最佳排列（式在 §5） |
| d_H（walls/floors/ceilings 非矩形實體） | ✅ | 廣義 Hausdorff：max{max_v D_pp(v,P′), max_v′ D_pp(v′,P)}（式 6） |
| F1@τ 主報表門檻 | ✅ | τ = 0.5（Table 2 的 F1@0.5） |
| Avg F1 的門檻集合 | 📎 | 沿 SceneScript [2] 慣例；具體集合附錄待查 |
| 深度指標 | ✅ | Δτ＝預測深度與 GT 深度差 ≤ τ cm 的像素比例；報 Δ5、Δ10；用輸入 camera poses 分別 render GT 幾何與預測 layout |

## 6. 對照數字（復現目標量級，Table 2 MULTIFLOOR3D）

Structures F1@0.5 0.40±0.10（Avg 0.38）；Doors 0.55±0.16（0.44）；Windows 0.43±0.29（0.38）；Stairs 0.42±0.48（0.41）；Δ5 61.1±9.2；Δ10 76.3±7.9；#Vertices ≈1957。
ScanNet++（DN-Splatter mesh 輸入）：Δ5 67.8、Δ10 84.7、#V 83.1；DN-Splatter 原始 mesh 本身 Δ5 84.1／Δ10 92.6（上限參考）。

## 7. ⚠️ 需要你處理 — 下載官方 supplementary（約 30 秒）

OpenReview 的附件下載對本機的 curl 觸發人機驗證（ChallengeRequiredError），我抓不下來。請有空時在瀏覽器：

1. 開 https://openreview.net/forum?id=5M5WdH659Y
2. 點標題下方的「Supplementary Material」下載 zip
3. 存到 `/home/ado/storage/HouseLayout3D/docs/supplementary.zip`

我拿到後會解壓精讀，把上表所有 📎 項目更新成 v2（屆時真正缺的才會列出來問你）。
