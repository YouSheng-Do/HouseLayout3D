> **⚠️ 2026-07-06 狀態註記**:本文件寫於發現官方 supplementary 之前，「pipeline 程式碼未開源」的前提已部分失效——
> OpenReview supplementary 內含官方 Stage 2/3 程式碼＋附錄＋16 場景官方預測（見 `docs/supplementary_inventory.md`）。
> 執行策略已修訂為「官方 code 為主幹、Stage 4/eval 依附錄自寫、eval-first」；現況以 `PROGRESS.md` 為準。
> 本文件的**執行紀律與溝通約定條款不受影響、仍然有效**。

# Project Brief: Reproducing MULTIFLOOR3D (HouseLayout3D)

> 這份 brief 是給 Claude Code 的執行指令。把它連同論文 PDF 和研究報告一起放進專案，
> 並在對話開頭要求 Claude Code **先完整讀過這三份文件再動手**。

---

## 0. 給執行者的最高原則（請先讀，且全程遵守）

1. **這不是「照 repo 跑」的任務，而是「依論文從零實作」的任務。** 官方 repo
   (https://houselayout3d.github.io/ → 連出的 GitHub) 只釋出了資料集下載與視覺化工具，
   **四階段 pipeline 的程式碼並未開源**。不要假設有現成的 `main.py` 可以跑；找不到就是沒有。

2. **一次只做一個階段。** 每個階段結束後**停下來**，輸出該階段的產物與一段簡短的自我檢查報告，
   等我確認後再進入下一階段。不要一路衝到底。

3. **先在單一場景跑通，再擴到全量。** 所有階段第一次都只在**一個 building / 一個場景**上驗證正確性。
   16 棟一起上是禁止的，直到 pipeline 全程在單場景驗證過。

3a. **當前策略：先用替代資料把程式跑通，MP3D 到位後再做正式驗證。** Matterport3D 授權審核中
   （數天），但**不要乾等**。Stage 1 的 DN-Splatter 只需要「一組 unposed RGB 影像」，因此用
   **DN-Splatter 官方範例資料**或**使用者自拍的房間影片抽幀**當替代場景，就能實際跑通並 debug
   全部四階段。「驗證」拆成兩層：**(A) 能不能跑（smoke test，用替代資料，現在就做）**、
   **(B) 跑得對不對（quantitative eval，需要 MP3D 真標註）**。現階段目標是把 (A) 全部做完，
   讓 code 處於「能執行、每階段產物長得對」的狀態；(B) 等資料到位再做。
   **嚴禁「只寫不跑、等 MP3D 才第一次執行」**——那會把等待變成延後爆的雷。

4. **論文沒寫清楚的參數，停下來問我，不要自己猜。** loss 權重、各種 threshold
   (τ_inter, τ_merge, τ_extend, τ 評估門檻)、初始化策略等若論文正文/supplementary 找不到明確值，
   **列出來問我**，並附上你建議的預設值與理由。絕對不要靜默填入猜測值然後繼續。
   在現階段（還沒有 MP3D、還不能調參）遇到這類參數時，把它們寫成**集中在一份 config 檔、
   有明確預設值、且標記 `# TODO: tune with MP3D` 的可調參數**，不要散落在各處程式碼裡。
   這樣 (B) 階段要調校時一目了然。

5. **需要我本人操作的事情，明確標記出來讓我做，不要嘗試代做。** 包含：Matterport3D EULA 申請、
   資料集下載授權、A5000 實驗室資源協調、任何需要帳號登入或簽署授權的步驟。

6. **回報用繁體中文或英文。**

---

## 1. 專案目標

重現 **MULTIFLOOR3D**——一個 training-free 的 3D layout estimation pipeline
（論文：HouseLayout3D, NeurIPS 2025 Datasets & Benchmarks track）。

最終目的有兩個：
- (a) 讓整條四階段 pipeline 在 HouseLayout3D 資料集上跑通並產生可評估的 layout。
- (b) 在跑通後，定位論文方法**目前 remaining 的缺陷**（尤其是樓梯偵測、頂點過多、
  透過大窗戶誤收室外元素、runtime 過長），作為後續改善的切入點。

**成功標準**：能在資料集的子集上復現論文 Table 2 量級的 F1 指標（不要求數字完全一致，
但量級與趨勢要對得上），且每個階段的中間產物都經過視覺化檢查。

---

## 2. 環境與資源

- **GPU**：主力一張 **A6000 (48GB)**；另有兩張 **A5000 (24GB)** 為實驗室共用資源，
  需要協調才能用（**協調由我負責**，你只需在需要多卡時明確告訴我哪個階段需要、需要多久）。
- GPU-heavy 階段依序是：Stage 1 的 3DGS/DN-Splatter 重建（最吃資源）、Metric3D 深度推論、
  Stage 2 的 OneFormer 推論。請在每階段開始前估算 VRAM 需求，若單張 A6000 放不下再提出需求。
- 相依套件之間**環境常有衝突**（DN-Splatter / SuperPoint Transformer / HOV-SG 各有各的
  CUDA / PyTorch 版本要求）。傾向用**多個獨立 conda 環境**、階段間以檔案（mesh/點雲/中間結果）
  交接，而不是硬塞進同一個 env。遇到衝突時把選項列給我判斷。

---

## 3. 前置條件檢查（Stage 0）

分成兩組。**A 組現在就要完成才能開始 Stage 1；B 組是 MP3D 到位後、正式量化驗證前才需要。**

### A 組 — 不需 MP3D，現在就開工（全部綠燈才進 Stage 1）
- [ ] 已確認官方 repo 實際釋出內容，並在報告中列出「有什麼、沒有什麼」
- [ ] conda / CUDA / GPU driver 環境可用，`nvidia-smi` 正常
- [ ] 相依套件環境已裝好（DN-Splatter / Metric3D / OneFormer / SPT / HOV-SG / CGAL 等，
      建議多個獨立 env）
- [ ] **已備妥替代場景資料**：優先用 DN-Splatter 官方範例資料；若不可得，用一段房間影片抽幀。
      這組資料就是 smoke test 全程的測試對象。

### B 組 — 需要 MP3D，之後才做（現在不是 blocker）
- [ ] Matterport3D EULA 已核准，MP3D 的 mesh + RGB-D + camera poses 可下載（**我負責申請，審核要數天**）
- [ ] HouseLayout3D 標註資料已下載並確認結構（從官方 repo / HuggingFace 取得）
- [ ] 選好一個「單場景量化驗證用」的 MP3D building

---

## 4. 四階段實作計劃（每階段結束都要停下來等我確認）

> **當前全部用替代場景資料實際執行（smoke test）**，目標是每階段的 code 能跑、產物長得對。
> 每個階段請輸出：程式碼、該階段的產物（存檔路徑）、一張或數張視覺化圖、
> 一段自我檢查（產物看起來對不對、有沒有明顯 artifact）、以及遇到的未定參數清單。
> **每個 Stage 都必須在替代場景上實際跑過、看過產物視覺化，才算完成**——不接受「寫完但沒執行」。
>
> **檢查點的手機友善設計（重要）**：我常常只能用手機遠端確認，無法在小螢幕上細看 3D 視覺化。
> 因此每個檢查點請**同時**輸出兩種東西：
> - **(1) 純文字摘要（手機用）**：3–6 行以內，講清楚「跑完了嗎、產物數量/大小合不合理、
>   有沒有明顯異常、你自己判斷這階段對不對、需不需要我回電腦看圖」。讓我光讀文字就能決定
>   「繼續 / 停下來等我回電腦細看」。
> - **(2) 視覺化圖檔（電腦用）**：存好路徑，留給我有空回電腦時細看。
>
> 若某階段你自己判斷產物明顯有問題，文字摘要第一行就直接標 **「⚠️ 建議回電腦確認」**。

### Stage 1 — Mesh Reconstruction from RGB
- 依 **DN-Splatter** 流程：COLMAP poses → 3DGS 重建 → 從 3DGS render 的深度做 **Poisson surface** → triangle mesh + per-frame depth maps。深度模型用 **Metric3D**。
- **先只跑單場景。** 產出 mesh 後做視覺化，確認幾何品質再往下。
- 停損點：mesh 有嚴重破洞 / 尺度錯誤 → 停下來討論，不要硬往 Stage 2 帶。

### Stage 2 — Layout Skeleton Extraction
- **OneFormer** 對輸入影像做 2D segmentation → 每張影像 back-project M=5000 取樣像素到最近 mesh vertex → 累積 class votes。
- 用 **superpoint 分群**（依 Robert et al. / SuperPoint Transformer 的 preprocessing）refine，取每個 vertex 所屬 cluster 的多數標籤。
- 分成四類：structural components / geometrically inaccurate surfaces (windows, mirrors) / objects / stairs。
- 產物：帶語意標籤的 mesh + 抽出的 layout skeleton + object/stair 子集。

### Stage 3 — Fitting a Layout Prototype
- 從 skeleton 初始化 planar 3D polygons，用 gradient descent 優化 vertex 位置與平面方程，三個 loss：
  - `L_geo = L_prox + L_empty`（L_empty 需要用 camera poses + depth 取樣 line segments）
  - `L_connect`（吸附共享邊界）
  - `L_simple`（消除非共享邊）
- Vertex merging：距離門檻合併 + **RDP** 簡化 + 合併相近同法向 polygon。
- Hole closing：object 投影到 floor、wall/ceiling 往地板延伸。
- **這階段未定參數最多**（各 threshold、loss 權重、初始化細節可能在 supplementary 或缺失）。
  把清單整理出來問我，附建議值。

### Stage 4 — Scene Graph from Prototype
- 識別 floors（合併相近高度的 floor polygon）→ 每層做 2D floorplan（floor + ceiling polygon 合併）。
- Room segmentation 用 **HOV-SG** 的演算法；opening 寬度 < 1.5m 判為 door。
- Stair detection：對 stair mesh 做 connected components 分群，連接樓層。
- Room extrusion：**Constrained Delaunay Triangulation**（CGAL）→ 三角形指派到 ceiling → extrude。
- Window detection：window pixel back-project 到牆面 → **DBSCAN** 分群 → 擬合矩形（LOF 濾 outlier）。
- 產物：最終 scene graph + 3D layout，接上評估腳本。

### 驗證（兩層，分開做）

**(A) Smoke test — 現在做，用替代場景資料。**
四階段全部串起來，在替代場景上從頭跑到尾，確認：能執行不報錯、每階段產物存得出來、
最終產出一個結構完整的 scene graph / layout。這層**不看 F1 數字對不對**，只看「跑不跑得動、
產物形狀對不對」。評估腳本（entity-distance F1、d_H、depth Δτ）此時就先寫好、能對著替代場景
的 layout 執行（哪怕沒有 ground truth 可比，先確保程式能跑）。

**(B) Quantitative eval — MP3D 到位後才做。**
接上 HouseLayout3D 官方標註，在單場景 / 小子集報 F1 與 depth 指標，對照論文 Table 2 的量級。
這時才回頭調 config 裡標 `# TODO: tune with MP3D` 的參數，讓數字對上。

---

## 5. 已知的最弱環節（跑通後優先檢視，作為改善切入點）

- **樓梯偵測**：Table 2 中 stair F1 = 0.42 ± 0.48，變異極大 → 很可能是最脆弱、最值得改的一環。
- **頂點過多 / 不夠 compact**：#Vertices 明顯高於 baseline。
- **透過大窗戶誤收室外元素**（論文 Limitations 明列）→ 產生 artifact。
- **runtime 長**：單場景 1–2 小時（RTX 4090），全量成本高。
- **窗戶 / 鏡面深度不準**：這類表面本來就被排除，偵測靠後處理，容易漏。

跑通後請針對這些做定量定位（哪個 class、哪類場景掉最多），再一起討論改善方向，
不要在 pipeline 還沒驗證前就先動手改方法。

---

## 6. 溝通約定

- 每階段結束 → 停 → 回報產物 + 自我檢查 + 未定參數清單 → 等我 confirm。
- **每次回報都以「純文字摘要（手機用）」開頭**，讓我能只用手機就決定繼續或喊停；視覺化圖檔另存路徑供電腦細看。
- **可以只用手機確認的**：環境/資料就緒狀態、某階段有沒有跑完不報錯、二選一的參數決策、要不要進下一階段。
- **建議回電腦看的**：mesh 品質、layout prototype、最終 layout 等需要看圖判斷的檢查點——這類請在摘要裡標「⚠️ 建議回電腦確認」。
- 遇到需要我操作的（授權、下載、多卡協調）→ 明確標記「需要你處理」。
- 缺參數 → 列清單問我，附建議值與理由，**不要靜默填猜測值**。
- 環境衝突 → 把選項與 trade-off 列給我，不要自行做重大取捨。
