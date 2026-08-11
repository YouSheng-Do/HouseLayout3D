> **⚠️ 2026-07-06 狀態註記**:本報告的核心前提「管線程式碼未開源」已部分失效——NeurIPS supplementary（OpenReview）
> 內含官方 Stage 2/3 程式碼＋附錄＋16 場景官方預測，見 `docs/supplementary_inventory.md` 與 `docs/param_inventory_v2.md`。
> 本報告保留為歷史背景；現況以 `PROGRESS.md` 為準。

# 重現 HouseLayout3D / MULTIFLOOR3D 的可行性與行動報告
# Reproducing HouseLayout3D & the MULTIFLOOR3D Training-Free Pipeline

## TL;DR
- **MULTIFLOOR3D 的管線程式碼尚未開源(截至 2026-07-03 已直接查證)**:官方 repo `github.com/HouseLayout3D/houselayout3d` 只有 4 個檔案(`README.md`、`main.py`、`visualize.py`、`requirements.txt`),`requirements.txt` 僅列 `datasets` / `pyviz3d` / `typer` / `numpy`(無版本鎖定),是純「資料集下載 + 視覺化」repo;四階段管線程式碼完全沒有釋出,要求開源的 issue #1(2025-12-07)與詢問 mesh 細節的 issue #2(2026-03-26)至今**都沒有作者回覆**。因此「重現」實質上等於「依論文從零重新實作」。
- **資料集可用但需 Matterport3D 授權**:HouseLayout3D 標註(3D 多邊形、門窗矩形、樓梯、nerfstudio 格式位姿)在 HuggingFace 以 MIT 授權釋出(僅 35.2 MB、26,618 rows),但底層網格與 RGB-D 來自 Matterport3D,必須先簽署 MP3D 學術 EULA(寄至 matterport3d@googlegroups.com、僅限非商業學術)。整個管線是 training-free,無需訓練權重,全靠現成模型。
- **硬體足夠但需耐心**:論文明載「MULTIFLOOR3D has a longer runtime (1–2 hours per scene on an NVIDIA GeForce RTX 4090) compared to feed-forward baselines (1–2 minutes)」(Bieri et al., arXiv:2512.02450)。你的 A6000(48GB)是理想單卡選擇,能容納最吃記憶體的 3DGS 階段;A5000×2 適合資料平行。跑完整 16 棟建築 benchmark 保守估需約 1.5–3 天連續運算。最弱、最值得改進的是**樓梯偵測(F1 0.42±0.48,變異極大)**、窗戶/戶外元素處理,以及過高的頂點數(1957 vs SceneScript 677)。

---

## Key Findings

1. **程式碼狀態(最關鍵)**:官方 GitHub 是純視覺化 repo,非管線 repo。`main.py`(10 行)只做 HuggingFace 資料集載入;`visualize.py`(81 行)只用 pyviz3d 渲染既有標註(預設場景 `1LXtFkjw3qL`)。四階段(mesh reconstruction、skeleton extraction、prototype fitting、scene graph)**零程式碼**,無 evaluation script、無 config、無 conda yml、無 Dockerfile。共 12 commits、35 stars,檔案樹自釋出以來未變。
2. **無 pretrained weights**:方法 training-free,所有神經組件(Metric3D 深度、OneFormer 分割、SPT superpoint、HOV-SG 房間分割)都用各自 off-the-shelf 權重。
3. **資料授權雙層**:HouseLayout3D 標註本身 MIT、可自由下載;但要用它們必須先取得 Matterport3D(簽 EULA、機構 email、非商業學術)。這是重現的第一個行政 gating,常需數天到數週。
4. **量化最弱環節**(論文 Table 2):MULTIFLOOR3D 全面勝 baseline 但絕對值仍低 — **Structure F1@0.5 = 0.40**(vs RoomFormer 0.24、SceneScript 0.28,近乎倍增兩個單樓層學習式 baseline)、Structures Avg F1 0.38、Doors F1@0.5 0.55、Windows 0.43、**Stairs 0.42±0.48**(標準差 ≈ 平均值,代表部分場景樓梯完全失敗)。頂點數 1957,遠不如 SceneScript(677)精簡。
5. **可插入的 2025–2026 後續技術**:VGGT(feed-forward 重建取代 COLMAP+3DGS)、SpatialLM(學習式 layout 估計)、Fast SceneScript、透明/玻璃表面深度模型,都是改進候選。

---

## Details

### 一、Repo / 資料集 / 授權盤點(任務一)

**專案頁**:https://houselayout3d.github.io — 連結 Paper(PDF)、Code(GitHub)、Dataset(HuggingFace)。作者:Valentin Bieri(ETH Zurich)、Marie-Julie Rakotosaona(Google)、Keisuke Tateno(Google)、Francis Engelmann(Stanford)、Leonidas Guibas(Stanford)。arXiv:2512.02450,NeurIPS 2025 Datasets & Benchmarks track。

**GitHub repo 現況(已逐檔查證)**:
- 檔案:`README.md`、`main.py`、`visualize.py`、`requirements.txt`;12 commits;35 stars。
- `requirements.txt` 全文:`datasets` / `pyviz3d` / `typer` / `numpy`(皆無版本鎖定)。**沒有 torch、nerfstudio、DN-Splatter、COLMAP、open3d 等任何重建/深度學習依賴**。
- **無 MULTIFLOOR3D 管線程式碼、無 evaluation、無 config、無環境規格檔**。
- issue #1「May I ask whether the baseline method Multi Floor 3D can be open-source?」(2025-12-07,作者 yangrongkun)與 issue #2 關於 mesh construction 的細節提問(2026-03-26,作者 Cynthia11281128)**都無任何作者/維護者回覆**。issue #2 提出的三個問題正是重現時最關鍵、論文未明示的細節:(a) 輸入是 MP3D 全景還是處理過的視角?(b) MP3D 全景重疊小,COLMAP 如何可靠估位姿?(c) DN-Splatter 是整棟一次跑還是逐房間跑?

**HuggingFace 資料集(`houselayout3d/HouseLayout3D`,MIT)**:
- 總檔 35.2 MB;subsets:doors(16 rows)、poses(26.6k rows)、stairs、structures、windows。
- 結構:`structures/{scene}.obj`(base layout 三角網格)、`layouts_split_by_entity/{scene}/*.obj`(逐 wall/ceiling/floor 實體)、`doors/{scene}.json`(4 角 + `normal` 開門方向)、`windows/{scene}.json`(4 角)、`stairs/{scene}/*.ply`、`poses/{scene}.json`(nerfstudio 格式相機內外參,供 layout 深度評估)。
- 下載:`git lfs install` + `git clone https://huggingface.co/datasets/houselayout3d/HouseLayout3D data`。
- **重要**:此處只有「標註 + 位姿」,不含 MP3D 原始 RGB-D 影像與帶紋理網格 — 需向 Matterport3D 另外申請。

**Matterport3D 授權**:須簽署 Terms of Use / Academic EULA,以機構 email 寄至 matterport3d@googlegroups.com 取得下載權;僅限非商業學術用途;散布衍生資料須附同一份 EULA。

**資料集規模**:16 棟建築、33 樓層、317 房間、>26,000 RGB-D 影格(規模與 ScanNet 驗證集相當);標註 292 doors、379 windows、34 staircases;每棟 1–5 層、4–40 房間;人工標註每棟 4–10 小時,用 Scasa PinPoint。

### 二、四階段管線與依賴對照(任務二)

因無官方程式碼,以下為依論文重新實作的藍圖,每個組件標出開源 repo、安裝難度、GPU 需求、雷點與授權。

**Stage 1 — Mesh Reconstruction(GPU 最重)**
- 流程:未定位 RGB → COLMAP(SfM 位姿)→ Metric3D 單目深度先驗 → DN-Splatter 訓練 3DGS → Poisson surface reconstruction 出三角網格 + 每影格深度圖。
- **DN-Splatter**(`maturk/dn-splatter`,WACV 2025):建在 `nerfstudio==1.1.3` + `gsplat==1.0.0` 之上,依賴 pymeshlab、vdbfusion、PyMCubes、omnidata-tools、pyrender。安裝中等偏難(nerfstudio 生態版本綁定嚴格)。
- **Metric3D**(`YvanYin/Metric3D`,ICCV 2023;v2 為 geometric foundation model):`torch.hub.load` 一行載入,支援 convnext_tiny→vit_giant2;ViT-small/large 單卡數 GB 即可。安裝容易。
- **COLMAP**:標準 SfM;MP3D 全景重疊小,位姿估計是已知痛點(issue #2 未獲官方回答)。
- **gsplat**(`nerfstudio-project/gsplat`,Apache-2.0):據 Ye et al.「gsplat」(JMLR vol.26, 24-1476)「gsplat achieves up to 10% less training time and 4× less memory than the original implementation」— MipNeRF360/A100、30k iters 記憶體由 9.0 GB 降至 5.6 GB。
- VRAM 參考:一般室內 3DGS 訓練 6–17GB(base 0.1–0.5M Gaussians ~10GB;dense 2–4M ~17GB,MILo/RTX 4090 文獻)。建築級整棟更吃記憶體 → **A6000 48GB 有明顯優勢**。

**Stage 2 — Layout Skeleton Extraction(GPU 中)**
- 流程:OneFormer 2D 分割 → 每影格取 M=5000 像素反投影到最近網格頂點累積類別票 → 依 Robert et al. superpoint 前處理分群 → 取群內多數標籤 → 分 4 類(結構元件 / 幾何不準表面〔窗鏡〕/ 物件 / 樓梯)。
- **OneFormer**(`SHI-Labs/OneFormer`,CVPR 2023;已進 HuggingFace transformers):推論只需影像 + task token,單卡輕鬆;類別對應 COCO。重現只需推論,容易。
- **Superpoint Transformer / SPT**(`drprojects/superpoint_transformer`,MIT;ICCV'23 + 3DV'24 SuperCluster):cut-pursuit superpoint 分割,GPU 加速。**已知雷點:FRNN GPU 近鄰搜尋庫安裝困難**(官方 install.sh 有專門處理,可退回較慢的 CPU nanoflann)。安裝難度偏高。

**Stage 3 — Layout Prototype Fitting(最佳化,GPU/CPU 混合)**
- 每個 superpoint 擬合成平面多邊形集合 P,梯度下降最佳化頂點位置與平面方程,三損失:L_geo = L_prox(頂點靠近多邊形)+ L_empty(用相機→深度反投影線段懲罰佔用觀測空區);L_connect(鼓勵共享邊界、避免縫隙);L_simple(懲罰非共享邊長以精簡)。
- 頂點合併:成對合併(τ_merge)+ **RDP** 逐多邊形簡化 + 合併相近同法向多邊形。RDP 有現成 python 套件(`fhirschmann/rdp`,MIT;或更快的 `fastrdp`)。
- 補洞:物件三角形投影到最近地板平面補地板洞;偵測牆多邊形向下法向邊、數 L 線段交會密度,低於 τ_extend 就延伸牆到地板/天花板。

**Stage 4 — Scene Graph(以 CPU 幾何演算法為主)**
- 辨識樓層(合併相近高度地板多邊形)→ 合併地板+天花板成 2D floorplan → 用 **HOV-SG** 房間分割切房間(開口寬 <1.5m 為門,否則 opening)→ 樓梯以 stair mesh 連通分量偵測連接樓層。
- 房間擠出:**CGAL 2D Constrained Delaunay Triangulation** 三角化 floorplan → 每三角形向上打射線指派天花板平面 → 擠出成封閉房殼。CGAL 三角化為 **GPL 授權**(商用需另談);有 SWIG python bindings(`CGAL/cgal-swig-bindings`);替代可用 `scipy.spatial.Delaunay`(需自加約束)或 header-only CDT 庫。
- 窗戶偵測:窗類別像素射線與牆相交 → **LOF**(Breunig 2000,scikit-learn 內建)濾離群 → 依牆實例切分 → 每面牆跑 **DBSCAN**(scikit-learn)→ ≥k=10 頂點的群擬合軸對齊矩形 → 高寬 >30cm 才判為窗。
- **HOV-SG**(`hovsg/HOV-SG`,RSS 2024):論文 Related Work 述其「combines BEV point-density maps with 2D object detection to construct a scene graph of floors, rooms, and objects, but without recovering their 3D geometry」,MULTIFLOOR3D **只重用它的房間分割子模組**。依賴 habitat-sim(conda 額外安裝、易出錯)、CLIP;安裝難度高;作者自承階段多、超參多、建圖耗時。建議只移植房間分割部分以避開 habitat-sim。

**純演算法組件**:Poisson reconstruction(Open3D / 原作者)、QSlim(僅 ablation 用)、RDP、DBSCAN、LOF、CDT — 皆成熟易得。

### 三、硬體與吞吐規劃(任務三)

- **論文基準**:單場景 1–2 小時 @ RTX 4090(24GB);SceneScript/RoomFormer 只需 1–2 分鐘(feed-forward)。RTX 4090 與 A6000 運算力相近,A6000 記憶體(48GB)是 4090 兩倍。
- **各階段 GPU 吃重程度**:
  - Stage 1(3DGS/DN-Splatter 訓練)= 最吃 VRAM 與時間,是 1–2 小時的主因。整棟建築 Gaussian 數量大,**48GB A6000 最能受益**;24GB A5000/4090 可能需降解析度(`--downscale-factor`)或分塊。
  - Metric3D 推論:逐影格,VRAM 小,但 26k+ 影格累積耗時。
  - OneFormer 推論:逐影格,單卡輕鬆,總時長受影格數影響。
  - Stage 2 SPT、Stage 3 最佳化:中等 VRAM;Stage 4 幾乎純 CPU 幾何。
- **建議配置**:
  - **主力用 A6000(48GB)**:單卡端到端跑完一個場景,無需分塊,最省心。
  - **A5000×2(24GB)**:適合**資料平行** — 不同建築/樓層分派到不同卡,wall-clock 幾乎減半;**不建議把單一 3DGS 訓練跨卡切**(多卡 3DGS 如 Grendel 複雜且非官方管線)。A5000 單卡也可跑,大場景 3DGS 需降解析度。
- **完整 benchmark 吞吐估算**:16 棟、33 樓層。以「每個獨立重建單元 1–2 小時」保守估,單卡 A6000 循序約 **1.5–3 天連續運算**(視是否逐樓層切分;COLMAP 對 26k 影格 SfM 本身可能數小時);A5000×2 資料平行可壓到約 1–1.5 天。第一次會多花時間在環境安裝(nerfstudio/DN-Splatter/FRNN/HOV-SG habitat-sim)與 COLMAP 調參,**建議先用 1–2 棟小建築打通再全量跑**。

### 四、限制與改進方向(任務四)

**論文自述限制**:
1. 執行時間遠長於 feed-forward baseline(1–2 小時 vs 1–2 分鐘)。
2. 難以移除透過大片窗戶看到的戶外元素 → artifacts。
3. 依賴啟發式規則而非學習式建築尺度推理。
4. 頂點數高、不精簡(1957 vs SceneScript 677 / RoomFormer 765)。
5. 窗/鏡等透明反射表面深度不準(Stage 2 特意將其歸為「幾何不準表面」排除)。

**由量化結果看最弱環節**:
- **樓梯 F1 = 0.42±0.48**:標準差 ≈ 平均,代表在不少場景樓梯完全失敗;這是 MULTIFLOOR3D 獨有能力,也最不穩定,是最高價值的改進點。
- **Structures Avg F1 僅 0.38**、Windows 0.44:雖勝 baseline 但絕對值仍低,牆/地板/天花板幾何與窗偵測都有大量空間。
- 頂點精簡度差:L_simple + RDP 尚不足以達到 SceneScript 等級緊緻度。

**可插入 / 可改進的 2025–2026 技術**:
- **重建加速**:**VGGT**(Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny;Oxford VGG + Meta,CVPR 2025 Best Paper,自 >13,000 投稿中選出;openaccess.thecvf.com 述其「simple and efficient, reconstructing images in under one second」,其 `demo_colmap.py` 可直接輸出 COLMAP 格式餵給 gsplat)或 MASt3R-SLAM 等 feed-forward 重建取代 COLMAP+3DGS+Poisson,可能把 Stage 1 從小時級壓到秒/分鐘級,直接攻擊「執行時間過長」。注意 VGGT 對長序列有 GPU 記憶體瓶頸,需分塊。
- **學習式 layout 推理**:**SpatialLM**(Yongsen Mao 等;Manycore Tech Inc. & HKUST,NeurIPS 2025,arXiv:2506.07491,repo `manycore-research/SpatialLM`):由開源 LLM 微調而成,從點雲直接偵測「architectural elements like walls, doors, windows, and oriented object bounding boxes」,訓練於 12,328 室內場景(54,778 房間)。可取代 Stage 3/4 的啟發式。**Fast SceneScript**(arXiv:2512.05597)則提升 SceneScript 速度與準確度。這些呼應論文結論呼籲的「跨整棟建築推理的學習式模型」。
- **窗/玻璃/戶外處理**:專門的透明與鏡面深度方法(Costanzino et al.「Learning Depth Estimation for Transparent and Mirror Surfaces」ICCV 2023,arXiv:2307.15052;以及 2025–2026 的玻璃表面深度先驗 pipeline)可改善 Stage 1 對窗戶的深度並協助濾除窗外幾何。
- **房間分割**:用學習式房間分割取代 HOV-SG 的啟發式 Voronoi/密度法,改善房間邊界與門/opening 判定。
- **精簡度**:引入更強多邊形正則化或可微分 CAD 表示,降低頂點數。

**建議的具體研究題目**:(a) 把 Stage 1 換成 VGGT/feed-forward 重建並量測 F1 與 runtime 取捨;(b) 針對樓梯設計專屬偵測模組以壓低 0.48 變異;(c) 用透明表面深度先驗處理大窗戶並評估戶外 artifact 減少;(d) 用 SpatialLM 類模型替換 Stage 3/4 啟發式做端到端學習式建築推理。

---

## Recommendations(分階段、可執行)

**第 0 階段 — 立即行動(第 1 週)**
1. 現在就寄出 Matterport3D EULA(機構 email)到 matterport3d@googlegroups.com — 這是最長的行政瓶頸,越早越好。
2. 同時 `git clone` HouseLayout3D 標註(HuggingFace,MIT,35MB)並用 `visualize.py` 熟悉標註格式與評估用位姿。
3. 在 GitHub issue #1/#2 留言或直接 email 通訊作者 Valentin Bieri / Francis Engelmann,詢問(a)是否計畫釋出管線程式碼、(b)issue #2 的 MP3D 輸入 / COLMAP / 分塊三個細節。這可能省下數週逆向工程。**觸發條件**:若一週內作者願分享程式碼或私下 preview,重現難度大幅下降,優先走該路。

**第 1 階段 — 環境與單場景打通(第 2–4 週)**
4. 在 A6000 上依序安裝並各自跑通:Metric3D(最易)→ OneFormer(HF transformers)→ DN-Splatter/nerfstudio/gsplat → SPT(注意 FRNN)→ CGAL/DBSCAN/LOF/RDP。HOV-SG 只移植房間分割子模組以避開 habitat-sim。
5. 先用 1 棟最小建築(少樓層少房間)端到端跑通四階段,對照 HouseLayout3D 標註用論文的 F1(entity distance / Hausdorff)與深度 Δ5/Δ10 驗證。**觸發條件**:若單場景 Structures F1 接近論文 0.38–0.40,代表實作正確可放大;若明顯偏低,先 debug Stage 1 網格品質與 Stage 2 分割對應。

**第 2 階段 — 全量 benchmark(第 5–7 週)**
6. 用 A6000 循序、或 A5000×2 資料平行(逐建築/逐樓層分派)跑完 16 棟。預留 1.5–3 天純運算 + debug 時間。監控 3DGS VRAM;A5000 大場景用 downscale 或分塊。

**第 3 階段 — 改進研究(第 8 週起)**
7. 選一個高槓桿方向做消融:最推薦 **(a) 用 VGGT 取代 Stage 1** 攻擊 runtime,或 **(b) 專屬樓梯偵測** 攻擊 0.42±0.48 的不穩定。
8. 次選:透明表面深度先驗改善窗/戶外;SpatialLM 取代啟發式 Stage 3/4。
**改變決策的門檻**:若 VGGT 重建的深度 Δ10 掉超過 ~10 個百分點,則保留 3DGS 只加速其他階段;若樓梯模組能把變異從 0.48 降到 <0.25,即值得寫成獨立貢獻。

---

## Caveats
- **最大風險**:管線程式碼未開源且作者未回應,所有「重現」都是從論文 + 附錄逆向實作。超參(τ_merge、τ_inter、τ_extend)、初始化、共享頂點實作、MP3D→COLMAP 前處理等細節論文說在 supplementary,務必細讀附錄;仍可能有無法從文字完全還原的工程細節。
- **runtime 與 VRAM 數字**:1–2 小時/場景是論文對 RTX 4090 的陳述;A6000/A5000 的實際時間會依分塊策略、影格取樣、COLMAP 設定而變,上文估算為推估而非實測。3DGS VRAM 數字引自一般室內場景文獻,建築級整棟可能更高。
- **授權**:CGAL 三角化為 GPL(商用需授權);MP3D 僅限非商業學術;HOV-SG / habitat-sim 安裝與授權另需確認。發表衍生資料須附 MP3D EULA。
- **後續技術為建議而非驗證**:VGGT、SpatialLM、透明表面深度等能否無縫接入此管線並提升指標,截至 2026-07 尚無人在 HouseLayout3D 上驗證,屬研究假設。
- 部分 GPU/VRAM 描述來自第三方部落格與相關論文,非 HouseLayout3D 官方數據,已於文中標明來源性質。

---

## Quick Reference — 依賴與連結

| 組件 | Repo | 授權 | 安裝難度 | GPU/VRAM |
|---|---|---|---|---|
| HouseLayout3D 標註 | huggingface.co/datasets/houselayout3d/HouseLayout3D | MIT | 易(git-lfs) | — |
| Matterport3D(底層資料) | niessner.github.io/Matterport | MP3D EULA(非商業學術) | 需簽署申請 | — |
| DN-Splatter | github.com/maturk/dn-splatter | 見 repo(nerfstudio Apache-2.0) | 中–難 | 高(3DGS 6–17GB+) |
| Metric3D / v2 | github.com/YvanYin/Metric3D | 見 repo | 易(torch.hub) | 低–中 |
| gsplat | github.com/nerfstudio-project/gsplat | Apache-2.0 | 中 | 中(比原版省 4×) |
| COLMAP | colmap.github.io | BSD | 中 | CPU/GPU |
| OneFormer | github.com/SHI-Labs/OneFormer | 見 repo(HF transformers) | 易(推論) | 中 |
| Superpoint Transformer (SPT) | github.com/drprojects/superpoint_transformer | MIT | 難(FRNN) | 中 |
| HOV-SG | github.com/hovsg/HOV-SG | 見 repo | 難(habitat-sim) | 中 |
| CGAL CDT | github.com/CGAL/cgal-swig-bindings | GPL(三角化) | 中 | CPU |
| RDP | github.com/fhirschmann/rdp | MIT | 易 | CPU |
| DBSCAN / LOF | scikit-learn | BSD | 易 | CPU |
| VGGT(改進) | github.com/facebookresearch/vggt | 見 repo | 中 | 中–高 |
| SpatialLM(改進) | github.com/manycore-research/SpatialLM | 見 repo | 中 | 中 |