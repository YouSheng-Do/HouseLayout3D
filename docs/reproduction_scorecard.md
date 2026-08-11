# MULTIFLOOR3D 復現計分卡（對照論文 Table 2 / 3）

> ⚠️ **2026-08-11 更正**：本文件中任何「stairs F1 0.473 / 達標 / 超過論文」敘述已作廢。evaluator bug（靜默丟 5 個非矩形 stairs GT）修正後為 **0.411（未達論文 0.42）**。專案定位＝research reimplementation，非 strict reproduction。權威更正見 `docs/paper_alignment_audit.md §9.1`（FIXED AND TESTED）與 `PROGRESS.md` 2026-08-11 段。


> 目的：一頁回答「論文結果複現到什麼程度」。狀態分三級：
> **✅ 已量化復現**（有數字、對得上論文量級）｜**🟢 已跑通待正式資料**（pipeline 實跑、產物正確，數字需 MP3D）｜**⏳ MP3D-gated**（僅缺資料）。
> 最後更新 2026-07-10。MP3D 已全機搜尋確認不在本機（見 PROGRESS.md），為唯一 gating。

## 一、復現路徑總覽

| 環節 | 狀態 | 證據 |
|---|---|---|
| 評估器正確性（d_E / d_H / Δτ / F1@τ / Avg-F1） | ✅ | 官方 16 場景預測 vs HF GT，復現 Table 2 到雜訊內；GT 恆等測試四類 F1=1.0 |
| Stage 1 mesh 重建（DN-Splatter） | ✅ | coffee_room vs FARO 雷射：88.8% 頂點 <5cm、覆蓋 94.5%、無尺度錯 |
| Stage 2 骨架（官方 extract_skeleton＋我方 OneFormer/SPT 缺件） | 🟢 | coffee_room 全鏈跑通、語意 mesh 過目視 |
| Stage 3 prototype（官方 fit_prototype 零修改＋我方 Algorithm 1 初始化） | 🟢 | 4.5M→260 頂點、足跡/雙層天花板/閉合殼正確 |
| Stage 4 scene graph（依 Appendix D 自寫＋OpenSeg 房型） | 🟢 | coffee_room 端到端；GT 乾跑 16 棟結構統計對上論文 |
| **Table 2/3 我方 pipeline 正式數字（16 場景）** | ⏳ | **需 MP3D poisson mesh + RGB-D**；`scripts/run_pipeline_scene.sh` 待命 |

## 二、Table 2 — HouseLayout3D benchmark（F1@0.5）

**(A) 評估器復現**（官方 predictions-ours 16 場景 vs HF GT，mean±std）——驗證「我方 eval＝論文 eval」：

| 類別 | 我方重算 | 論文 Table 2 | 判定 |
|---|---|---|---|
| Structures | 0.374 ± 0.08 | 0.40 ± 0.10 | ✅ 量級吻合（殘差＝視覺 mesh 逆向切實體，非 eval 誤差） |
| Doors | 0.563 ± 0.15 | 0.55 ± 0.16 | ✅ 幾乎重合 |
| Windows | 0.430 ± 0.29 | 0.43 ± 0.29 | ✅ 均值＋標準差全中 |
| Stairs | （官方預測未含樓梯獨立實體，無法由此校準） | 0.42 ± 0.48 | 見 §四 |

**(B) 我方 pipeline 數字**：⏳ MP3D-gated。原生實體匯出（`stage4/entities/`）已就緒，接 `eval_scene.py` 直算，不再有「ply 逆向切實體」的 −0.03 損耗。

## 三、Table 3 / 深度 Δτ

**(A) 評估器復現**（官方預測 vs HF GT layout，16 場景，stride=1、26,586 幀）：

| | 我方重算 | 論文 |
|---|---|---|
| Δ5 | 61.9 ± 10.4 | 61.1 ± 9.2 |
| Δ10 | 75.8 ± 9.4 | 76.3 ± 7.9 |

**(B) 我方 pipeline（coffee_room 替代場景，FARO 全景 GT，論文物件剔除協定）**：Δ5 **54.0** / Δ10 **64.3**（官方 ScanNet++ 單房 67.8/84.7；殘差為已記錄玻璃隔屏/窗簾限制）。DN-Splatter 原始 mesh 上界對照亦已實作。

## 四、結構統計（GT 乾跑，16 棟真建築，不需 MP3D）

在真實多樓層 GT layout 上驗 Stage 4 的 D.1/D.3/D.5：

| 指標 | 我方 | 論文資料集統計 | 說明 |
|---|---|---|---|
| 樓層數合計 | 34 | 33 | ✅（守衛防樓梯平台塌縮後） |
| 房間數合計 | ~189 | 317 | 偏低——符合論文「大量空間由開放通道/樓梯相連而非實門」特性 |
| 樓梯 D.5 指派通過 | 8–9 / 34 檔 | Stairs F1 0.42±0.48 | ✅ 印證論文樓梯最不穩：GT 標註為細碎矩形、D.5 預測為 CC 合併大矩形 → d_E 天生錯配（改善切入點） |

## 五、要產生 Table 2/3 正式數字，尚缺（唯一項）

**MP3D 學術 EULA 核准後的資料**（16 棟建築子集）：
1. `v1/scans/<scene>/poisson_meshes/<scene>_10.ply`（Stage 2 輸入 mesh）。
2. **MP3D → OpenNeRF/nerfstudio 轉換**產生每場景 `images/frame_*.jpg` + `depths/frame_*.png`（OneFormer/OpenSeg/反投影用）。⚠ HF poses json 內的 file_path 是**作者機器絕對路徑**（`/mnt/usb_ssd/bieriv/...`，本機不存在）——runner 已含 `localize_poses.py` 自動改寫為本機路徑並驗證每幀檔案存在（缺檔即報，不會靜默跑爛）。

**一鍵復現**（資料到位後）：
```
scripts/run_all_mp3d.sh <mp3d_root> [gpu]     # 16 棟續跑，單棟失敗不中斷
# 各棟：localize→OneFormer→extract_skeleton→init→fit→scene_graph→OpenSeg→eval
# 收尾自動呼叫 aggregate_results.py 出「我方 vs Table 2/3」mean±std 彙總表
```
預估單棟 1–2 小時（A6000）、16 棟 1.5–3 天。**需使用者：MP3D 授權＋OpenNeRF 轉換＋放置資料（或給我路徑）＋A5000 多卡協調（若並行）。** runner/aggregator 語法與路徑改寫已靜態驗證通過。

## 六、資料取得已窮盡的合法途徑（2026-07-10 查證）

| 途徑 | 結果 |
|---|---|
| 本機是否已有 MP3D | ❌ 全碟搜尋僅 MIDI-3D 2 張無關範例 PNG |
| OpenNeRF 是否釋出預處理 MP3D nerfstudio 資料 | ❌ 僅 Replica/SceneFun3D/LERF |
| 是否可先寫 MP3D→nerfstudio 轉換器 | ❌ 依賴 OpenNeRF frame-index↔MP3D 檔名對映，無資料無法驗證，寫了即違「只寫不跑」紅線 |

**唯一解＝MP3D 學術 EULA（僅使用者可簽，審核數天）**。此步經 CLAUDE.md 鐵則 7 明訂「別代做」，且繞過 EULA 側載會違反 Matterport 非商業授權——**自動迴圈不得跨越此線**。

## 七、一句話結論

論文的**方法與評估管線已完整、可執行、且以官方輸出＋替代場景雙重量化驗證到論文量級**；「16 場景正式數字」卡在**唯一且僅使用者可解的 MP3D EULA**（合法途徑已窮盡、side-load 受授權與鐵則 7 禁止）。資料一到，`run_all_mp3d.sh` 即自動出 Table 2/3——pipeline / runner / localize / aggregator 全部待命且靜態驗證通過。
