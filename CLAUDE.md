# CLAUDE.md — MULTIFLOOR3D 重現專案鐵則

> 每個 session 都會讀到這份。這裡是不能妥協的紅線；完整說明見 project brief。
> 語言：一律用繁體中文或英文回報。

## 每個 session 必讀：論文一致性
- 開始修改 pipeline、evaluator、實驗結論或對外說明前，先讀
  `docs/paper_alignment_audit.md`。這是截至 2026-08-10 對論文、Appendix、作者
  supplementary code、本地 runtime 與 evaluator 的交叉稽核。
- 本專案目前研究目標是 **annotated 2D floorplan**；涉及實作順序、範圍、產物 schema、
  2D evaluator 或時程時，再讀 `docs/annotated_floorplan_roadmap.md`。3D extrusion/depth
  不在目前主線，不得讓它阻擋 canonical 2D artifact。
- 不得把本地 heuristic/改良方法直接稱為 paper-faithful；依該文件區分
  **Paper-faithful**、**Official-code-faithful**、**Local engineering variant**、
  **Unverified assumption**。
- `PROGRESS.md` 保留實驗歷史與最新執行紀錄；若其中舊結論與 alignment audit 或目前
  source code 衝突，以論文／作者 code／目前 runtime 的交叉證據為準，並同步更新文件。

## 任務本質（2026-07-06 修訂）
- 官方 GitHub repo 只有資料集下載 + 視覺化工具;但 **NeurIPS supplementary
  (`docs/supplementary/`)含官方 Stage 2/3 程式碼、附錄、16 場景官方預測**。
- 策略(使用者拍板):**官方 code 為主幹**——Stage 2 用 `extract_skeleton.py`、
  Stage 3 用 `fit_prototype.py`（Matterport runtime 參數以官方
  `mesh_fitting_3D/polygon_fitting_config.py` 為準；`docs/param_inventory_v2.md` 有已知舊數值）;
  **Stage 4 與評估腳本在 zip 裡是空檔,依 Appendix D 規格自寫**;OpenSeg 房型分類一開始就裝。
- **Eval-first**:先用 `predictions-ours/` 對 HF GT 校準評估腳本(不需 MP3D),再跑 pipeline。
- 執行進度以 **PROGRESS.md 最新日期段落**為準；paper/code alignment 以
  `docs/paper_alignment_audit.md` 為準；brief/research.md 是歷史文件。

## 執行紀律（每一條都必須遵守）
1. **一次只做一個階段。** 每階段結束就**停下來**,等我確認才進下一階段。不要一路衝到底。
2. **先單場景跑通,再擴全量。** 每個階段第一次只在**一個場景**上驗證。禁止一次上 16 棟,
   直到 pipeline 全程在單場景通過。
3. **所有修改都要實跑,不是空寫。** MP3D 16-scene pipeline 已有正式執行紀錄；新修改先用
   單一 MP3D 場景或適當 fixture 驗證，通過檢查點後才擴全量。「寫完但沒執行」不算完成。
   另把 smoke-test 的「能不能跑」與 MP3D benchmark 的「跑得對不對」分開回報。
4. **缺參數→停下來問,不要猜。** loss 權重、各種 threshold、初始化若論文沒給明確值,
   列清單問我 + 附建議值與理由。現階段先寫成集中在一份 config、標 `# TODO: tune with MP3D`
   的可調參數,**絕不靜默填猜測值往下走**。
5. **卡關就停,不要無限迴圈。** 同一個問題連續嘗試數次仍未解決,停下來寫報告等我,
   不要一直重試燒 GPU 時數。
6. **危險操作一定停下來問。** 刪資料、覆寫、動到共用 GPU 資源——先問我。
7. **需要我本人操作的,明確標記讓我做,別代做。** MP3D 授權、資料下載、A5000 協調、
   任何要登入或簽授權的步驟。共用 GPU(A5000)要用前先問我協調。

## 回報格式（每個檢查點）
- **先給「手機用純文字摘要」開頭**(3–6 行:跑完了嗎、產物合不合理、有無異常、
  你的判斷、需不需要我回電腦看圖),讓我光讀文字就能決定繼續或喊停。
- 視覺化圖檔另存路徑,供我回電腦細看。
- 產物明顯有問題時,摘要第一行標「⚠️ 建議回電腦確認」。
- 環境衝突→把選項與 trade-off 列給我,不要自行做重大取捨。

## 進度紀錄
- 把每階段的進度、關鍵決策、未定參數,寫進一份 `PROGRESS.md`(不要只留在對話裡)。
  這樣跨 session 或重開對話都接得上。
