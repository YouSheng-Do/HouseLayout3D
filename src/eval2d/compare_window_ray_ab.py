"""Report the formal legacy-window-rays vs paper-plus-outdoor A/B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "window_rays_ab_v0_1" / "all")
EVAL = "eval2d_windows_v0_2_endpoint_hungarian_all"
OUT = BASE / "comparison"
VARIANTS = ("legacy_three_classes", "paper_plus_outdoor")
LABELS = {
    "legacy_three_classes": "legacy：window/blind/curtain",
    "paper_plus_outdoor": "paper classes：+outdoor",
}
COLORS = {
    "legacy_three_classes": "#287C8E",
    "paper_plus_outdoor": "#D56A4A",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    summaries = {
        variant: json.load((BASE / variant / EVAL / "summary.json").open())
        for variant in VARIANTS
    }
    scores = {
        variant: json.load((BASE / variant / EVAL / "scores.json").open())
        for variant in VARIANTS
    }
    scenes = sorted(scores[VARIANTS[0]])
    rows = []
    for scene in scenes:
        old = scores["legacy_three_classes"][scene]
        paper = scores["paper_plus_outdoor"][scene]
        old_count = old["counts"]["0.5"]
        paper_count = paper["counts"]["0.5"]
        old_pred = sum(row["pred_windows"] for row in old["per_level"])
        paper_pred = sum(row["pred_windows"] for row in paper["per_level"])
        gt_count = sum(row["gt_windows"] for row in old["per_level"])
        diagnostics = json.load(
            (BASE / "paper_plus_outdoor" /
             f"{scene}.diagnostics.json").open())
        rows.append({
            "scene": scene,
            "legacy_f1@0.5": f1(old_count),
            "paper_f1@0.5": f1(paper_count),
            "delta_f1@0.5": f1(paper_count) - f1(old_count),
            "legacy_counts@0.5": old_count,
            "paper_counts@0.5": paper_count,
            "gt_windows": gt_count,
            "legacy_pred_windows": old_pred,
            "paper_pred_windows": paper_pred,
            "paper_outdoor_evidence_records": diagnostics["summary"][
                "outdoor_supported_records"],
            "paper_room_associated": diagnostics["summary"]["room_associated"],
        })

    wins = sum(row["delta_f1@0.5"] > 1e-12 for row in rows)
    losses = sum(row["delta_f1@0.5"] < -1e-12 for row in rows)
    ties = len(rows) - wins - losses
    selected = {variant: summaries[variant]["selected"] for variant in VARIANTS}
    old_metric = selected["legacy_three_classes"]["metrics"]["windows@0.5"]
    paper_metric = selected["paper_plus_outdoor"]["metrics"]["windows@0.5"]

    ordered = sorted(rows, key=lambda row: row["delta_f1@0.5"])
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=160)
    y = np.arange(len(ordered))
    old_values = [row["legacy_f1@0.5"] for row in ordered]
    paper_values = [row["paper_f1@0.5"] for row in ordered]
    for index in range(len(ordered)):
        ax.plot([old_values[index], paper_values[index]], [index, index],
                color="#B8C2CC", lw=1.8, zorder=1)
    ax.scatter(old_values, y, color=COLORS["legacy_three_classes"], s=50,
               label=LABELS["legacy_three_classes"], zorder=2)
    ax.scatter(paper_values, y, color=COLORS["paper_plus_outdoor"], s=50,
               label=LABELS["paper_plus_outdoor"], zorder=2)
    ax.set_yticks(y, [row["scene"] for row in ordered])
    ax.set_xlim(0, max(0.72, max(old_values + paper_values) + 0.04))
    ax.set_xlabel("Windows F1@0.5m（2D endpoint-max Hungarian）")
    ax.set_title("Outdoor rays：endpoint-aware same-input A/B")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    paired_path = OUT / "paired_window_f1.png"
    fig.savefig(paired_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    ordered = sorted(rows, key=lambda row: row["scene"])
    x = np.arange(len(ordered)); width = 0.27
    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=160)
    ax.bar(x - width, [row["gt_windows"] for row in ordered], width,
           color="#D9DEE3", label="released GT")
    ax.bar(x, [row["legacy_pred_windows"] for row in ordered], width,
           color=COLORS["legacy_three_classes"],
           label=LABELS["legacy_three_classes"])
    ax.bar(x + width, [row["paper_pred_windows"] for row in ordered], width,
           color=COLORS["paper_plus_outdoor"],
           label=LABELS["paper_plus_outdoor"])
    ax.set_xticks(x, [row["scene"] for row in ordered], rotation=42, ha="right")
    ax.set_ylabel("Window segments（scene total）")
    ax.set_title("加入 outdoor 後候選數增加：709 vs 637（GT 379）")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    count_path = OUT / "window_count_comparison.png"
    fig.savefig(count_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "experiment": "window_rays_ab_v0_1",
        "checkpoint": "phase3_window_rays_ab_v0_1",
        "status": "frozen_same_input_window_annotation_ab",
        "metric_contract": (
            "released HouseLayout3D 3D rectangle bottom edge -> 2D segment; "
            "strict predicted/GT level alignment; orientation-invariant "
            "Hungarian endpoint-max distance"),
        "generation_contract": (
            "same Stage-4 inputs and frozen watershed rooms; prototype loaded "
            "once per scene; only ray class set changes; no GT/score access"),
        "selected": selected,
        "paired_scene_outcome": {
            "paper_better": wins, "paper_worse": losses, "tie": ties,
        },
        "aggregate_delta@0.5": {
            "tp": paper_metric["tp"] - old_metric["tp"],
            "fp": paper_metric["fp"] - old_metric["fp"],
            "fn": paper_metric["fn"] - old_metric["fn"],
            "precision": paper_metric["p"] - old_metric["p"],
            "recall": paper_metric["r"] - old_metric["r"],
            "f1": paper_metric["f1"] - old_metric["f1"],
        },
        "per_scene": rows,
        "limitations": [
            "2D window metric is newly defined and is not a paper-reported metric.",
            "DBSCAN, LOF, and ray-length slack remain local assumptions.",
            "Window-room association is nearest-geometry candidate, not independent truth.",
            "All 16 scenes were observed before the prospective split freeze.",
        ],
    }
    comparison_path = OUT / "comparison.json"
    comparison_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def table_row(partition, variant):
        block = (summaries[variant]["selected"] if partition == "all"
                 else summaries[variant][partition])
        value = block["metrics"]["windows@0.5"]
        return (f"{block['pred_windows']} | {value['p']:.3f} | "
                f"{value['r']:.3f} | {value['f1']:.3f}")

    recall_change = "相同" if abs(paper_metric["r"] - old_metric["r"]) < 1e-12 else "改變"
    report = f"""# Window rays：legacy three classes vs paper + outdoor

> checkpoint：`phase3_window_rays_ab_v0_1` · CPU-only · 16 scenes · evaluator：`eval2d_windows_v0_2_endpoint_hungarian`

## 結論

依論文 class list 加入 `outdoor` rays 後，all-16 recall {recall_change}（**{old_metric['r']:.3f} → {paper_metric['r']:.3f}**），新增 **{paper_metric['tp']-old_metric['tp']}** 個 released GT window matches；prediction 增加 72 個，FP 增加 **{paper_metric['fp']-old_metric['fp']}**，precision 由 **{old_metric['p']:.3f} → {paper_metric['p']:.3f}**，Windows@0.5 F1 因而由 **{old_metric['f1']:.3f} 降至 {paper_metric['f1']:.3f}**。

這是 paper-alignment 成功、品質沒有提升的明確案例。`paper_plus_outdoor` 應保留為 paper-spec branch；研究用 best branch 暫時保留三類 control，後續若改善 outdoor rays，必須用 geometry/semantic gating 並標為 local extension。

## 結果

| split / variant | Pred windows | Precision | Recall | F1@0.5 |
|---|---:|---:|---:|---:|
| dev / legacy | {table_row('dev', 'legacy_three_classes')} |
| dev / +outdoor | {table_row('dev', 'paper_plus_outdoor')} |
| held-out / legacy | {table_row('held_out', 'legacy_three_classes')} |
| held-out / +outdoor | {table_row('held_out', 'paper_plus_outdoor')} |
| all / legacy | {table_row('all', 'legacy_three_classes')} |
| all / +outdoor | {table_row('all', 'paper_plus_outdoor')} |

paired scenes：paper 較好 **{wins}/16**、較差 **{losses}/16**、平手 **{ties}/16**。dev 與 held-out 都是 recall 不變、precision/F1 下降，方向一致。

## 評估與 annotation contract

- GT：released `external/houselayout3d/data/windows/*.json` 3D rectangle 的 bottom edge 投影成 2D segment。
- level：GT 與 prediction 各自先依 floor elevation 指派，再使用 strict level alignment。
- matching：2D segment endpoint-max distance（端點方向取較小者）Hungarian；報告 0.2 m／0.5 m，本表使用 0.5 m。中心正確但長度或方向錯誤不算命中。
- canonical windows 保存 2D segment、direct Stage-3 wall PID、nearest-room candidate、confidence 與 ray-class evidence sidecar。
- generator 不讀 GT 或 score；prototype 每棟 load 一次；沒有 Stage 2/3 rerun、room segmentation、extrusion 或 GPU。

## 限制

- 這個 2D Windows metric 是本專案新增指標，不是論文 Table 2 的 3D rectangle metric。
- 加入 `outdoor` 符合論文 class list；DBSCAN、LOF、ray slack 仍是 local assumptions。
- room association 是 nearest-geometry candidate，不是人工 truth。
- 不能從本 A/B 宣稱 outdoor rays 沒有價值；目前只能說未加 gating 時 false positives 超過新增 true positives。

## Artifacts

- `../legacy_three_classes/`
- `../paper_plus_outdoor/`
- `comparison.json`
- `paired_window_f1.png`
- `window_count_comparison.png`
"""
    report_path = OUT / "RESULTS.md"
    report_path.write_text(report)
    manifest = {
        "inputs": {
            str((BASE / variant / EVAL / "summary.json").relative_to(ROOT)):
                sha256(BASE / variant / EVAL / "summary.json")
            for variant in VARIANTS
        },
        "outputs": {
            path.name: sha256(path)
            for path in (comparison_path, report_path, paired_path, count_path)
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"window comparison -> {report_path}")
    print(f"paired paper better/worse/tie = {wins}/{losses}/{ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
