"""Generate the formal retained-feature Room-type A/B report."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "room_types_ab_v0_1" / "all")
EVAL = "eval2d_room_types_v0_1_explicit_crosswalk_all"
OUT = BASE / "comparison"
VARIANTS = ("sample_point_room_mean", "mesh_vertex_k5_room_mean")
LABELS = {
    "sample_point_room_mean": "direct retained samples (best control)",
    "mesh_vertex_k5_room_mean": "mesh vertices k=5 (paper-spec interpretation)",
}
COLORS = {
    "sample_point_room_mean": "#287C8E",
    "mesh_vertex_k5_room_mean": "#D56A4A",
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
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def setup_font():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def diagnostics_summary(variant, scenes):
    classes = Counter()
    margins = []
    totals = Counter()
    for scene in scenes:
        payload = json.load((BASE / variant /
                             f"{scene}.diagnostics.json").open())
        totals.update(payload["summary"])
        totals["ambiguous_hits"] += payload["aggregation"]["assignment"][
            "ambiguous_hits"]
        totals["assigned_features_or_vertices"] += payload[
            "aggregation"]["assignment"]["assigned"]
        totals["unassigned_features_or_vertices"] += payload[
            "aggregation"]["assignment"]["unassigned"]
        for room in payload["rooms"]:
            classes[room["type"]] += 1
            if room["confidence"] is not None:
                margins.append(room["confidence"])
    return {
        "predicted_class_counts": dict(sorted(classes.items())),
        "median_top1_top2_cosine_margin": float(np.median(margins)),
        **dict(totals),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
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
        row = {"scene": scene}
        for variant in VARIANTS:
            score = scores[variant][scene]
            row[f"{variant}_room_type_f1"] = f1(
                score["room_type_counts"])
            row[f"{variant}_matched_mappable"] = score["matched_mappable"]
            row[f"{variant}_conditional_top1"] = (
                score["top1_correct"] / score["matched_mappable"]
                if score["matched_mappable"] else None)
        row["delta_room_type_f1_vertex_minus_sample"] = (
            row["mesh_vertex_k5_room_mean_room_type_f1"] -
            row["sample_point_room_mean_room_type_f1"])
        rows.append(row)

    wins = sum(row["delta_room_type_f1_vertex_minus_sample"] > 1e-12
               for row in rows)
    losses = sum(row["delta_room_type_f1_vertex_minus_sample"] < -1e-12
                 for row in rows)
    ties = len(rows) - wins - losses
    diagnostics = {
        variant: diagnostics_summary(variant, scenes)
        for variant in VARIANTS
    }

    ordered = sorted(rows,
                     key=lambda row: row[
                         "delta_room_type_f1_vertex_minus_sample"])
    y = np.arange(len(ordered))
    sample_values = [row["sample_point_room_mean_room_type_f1"]
                     for row in ordered]
    vertex_values = [row["mesh_vertex_k5_room_mean_room_type_f1"]
                     for row in ordered]
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=160)
    for index in range(len(ordered)):
        ax.plot([sample_values[index], vertex_values[index]], [index, index],
                color="#B8C2CC", lw=1.8, zorder=1)
    ax.scatter(sample_values, y, color=COLORS[VARIANTS[0]], s=50,
               label=LABELS[VARIANTS[0]], zorder=2)
    ax.scatter(vertex_values, y, color=COLORS[VARIANTS[1]], s=50,
               label=LABELS[VARIANTS[1]], zorder=2)
    ax.set_yticks(y, [row["scene"] for row in ordered])
    ax.set_xlim(0, max(0.65, max(sample_values + vertex_values) + 0.04))
    ax.set_xlabel("Room+type F1（all rooms；strict level＋room IoU>0.5）")
    ax.set_title("Room type aggregation：same-base paired A/B")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    paired_path = OUT / "paired_room_type_f1.png"
    fig.savefig(paired_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    classes = sorted(set().union(*(
        diagnostics[variant]["predicted_class_counts"]
        for variant in VARIANTS)))
    x = np.arange(len(classes)); width = 0.38
    fig, ax = plt.subplots(figsize=(14.5, 7.0), dpi=160)
    for offset, variant in ((-width / 2, VARIANTS[0]),
                            (width / 2, VARIANTS[1])):
        ax.bar(x + offset,
               [diagnostics[variant]["predicted_class_counts"].get(name, 0)
                for name in classes], width, color=COLORS[variant],
               label=LABELS[variant])
    ax.set_xticks(x, classes, rotation=42, ha="right")
    ax.set_ylabel("Predicted canonical rooms")
    ax.set_title("Class distribution reveals bedroom/entrance collapse")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    class_path = OUT / "predicted_class_distribution.png"
    fig.savefig(class_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected = {variant: summaries[variant]["selected"]
                for variant in VARIANTS}
    payload = {
        "experiment": "room_types_ab_v0_1",
        "checkpoint": "phase3_room_types_ab_v0_1",
        "status": "frozen_same_base_retained_feature_room_type_ab",
        "metric_contract": (
            "strict level alignment; room IoU>0.5 Hungarian; all-room "
            "Room+type F1 plus conditional top1/top3 only for "
            "geometry-matched rooms with an explicit non-null GT crosswalk"),
        "generation_contract": (
            "same frozen canonical rooms, retained OpenSeg samples, 15 CLIP "
            "text embeddings, and cosine argmax; aggregation path is the "
            "only experimental variable; no GT/score access during generation"),
        "selected": selected,
        "diagnostics": diagnostics,
        "paired_scene_outcome": {
            "vertex_better": wins, "vertex_worse": losses, "tie": ties,
        },
        "per_scene": rows,
        "decision": {
            "annotated_best": "sample_point_room_mean",
            "paper_alignment": "mesh_vertex_k5_room_mean",
            "room_types_reliable": False,
            "prune_rooms": False,
        },
        "limitations": [
            "The released Stage-4 author implementation is absent; mesh-vertex association is a reproducible paper-spec interpretation.",
            "Only 221/325 GT rooms have a conservative mapping to the paper 15-class ontology.",
            "Conditional accuracy excludes geometry misses and unmappable GT labels.",
            "All 16 scenes were observed before the prospective split freeze.",
        ],
    }
    comparison_path = OUT / "comparison.json"
    comparison_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def table_row(partition, variant):
        block = (summaries[variant]["selected"] if partition == "all"
                 else summaries[variant][partition])
        metric = block["room+type_all_rooms"]
        return (f"{metric['p']:.3f} | {metric['r']:.3f} | "
                f"{metric['f1']:.3f} | "
                f"{block['conditional_mappable_top1_accuracy']:.3f} | "
                f"{block['conditional_mappable_top3_accuracy']:.3f} | "
                f"{block['matched_mappable']}")

    sample = selected[VARIANTS[0]]
    vertex = selected[VARIANTS[1]]
    report = f"""# Room types：direct samples vs mesh-vertex k=5 A/B

> checkpoint：`phase3_room_types_ab_v0_1` · CPU-only retained features · 16 scenes

## 結論

paper-spec `mesh_vertex_k5_room_mean` 路徑已建立，但沒有改善 Room-type 品質。
all-16 end-to-end Room+type F1 為 **{sample['room+type_all_rooms']['f1']:.3f}**
（direct sample control）vs **{vertex['room+type_all_rooms']['f1']:.3f}**
（mesh vertex k=5）；geometry-matched 且 GT type 可保守映射的 rooms 上，Top-1 為
**{sample['conditional_mappable_top1_accuracy']:.3f} vs {vertex['conditional_mappable_top1_accuracy']:.3f}**。

因此 annotated-best 暫採 `sample_point_room_mean`，paper branch 保留
`mesh_vertex_k5_room_mean` 作 alignment/ablation。兩者皆維持
`room_types_reliable=false`，且不依低信心分類刪除 room polygons。

## 正式結果

| split / aggregation | Precision | Recall | Room+type F1 | conditional Top-1 | Top-3 | matched+mappable n |
|---|---:|---:|---:|---:|---:|---:|
| dev / direct samples | {table_row('dev', VARIANTS[0])} |
| dev / mesh vertices k=5 | {table_row('dev', VARIANTS[1])} |
| held-out / direct samples | {table_row('held_out', VARIANTS[0])} |
| held-out / mesh vertices k=5 | {table_row('held_out', VARIANTS[1])} |
| all / direct samples | {table_row('all', VARIANTS[0])} |
| all / mesh vertices k=5 | {table_row('all', VARIANTS[1])} |

兩法使用完全相同 geometry，故 Room F1 都是 **{sample['room']['f1']:.3f}**
（TP/FP/FN={sample['room']['tp']}/{sample['room']['fp']}/{sample['room']['fn']}）。
paired scenes 中 vertex 較好 **{wins}/16**、較差 **{losses}/16**、平手
**{ties}/16**；dev 與 held-out 均未勝過 direct-sample control。

## 指標應如何解讀

- GT 共 325 rooms，只有 **221（68.0%）**能以保守 crosswalk 對應到 paper 15
  classes；104 rooms（如 hallway、dining、closet）刻意不為拉高分數而硬合併。
- `Room+type F1` 是嚴格 end-to-end 指標：geometry miss、type 錯誤及無可映射
  ontology 的 rooms 都會降低分數，因此不是純 classifier accuracy。
- conditional Top-1/Top-3 只在 **139** 個 geometry-matched＋mappable rooms 上計算，
  可看分類訊號，但不能掩蓋 geometry misses 或 ontology coverage。
- direct sample 的 predicted types 高度集中在 bedroom（126/316）與 entrance
  （107/316），Top-1/Top-2 cosine margin median 僅
  **{diagnostics[VARIANTS[0]]['median_top1_top2_cosine_margin']:.4f}**；目前不適合稱為
  reliable annotation。

舊報告的 Room+type F1 0.144 使用不同 artifact/evaluator ontology contract，不能直接
拿來宣稱本輪提升 0.043；本 checkpoint 的可信比較只有同一列內的兩個 aggregation。

## Paper alignment boundary

- Appendix D.4：OpenSeg pixel-aligned features 依 mesh segmentation 相同方式投影到
  mesh vertices，再對 room 中 vertices 取平均，以 CLIP 分 15 classes。
- 本 `mesh_vertex_k5_room_mean` 使用 retained OpenSeg samples、structural mesh vertices
  與 mesh segmentation 的 k=5 KNN；保存完整 15 cosine scores。
- 作者 Stage-4 implementation 未釋出，因此 vertex-to-room association 與 retained
  sampling 細節是 reproducible local interpretation，不稱 author-code reproduction。
- paper 的 last-five outdoor leaf rule只記成 candidate：direct sample 45 rooms、vertex
  41 rooms；全部保留，避免低 confidence class 誤刪 geometry。

## 執行與 artifacts

- CPU-only；未重跑 OpenSeg/CLIP inference、Stage 2/3/4、room segmentation、pipeline
  或 GPU。
- 16 scenes 各 variant 都輸出 canonical v0.2 JSON＋diagnostics＋hash manifest。
- direct samples 有 315/316 rooms 具 features；mesh vertices 316/316。
- `../sample_point_room_mean/`
- `../mesh_vertex_k5_room_mean/`
- `comparison.json`
- `paired_room_type_f1.png`
- `predicted_class_distribution.png`
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
            for path in (comparison_path, report_path, paired_path, class_path)
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"room-type comparison -> {report_path}")
    print(f"paired vertex better/worse/tie = {wins}/{losses}/{ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
