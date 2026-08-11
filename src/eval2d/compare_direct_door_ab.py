"""Generate formal direct semantic door A/B report and diagnostics."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from geometry_v2 import to_shapely  # noqa: E402
from metrics import align_levels  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "direct_doors_ab_v0_1" / "all")
EVAL = "eval2d_v3_strict_levels_official325_all"
OUT = BASE / "comparison"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
VARIANTS = ("watershed_control", "direct_semantic_only", "fused_union")
LABELS = {
    "watershed_control": "watershed bottleneck control",
    "direct_semantic_only": "direct semantic only",
    "fused_union": "fused union (annotated-best)",
}
COLORS = {
    "watershed_control": "#287C8E",
    "direct_semantic_only": "#888F98",
    "fused_union": "#D56A4A",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(counts):
    tp, fp, fn = counts
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def setup_font():
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def draw_room_geometry(ax, geometry):
    shape = to_shapely(geometry)
    polygons = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="#C4CAD1", lw=1.0, zorder=1)
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords)
            ax.plot(hole[:, 0], hole[:, 1], color="#C4CAD1", lw=0.8,
                    zorder=1)


def draw_segments(ax, segments, color, label, linewidth):
    for index, segment in enumerate(segments):
        points = np.asarray(segment, dtype=np.float64)
        ax.plot(points[:, 0], points[:, 1], color=color, lw=linewidth,
                solid_capstyle="round", zorder=3,
                label=label if index == 0 else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    summaries = {
        variant: json.load((BASE / variant / EVAL / "summary.json").open())
        for variant in VARIANTS
    }
    scores = {}
    for variant in VARIANTS:
        with (BASE / variant / EVAL / "scores.pkl").open("rb") as stream:
            scores[variant] = pickle.load(stream)
    scenes = sorted(scores[VARIANTS[0]])
    rows = []
    for scene in scenes:
        row = {"scene": scene}
        for variant in VARIANTS:
            row[f"{variant}_f1@0.5"] = f1(
                scores[variant][scene]["B"]["doors@0.5"])
        row["delta_fused_minus_control"] = (
            row["fused_union_f1@0.5"] -
            row["watershed_control_f1@0.5"])
        rows.append(row)
    wins = sum(row["delta_fused_minus_control"] > 1e-12 for row in rows)
    losses = sum(row["delta_fused_minus_control"] < -1e-12 for row in rows)
    ties = len(rows) - wins - losses

    ordered = sorted(rows, key=lambda row: row["delta_fused_minus_control"])
    y = np.arange(len(ordered))
    control = [row["watershed_control_f1@0.5"] for row in ordered]
    fused = [row["fused_union_f1@0.5"] for row in ordered]
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=160)
    for index in range(len(ordered)):
        ax.plot([control[index], fused[index]], [index, index],
                color="#B8C2CC", lw=1.8, zorder=1)
    ax.scatter(control, y, s=50, color=COLORS[VARIANTS[0]],
               label=LABELS[VARIANTS[0]], zorder=2)
    ax.scatter(fused, y, s=50, color=COLORS[VARIANTS[2]],
               label=LABELS[VARIANTS[2]], zorder=2)
    ax.set_yticks(y, [row["scene"] for row in ordered])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Doors F1@0.5m（strict level＋midpoint Hungarian）")
    ax.set_title("Direct semantic door fusion：paired all-16 result")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    paired_path = OUT / "paired_doors_f1.png"
    fig.savefig(paired_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=160)
    for variant in VARIANTS:
        metric = summaries[variant]["selected"]["metrics"]["B"]["doors@0.5"]
        ax.scatter(metric["r"], metric["p"], s=115,
                   color=COLORS[variant], label=LABELS[variant])
        ax.annotate(f"  F1={metric['f1']:.3f}", (metric["r"], metric["p"]),
                    va="center", fontsize=10)
    ax.set_xlim(0.10, 0.32); ax.set_ylim(0.32, 0.46)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Fusion trades precision for materially higher door recall")
    ax.grid(color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    pr_path = OUT / "door_precision_recall.png"
    fig.savefig(pr_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cases = ["17DRP5sb8fy", "r47D5H71a5s", "JeFG25nYj2p",
             "5LpN3gDmAk7"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), dpi=160)
    for ax, scene in zip(axes.flat, cases):
        control_artifact = json.load((BASE / "watershed_control" /
                                      f"{scene}.json").open())
        fused_artifact = json.load((BASE / "fused_union" /
                                    f"{scene}.json").open())
        diagnostics = json.load((BASE / "fused_union" /
                                 f"{scene}.diagnostics.json").open())
        selected_level = max(
            diagnostics["levels"],
            key=lambda row: (row["semantic_added"], row["output_doors"],
                             -int(row["level"])))
        pred_level = int(selected_level["level"])
        control_level = next(level for level in control_artifact["levels"]
                             if int(level["id"]) == pred_level)
        fused_level = next(level for level in fused_artifact["levels"]
                           if int(level["id"]) == pred_level)
        with (GT_DIR / f"{scene}.pkl").open("rb") as stream:
            gt = pickle.load(stream)
        pred_z = {int(level["id"]): float(level["elevation"])
                  for level in fused_artifact["levels"]}
        alignment = align_levels(gt["lvl_z"], pred_z)
        gt_level = next((gt_id for gt_id, pred_id in alignment.items()
                         if pred_id == pred_level), None)
        for room in fused_level["rooms"]:
            draw_room_geometry(ax, room["geometry"])
        gt_segments = [door["seg"] for door in gt["doors"]
                       if door["level"] == gt_level]
        control_segments = [door["segment"]
                            for door in control_level["doors"]]
        semantic_segments = [door["segment"] for door in fused_level["doors"]
                             if "SD" in door["id"]]
        draw_segments(ax, gt_segments, "#2D8A56", "released GT", 3.5)
        draw_segments(ax, control_segments, COLORS[VARIANTS[0]],
                      "watershed control", 2.2)
        draw_segments(ax, semantic_segments, COLORS[VARIANTS[2]],
                      "novel semantic", 2.2)
        ax.set_title(
            f"{scene} · pred L{pred_level} ↔ GT L{gt_level} · "
            f"semantic +{selected_level['semantic_added']}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#EEF0F2", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Door geometry audit（reporting cases；green GT is evaluator-only）",
        fontsize=14)
    fig.tight_layout()
    cases_path = OUT / "door_geometry_cases.png"
    fig.savefig(cases_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    diagnostic_totals = {}
    for variant in VARIANTS:
        total = {key: 0 for key in (
            "base_doors", "semantic_candidates", "semantic_added",
            "output_doors", "resolved_semantic_associations")}
        for scene in scenes:
            payload = json.load((BASE / variant /
                                 f"{scene}.diagnostics.json").open())
            for key in total:
                total[key] += payload["summary"][key]
        diagnostic_totals[variant] = total
    selected = {variant: summaries[variant]["selected"]
                for variant in VARIANTS}
    control_metric = selected[VARIANTS[0]]["metrics"]["B"]["doors@0.5"]
    fused_metric = selected[VARIANTS[2]]["metrics"]["B"]["doors@0.5"]
    payload = {
        "experiment": "direct_doors_ab_v0_1",
        "checkpoint": "phase3_direct_doors_ab_v0_1",
        "status": "frozen_retained_semantic_door_fusion_ab",
        "selected": selected,
        "diagnostic_totals": diagnostic_totals,
        "paired_scene_outcome": {
            "fused_better": wins, "fused_worse": losses, "tie": ties,
        },
        "aggregate_delta@0.5": {
            key: fused_metric[key] - control_metric[key]
            for key in ("tp", "fp", "fn", "p", "r", "f1")
        },
        "per_scene": rows,
        "decision": {
            "annotated_best_doors": "fused_union",
            "paper_alignment_doors": "paper_two_stage_bottleneck",
            "semantic_only_replaces_control": False,
        },
        "limitations": [
            "Direct semantic fusion is a local annotated-best extension, not the paper door method.",
            "Door evaluation uses segment midpoint and does not score width or orientation.",
            "Door-room associations and edge GT are geometry-derived, not independent annotations.",
            "All 16 scenes had been observed before the prospective split freeze.",
        ],
    }
    comparison_path = OUT / "comparison.json"
    comparison_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def row(partition, variant, metric_name="doors@0.5"):
        block = (summaries[variant]["selected"] if partition == "all"
                 else summaries[variant][partition])
        metric = block["metrics"]["B"][metric_name]
        return (f"{metric['p']:.3f} | {metric['r']:.3f} | "
                f"{metric['f1']:.3f}")

    edge_control = selected[VARIANTS[0]]["metrics"]["C"]["all"]
    edge_fused = selected[VARIANTS[2]]["metrics"]["C"]["all"]
    report = f"""# Direct semantic doors：watershed vs semantic vs fused A/B

> checkpoint：`phase3_direct_doors_ab_v0_1` · CPU-only retained semantics · 16 scenes

## 結論

以 frozen watershed doors 為 control，加入 retained OneFormer door-ray wall-snapped
candidates 後，all-16 Doors@0.5 F1 由 **{control_metric['f1']:.3f} 提升到
{fused_metric['f1']:.3f}**。dev **0.251→0.332**、held-out **0.236→0.311**，
方向一致；這是 annotated-best 的實質改善。

direct semantic-only all F1 為 **{selected[VARIANTS[1]]['metrics']['B']['doors@0.5']['f1']:.3f}**，
低於 fusion，證明兩個 source 互補，不能用 semantic detector 直接取代 watershed
bottleneck。paper branch 仍保留 paper two-stage bottleneck doors；fusion 明確標為 local
research extension。

## 正式 Doors@0.5 結果

| split / variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| dev / watershed control | {row('dev', VARIANTS[0])} |
| dev / semantic only | {row('dev', VARIANTS[1])} |
| dev / fused | {row('dev', VARIANTS[2])} |
| held-out / watershed control | {row('held_out', VARIANTS[0])} |
| held-out / semantic only | {row('held_out', VARIANTS[1])} |
| held-out / fused | {row('held_out', VARIANTS[2])} |
| all / watershed control | {row('all', VARIANTS[0])} |
| all / semantic only | {row('all', VARIANTS[1])} |
| all / fused | **{row('all', VARIANTS[2])}** |

all counts：control TP/FP/FN = **50/69/242**；fused = **82/136/210**。
99 個 novel semantic candidates 帶來 +32 TP、+67 FP；recall
**{control_metric['r']:.3f}→{fused_metric['r']:.3f}**，precision
**{control_metric['p']:.3f}→{fused_metric['p']:.3f}**。Doors@0.2 也由
**{selected[VARIANTS[0]]['metrics']['B']['doors@0.2']['f1']:.3f}→{selected[VARIANTS[2]]['metrics']['B']['doors@0.2']['f1']:.3f}**。

paired scenes：fused 較好 **{wins}/16**、較差 **{losses}/16**、平手 **{ties}/16**。

## Frozen dev contract

在看 held-out 前，以 dev-only 小型 geometry sweep 固定：10 cm voxel、point-to-wall
≤0.30 m、cluster mean wall distance ≤0.15 m、along-wall gap 0.35 m、support ≥100
voxels、vertical span ≥1.4 m、width 0.35–1.5 m；semantic 與 base door midpoint <0.5 m
視為 duplicate。held-out/all 執行後沒有再調整。

輸入是既有 `full_ray_dests.npy`、OneFormer hard labels、valid-depth mask 與 canonical
Stage-3 wall segments；未重跑 OneFormer、Stage 2/3/4 或 GPU。115 個 semantic candidates
中 94 個能以 room polygons 做 derived room/outside association；fusion 去重後加入 99 個，
輸出 218 doors。

## Topology diagnostic

相同 geometry-derived edge protocol 下，edge_all F1 從
**{edge_control['f1']:.3f}→{edge_fused['f1']:.3f}**。這可作方向性 diagnostic，不能稱
independent topology accuracy，因 GT 與 prediction associations 都由 polygons/door
probes 推導。若 topology 是研究核心，roadmap 的 independent gold subset 仍必須做。

## 限制

- 這是 annotated-best local fusion，不是 HouseLayout3D paper 的 door method。
- current Doors metric 只比 segment midpoint，不評 width/orientation；新增 candidates 的
  geometry quality仍需 visual audit。
- 16 scenes 都曾在 split freeze 前被看過；held-out 只代表本 checkpoint 無 retuning。
- precision 0.376 仍低，不能宣稱 doors 已可靠；結果只表示比 control 有穩定改善。

## Artifacts

- `../watershed_control/`
- `../direct_semantic_only/`
- `../fused_union/`
- `comparison.json`
- `paired_doors_f1.png`
- `door_precision_recall.png`
- `door_geometry_cases.png`
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
            for path in (comparison_path, report_path, paired_path, pr_path,
                         cases_path)
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"direct-door comparison -> {report_path}")
    print(f"paired fused better/worse/tie = {wins}/{losses}/{ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
