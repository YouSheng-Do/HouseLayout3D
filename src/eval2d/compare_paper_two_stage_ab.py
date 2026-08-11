"""Summarize and visualize the frozen same-input D.3 A/B checkpoint.

This script is reporting-only.  It reads canonical predictions, frozen GT and
artifact-only evaluator outputs; it never invokes Stage 4 or changes either
method's parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from canonical_io import load_canonical  # noqa: E402
from geometry_v2 import to_shapely  # noqa: E402
from metrics import align_levels, match_rooms_iou, prf  # noqa: E402
from split_io import load_split  # noqa: E402


EXPERIMENT = (ROOT / "outputs" / "eval2d" / "experiments" /
              "paper_two_stage_ab_v0_1" / "all")
EVAL_NAME = "eval2d_v3_strict_levels_official325_all"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
SPLIT_PATH = ROOT / "configs" / "eval2d" / "split_v0_1.json"
OUT = EXPERIMENT / "comparison"
FROZEN_EVAL = (ROOT / "outputs" / "eval2d" / "canonical" /
               "watershed_v3_pre_report_v0_2" /
               "eval2d_v3_strict_levels_official325_all")
METHODS = ("paper_spec_two_stage", "watershed_v3_rerun")
LABELS = {
    "paper_spec_two_stage": "paper_spec_two_stage",
    "watershed_v3_rerun": "watershed_v3（same-input rerun）",
}
COLORS = {
    "paper_spec_two_stage": "#D56A4A",
    "watershed_v3_rerun": "#287C8E",
}
PALETTE = ("#A9D6E5", "#89C2D9", "#61A5C2", "#B8D8BA", "#E8C07D",
           "#D8B4E2", "#F0A6A6", "#A8DADC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    summaries, scores, predictions = {}, {}, {method: {} for method in METHODS}
    for method in METHODS:
        method_dir = EXPERIMENT / method
        eval_dir = method_dir / EVAL_NAME
        summaries[method] = json.load((eval_dir / "summary.json").open())
        with (eval_dir / "scores.pkl").open("rb") as stream:
            scores[method] = pickle.load(stream)
        for path in sorted(method_dir.glob("*.json")):
            if path.name.endswith(".diagnostics.json") or path.name == "collection_manifest.json":
                continue
            predictions[method][path.stem] = load_canonical(json.load(path.open()))
    split = load_split(SPLIT_PATH)
    gt = {}
    for scene in split["dev"] + split["held_out"]:
        with (GT_DIR / f"{scene}.pkl").open("rb") as stream:
            gt[scene] = pickle.load(stream)
    return summaries, scores, predictions, split, gt


def metric_block(summary, partition):
    key = "selected" if partition == "all" else partition
    block = summary[key]
    return {
        "room": block["metrics"]["A"]["room"],
        "matched_iou": block["matched_iou"],
        "corner_0_1": block["metrics"]["A"]["corner@0.1"],
        "doors_0_5": block["metrics"]["B"]["doors@0.5"],
        "edge_all": block["metrics"]["C"]["all"],
    }


def included_gt_rooms(gt_scene):
    return [room for room in gt_scene["rooms"]
            if room["in_roomset"] and not room.get("is_stairs")]


def per_level_room_result(gt_scene, prediction, gt_level):
    mapping = align_levels(gt_scene["lvl_z"], prediction.get("level_z", {}))
    pred_level = mapping.get(gt_level)
    gt_rooms = [room for room in included_gt_rooms(gt_scene)
                if room["level"] == gt_level]
    pred_rooms = [room for room in prediction["rooms"]
                  if room["level"] == pred_level] if pred_level is not None else []
    matching, _ = match_rooms_iou(pred_rooms, gt_rooms, 0.5)
    counts = [len(matching), len(pred_rooms) - len(matching),
              len(gt_rooms) - len(matching)]
    return {
        "gt_level": gt_level,
        "pred_level": pred_level,
        "gt_rooms": gt_rooms,
        "pred_rooms": pred_rooms,
        "counts": counts,
        "f1": prf(*counts)["f1"],
    }


def choose_case_level(scene, gt_scene, predictions):
    candidates = []
    for gt_level in sorted({room["level"] for room in included_gt_rooms(gt_scene)}):
        paper = per_level_room_result(
            gt_scene, predictions["paper_spec_two_stage"][scene], gt_level)
        water = per_level_room_result(
            gt_scene, predictions["watershed_v3_rerun"][scene], gt_level)
        candidates.append((abs(paper["f1"] - water["f1"]),
                           abs(len(paper["pred_rooms"]) - len(water["pred_rooms"])),
                           gt_level, paper, water))
    if not candidates:
        raise RuntimeError(f"{scene}: no included GT levels")
    return max(candidates, key=lambda row: row[:2])


def polygon_parts(value):
    geometry = to_shapely(value)
    if geometry.is_empty:
        return []
    return [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)


def draw_rooms(ax, rooms, geometry_key, title):
    for index, room in enumerate(rooms):
        value = room[geometry_key] if geometry_key in room else room["poly"]
        for polygon in polygon_parts(value):
            exterior = np.asarray(polygon.exterior.coords)
            ax.fill(exterior[:, 0], exterior[:, 1],
                    color=PALETTE[index % len(PALETTE)], alpha=0.75,
                    ec="#263238", lw=0.8)
            for interior in polygon.interiors:
                hole = np.asarray(interior.coords)
                ax.fill(hole[:, 0], hole[:, 1], color="white", ec="#263238",
                        lw=0.5)
        geometry = to_shapely(value)
        if not geometry.is_empty:
            point = geometry.representative_point()
            ax.text(point.x, point.y, str(room["idx"]), ha="center", va="center",
                    fontsize=6, color="#1F2933")
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_aspect("equal")
    ax.axis("off")


def set_shared_bounds(axes, room_groups):
    bounds = []
    for rooms, geometry_key in room_groups:
        for room in rooms:
            value = room[geometry_key] if geometry_key in room else room["poly"]
            geometry = to_shapely(value)
            if not geometry.is_empty:
                bounds.append(geometry.bounds)
    if not bounds:
        return
    x0 = min(value[0] for value in bounds); y0 = min(value[1] for value in bounds)
    x1 = max(value[2] for value in bounds); y1 = max(value[3] for value in bounds)
    margin = max(x1 - x0, y1 - y0) * 0.04 + 0.1
    for ax in axes:
        ax.set_xlim(x0 - margin, x1 + margin)
        ax.set_ylim(y0 - margin, y1 + margin)


def make_plots(rows, gt, predictions):
    scenes = [row["scene"] for row in sorted(rows, key=lambda row: row["delta_f1"])]
    paper = [next(row for row in rows if row["scene"] == scene)["paper_f1"]
             for scene in scenes]
    water = [next(row for row in rows if row["scene"] == scene)["watershed_f1"]
             for scene in scenes]
    held = {row["scene"]: row["partition"] == "held_out" for row in rows}

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=160)
    y = np.arange(len(scenes))
    for index, scene in enumerate(scenes):
        ax.plot([paper[index], water[index]], [index, index], color="#B8C2CC",
                lw=1.8, zorder=1)
    ax.scatter(paper, y, s=52, color=COLORS["paper_spec_two_stage"],
               label=LABELS["paper_spec_two_stage"], zorder=2)
    ax.scatter(water, y, s=52, color=COLORS["watershed_v3_rerun"],
               label=LABELS["watershed_v3_rerun"], zorder=2)
    labels = [f"{scene}  {'H' if held[scene] else 'D'}" for scene in scenes]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.02); ax.set_xlabel("Room F1（strict level, IoU ≥ 0.5）")
    ax.set_title("Same-input paired Room F1：two-stage 多數場景低於 watershed")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.01, -0.09, "D = dev；H = prospective held-out（16 scenes 均曾被看過，非 untouched test）",
            transform=ax.transAxes, fontsize=8.5, color="#52606D")
    fig.tight_layout()
    f1_path = OUT / "paired_room_f1.png"
    fig.savefig(f1_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    ordered = sorted(rows, key=lambda row: row["scene"])
    x = np.arange(len(ordered)); width = 0.27
    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=160)
    ax.bar(x - width, [row["gt_rooms"] for row in ordered], width,
           label="GT rooms", color="#D9DEE3")
    ax.bar(x, [row["paper_rooms"] for row in ordered], width,
           label=LABELS["paper_spec_two_stage"], color=COLORS["paper_spec_two_stage"])
    ax.bar(x + width, [row["watershed_rooms"] for row in ordered], width,
           label=LABELS["watershed_v3_rerun"], color=COLORS["watershed_v3_rerun"])
    ax.set_xticks(x, [row["scene"] for row in ordered], rotation=42, ha="right")
    ax.set_ylabel("Room count（scene total）")
    ax.set_title("Room-count diagnosis：paper_spec_two_stage 系統性 over-segmentation")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    count_path = OUT / "room_count_comparison.png"
    fig.savefig(count_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best = max(rows, key=lambda row: row["delta_f1"])
    worst = min(rows, key=lambda row: row["delta_f1"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)
    case_meta = []
    for row_index, case in enumerate((best, worst)):
        scene = case["scene"]
        _, _, gt_level, paper_level, water_level = choose_case_level(
            scene, gt[scene], predictions)
        gt_rooms = [room for room in included_gt_rooms(gt[scene])
                    if room["level"] == gt_level]
        draw_rooms(axes[row_index, 0], gt_rooms, "poly",
                   f"GT · L{gt_level} · {len(gt_rooms)} rooms")
        draw_rooms(axes[row_index, 1], paper_level["pred_rooms"], "geometry",
                   f"paper · L{paper_level['pred_level']} · {len(paper_level['pred_rooms'])} rooms · F1 {paper_level['f1']:.2f}")
        draw_rooms(axes[row_index, 2], water_level["pred_rooms"], "geometry",
                   f"watershed · L{water_level['pred_level']} · {len(water_level['pred_rooms'])} rooms · F1 {water_level['f1']:.2f}")
        groups = [(gt_rooms, "poly"),
                  (paper_level["pred_rooms"], "geometry"),
                  (water_level["pred_rooms"], "geometry")]
        set_shared_bounds(axes[row_index], groups)
        label = "two-stage 最大改善" if row_index == 0 else "two-stage 最大退步"
        axes[row_index, 0].text(
            0.0, 1.13,
            f"{label}：{scene} · scene ΔF1 {case['delta_f1']:+.3f}",
            transform=axes[row_index, 0].transAxes, fontsize=11, weight="bold")
        case_meta.append({
            "case": "largest_gain" if row_index == 0 else "largest_drop",
            "scene": scene,
            "scene_delta_f1": case["delta_f1"],
            "gt_level": gt_level,
            "paper_pred_level": paper_level["pred_level"],
            "watershed_pred_level": water_level["pred_level"],
            "paper_level_f1": paper_level["f1"],
            "watershed_level_f1": water_level["f1"],
        })
    fig.suptitle("paper_spec_two_stage vs watershed_v3：paired qualitative cases",
                 fontsize=15, weight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "相同 Stage-4 inputs、相同 predicted levels、相同 polygonizer；圖中只比較 room geometry。",
             ha="center", fontsize=9, color="#52606D")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    cases_path = OUT / "paired_cases.png"
    fig.savefig(cases_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [f1_path, count_path, cases_path], case_meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    summaries, scores, predictions, split, gt = load_inputs()
    frozen_summary = json.load((FROZEN_EVAL / "summary.json").open())
    with (FROZEN_EVAL / "scores.pkl").open("rb") as stream:
        frozen_scores = pickle.load(stream)
    partition_of = {scene: "dev" for scene in split["dev"]}
    partition_of.update({scene: "held_out" for scene in split["held_out"]})
    scenes = sorted(partition_of)
    rows = []
    for scene in scenes:
        paper_score = scores["paper_spec_two_stage"][scene]
        water_score = scores["watershed_v3_rerun"][scene]
        paper_f1 = prf(*paper_score["A"]["room"])["f1"]
        water_f1 = prf(*water_score["A"]["room"])["f1"]
        rows.append({
            "scene": scene,
            "partition": partition_of[scene],
            "gt_rooms": len(included_gt_rooms(gt[scene])),
            "paper_rooms": len(predictions["paper_spec_two_stage"][scene]["rooms"]),
            "watershed_rooms": len(predictions["watershed_v3_rerun"][scene]["rooms"]),
            "paper_f1": paper_f1,
            "watershed_f1": water_f1,
            "delta_f1": paper_f1 - water_f1,
            "paper_room_counts": paper_score["A"]["room"],
            "watershed_room_counts": water_score["A"]["room"],
        })

    plot_paths, case_meta = make_plots(rows, gt, predictions)
    metrics = {
        partition: {
            method: metric_block(summaries[method], partition)
            for method in METHODS
        } for partition in ("dev", "held_out", "all")
    }
    rc_stats = {}
    for method in METHODS:
        errors = [value for scene in scenes
                  for value in scores[method][scene]["rc_err"]]
        rc_stats[method] = {
            "mean_signed_per_evaluated_level": float(np.mean(errors)),
            "mean_absolute_per_evaluated_level": float(np.mean(np.abs(errors))),
            "n_level_records": len(errors),
        }
    wins = sum(row["delta_f1"] > 1e-12 for row in rows)
    losses = sum(row["delta_f1"] < -1e-12 for row in rows)
    ties = len(rows) - wins - losses
    frozen_changed_scenes = [
        scene for scene in scenes
        if frozen_scores[scene]["A"]["room"] !=
        scores["watershed_v3_rerun"][scene]["A"]["room"]
    ]
    frozen_room = frozen_summary["selected"]["metrics"]["A"]["room"]
    rerun_room = metrics["all"]["watershed_v3_rerun"]["room"]
    payload = {
        "experiment": "paper_two_stage_ab_v0_1",
        "checkpoint": "phase2_paper_two_stage_ab_v0_1",
        "status": "frozen_same_input_ab",
        "generation_contract": (
            "same Stage-4 files; prototype loaded once; identify_levels once; "
            "deep-copied levels; method-only segmentation difference; CPU-only; "
            "no GT or evaluator scores available to generator"),
        "paper_faithfulness_boundary": (
            "Appendix D.3 width/order/door rule implemented. Morphology internals "
            "are an explicit reproducible interpretation, not released author code "
            "or a literal HOV-SG port."),
        "metrics": metrics,
        "room_count_error": rc_stats,
        "paired_scene_outcome": {
            "paper_better": wins, "paper_worse": losses, "tie": ties,
        },
        "historical_frozen_watershed_reference": {
            "room": frozen_room,
            "matched_iou": frozen_summary["selected"]["matched_iou"],
            "same_input_rerun_room_f1_delta": (
                rerun_room["f1"] - frozen_room["f1"]),
            "scenes_with_room_count_metric_change": frozen_changed_scenes,
            "explanation": (
                "The same-input rerun uses structured v0.2 mask polygonization. "
                "The historical frozen source stored only one legacy exterior "
                "ring; WYY7iVyf5p8 changes by one IoU-threshold match."),
        },
        "per_scene": rows,
        "visual_cases": case_meta,
        "limitations": [
            "All 16 scenes had been observed before split freeze; held_out is prospective, not untouched.",
            "Room types were set to unknown in both methods and must not be compared.",
            "Door and graph metrics are diagnostic because their GT associations are derived/lower-confidence.",
            "Matched IoU is conditional on successful IoU>=0.5 room matches.",
        ],
    }
    comparison_path = OUT / "comparison.json"

    def row(partition, method):
        value = metrics[partition][method]
        room = value["room"]
        return (f"{room['p']:.3f} | {room['r']:.3f} | {room['f1']:.3f} | "
                f"{value['matched_iou']['mean']:.3f} ({value['matched_iou']['n']})")

    paper_all = metrics["all"]["paper_spec_two_stage"]
    water_all = metrics["all"]["watershed_v3_rerun"]
    paper_doors = paper_all["doors_0_5"]
    water_doors = water_all["doors_0_5"]
    opening_status = {method: {} for method in METHODS}
    for method in METHODS:
        counts = {}
        for path in (EXPERIMENT / method).glob("*.diagnostics.json"):
            diagnostics = json.load(path.open())
            for level in diagnostics["levels"]:
                for opening in level["bottlenecks"]:
                    status = opening.get("canonical_export_status", "legacy")
                    counts[status] = counts.get(status, 0) + 1
        opening_status[method] = counts
    payload["door_annotation"] = {
        "paper_doors@0.5": paper_doors,
        "watershed_doors@0.5": water_doors,
        "opening_export_status": opening_status,
        "paper_contract": (
            "oriented bottleneck rectangle; width <1.5m; direct room pair; "
            "drop canonical door if either room geometry was not exported"),
    }
    comparison_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                          sort_keys=True) + "\n")
    best = max(rows, key=lambda value: value["delta_f1"])
    worst = min(rows, key=lambda value: value["delta_f1"])
    report = f"""# paper_spec_two_stage vs watershed_v3 — Same-input A/B v0.1

> checkpoint：`phase2_paper_two_stage_ab_v0_1` · CPU-only · 16 scenes · evaluator：`eval2d_v3_strict_levels` · GT：`mp3d_house_floor_v0_1`（official325）

## 結論

在這個固定輸入、固定 evaluator 的 A/B 中，`paper_spec_two_stage` **沒有優於**目前的 `watershed_v3`。全量 Room F1 是 **{paper_all['room']['f1']:.3f} vs {water_all['room']['f1']:.3f}**（Δ={paper_all['room']['f1']-water_all['room']['f1']:+.3f}）。主因不是成功配對房間的形狀品質：matched-only mean IoU 幾乎相同（{paper_all['matched_iou']['mean']:.3f} vs {water_all['matched_iou']['mean']:.3f}）；而是 two-stage 的 precision {paper_all['room']['p']:.3f} 明顯低於 watershed 的 {water_all['room']['p']:.3f}，呈現系統性 over-segmentation。

因此現階段應保留兩條線：`paper_spec_two_stage` 作 paper-alignment / ablation；研究用 annotated floorplan 主線繼續使用 `watershed_v3`，不能因方法名稱較接近論文就替換較好的 baseline。

## 量化結果

| split / method | Room P | Room R | Room F1 | matched IoU (n) |
|---|---:|---:|---:|---:|
| dev / paper_spec | {row('dev', 'paper_spec_two_stage')} |
| dev / watershed | {row('dev', 'watershed_v3_rerun')} |
| held-out / paper_spec | {row('held_out', 'paper_spec_two_stage')} |
| held-out / watershed | {row('held_out', 'watershed_v3_rerun')} |
| all / paper_spec | {row('all', 'paper_spec_two_stage')} |
| all / watershed | {row('all', 'watershed_v3_rerun')} |

`held_out` 只代表 split freeze 後沒有再用來改 heuristic；16 棟過去都曾被看過，所以不是 untouched test。dev 與 held-out 結論同方向。

### 與歷史 frozen watershed headline 的關係

本 A/B 的 watershed 是同輸入 fresh rerun（F1 **{rerun_room['f1']:.3f}**），不是直接拿歷史 frozen canonical（F1 **{frozen_room['f1']:.3f}**）充當對照。兩者只差 **{rerun_room['tp']-frozen_room['tp']:+d} TP**，發生在 `{', '.join(frozen_changed_scenes)}`：舊 frozen source 只能保存單一 exterior ring，新 rerun 由 mask 輸出 structured v0.2 topology；該場景一個 room 由舊 Polygon 變成 2-component MultiPolygon，剛好跨過 IoU=0.5 matching threshold。這個 -0.003 差異是 representation hardening 的結果，不是 A/B segmentation input 不一致。公平比較仍應使用 fresh rerun，因為兩法才共享同一個 polygonizer。

## Failure mechanism

- paired scenes：paper 較好 **{wins}/16**、較差 **{losses}/16**、平手 **{ties}/16**。
- 每個 evaluated level 的 signed room-count error：paper **{rc_stats['paper_spec_two_stage']['mean_signed_per_evaluated_level']:+.2f}**，watershed **{rc_stats['watershed_v3_rerun']['mean_signed_per_evaluated_level']:+.2f}**。
- 每層 room-count MAE：paper **{rc_stats['paper_spec_two_stage']['mean_absolute_per_evaluated_level']:.2f}**，watershed **{rc_stats['watershed_v3_rerun']['mean_absolute_per_evaluated_level']:.2f}**。
- paper 最大 scene gain：`{best['scene']}`，ΔF1 {best['delta_f1']:+.3f}；最大 drop：`{worst['scene']}`，ΔF1 {worst['delta_f1']:+.3f}。

這表示 2.5 m coarse split 被保留、再以 1.5 m refinement 新增小 cell 的規則，在目前帶噪牆／floorplan 輸入上會放大窄連接與碎片；但少數原本 watershed 欠分割的場景確實受益。

## Paper door/opening annotation

paper path 現在保存 bottleneck 的 oriented 2D rectangle、segment、width、stage 與 direct `room_a/room_b`；`width <1.5m` 才輸出 door，較寬者保留在 diagnostics 作 opening。若任一相鄰 room 沒有成功 polygonize，該 bottleneck 仍留在 diagnostics，但以 `dropped_missing_room_geometry` 標記，不建立 dangling canonical door/edge。

全 16 scenes 有 **{opening_status['paper_spec_two_stage'].get('exported_direct_door', 0)}** 個有效 paper doors、**{opening_status['paper_spec_two_stage'].get('diagnostic_opening_only', 0)}** 個 non-door openings、**{opening_status['paper_spec_two_stage'].get('dropped_missing_room_geometry', 0)}** 個因缺 room geometry 未輸出的 door candidates。Doors@0.5 是 paper **{paper_doors['f1']:.3f}**（P {paper_doors['p']:.3f} / R {paper_doors['r']:.3f}）vs watershed **{water_doors['f1']:.3f}**（P {water_doors['p']:.3f} / R {water_doors['r']:.3f}）。因此 paper bottleneck-only doors 目前不足以作研究用 annotation 主線；下一個 best-variant 槓桿仍是 direct semantic door detection/fusion。

## 「paper-spec」的精確邊界

[HouseLayout3D Appendix D.3](https://openreview.net/pdf/e20998737506e658d49d8d9d073931ac459638c7.pdf) 明載兩次 HOV-SG-style morphology segmentation（2.5 m，再 1.5 m）及 `<1.5 m` door rule；但公開文件沒有給足 morphology 的完整 implementation details，而 HouseLayout3D 的 Stage-4 author implementation 未提供。官方 [HOV-SG repository](https://github.com/hovsg/HOV-SG) 的公開 segmentation 是 distance-transform / watershed pipeline，也沒有直接暴露這組 2.5/1.5 m 參數。

本實作因此把 bottleneck width `w` 定義為 Euclidean erosion radius `w/2`，以 erosion 後 connected components 為 seeds、nearest-seed 回填，並在每個 2.5 m cell 內獨立做 1.5 m refinement。它實作了論文可觀察的 width / order / door contract，但必須稱為 **reproducible paper-spec interpretation**，不能稱 author-code reproduction 或 literal HOV-SG port。

兩法共用 0.08 m wall raster adapter、5 cm grid、相同 predicted levels 與 v0.2 hierarchy-aware polygonizer；0.08 m 並非 D.3 明訂參數。

## 可視化與資料

- paired Room F1：`paired_room_f1.png`
- scene room counts：`room_count_comparison.png`
- 最大 gain / drop floorplan：`paired_cases.png`
- machine-readable comparison：`comparison.json`
- canonical predictions：`../paper_spec_two_stage/`、`../watershed_v3_rerun/`
- exact input hashes：`../shared_input_manifests/`

## 限制與不可宣稱

- Room+type 在兩法都刻意設為 `unknown`，這個 A/B 不比較 room type。
- matched IoU 只對 IoU≥0.5 的成功 matched rooms 取平均，不能解讀成所有房間平均 IoU。
- door / connectivity 指標只作診斷；目前 GT graph/association 不是獨立完整 annotation。
- 此結果只隔離 Stage-4a room segmentation；不代表整條論文 pipeline 的最終 3D reproduction。
- 不可宣稱本 two-stage 是作者 hidden code，或證明論文方法本身較差；它只證明這個明示規格 interpretation 在目前輸入上的結果。

## 重現

```bash
CUDA_VISIBLE_DEVICES="" envs/geometry/bin/python src/eval2d/run_paper_two_stage_ab.py \\
  --partition all --checkpoint-name phase2_paper_two_stage_ab_v0_1

CUDA_VISIBLE_DEVICES="" envs/geometry/bin/python src/eval2d/evaluate_canonical.py \\
  --canonical-dir outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/paper_spec_two_stage \\
  --partition all --checkpoint-name phase2_paper_two_stage_ab_v0_1 \\
  --skip-expected-equivalence

CUDA_VISIBLE_DEVICES="" envs/geometry/bin/python src/eval2d/evaluate_canonical.py \\
  --canonical-dir outputs/eval2d/experiments/paper_two_stage_ab_v0_1/all/watershed_v3_rerun \\
  --partition all --checkpoint-name phase2_paper_two_stage_ab_v0_1 \\
  --skip-expected-equivalence

envs/geometry/bin/python src/eval2d/compare_paper_two_stage_ab.py
```
"""
    report_path = OUT / "RESULTS.md"
    report_path.write_text(report)
    manifest = {
        "inputs": {
            os.path.relpath(EXPERIMENT / method / EVAL_NAME / "summary.json", ROOT):
                sha256(EXPERIMENT / method / EVAL_NAME / "summary.json")
            for method in METHODS
        },
        "outputs": {
            path.name: sha256(path)
            for path in [comparison_path, report_path] + plot_paths
        },
    }
    manifest["inputs"][os.path.relpath(FROZEN_EVAL / "summary.json", ROOT)] = sha256(
        FROZEN_EVAL / "summary.json")
    manifest["inputs"][os.path.relpath(FROZEN_EVAL / "scores.pkl", ROOT)] = sha256(
        FROZEN_EVAL / "scores.pkl")
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                        sort_keys=True) + "\n")
    print(f"comparison report -> {report_path}")
    print(f"paired outcome: paper better/worse/tie = {wins}/{losses}/{ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
