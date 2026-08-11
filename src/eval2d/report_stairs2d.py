"""Generate the durable report and visual diagnostics for stairs2d v0.1."""
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from evaluate_stairs2d import canonical_stairs  # noqa: E402
from geometry_v2 import to_shapely  # noqa: E402
from stairs2d import official_stair_entities  # noqa: E402


BASE = (ROOT / "outputs" / "eval2d" / "experiments" /
        "stairs2d_v0_1" / "all" / "annotated_best")
EVAL = "eval2d_stairs_v0_1_footprint_iou_hungarian_all"
GT_DIR = (ROOT / "outputs" / "eval2d" / "gt_candidates" /
          "mp3d_house_floor_v0_1")
STAIR_GT_DIR = ROOT / "external" / "houselayout3d" / "data" / "stairs"
OUT = BASE / "report"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_font():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def polygons(geometry):
    shape = to_shapely(geometry)
    if shape.geom_type == "Polygon":
        return [shape]
    return list(shape.geoms) if shape.geom_type == "MultiPolygon" else []


def draw_geometry(ax, geometry, color, fill=False, label=None, linewidth=2.0):
    first = True
    for polygon in polygons(geometry):
        xy = np.asarray(polygon.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=linewidth,
                label=label if first else None)
        if fill:
            ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=0.16)
        for hole in polygon.interiors:
            ring = np.asarray(hole.coords)
            ax.plot(ring[:, 0], ring[:, 1], color=color, lw=1.2)
        first = False


def scene_entities(scene):
    artifact = json.load((BASE / f"{scene}.json").open())
    _, predicted = canonical_stairs(artifact)
    with (GT_DIR / f"{scene}.pkl").open("rb") as stream:
        gt = pickle.load(stream)
    official = official_stair_entities(scene, gt["lvl_z"], STAIR_GT_DIR)
    return official, predicted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    summary_path = BASE / EVAL / "summary.json"
    scores_path = BASE / EVAL / "scores.json"
    summary = json.load(summary_path.open())
    scores = json.load(scores_path.open())
    scenes = sorted(scores)

    ordered = sorted(
        scenes,
        key=lambda scene: (
            scores[scene]["counts"][0] + scores[scene]["counts"][2] == 0
            and scores[scene]["counts"][1] == 0,
            summary["selected"]["per_scene_f1"][scene]
            if summary["selected"]["per_scene_f1"][scene] is not None else -1,
            scene),
    )
    values = [summary["selected"]["per_scene_f1"][scene]
              for scene in ordered]
    display = [0.0 if value is None else value for value in values]
    colors = ["#D7DCE1" if value is None else "#D56A4A" for value in values]
    fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=160)
    ax.bar(np.arange(len(ordered)), display, color=colors)
    ax.set_xticks(np.arange(len(ordered)), ordered, rotation=42, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Stair-footprint F1 (IoU>0.5)")
    ax.set_title("16-scene stair footprint：5/12 predicted entities matched")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    for index, value in enumerate(values):
        if value is None:
            ax.text(index, 0.025, "N/A", ha="center", va="bottom",
                    rotation=90, fontsize=8, color="#59636E")
    fig.tight_layout()
    f1_path = OUT / "per_scene_stair_f1.png"
    fig.savefig(f1_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cases = ["S9hNv5qa7GM", "e9zR4mvMWw7", "WYY7iVyf5p8",
             "r47D5H71a5s"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), dpi=160)
    for ax, scene in zip(axes.flat, cases):
        gt, pred = scene_entities(scene)
        for index, item in enumerate(gt):
            draw_geometry(ax, item["geometry"], "#287C8E", fill=True,
                          label="released GT entities" if index == 0 else None,
                          linewidth=1.8)
        for index, item in enumerate(pred):
            draw_geometry(ax, item["geometry"], "#D56A4A", fill=False,
                          label="D.5 merged prediction" if index == 0 else None,
                          linewidth=2.6)
        counts = scores[scene]["counts"]
        value = summary["selected"]["per_scene_f1"][scene]
        title_f1 = "N/A" if value is None else f"{value:.3f}"
        ax.set_title(
            f"{scene} · GT {counts[0]+counts[2]} / pred {counts[0]+counts[1]} "
            f"· F1 {title_f1}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#EEF0F2", lw=0.6)
        if gt or pred:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Stair footprint overlays（樓層疊畫；僅診斷 geometry，不是 link GT）",
        fontsize=14)
    fig.tight_layout()
    cases_path = OUT / "stair_footprint_cases.png"
    fig.savefig(cases_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected = summary["selected"]
    metric = selected["metrics"]["stair_footprint_iou>0.5"]
    no_prediction = sum(
        scores[scene]["counts"][0] + scores[scene]["counts"][2] > 0
        and scores[scene]["counts"][0] + scores[scene]["counts"][1] == 0
        for scene in scenes)
    report = f"""# Stairs 2D footprint checkpoint v0.1

> checkpoint：`phase3_stairs2d_v0_1` · CPU-only · 16 scenes · artifact-only

## 結論

existing Stage-4 D.5 outputs 產生 **{selected['predicted_stairs']}** 個 stair
predictions；released GT 有 **{selected['gt_stair_entities']}** 個 mesh entities。
strict level alignment＋footprint IoU>0.5 Hungarian 的結果為：

| split | Pred | GT | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| dev | {summary['dev']['predicted_stairs']} | {summary['dev']['gt_stair_entities']} | {summary['dev']['metrics']['stair_footprint_iou>0.5']['p']:.3f} | {summary['dev']['metrics']['stair_footprint_iou>0.5']['r']:.3f} | {summary['dev']['metrics']['stair_footprint_iou>0.5']['f1']:.3f} |
| held-out | {summary['held_out']['predicted_stairs']} | {summary['held_out']['gt_stair_entities']} | {summary['held_out']['metrics']['stair_footprint_iou>0.5']['p']:.3f} | {summary['held_out']['metrics']['stair_footprint_iou>0.5']['r']:.3f} | {summary['held_out']['metrics']['stair_footprint_iou>0.5']['f1']:.3f} |
| all | **{selected['predicted_stairs']}** | **{selected['gt_stair_entities']}** | **{metric['p']:.3f}** | **{metric['r']:.3f}** | **{metric['f1']:.3f}** |

all counts：TP/FP/FN = **{metric['tp']}/{metric['fp']}/{metric['fn']}**。

## 如何解讀 0.217

這不是單純的 geometry quality：D.5 先以 0.4 m nearby-component merge 合併整座
樓梯，released annotation 則常把同一物理樓梯拆成 flights/landings。12 個 prediction
即使全部命中 34 個 entity，entity-level recall 上限也只有 12/34=0.353；目前實際命中
5/12 predictions。另有 **{no_prediction}** 個含 GT stairs 的 scenes 完全沒有 prediction。

因此本分數適合揭露 strict released-entity contract 與 failure modes，不應直接宣稱
「annotated floorplan 的樓梯只有 21.7% 可用」。若研究需要 physical-staircase 指標，
下一版必須先建立人工 grouping/link gold subset，不能為提高數字事後用 prediction 的
0.4 m 規則合併 GT。

## Multi-level links

canonical artifacts 保留 D.5 的 direct prediction associations：12/12 有完整
`from_level/to_level` 與 ordered adjacent rooms，reference checks 全通過；其中 e9zR 有
1 個 link 跨過非相鄰 predicted level index，已列入 audit。這是 **coverage，不是
accuracy**。

正式 Stair-link metric 為 **N/A**：released 34 meshes 沒有官方 from/to-level 或
adjacent-room 標註，且不從 mesh z 猜 `to_level`。footprint evaluator 只用
`highest floor <= min(mesh_z)+0.15m` 指派 owning floor，並明確不把它當 link GT。

## Contract 與限制

- GT footprint：每個 released mesh 的 XY triangle union；34/34 保留，不 grouping。
- prediction footprint：D.5 OBB 的 endpoint storage order `01/23` 轉為有效 perimeter
  `0-1-3-2`；舊 3D d_E/d_H evaluator 不受影響。
- 本方法的 0.4 m merge、0.3 m² minimum area、0.4 m minimum rise、0.5 m room
  assignment rejection 都是 local best-variant assumptions。
- 2D Stair F1 **{metric['f1']:.3f}** 與既有 3D Stairs F1 **0.411** 是不同 metric，
  不可互相比大小或混寫。
- 全程只轉換既有 Stage-4 JSON；未重跑 stair detection、Stage 2/3、pipeline、extrusion
  或 GPU。

## Artifacts

- `../*.json`：16 份 annotated-best canonical floorplans（rooms＋windows＋stairs）。
- `../{EVAL}/`：scores／summary／hash manifest。
- `per_scene_stair_f1.png`
- `stair_footprint_cases.png`
"""
    report_path = OUT / "RESULTS.md"
    report_path.write_text(report)
    manifest = {
        "inputs": {
            str(summary_path.relative_to(ROOT)): sha256(summary_path),
            str(scores_path.relative_to(ROOT)): sha256(scores_path),
        },
        "outputs": {
            path.name: sha256(path)
            for path in (report_path, f1_path, cases_path)
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"stairs2d report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
