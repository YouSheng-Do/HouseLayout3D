"""失敗像素分類診斷：把 Δ5 失敗的像素拆成可行動的類別，並以官方預測為參照組。

類別（GT-valid 像素上）：
  ok5      |zp-zg| ≤ 5cm         （通過）
  off5_10  5 < |zp-zg| ≤ 10cm    （小位移：fit 精度問題 → 收緊/正則化可救）
  front    zp < zg - 10cm         （前方多餘幾何：多建了牆/假牆）
  behind   zp > zg + 10cm（有限）  （漏光：視線穿縫/破洞打到更遠表面，或牆被大幅後移）
  nohit    zp = inf               （破洞：視線直接穿出 prototype）

behind+nohit 高 → init 碎片縫隙/覆蓋是主因（合併/補洞有效）；
off5_10 高    → fit 精度是主因（容差/正則化有效）。
用法: python depth_error_diagnosis.py --scenes 2t7WUuJeko7 e9zR4mvMWw7 jtcxE69GiFV --stride 10
"""
import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from depth_metrics import _make_scene, _z_factor, render_z  # noqa: E402
from gt_loader import load_scene_gt  # noqa: E402

ROOT = "/home/ado/storage/HouseLayout3D"
OFFICIAL = f"{ROOT}/docs/supplementary/supplementary/predictions-ours"
CATS = ["ok5", "off5_10", "front", "behind", "nohit"]


def classify(zg: np.ndarray, zp: np.ndarray) -> np.ndarray:
    """回傳類別索引圖（-1=GT invalid）。"""
    out = np.full(zg.shape, -1, dtype=np.int8)
    valid = np.isfinite(zg)
    pinf = valid & ~np.isfinite(zp)
    both = valid & np.isfinite(zp)
    d = zp - zg
    out[both & (np.abs(d) <= 0.05)] = 0
    out[both & (np.abs(d) > 0.05) & (np.abs(d) <= 0.10)] = 1
    out[both & (d < -0.10)] = 2
    out[both & (d > 0.10)] = 3
    out[pinf] = 4
    return out


def diagnose(gt_mesh, pred_mesh, poses, stride):
    sg, sp = _make_scene(gt_mesh), _make_scene(pred_mesh)
    frames = poses["frames"][::stride]
    counts = np.zeros(5, dtype=np.int64)
    zfac_cache = {}
    per_frame = []  # (fail_ratio, frame, zg, zp) 供挑最差幀出圖
    for fr in frames:
        h, w = fr["h"], fr["w"]
        fx, fy, cx, cy = fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]
        key = (h, w, fx, fy, cx, cy)
        if key not in zfac_cache:
            zfac_cache[key] = _z_factor(h, w, fx, fy, cx, cy)
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w[0:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        zg = render_z(sg, w2c, h, w, fx, fy, cx, cy, zfac_cache[key])
        zp = render_z(sp, w2c, h, w, fx, fy, cx, cy, zfac_cache[key])
        cat = classify(zg, zp)
        n_valid = int((cat >= 0).sum())
        if n_valid == 0:
            continue
        counts += np.bincount(cat[cat >= 0], minlength=5)
        fail = 1.0 - float((cat == 0).sum()) / n_valid
        per_frame.append((fail, fr, cat))
    total = counts.sum()
    pct = {c: 100.0 * counts[i] / max(total, 1) for i, c in enumerate(CATS)}
    return pct, per_frame


def render_maps(scene, per_frame_ours, per_frame_off, out_png, k=3):
    """最差 k 幀（依我方 fail 比例）：RGB | 我方類別圖 | 官方類別圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    palette = np.array([[46, 160, 67],    # ok5 綠
                        [255, 212, 59],   # off5_10 黃
                        [255, 137, 4],    # front 橙
                        [219, 55, 55],    # behind 紅
                        [30, 30, 30],     # nohit 黑
                        [235, 235, 235]], dtype=np.uint8)  # invalid 淺灰(索引5)
    off_by_stem = {os.path.basename(fr["file_path"]): cat
                   for _, fr, cat in per_frame_off}
    worst = sorted(per_frame_ours, key=lambda x: -x[0])[:k]
    fig, axes = plt.subplots(k, 3, figsize=(13, 3.4 * k))
    for i, (fail, fr, cat) in enumerate(worst):
        stem = os.path.basename(fr["file_path"])
        # HF poses 的 file_path 是作者機器絕對路徑 → 映射回本機 MP3D
        rgb_path = f"{ROOT}/data/mp3d/v1/scans/{scene}/{scene}/undistorted_color_images/{stem}"
        if not os.path.exists(rgb_path):
            rgb_path = fr["file_path"]
        img = Image.open(rgb_path) if os.path.exists(rgb_path) else None
        cat_img = palette[np.where(cat < 0, 5, cat)]
        axes[i, 0].imshow(img) if img is not None else axes[i, 0].text(.5, .5, "no RGB", ha="center")
        axes[i, 0].set_title(f"RGB {stem[:16]}", fontsize=8)
        axes[i, 1].imshow(cat_img)
        axes[i, 1].set_title(f"ours fail={fail:.0%}", fontsize=8)
        oc = off_by_stem.get(stem)
        if oc is not None:
            axes[i, 2].imshow(palette[np.where(oc < 0, 5, oc)])
            ofail = 1 - float((oc == 0).sum()) / max((oc >= 0).sum(), 1)
            axes[i, 2].set_title(f"official fail={ofail:.0%}", fontsize=8)
        else:
            axes[i, 2].axis("off")
        for ax in axes[i]:
            ax.axis("off")
    handles = [Patch(color=palette[j] / 255, label=CATS[j]) for j in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"{scene} — worst frames (by ours fail%)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["2t7WUuJeko7", "e9zR4mvMWw7", "jtcxE69GiFV"])
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-dir", default=f"{ROOT}/outputs/mp3d_reports")
    args = ap.parse_args()

    hdr = f"{'scene':>12s} | {'who':>8s} | " + " ".join(f"{c:>8s}" for c in CATS)
    print(hdr)
    print("-" * len(hdr))
    for scene in args.scenes:
        t0 = time.time()
        gt = load_scene_gt(scene, load_poses=True)
        ours = o3d.io.read_triangle_mesh(f"{ROOT}/outputs/mp3d/{scene}/stage4/combined.ply")
        official = o3d.io.read_triangle_mesh(f"{OFFICIAL}/{scene}/combined.ply")
        pct_o, pf_o = diagnose(gt.structures_mesh, ours, gt.poses, args.stride)
        pct_f, pf_f = diagnose(gt.structures_mesh, official, gt.poses, args.stride)
        for who, pct in (("ours", pct_o), ("official", pct_f)):
            print(f"{scene:>12s} | {who:>8s} | " +
                  " ".join(f"{pct[c]:>7.1f}%" for c in CATS))
        out_png = os.path.join(args.out_dir, f"_diag_{scene}.png")
        render_maps(scene, pf_o, pf_f, out_png)
        print(f"{'':>12s} | 圖 → {out_png}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
