"""Stage 4 / D.4：OpenSeg 每像素 CLIP-aligned 特徵抽取＋反投影（openseg env, TF）。

用法（OpenScene/OpenNeRF 生態的標準 serving 介面）：
  saved_model.signatures['serving_default'](inp_image_bytes=..., inp_text_emb=zeros(1,1,768))
輸出：subsample 像素的 3D 點 + 768-d 特徵（float16），供 room_classify 聚合。
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--poses-file", required=True, help="gt-frame 轉換後的 transformations json（絕對路徑版）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-stride", type=int, default=8)
    ap.add_argument("--pixels-per-frame", type=int, default=2000)
    ap.add_argument("--depth-scale", type=float, default=0.001)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import tensorflow as tf2
    import tensorflow.compat.v1 as tf
    from PIL import Image

    print("[openseg] loading saved model ...", flush=True)
    model = tf2.saved_model.load(args.model_dir)
    text_emb = tf.zeros([1, 1, 768])

    with open(args.poses_file) as f:
        poses = json.load(f)
    frames = poses["frames"][:: args.frame_stride]
    print(f"[openseg] {len(frames)} frames (stride {args.frame_stride})", flush=True)

    rng = np.random.RandomState(0)
    all_pts, all_feats = [], []
    for i, fr in enumerate(frames):
        with open(fr["file_path"], "rb") as f:
            img_bytes = f.read()
        res = model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(img_bytes),
            inp_text_emb=text_emb)
        info = res["image_info"]
        crop = [int(info[0, 0] * info[2, 0]), int(info[0, 1] * info[2, 1])]
        feat = res["ppixel_ave_feat"][:, : crop[0], : crop[1]]

        H, W = fr["h"], fr["w"]
        feat = tf2.image.resize(feat, [H, W], method="nearest")[0].numpy().astype(np.float16)

        depth = np.array(Image.open(fr["depth_file_path"])).astype(np.float32) * args.depth_scale
        assert depth.shape == (H, W)
        valid = depth > 0
        ys, xs = np.where(valid)
        if len(ys) == 0:
            continue
        sel = rng.choice(len(ys), min(args.pixels_per_frame, len(ys)), replace=False)
        ys, xs = ys[sel], xs[sel]

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w[0:3, 1:3] *= -1  # OpenGL→OpenCV
        z = depth[ys, xs]
        x_cam = (xs + 0.5 - fr["cx"]) / fr["fl_x"] * z
        y_cam = (ys + 0.5 - fr["cy"]) / fr["fl_y"] * z
        pts_cam = np.c_[x_cam, y_cam, z, np.ones(len(z))]
        pts_w = (c2w @ pts_cam.T).T[:, :3]

        all_pts.append(pts_w.astype(np.float32))
        all_feats.append(feat[ys, xs])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)

    pts = np.concatenate(all_pts)
    feats = np.concatenate(all_feats)
    np.save(f"{args.out_dir}/openseg_points.npy", pts)
    np.save(f"{args.out_dir}/openseg_feats.npy", feats)
    print(f"[openseg] saved {len(pts)} points × {feats.shape[1]}d → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
