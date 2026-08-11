"""Stage 3 前置：把 Stage 2 的細粒度標籤（保留 curtain/window_blind 供 Stage 4 窗偵測）
重映射成官方 fit_prototype 期望的粗粒度詞彙（cabinet/curtain/window_blind → surface）。

輸出到 <out-dir>/：labels.npy、cwf_classes.npy、ray_classes.npy、polygon_info_coarse.json
"""
import argparse
import json
import os

import numpy as np

FINE_TO_COARSE = {
    "wall": "wall", "ceiling": "ceiling", "floor": "floor",
    "cabinet": "surface", "curtain": "surface", "window_blind": "surface",
    "door": "door", "window": "window", "mirror": "mirror",
    "outdoor": "outdoor", "stairs": "stairs", "object": "object",
}
COARSE_LABELS = ["wall", "ceiling", "floor", "surface", "door",
                 "window", "mirror", "outdoor", "stairs", "object"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", required=True)
    ap.add_argument("--polygon-info", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    s2 = args.stage2_dir

    fine = [str(x) for x in np.load(f"{s2}/simplified_segmentation_labels.npy")]
    col_map = np.array([COARSE_LABELS.index(FINE_TO_COARSE[n]) for n in fine])

    probs = np.load(f"{s2}/ceiling_wall_floor_mesh_classes.npy").astype(np.float32)
    coarse_probs = np.zeros((len(probs), len(COARSE_LABELS)), dtype=np.float32)
    for fi, ci in enumerate(col_map):
        coarse_probs[:, ci] += probs[:, fi]
    np.save(f"{args.out_dir}/cwf_classes.npy", coarse_probs.astype(np.float16))

    rays = np.load(f"{s2}/hard_labels_simplified_segmentations.npy")
    np.save(f"{args.out_dir}/ray_classes.npy", col_map[rays].astype(np.uint8))

    np.save(f"{args.out_dir}/labels.npy", np.array(COARSE_LABELS))

    with open(args.polygon_info) as f:
        pinfo = json.load(f)
    for v in pinfo.values():
        v["class"] = FINE_TO_COARSE.get(v["class"], v["class"])
    with open(f"{args.out_dir}/polygon_info_coarse.json", "w") as f:
        json.dump(pinfo, f)

    from collections import Counter
    print("coarse 標籤:", COARSE_LABELS)
    print("polygon 類別:", dict(Counter(v["class"] for v in pinfo.values())))
    print("probs 形狀:", coarse_probs.shape, "| ray 樣本:", np.bincount(col_map[rays], minlength=10)[:6])


if __name__ == "__main__":
    main()
