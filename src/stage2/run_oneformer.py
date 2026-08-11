"""Stage 2 缺件 (a)：OneFormer 推論 wrapper——補上 supplementary 缺的 house_layout_inference.py。

對每張輸入影像跑 OneFormer COCO semantic segmentation，依 Appendix Table 5 映射成
簡化類別，輸出 uint8 標籤 PNG（與影像同名）＋ labels.txt，供官方 extract_skeleton.py 使用。
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_to_simplified import SIMPLIFIED_LABELS, build_lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="影像目錄或 glob")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="shi-labs/oneformer_coco_swin_large")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 張（smoke 用）")
    ap.add_argument("--viz-first", type=int, default=3, help="前 N 張另存彩色視覺化")
    args = ap.parse_args()

    if os.path.isdir(args.images):
        paths = sorted(glob.glob(os.path.join(args.images, "*")))
    else:
        paths = sorted(glob.glob(args.images))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if args.limit:
        paths = paths[: args.limit]
    assert paths, f"找不到影像: {args.images}"
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    processor = OneFormerProcessor.from_pretrained(args.model)
    model = OneFormerForUniversalSegmentation.from_pretrained(args.model).cuda().eval()
    lut = build_lut({int(k): v for k, v in model.config.id2label.items()})

    with open(os.path.join(args.out_dir, "labels.txt"), "w") as f:
        f.write("\n".join(SIMPLIFIED_LABELS) + "\n")

    palette = np.array([
        [200, 120, 40], [60, 120, 60], [160, 160, 160], [140, 100, 200], [220, 40, 40],
        [200, 180, 60], [120, 200, 200], [40, 90, 220], [240, 120, 200], [90, 60, 30],
        [255, 240, 60], [110, 110, 110]], dtype=np.uint8)

    t0 = time.time()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        sem = processor.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0]
        simp = lut[sem.cpu().numpy()]
        stem = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(simp, mode="L").save(os.path.join(args.out_dir, stem + ".png"))
        if i < args.viz_first:
            Image.fromarray(palette[simp]).save(os.path.join(args.out_dir, stem + "_viz.png"))
        if (i + 1) % 25 == 0 or i == len(paths) - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{len(paths)}  ({dt/(i+1):.2f}s/img, 剩餘約 {(len(paths)-i-1)*dt/(i+1)/60:.1f} min)",
                  flush=True)

    # 統計最後一張的類別占比（sanity）
    from collections import Counter
    cnt = Counter(simp.ravel().tolist())
    tot = simp.size
    stats = ", ".join(f"{SIMPLIFIED_LABELS[k]}:{v/tot*100:.0f}%" for k, v in cnt.most_common(6))
    print(f"完成 {len(paths)} 張 → {args.out_dir}；最後一張類別占比: {stats}")


if __name__ == "__main__":
    main()
