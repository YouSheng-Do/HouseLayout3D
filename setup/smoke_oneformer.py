"""Stage 0 smoke check: OneFormer COCO panoptic 單圖分割（用替代場景的真室內影像）。"""
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image

IMG = "/home/ado/storage/HouseLayout3D/data/substitute_scene/room_datasets/coffee_room/iphone/long_capture/images/frame_000001.jpg"
OUT_DIR = "/home/ado/storage/HouseLayout3D/outputs/smoke"
MODEL = "shi-labs/oneformer_coco_swin_large"
os.makedirs(OUT_DIR, exist_ok=True)

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

if not os.path.exists(IMG):
    import glob
    cands = sorted(glob.glob(os.path.dirname(IMG) + "/*"))
    IMG = cands[0]
image = Image.open(IMG).convert("RGB")

processor = OneFormerProcessor.from_pretrained(MODEL)
model = OneFormerForUniversalSegmentation.from_pretrained(MODEL).cuda().eval()

inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]])[0]
seg = result["segmentation"].cpu().numpy()
id2label = model.config.id2label

print(f"image: {IMG}")
print(f"segmentation shape: {seg.shape}, n segments: {len(result['segments_info'])}")
counts = Counter()
for info in result["segments_info"]:
    name = id2label[info["label_id"]]
    frac = (seg == info["id"]).mean()
    counts[name] += frac
for name, frac in counts.most_common(12):
    print(f"  {name:<28s} {frac*100:5.1f}%")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)
palette = rng.randint(0, 255, (seg.max() + 2, 3))
seg_rgb = palette[seg + 1]
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1); plt.imshow(image); plt.title("input (coffee_room)"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(seg_rgb); plt.title("OneFormer COCO panoptic"); plt.axis("off")
handles = []
for info in result["segments_info"][:14]:
    c = palette[info["id"] + 1] / 255.0
    handles.append(plt.Line2D([0], [0], marker="s", ls="", color=c,
                              label=id2label[info["label_id"]]))
plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
out = os.path.join(OUT_DIR, "oneformer_smoke.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
