"""Stage 0 smoke check: Metric3D vit_small 單圖深度推論（依官方 README recipe）。"""
import os

import cv2
import numpy as np
import torch

IMG = "/home/ado/storage/HouseLayout3D/external/Metric3D/data/wild_demo/david-kohler-VFRTXGw1VjU-unsplash.jpg"
OUT_DIR = "/home/ado/storage/HouseLayout3D/outputs/smoke"
os.makedirs(OUT_DIR, exist_ok=True)

rgb_origin = cv2.imread(IMG)[:, :, ::-1]
h0, w0 = rgb_origin.shape[:2]
# smoke 用假設內參（fx=fy=1000）
intrinsic = [1000.0, 1000.0, w0 / 2, h0 / 2]

# --- 官方 ViT 前處理：resize 短邊進 (616,1064)、pad、normalize ---
input_size = (616, 1064)
scale = min(input_size[0] / h0, input_size[1] / w0)
rgb = cv2.resize(rgb_origin, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
intrinsic = [v * scale for v in intrinsic]
h, w = rgb.shape[:2]
pad_h, pad_w = input_size[0] - h, input_size[1] - w
ph, pw = pad_h // 2, pad_w // 2
rgb = cv2.copyMakeBorder(rgb, ph, pad_h - ph, pw, pad_w - pw, cv2.BORDER_CONSTANT,
                         value=[123.675, 116.28, 103.53])
pad_info = [ph, pad_h - ph, pw, pad_w - pw]
mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
t = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
t = torch.div((t - mean), std)[None].cuda()

model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
model = model.cuda().eval()
with torch.no_grad():
    pred_depth, confidence, _ = model.inference({"input": t})

pred_depth = pred_depth.squeeze()
pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0] - pad_info[1],
                        pad_info[2]:pred_depth.shape[1] - pad_info[3]]
pred_depth = torch.nn.functional.interpolate(pred_depth[None, None], (h0, w0), mode="bilinear").squeeze()
pred_depth = pred_depth * (intrinsic[0] / 1000.0)  # canonical -> metric
pred_depth = torch.clamp(pred_depth, 0, 300)
d = pred_depth.cpu().numpy()
print(f"depth stats: min {d.min():.2f}  max {d.max():.2f}  median {np.median(d):.2f} (m), shape {d.shape}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(rgb_origin); plt.title("input"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(d, cmap="turbo"); plt.title("Metric3D vit_small depth (m)")
plt.colorbar(fraction=0.04); plt.axis("off")
out = os.path.join(OUT_DIR, "metric3d_smoke.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
