"""Stage 0 smoke check: SPT 前處理關鍵依賴（FRNN GPU 近鄰、pgeof 幾何特徵、cut-pursuit）。"""
import numpy as np
import torch

# 玩具點雲：兩面牆 + 地板
rng = np.random.RandomState(0)
n = 3000
floor = np.c_[rng.uniform(0, 4, n), rng.uniform(0, 4, n), rng.normal(0, 0.01, n)]
wall1 = np.c_[rng.uniform(0, 4, n), rng.normal(0, 0.01, n), rng.uniform(0, 2.5, n)]
wall2 = np.c_[rng.normal(0, 0.01, n), rng.uniform(0, 4, n), rng.uniform(0, 2.5, n)]
pts = np.vstack([floor, wall1, wall2]).astype(np.float32)

# 1) FRNN GPU 近鄰
import frnn
p = torch.from_numpy(pts).cuda()[None]
lengths = torch.tensor([pts.shape[0]], dtype=torch.long, device="cuda")
dists, idxs, _, _ = frnn.frnn_grid_points(p, p, lengths, lengths, K=16, r=0.5)
assert idxs.shape == (1, pts.shape[0], 16)
print(f"FRNN: {pts.shape[0]} pts, K=16 近鄰 OK (GPU)")

# 2) pgeof 幾何特徵（linearity/planarity/scattering...）
import pgeof
nn = idxs[0].cpu().numpy().astype("uint32")
nn_ptr = np.arange(0, nn.size + 1, 16, dtype="uint32")
feats = pgeof.compute_features(pts, nn.ravel(), nn_ptr, 5)
print(f"pgeof: features {feats.shape} OK, planarity 平均 {feats[:, 1].mean():.2f}")

# 3) cut-pursuit（superpoint 分割核心；SPT 用 cp_d0_dist）
from pycut_pursuit import cp_d0_dist
fn = getattr(cp_d0_dist, "cp_d0_dist", None)
assert callable(fn), f"cp_d0_dist 模組內容: {dir(cp_d0_dist)}"
print("pycut_pursuit.cp_d0_dist import OK")

print("SPT 依賴 smoke 全部通過")
