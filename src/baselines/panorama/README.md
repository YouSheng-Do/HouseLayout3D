# Panorama layout baseline

Measurement-only DOPNet/HorizonNet evaluation on the 16 local Matterport3D
scenes. The implementation keeps per-viewpoint layout quality separate from
building-scale polygon assembly and uses the frozen
`eval2d_v3_strict_levels` + `mp3d_house_floor_v0_1` protocol.

## Reproduction

The complete sequence is:

```bash
MPLCONFIGDIR=/tmp/mpl-panorama \
  /home/ado/storage/conda_envs/panorama/bin/python \
  src/baselines/panorama/prepare.py --stitch --align --workers 8
CUDA_VISIBLE_DEVICES=0 /home/ado/storage/conda_envs/panorama/bin/python \
  src/baselines/panorama/infer_horizonnet.py
CUDA_VISIBLE_DEVICES=1 /home/ado/storage/conda_envs/panorama/bin/python \
  src/baselines/panorama/infer_dopnet.py
/home/ado/storage/conda_envs/panorama_eval/bin/python \
  src/baselines/panorama/evaluate.py
MPLCONFIGDIR=/tmp/mpl-panorama \
  /home/ado/storage/conda_envs/panorama_eval/bin/python \
  src/baselines/panorama/report.py
```

All generated artifacts are written under
`outputs/eval2d/baselines/panorama_layout_v0_1`; source skyboxes and existing
baseline artifacts are never modified.

## Frozen upstream inputs

- DOPNet: `zhijieshen-bjtu/DOPNet` commit
  `35c195af211ec158f1c14f6b66c60f8daf41ee5c`; MP3D checkpoint SHA-256
  `0dcc7929fe92cc5023691ecdb756dcfa6b8f8958524b63b8c30df2e24f3bb8cb`.
- HorizonNet: `sunset1995/HorizonNet` commit
  `43b7fdea131e306f781a57b5476dd37390040fa5c`; MP3D checkpoint SHA-256
  `bdef2e337133f8f2937f99e7643cbef7be56a745a1fe3dad00d855bdcadd0c96`.

No training or fine-tuning is performed. DOPNet is loaded with its published
`resnet34` MP3D state and official Manhattan postprocessing; HorizonNet uses
its published MP3D state and general-layout postprocessing.

### Skybox-to-world orientation

The 18 `matterport_camera_poses` files are six rotations at three tilt angles;
their second index is not a cubemap-face index. Following the Matterport3D pose
construction reported by EDM (CVPR 2025), the 12th one-based pose
(`pose_1_5`) is aligned with the second skybox image. PanFusion's released
stitcher identifies that image as cubemap **left**. Consequently panorama
`(right, forward, up)` maps to pose-1-5
`(-camera-forward, camera-right, -camera-down)`.

This convention was frozen after a first-scene smoke test exposed the 90°
error caused by treating `pose_1_2` as skybox face 2. It is a single global
axis mapping, not a per-view or GT-optimized rotation. Reference:
<https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_EDM_Equirectangular_Projection-Oriented_Dense_Kernelized_Feature_Matching_CVPR_2025_paper.pdf>.

## Compatibility-only upstream edits

The old repositories required four mechanical compatibility changes. They do
not alter learned weights, thresholds, or layout algorithms:

- DOPNet's fixed sampling tensor is a device-agnostic non-persistent buffer.
- DOPNet skips a redundant ImageNet initialization because the released
  checkpoint contains the full encoder.
- OpenCV 4's tuple returned by `findContours` is converted to a list before
  the official in-place sort.
- HorizonNet's only one-component `sklearn` PCA call is expressed as the
  mathematically equivalent leading NumPy SVD vector.

The isolated inference environment uses Torch 1.9 / CUDA 11.1 and
`mmcv-full==1.7.2`. Evaluation/reporting uses a separate NumPy 2.1 / Shapely
2.0 environment because the frozen official325 pickle artifacts were written
with NumPy 2 module paths; the evaluator source itself is unchanged.

## Visual preparation checks

Before scaling beyond the first scene, the stitched and VP-aligned images were
inspected for seams, face ordering, poles, horizon, and genuinely low camera
heights. Audited examples include:

- `data/stitched/17DRP5sb8fy/00ebbf3782c64d74aaf7dd39cd561175.png`
- `data/stitched/17DRP5sb8fy/3577f3b954094af3a80a1208ed3cd337.png`
- `data/aligned/17DRP5sb8fy/00ebbf3782c64d74aaf7dd39cd561175.png`
- `data/stitched/e9zR4mvMWw7/ac03b99e3f3642be80b4d24fde0af03a.png`
- `data/aligned/e9zR4mvMWw7/ac03b99e3f3642be80b4d24fde0af03a.png`
- `data/stitched/p5wJjkQkbXX/b63c954def2a4e86b44707384d727391.png`

All paths above are relative to the panorama baseline output directory.
