#!/usr/bin/env bash
# 對 Stage 2–3 已完成、Stage 4 失敗的棟做 Stage-4-only 重試（scene_graph→OpenSeg→classify→eval→報告→清理）。
# 用法: ./retry_stage4.sh <GPU> <scene1> [scene2 ...]
set -uo pipefail
GPU=${1:?gpu}; shift
ROOT=/home/ado/storage/HouseLayout3D
source /home/ado/anaconda3/etc/profile.d/conda.sh
for S in "$@"; do
  O=$ROOT/outputs/mp3d/$S
  echo "=== [$S] Stage-4 重試 ==="
  if [ ! -f $O/stage3/fit/fitted_mesh.ply ]; then echo "  ✗ 缺 fitted_mesh，需完整重跑"; continue; fi
  conda activate $ROOT/envs/geometry
  python $ROOT/src/stage4/scene_graph.py \
    --fitted $O/stage3/fit/fitted_mesh.ply --skeleton $O/skeleton/ceiling_wall_floor_mesh.ply \
    --probs $O/stage3/coarse/cwf_classes.npy --labels $O/stage3/coarse/labels.npy \
    --out-dir $O/stage4 --stage2-dir $O/skeleton --stair-mesh $O/skeleton/stair_mesh.ply || { echo "  ✗ scene_graph 又失敗"; conda deactivate; continue; }
  conda deactivate
  conda activate $ROOT/envs/openseg
  CUDA_VISIBLE_DEVICES=$GPU python $ROOT/src/stage4/openseg_features.py \
    --model-dir $ROOT/.cache/models/openseg_exported_clip --poses-file $O/poses_mp3d.json \
    --out-dir $O/stage4 --frame-stride 8 --depth-scale 0.00025 || echo "  (openseg 失敗，跳過房型)"
  conda deactivate
  conda activate $ROOT/envs/geometry
  python $ROOT/src/stage4/room_classify.py \
    --fitted $O/stage3/fit/fitted_mesh.ply --skeleton $O/skeleton/ceiling_wall_floor_mesh.ply \
    --probs $O/stage3/coarse/cwf_classes.npy --labels $O/stage3/coarse/labels.npy \
    --openseg-dir $O/stage4 --text-emb $ROOT/.cache/models/room_text_emb.npy \
    --scene-graph $O/stage4/scene_graph.json || true
  python $ROOT/src/eval/eval_scene.py --scene $S --pred-dir $O/stage4
  python - <<EOF
import sys, json; sys.path.insert(0,'$ROOT/src/eval')
import open3d as o3d
from depth_metrics import scene_delta_tau
from gt_loader import load_scene_gt
gt = load_scene_gt('$S')
pred = o3d.io.read_triangle_mesh('$O/stage4/combined.ply')
res = scene_delta_tau(gt.structures_mesh, pred, gt.poses, stride=1)
print(f"[$S] Δ5={res[5.0]:.1f}  Δ10={res[10.0]:.1f}")
EOF
  python $ROOT/src/eval/scene_report.py --scene $S --pred-dir $O/stage4 --out $ROOT/outputs/mp3d_reports/$S.png || true
  python3 - <<PYEOF
import os, glob
big=['skeleton/ceiling_wall_floor_mesh.ply','skeleton/ceiling_wall_floor_mesh_classes.npy',
     'skeleton/point_cloud.ply','skeleton/objects_mesh.ply','skeleton/objects_mesh_classes.npy',
     'skeleton/vertex_probabilities.npy','skeleton/vertex_hard_assignments.npy']
freed=0
for f in [os.path.join('$O',x) for x in big]+glob.glob('$O/skeleton/spt/*.ply'):
    if os.path.exists(f): freed+=os.path.getsize(f); os.remove(f)
print(f"  重試後清理：釋放 {freed/1e9:.1f} GB")
PYEOF
  echo "=== [$S] 重試完成 ==="
done
