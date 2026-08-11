set -eo pipefail
ROOT=/home/ado/storage/HouseLayout3D; S=p5wJjkQkbXX; O=$ROOT/outputs/mp3d/$S
source /home/ado/anaconda3/etc/profile.d/conda.sh; conda activate $ROOT/envs/geometry
python $ROOT/src/stage3/init_polygons.py --stage2-dir $O/skeleton --out-dir $O/stage3 --max-polygons 600
python $ROOT/src/stage3/make_coarse_labels.py --stage2-dir $O/skeleton --polygon-info $O/stage3/polygon_info.json --out-dir $O/stage3/coarse
cd $ROOT/docs/supplementary/supplementary/multi-floor-3d-code
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:mesh_fitting_3D python -u fit_prototype.py --scene-type matterport \
  --rectified-ply-path $O/stage3/clean_edge_mesh.ply --target-pcd-path $O/skeleton/ceiling_wall_floor_mesh.ply \
  --target-vertex-classes $O/stage3/coarse/cwf_classes.npy --target-vertex-class-names $O/stage3/coarse/labels.npy \
  --polygon-info-path $O/stage3/coarse/polygon_info_coarse.json \
  --target-pcd-ray-origins-path $O/skeleton/full_ray_origins.npy --target-pcd-ray-dests-path $O/skeleton/full_ray_dests.npy \
  --object-mesh $O/skeleton/objects_mesh.ply --ray-classes $O/stage3/coarse/ray_classes.npy --output-dir $O/stage3/fit
cd $ROOT
python $ROOT/src/stage4/scene_graph.py --fitted $O/stage3/fit/fitted_mesh.ply --skeleton $O/skeleton/ceiling_wall_floor_mesh.ply \
  --probs $O/stage3/coarse/cwf_classes.npy --labels $O/stage3/coarse/labels.npy \
  --out-dir $O/stage4 --stage2-dir $O/skeleton --stair-mesh $O/skeleton/stair_mesh.ply
python - <<'EOF'
import sys, json; sys.path.insert(0,'/home/ado/storage/HouseLayout3D/src/eval')
import open3d as o3d
from depth_metrics import scene_delta_tau
from gt_loader import load_scene_gt
S='p5wJjkQkbXX'
gt=load_scene_gt(S)
pred=o3d.io.read_triangle_mesh(f'/home/ado/storage/HouseLayout3D/outputs/mp3d/{S}/stage4/combined.ply')
r=scene_delta_tau(gt.structures_mesh,pred,gt.poses,stride=4)
print(f'[p5w 自適應K驗證] Δ5={r[5.0]:.1f} Δ10={r[10.0]:.1f}  (舊: 退化 0；官方: 56.1/69.8)')
EOF
