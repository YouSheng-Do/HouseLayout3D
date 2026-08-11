set -eo pipefail
ROOT=/home/ado/storage/HouseLayout3D; S=17DRP5sb8fy; O=$ROOT/outputs/mp3d/$S
source /home/ado/anaconda3/etc/profile.d/conda.sh; conda activate $ROOT/envs/geometry
echo "=== init v2 (contour-rdp 0.03) ==="
python $ROOT/src/stage3/init_polygons.py --stage2-dir $O/skeleton --out-dir $O/stage3_v2 --min-unassigned 2000 --max-polygons 400 --contour-rdp 0.03
python $ROOT/src/stage3/make_coarse_labels.py --stage2-dir $O/skeleton --polygon-info $O/stage3_v2/polygon_info.json --out-dir $O/stage3_v2/coarse
echo "=== fit v2 (官方 config) ==="
cd $ROOT/docs/supplementary/supplementary/multi-floor-3d-code
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:mesh_fitting_3D python -u fit_prototype.py --scene-type matterport \
  --rectified-ply-path $O/stage3_v2/clean_edge_mesh.ply --target-pcd-path $O/skeleton/ceiling_wall_floor_mesh.ply \
  --target-vertex-classes $O/stage3_v2/coarse/cwf_classes.npy --target-vertex-class-names $O/stage3_v2/coarse/labels.npy \
  --polygon-info-path $O/stage3_v2/coarse/polygon_info_coarse.json \
  --target-pcd-ray-origins-path $O/skeleton/full_ray_origins.npy --target-pcd-ray-dests-path $O/skeleton/full_ray_dests.npy \
  --object-mesh $O/skeleton/objects_mesh.ply --ray-classes $O/stage3_v2/coarse/ray_classes.npy --output-dir $O/stage3_v2/fit
echo "stage3_v2 done"
