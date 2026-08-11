#!/usr/bin/env bash
# MULTIFLOOR3D 單場景 pipeline（MP3D 版）：Stage 2 → 3 → 4 → eval。
# 用法: ./run_pipeline_scene.sh <SCENE_ID> <MP3D_ROOT> [GPU_ID]
#   MP3D_ROOT 需含: v1/scans/<scene>/poisson_meshes/<scene>_10.ply
#   poses 用 HF 標註: external/houselayout3d/data/poses/<scene>.json（OpenNeRF 式）
#   ⚠ MP3D 影像/深度路徑需與 poses json 內 file_path/depth_file_path 一致（見 PROGRESS.md）
set -eo pipefail
SCENE=${1:?scene id}
MP3D=${2:?mp3d root}
GPU=${3:-0}
# DATA_ROOT（選填）：本機 MP3D→OpenNeRF/nerfstudio 場景根（含 images/ depths/）。
# 給定時改寫 HF poses 的作者機器絕對路徑為本機路徑；預設猜 <MP3D>/nerfstudio/matterport_<scene>。
DEPTH_SCALE=${4:-0.00025}   # MP3D 16-bit 深度 0.25mm/unit
ROOT=/home/ado/storage/HouseLayout3D
OUT=$ROOT/outputs/mp3d/$SCENE
# MP3D zip 內含一層 scene-id 資料夾 → 真實場景根雙層
SCENE_ROOT=$MP3D/v1/scans/$SCENE/$SCENE
MESH=$SCENE_ROOT/poisson_meshes/${SCENE}_10.ply
CONF=$SCENE_ROOT/undistorted_camera_parameters/$SCENE.conf
POSES=$OUT/poses_mp3d.json
source /home/ado/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/envs/geometry
mkdir -p $OUT
if [ -f $POSES ]; then echo "=== [$SCENE] poses 已存在，跳過 ==="; else
echo "=== [$SCENE] 從 MP3D .conf 建 poses（原生座標，已驗證同 GT 框）==="
python $ROOT/src/stage2/mp3d_to_poses.py --conf $CONF --data-root $SCENE_ROOT --out $POSES
fi
conda deactivate

if [ -f $OUT/oneformer/labels.txt ]; then echo "=== [$SCENE] Stage 2a: OneFormer 已存在，跳過 ==="; else
echo "=== [$SCENE] Stage 2a: OneFormer PNGs ==="
conda activate $ROOT/envs/oneformer
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$ROOT/.cache/hf python $ROOT/src/stage2/run_oneformer.py \
  --images "$(python -c "import json;p=json.load(open('$POSES'));import os;print(os.path.dirname(p['frames'][0]['file_path']))")" \
  --out-dir $OUT/oneformer
conda deactivate
fi

echo "=== [$SCENE] Stage 2b: extract_skeleton（官方 patched）==="
conda activate $ROOT/envs/spt
CUDA_VISIBLE_DEVICES=$GPU python $ROOT/src/stage2/extract_skeleton_patched.py \
  --output-dir $OUT/skeleton --poses-file $POSES --mesh-file $MESH \
  --seg-dir $OUT/oneformer --depth-scale $DEPTH_SCALE --samples-per-frame 3000
conda deactivate

echo "=== [$SCENE] Stage 3: init + coarse + fit（官方）==="
conda activate $ROOT/envs/geometry
python $ROOT/src/stage3/init_polygons.py --stage2-dir $OUT/skeleton --out-dir $OUT/stage3 \
  --max-polygons 600
python $ROOT/src/stage3/make_coarse_labels.py --stage2-dir $OUT/skeleton \
  --polygon-info $OUT/stage3/polygon_info.json --out-dir $OUT/stage3/coarse
cd $ROOT/docs/supplementary/supplementary/multi-floor-3d-code
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.:mesh_fitting_3D python -u fit_prototype.py \
  --scene-type matterport \
  --rectified-ply-path $OUT/stage3/clean_edge_mesh.ply \
  --target-pcd-path $OUT/skeleton/ceiling_wall_floor_mesh.ply \
  --target-vertex-classes $OUT/stage3/coarse/cwf_classes.npy \
  --target-vertex-class-names $OUT/stage3/coarse/labels.npy \
  --polygon-info-path $OUT/stage3/coarse/polygon_info_coarse.json \
  --target-pcd-ray-origins-path $OUT/skeleton/full_ray_origins.npy \
  --target-pcd-ray-dests-path $OUT/skeleton/full_ray_dests.npy \
  --object-mesh $OUT/skeleton/objects_mesh.ply \
  --ray-classes $OUT/stage3/coarse/ray_classes.npy \
  --output-dir $OUT/stage3/fit
cd $ROOT

echo "=== [$SCENE] Stage 4: scene graph + 實體匯出 ==="
python $ROOT/src/stage4/scene_graph.py \
  --fitted $OUT/stage3/fit/fitted_mesh.ply --skeleton $OUT/skeleton/ceiling_wall_floor_mesh.ply \
  --probs $OUT/stage3/coarse/cwf_classes.npy --labels $OUT/stage3/coarse/labels.npy \
  --out-dir $OUT/stage4 --stage2-dir $OUT/skeleton --stair-mesh $OUT/skeleton/stair_mesh.ply
conda deactivate

echo "=== [$SCENE] Stage 4b: OpenSeg 房型（D.4）==="
conda activate $ROOT/envs/openseg
CUDA_VISIBLE_DEVICES=$GPU python $ROOT/src/stage4/openseg_features.py \
  --model-dir $ROOT/.cache/models/openseg_exported_clip --poses-file $POSES \
  --out-dir $OUT/stage4 --frame-stride 8 --depth-scale $DEPTH_SCALE
conda deactivate
conda activate $ROOT/envs/geometry
python $ROOT/src/stage4/room_classify.py \
  --fitted $OUT/stage3/fit/fitted_mesh.ply --skeleton $OUT/skeleton/ceiling_wall_floor_mesh.ply \
  --probs $OUT/stage3/coarse/cwf_classes.npy --labels $OUT/stage3/coarse/labels.npy \
  --openseg-dir $OUT/stage4 --text-emb $ROOT/.cache/models/room_text_emb.npy \
  --scene-graph $OUT/stage4/scene_graph.json

echo "=== [$SCENE] Eval: F1（原生實體）＋ Δτ ==="
if [ -f "$ROOT/external/houselayout3d/data/doors/$SCENE.json" ]; then
  python $ROOT/src/eval/eval_scene.py --scene $SCENE --pred-dir $OUT/stage4
  python - <<EOF
import sys, json; sys.path.insert(0,'$ROOT/src/eval')
import open3d as o3d
from depth_metrics import scene_delta_tau
from gt_loader import load_scene_gt
gt = load_scene_gt('$SCENE')
pred = o3d.io.read_triangle_mesh('$OUT/stage4/combined.ply')
res = scene_delta_tau(gt.structures_mesh, pred, gt.poses, stride=1)
print(f"[$SCENE] Δ5={res[5.0]:.1f}  Δ10={res[10.0]:.1f}")
EOF
else
  echo "  ⚠ 無 HF GT（$SCENE 非 benchmark 場景，如整合測試）→ 跳過 eval，僅驗 pipeline 產物"
  ls -la $OUT/stage4/entities/ | head -3
fi

# 每棟診斷圖（保留，供逐棟目視——即使中間產物被清理）
python $ROOT/src/eval/scene_report.py --scene $SCENE --pred-dir $OUT/stage4 \
  --out $ROOT/outputs/mp3d_reports/$SCENE.png 2>/dev/null || echo "  (診斷圖略過)"

# 逐棟清理大中間產物（保留 stage4 成品/scene_graph/poses/fitted_mesh/診斷圖）
# 設 KEEP_INTERMEDIATES=1 可停用（想細看某棟中間產物時）
if [ "${KEEP_INTERMEDIATES:-0}" != "1" ]; then
  python3 - <<PYEOF
import os, glob
# 2026-08-04：改為保留 Stage-4 重跑所需輸入（fitted_mesh + ceiling_wall_floor_mesh +
# objects_mesh + full_ray_* + coarse/，共 ~170MB/棟），讓 watershed 調參後只需重跑 Stage 4
# （~5min/棟）而非整條 pipeline（~18h）。僅刪真正巨大且 Stage-4 不需的轉檔。
big = ['skeleton/mesh.ply','skeleton/ceiling_wall_floor_mesh_classes.npy',
       'skeleton/point_cloud.ply','skeleton/objects_mesh_classes.npy',
       'skeleton/vertex_probabilities.npy','skeleton/vertex_hard_assignments.npy']
rm = [os.path.join('$OUT', f) for f in big]
rm += glob.glob('$OUT/skeleton/spt/*.ply')
rm += glob.glob('$OUT/stage3*/fit/fitted_mesh_[0-9]*.ply')  # 保留最終 fitted_mesh.ply
freed = 0
for p in rm:
    if os.path.exists(p): freed += os.path.getsize(p); os.remove(p)
print(f"  逐棟清理：釋放 {freed/1e9:.1f} GB（保留 Stage-4 輸入：可 5min/棟重跑調參）")
PYEOF
fi
echo "=== [$SCENE] 完成 → $OUT ==="
