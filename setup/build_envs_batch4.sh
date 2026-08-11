#!/usr/bin/env bash
# Stage 0 batch 4（流程修訂後新增）: geometry env 補 CGAL+torch（跑官方 Stage 3）+ 新建 openseg env（TF）
set -o pipefail
ROOT=/home/ado/storage/HouseLayout3D
export PIP_CACHE_DIR=$ROOT/.cache/pip
source /home/ado/anaconda3/etc/profile.d/conda.sh
STATUS=""
MF3D=$ROOT/docs/supplementary/supplementary/multi-floor-3d-code

echo "=== [1/2] geometry env += CGAL + torch ($(date +%H:%M:%S)) ==="
conda activate $ROOT/envs/geometry
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q cgal "numpy<2" 2>&1 | tail -1
if ! python -c "from CGAL.CGAL_Kernel import Point_2" 2>/dev/null; then
  echo "pip cgal 失敗，改用 conda-forge"
  conda install -y -q -p $ROOT/envs/geometry -c conda-forge cgal 2>&1 | tail -2
fi
if python -c "
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Constrained_triangulation_2
import torch
print('CGAL bindings OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())
"; then STATUS="$STATUS\ngeometry+cgal+torch: OK"; else STATUS="$STATUS\ngeometry+cgal+torch: FAIL"; fi

echo "--- 官方 mesh_fitting_3D import smoke ---"
if PYTHONPATH=$MF3D/mesh_fitting_3D python -c "
import cgal_triangulations, merge_split_util, geometry_utils
import differentiable_3D_polygon_stuctures
print('official mesh_fitting_3D imports OK (pytorch3d fallback path)')
"; then STATUS="$STATUS\nofficial-stage3-imports: OK"; else STATUS="$STATUS\nofficial-stage3-imports: FAIL"; fi
conda deactivate

echo "=== [2/2] openseg env（TensorFlow）($(date +%H:%M:%S)) ==="
conda create --prefix $ROOT/envs/openseg python=3.10 -y -q
conda activate $ROOT/envs/openseg
pip install -q "tensorflow[and-cuda]==2.15.1" pillow "numpy<2" 2>&1 | tail -2
if python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('tensorflow', tf.__version__, '| GPUs:', gpus if gpus else 'NONE (CPU fallback)')
" 2>&1 | grep -v "^20\|oneDNN\|TF-TRT"; then STATUS="$STATUS\nopenseg-tf: OK"; else STATUS="$STATUS\nopenseg-tf: FAIL"; fi
conda deactivate

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
