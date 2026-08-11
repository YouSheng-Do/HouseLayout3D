#!/usr/bin/env bash
# Stage 0 batch 2: geometry env + oneformer transformers 修正 + dnsplatter env（tiny-cuda-nn/gsplat 需編譯，最久）
set -o pipefail
ROOT=/home/ado/storage/HouseLayout3D
export PIP_CACHE_DIR=$ROOT/.cache/pip
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TCNN_CUDA_ARCHITECTURES=86
export MAX_JOBS=16
source /home/ado/anaconda3/etc/profile.d/conda.sh
STATUS=""

echo "=== [1/3] geometry env ($(date +%H:%M:%S)) ==="
conda create --prefix $ROOT/envs/geometry python=3.10 -y -q
conda activate $ROOT/envs/geometry
pip install -q open3d trimesh shapely scikit-learn rdp triangle pyviz3d networkx matplotlib opencv-python-headless plyfile scikit-fmm "numpy<2" scipy tqdm
pip install -q PythonCDT || echo "PythonCDT 不在 pip 上（fallback: triangle 已裝）"
if python -c "import open3d, trimesh, shapely, sklearn, rdp, triangle, pyviz3d, networkx, skfmm; print('geometry: all imports OK, open3d', open3d.__version__)"; then
  STATUS="$STATUS\ngeometry: OK"
else
  STATUS="$STATUS\ngeometry: FAIL"
fi
conda deactivate

echo "=== [2/3] oneformer transformers<5 修正 ($(date +%H:%M:%S)) ==="
conda activate $ROOT/envs/oneformer
pip install -q "transformers<5"
if python -c "
import torch, transformers
from transformers import OneFormerProcessor
print('oneformer fixed: torch', torch.__version__, '| transformers', transformers.__version__)
"; then
  STATUS="$STATUS\noneformer-fix: OK"
else
  STATUS="$STATUS\noneformer-fix: FAIL"
fi
conda deactivate

echo "=== [3/3] dnsplatter env ($(date +%H:%M:%S)) ==="
conda create --prefix $ROOT/envs/dnsplatter python=3.10 -y -q
conda activate $ROOT/envs/dnsplatter
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q ninja cmake "numpy<2"
echo "--- tiny-cuda-nn（編譯，需時較久）---"
pip install -q git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 2>&1 | tail -5
echo "--- dn-splatter -e install（帶入 nerfstudio 1.1.3 / gsplat 1.0.0）---"
pip install -q -e $ROOT/external/dn-splatter 2>&1 | tail -5
if python -c "import torch, tinycudann, nerfstudio, gsplat; print('dnsplatter: torch', torch.__version__, '| nerfstudio', nerfstudio.__version__, '| gsplat', gsplat.__version__)"; then
  STATUS="$STATUS\ndnsplatter: OK"
else
  STATUS="$STATUS\ndnsplatter: FAIL"
fi
conda deactivate

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
