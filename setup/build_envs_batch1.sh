#!/usr/bin/env bash
# Stage 0 batch 1: metric3d + oneformer conda envs（循序建立，共用 pip cache 以重用 torch wheel）
set -o pipefail
ROOT=/home/ado/storage/HouseLayout3D
export PIP_CACHE_DIR=$ROOT/.cache/pip
source /home/ado/anaconda3/etc/profile.d/conda.sh

STATUS=""

echo "=== [1/2] metric3d env ($(date +%H:%M:%S)) ==="
conda create --prefix $ROOT/envs/metric3d python=3.10 -y -q
conda activate $ROOT/envs/metric3d
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q timm opencv-python-headless matplotlib "numpy<2"
if python -c "import torch, timm, cv2; print('metric3d: torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"; then
  STATUS="$STATUS\nmetric3d: OK"
else
  STATUS="$STATUS\nmetric3d: FAIL"
fi
conda deactivate

echo "=== [2/2] oneformer env ($(date +%H:%M:%S)) ==="
conda create --prefix $ROOT/envs/oneformer python=3.10 -y -q
conda activate $ROOT/envs/oneformer
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers pillow scipy "numpy<2"
if python -c "import torch, transformers; print('oneformer: torch', torch.__version__, '| cuda avail:', torch.cuda.is_available(), '| transformers', transformers.__version__)"; then
  STATUS="$STATUS\noneformer: OK"
else
  STATUS="$STATUS\noneformer: FAIL"
fi
conda deactivate

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
