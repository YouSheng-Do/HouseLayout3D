#!/usr/bin/env bash
# Stage 0 batch 3: oneformer transformers 修正 + dnsplatter 編譯依賴修復 + SPT install.sh
set -o pipefail
ROOT=/home/ado/storage/HouseLayout3D
export PIP_CACHE_DIR=$ROOT/.cache/pip
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TCNN_CUDA_ARCHITECTURES=86
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=16
source /home/ado/anaconda3/etc/profile.d/conda.sh
STATUS=""

echo "=== [1/3] oneformer: transformers==4.44.2 ($(date +%H:%M:%S)) ==="
conda activate $ROOT/envs/oneformer
pip install -q "transformers==4.44.2" "numpy<2"
if python -c "
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
print('oneformer: torch', torch.__version__, '| OneFormer classes import OK')
"; then STATUS="$STATUS\noneformer: OK"; else STATUS="$STATUS\noneformer: FAIL"; fi
conda deactivate

echo "=== [2/3] dnsplatter: 編譯依賴修復 ($(date +%H:%M:%S)) ==="
conda activate $ROOT/envs/dnsplatter
pip install -q "setuptools<81" wheel ninja cmake cython "numpy<2"
echo "--- PyMCubes 0.1.2 (no-build-isolation) ---"
pip install -q --no-build-isolation "PyMCubes==0.1.2" 2>&1 | tail -2
echo "--- tiny-cuda-nn (no-build-isolation, 編譯) ---"
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" 2>&1 | tail -3
echo "--- gsplat 1.0.0（先試官方 wheel index，失敗則原始碼編譯）---"
pip install -q gsplat==1.0.0 --index-url https://docs.gsplat.studio/whl/pt21cu118 2>&1 | tail -2 || pip install --no-build-isolation gsplat==1.0.0 2>&1 | tail -3
echo "--- dn-splatter -e ---"
pip install -q -e $ROOT/external/dn-splatter 2>&1 | tail -3
if python -c "
import torch, tinycudann, gsplat, nerfstudio, dn_splatter
print('dnsplatter: torch', torch.__version__, '| gsplat', gsplat.__version__, '| nerfstudio', nerfstudio.__version__)
"; then STATUS="$STATUS\ndnsplatter: OK"; else STATUS="$STATUS\ndnsplatter: FAIL"; fi
conda deactivate

echo "=== [3/3] SPT install.sh ($(date +%H:%M:%S)) ==="
cd $ROOT/external/superpoint_transformer
bash install.sh 2>&1 | tail -15
conda activate spt
if python -c "
import torch, torch_geometric, pgeof
import frnn
print('spt: torch', torch.__version__, '| pyg', torch_geometric.__version__, '| FRNN OK')
"; then STATUS="$STATUS\nspt: OK"; else STATUS="$STATUS\nspt: FAIL(檢查 FRNN)"; fi
conda deactivate

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
