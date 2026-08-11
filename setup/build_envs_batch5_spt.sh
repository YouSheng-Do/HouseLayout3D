#!/usr/bin/env bash
# Stage 0 batch 5: 非互動補完 spt env（install.sh 卡死後的接力；跳過 jupyter/wandb 等訓練雜項）
set -o pipefail
ROOT=/home/ado/storage/HouseLayout3D
export PIP_CACHE_DIR=$ROOT/.cache/pip
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=16
source /home/ado/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/envs/spt
STATUS=""

echo "=== torch 2.2.0 cu118 ($(date +%H:%M:%S)) ==="
pip install -q torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q "numpy<2" torchmetrics==0.11.4

echo "=== pyg wheels ==="
pip install -q pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -q torch_geometric==2.3.0

echo "=== SPT 其餘 pip 依賴（精簡版）==="
pip install -q matplotlib plyfile h5py colorhash numba pytorch-lightning pyrootutils hydra-core hydra-colorlog "rich<=14.0" open3d torch-ransac3d "numpy<2"
pip install -q pgeof pycut-pursuit pygrid-graph torch-graph-components 2>&1 | tail -2

echo "=== FRNN（唯一原始碼編譯）==="
cd $ROOT/external/superpoint_transformer
if [ ! -d src/dependencies/FRNN ]; then
  git clone -q --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
fi
pip install -q "setuptools<81" wheel ninja
cd src/dependencies/FRNN/external/prefix_sum && pip install -q --no-build-isolation . 2>&1 | tail -2
cd ../.. && pip install --no-build-isolation . 2>&1 | tail -3
cd $ROOT

echo "=== 驗證 ==="
if python -c "
import torch, torch_geometric, pgeof
import frnn
print('spt: torch', torch.__version__, '| pyg', torch_geometric.__version__, '| frnn/pgeof import OK | cuda', torch.cuda.is_available())
"; then STATUS="spt: OK"; else STATUS="spt: FAIL"; fi

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
