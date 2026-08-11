#!/usr/bin/env bash
# Stage 0 batch 6: spt env 收尾——本地 wheel 換裝 torch 2.2.0+cu118、編 FRNN、驗證
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

echo "=== 換裝 torch 2.2.0+cu118（本地 wheel）($(date +%H:%M:%S)) ==="
pip uninstall -y -q torch torchvision 2>/dev/null
pip install -q $ROOT/.cache/wheels/torch-2.2.0%2Bcu118-cp38-cp38-linux_x86_64.whl $ROOT/.cache/wheels/torchvision-0.17.0%2Bcu118-cp38-cp38-linux_x86_64.whl "numpy<2"
python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())" || exit 1

echo "=== FRNN 編譯 ($(date +%H:%M:%S)) ==="
pip install -q "setuptools<81" wheel ninja
cd $ROOT/external/superpoint_transformer/src/dependencies/FRNN/external/prefix_sum && pip install -q --no-build-isolation . 2>&1 | tail -2
cd ../.. && pip install --no-build-isolation . 2>&1 | tail -3
cd $ROOT

echo "=== 驗證 ($(date +%H:%M:%S)) ==="
if python -c "
import torch, torch_geometric, torch_scatter, torch_cluster, pgeof
import frnn
print('spt: torch', torch.__version__, '| pyg', torch_geometric.__version__, '| scatter/cluster/pgeof/frnn OK | cuda', torch.cuda.is_available())
"; then STATUS="spt: OK"; else STATUS="spt: FAIL"; fi

echo "=== SUMMARY ($(date +%H:%M:%S)) ==="
echo -e "$STATUS"
