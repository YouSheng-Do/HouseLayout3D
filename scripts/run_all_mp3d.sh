#!/usr/bin/env bash
# 全 16 場景復現：跨多 GPU 動態派工（大棟先跑以平衡負載）→ 每棟自清理 → 彙總對照 Table 2/3。
# 用法: ./run_all_mp3d.sh <MP3D_ROOT> [GPU清單,逗號分隔，預設 0,1,2]
set -uo pipefail
MP3D=${1:?mp3d root}
IFS=',' read -ra GPUS <<< "${2:-0,1,2}"
ROOT=/home/ado/storage/HouseLayout3D
LOGDIR=$ROOT/setup/logs/mp3d_run
mkdir -p $LOGDIR $ROOT/outputs/mp3d_reports
: > $LOGDIR/_progress.txt

# 依影格數大→小排序（負載平衡）；17DRP5sb8fy 已跑過，如需重跑可保留
SCENES=(p5wJjkQkbXX jtcxE69GiFV 5LpN3gDmAk7 1LXtFkjw3qL TbHJrupSAjP S9hNv5qa7GM \
        JeFG25nYj2p e9zR4mvMWw7 YFuZgdQ5vWj JmbYfDe2QKZ WYY7iVyf5p8 r47D5H71a5s \
        i5noydFURQK HxpKQynjfin 2t7WUuJeko7 17DRP5sb8fy)

declare -A slot   # gpu -> pid
launch() {  # $1=scene $2=gpu
  ( echo "$(date +%H:%M:%S) ▶ $1 start (GPU $2)" >> $LOGDIR/_progress.txt
    bash $ROOT/scripts/run_pipeline_scene.sh "$1" "$MP3D" "$2" > $LOGDIR/$1.log 2>&1
    rc=$?
    line=$(grep -E "F1@0.5|Δ5=" $LOGDIR/$1.log | tr '\n' ' ')
    echo "$(date +%H:%M:%S) $( [ $rc -eq 0 ] && echo ✅ || echo ❌ ) $1 rc=$rc  $line" >> $LOGDIR/_progress.txt ) &
  slot[$2]=$!
}

for S in "${SCENES[@]}"; do
  mesh=$MP3D/v1/scans/$S/$S/poisson_meshes/${S}_10.ply
  if [ ! -f "$mesh" ]; then echo "⏭  $S 無資料（$mesh），跳過" | tee -a $LOGDIR/_progress.txt; continue; fi
  assigned=""
  while [ -z "$assigned" ]; do
    for g in "${GPUS[@]}"; do
      pid=${slot[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then assigned=$g; break; fi
    done
    [ -z "$assigned" ] && sleep 10
  done
  echo "▶ $S → GPU $assigned"
  launch "$S" "$assigned"
  sleep 3
done
wait

echo "=== 全部完成，彙總對照 Table 2/3 ==="
source /home/ado/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/envs/geometry
python $ROOT/src/eval/aggregate_results.py --pred-root $ROOT/outputs/mp3d | tee $LOGDIR/_table.txt
echo "逐棟診斷圖：$ROOT/outputs/mp3d_reports/*.png"
