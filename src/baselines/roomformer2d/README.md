# RoomFormer 2D baseline

This pipeline evaluates the official semantic-rich Structured3D RoomFormer
checkpoint on the 16 local HouseLayout3D/Matterport3D scenes in two modes:

- `per_floor`: merge sampled surface points from all evaluated regions on one
  MP3D level (32 inputs).
- `per_room`: run one sampled MP3D region at a time and concatenate predictions
  in the native metric frame (325 inputs).

The HouseLayout3D supplement specifies uniform mesh-surface sampling but does
not publish a point count. This implementation records its deterministic
sampling assumptions in `input_manifest.json`.

## Reproduce

From `/home/ado/storage/HouseLayout3D`:

```bash
BASE=outputs/eval2d/baselines/roomformer_hl3d_2d_v1

envs/hl3d/bin/python src/baselines/roomformer2d/prepare_inputs.py \
  --out "$BASE/input_data"

CUDA_VISIBLE_DEVICES=0 /home/ado/storage/conda_envs/roomformer/bin/python \
  src/baselines/roomformer2d/infer.py \
  --roomformer-root /home/ado/storage/RoomFormer \
  --checkpoint /home/ado/storage/RoomFormer/checkpoints/roomformer_stru3d_semantic_rich.pth \
  --input-root "$BASE/input_data" --manifest "$BASE/input_data/input_manifest.json" \
  --out "$BASE/per_floor" --mode per_floor --batch-size 4 --save-overlays

CUDA_VISIBLE_DEVICES=0 /home/ado/storage/conda_envs/roomformer/bin/python \
  src/baselines/roomformer2d/infer.py \
  --roomformer-root /home/ado/storage/RoomFormer \
  --checkpoint /home/ado/storage/RoomFormer/checkpoints/roomformer_stru3d_semantic_rich.pth \
  --input-root "$BASE/input_data" --manifest "$BASE/input_data/input_manifest.json" \
  --out "$BASE/per_room" --mode per_room --batch-size 4 --save-overlays

python src/baselines/roomformer2d/evaluate.py --root "$BASE"
python src/baselines/roomformer2d/export_canonical.py --root "$BASE"
python src/baselines/roomformer2d/visualize_comparison.py \
  --root "$BASE" --out "$BASE/comparisons"
```

The checkpoint SHA-256 is
`a130c72bfcd9cb52eb9107c450cb52853ecbc549a576e745fe56d58e32f90e7b`.
Checkpoint loading must report zero missing and zero unexpected keys.

These are local 2D diagnostic metrics. They are not the 3D metrics in
HouseLayout3D Table 2.
