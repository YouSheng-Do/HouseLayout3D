# RoomFormer 3D lifting on HouseLayout3D

This stage consumes the frozen canonical predictions produced by
`src/baselines/roomformer2d` and applies the baseline rules in HouseLayout3D
Appendix E.1:

- floor and ceiling: lower and upper 5% quantiles of the sampled input-point
  height distribution;
- doors: floor level to 2.10 m above the floor;
- windows: the centred 80% of wall height;
- stairs: no prediction (RoomFormer does not predict stairs).

The lower/upper quantile interpretation is explicit because the supplement's
phrase “based on the 5%-quantile” does not spell out both tails separately.

## Reproduce

From `/home/ado/storage/HouseLayout3D`:

```bash
TWO_D=outputs/eval2d/baselines/roomformer_hl3d_2d_v1
THREE_D=outputs/eval3d/baselines/roomformer_hl3d_3d_v1

envs/geometry/bin/python src/baselines/roomformer3d/lift.py \
  --canonical-root "$TWO_D" \
  --input-manifest "$TWO_D/input_data/input_manifest.json" \
  --out "$THREE_D"

envs/geometry/bin/python src/baselines/roomformer3d/evaluate.py \
  --pred-root "$THREE_D"

envs/geometry/bin/python src/baselines/roomformer3d/depth_evaluate.py \
  --pred-root "$THREE_D" --stride 1 --resume

# Re-run once to merge depth into RESULTS_3D.md.
envs/geometry/bin/python src/baselines/roomformer3d/evaluate.py \
  --pred-root "$THREE_D"

MPLCONFIGDIR=/tmp/mpl-roomformer3d envs/geometry/bin/python \
  src/baselines/roomformer3d/visualize.py \
  --baseline-root "$TWO_D" \
  --input-manifest "$TWO_D/input_data/input_manifest.json" \
  --scene 2t7WUuJeko7 --out "$THREE_D/visualizations/2t7WUuJeko7.png"
```

Each scene directory contains individual structure-plane PLY files, 3D door
and window JSON, an explicitly empty stairs JSON, a coloured `combined.ply`,
and a lifting manifest. Evaluation uses the calibrated evaluator in `src/eval`:
generalized Hausdorff distance for structures and rectangular entity distance
for doors/windows.

Depth Δ5/Δ10 uses shared rays for GT, per-floor and per-room layouts and writes
one resumable JSON file per scene. `--stride 1` is the full Table-2 protocol.
