# Panorama-based layout baselines — HouseLayout3D

> Measurement-only experiment: published MP3D checkpoints, no training or fine-tuning. Building scoring uses the unchanged `eval2d_v3_strict_levels` + `mp3d_house_floor_v0_1` protocol (325 rooms / 32 levels).

## Executive result

- **DOPNet**: per-view GT-room IoU mean/median **0.597/0.632** (n=1390); assembled Room P/R/F1 **0.594/0.643/0.617** at the dev-selected merge threshold 0.40; conditional matched IoU **0.776** (n=209).
- **HorizonNet**: per-view GT-room IoU mean/median **0.582/0.609** (n=1388); assembled Room P/R/F1 **0.537/0.646/0.587** at the dev-selected merge threshold 0.40; conditional matched IoU **0.769** (n=210).

The local-layout score and assembled score are reported separately: the former tests the model; the latter also includes duplicate merging and building coverage. Doors, room types, and connectivity are **N/A** for both panorama models.

The published ~79% MatterportLayout number is 3D IoU against that dataset's dedicated physical-enclosure layout labels. Our diagnostic is 2D IoU against MP3D `.house` semantic region polygons, which can split visually open spaces. It is useful for this downstream floorplan task but is not a like-for-like reproduction of the 79% metric.

## 1. Data preparation and pose verification

All **16** scenes contain skyboxes: **1477 viewpoints / 8862 JPEG faces**, exactly six 1024×1024 faces per viewpoint. Faces were stitched as MP3D U/L/F/R/B/D to 1024×512 equirectangular PNG, then Manhattan-VP aligned with HorizonNet's published preprocessing.

Room coverage is **311/325 = 95.69%** for Room-metric regions; including stairs it is 328/348 = 94.25%. Therefore the direct-observation Room recall ceiling is at most 0.957 before any layout or merge error.

Level assignment used nearest robust camera-z center, computed per scene from non-stair official panorama records. This avoids copying the single region-level of stair regions that physically span floors. Non-stair disagreements are disclosed below.

Orientation uses `pose_1_5`, whose forward direction is skybox image 1 / cubemap left under the [EDM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_EDM_Equirectangular_Projection-Oriented_Dense_Kernelized_Feature_Matching_CVPR_2025_paper.pdf) Matterport3D convention. Therefore panorama right/forward/up maps to -camera-forward/camera-right/-camera-down. A first-scene smoke test caught and rejected the incorrect `pose_1_2 = skybox face 2` assumption; the corrected mapping is global and frozen, not fitted per panorama.

| scene | views | views by z-level | rooms covered | indoor+stairs covered | exact pose-in-room | outside >2cm / unassigned | non-stair level disagreements |
|---|---:|---|---:|---:|---:|---:|---:|
| 17DRP5sb8fy | 48 | L0:48 | 10/10 | 10/10 | 44/48 | 3 / 0 | 0 |
| 1LXtFkjw3qL | 131 | L0:46, L1:63, L2:22 | 25/27 | 27/31 | 130/131 | 0 / 0 | 0 |
| 2t7WUuJeko7 | 37 | L0:37 | 6/6 | 6/6 | 36/37 | 1 / 0 | 0 |
| 5LpN3gDmAk7 | 140 | L0:70, L1:70 | 25/27 | 26/28 | 126/140 | 8 / 0 | 0 |
| HxpKQynjfin | 44 | L0:44 | 8/8 | 8/8 | 41/44 | 1 / 0 | 0 |
| JeFG25nYj2p | 92 | L0:92 | 19/22 | 19/22 | 91/92 | 0 / 0 | 0 |
| JmbYfDe2QKZ | 81 | L0:47, L1:34 | 17/18 | 18/19 | 76/81 | 4 / 0 | 0 |
| S9hNv5qa7GM | 110 | L0:61, L1:49 | 18/18 | 19/19 | 106/110 | 2 / 0 | 0 |
| TbHJrupSAjP | 116 | L0:32, L1:50, L2:34 | 26/26 | 28/28 | 115/116 | 0 / 0 | 0 |
| WYY7iVyf5p8 | 78 | L0:14, L1:29, L2:27, L3:8 | 20/21 | 22/24 | 78/78 | 0 / 0 | 1 |
| YFuZgdQ5vWj | 89 | L0:38, L1:51 | 17/18 | 18/19 | 86/89 | 2 / 0 | 0 |
| e9zR4mvMWw7 | 91 | L0:5, L1:48, L2:38 | 22/22 | 24/24 | 86/91 | 2 / 3 | 0 |
| i5noydFURQK | 56 | L0:22, L1:34 | 13/13 | 14/14 | 54/56 | 2 / 0 | 0 |
| jtcxE69GiFV | 148 | L0:97, L1:51 | 33/35 | 36/38 | 148/148 | 0 / 0 | 0 |
| p5wJjkQkbXX | 155 | L0:31, L1:124 | 34/35 | 35/37 | 154/155 | 0 / 1 | 2 |
| r47D5H71a5s | 61 | L0:61 | 18/19 | 18/21 | 58/61 | 2 / 0 | 0 |

Empirical frame check: 1429/1477 positions are inside their official region exactly, 1446/1477 inside or within 2 cm; 27 are farther than 2 cm and 4 official records remain unassigned. Pose-file vs `.house` position max delta is 0.036 m. Per-scene overlays are in `outputs/eval2d/baselines/panorama_layout_v0_1/data/pose_overlays/`. Visual seam/pole checks were made before full inference; sample paths are recorded in the run README.

## 2. Per-viewpoint room-layout quality

Only viewpoints assigned to a non-stair Room-metric GT region enter this diagnostic. The GT polygon is used only for scoring, never for inference or scaling.

| model | successful / failed views | scored in-room views | GT rooms represented | mean | q25 | median | q75 | min–max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DOPNet | 1477 / 0 | 1390 | 311 | 0.597 | 0.395 | 0.632 | 0.821 | 0.000–0.966 |
| HorizonNet | 1475 / 2 | 1388 | 311 | 0.582 | 0.390 | 0.609 | 0.809 | 0.000–0.981 |

Room-balanced view: first average all viewpoint IoUs within each represented GT room, then summarize those room means (so heavily sampled rooms do not dominate).

| model | represented GT rooms | mean of room means | median of room means |
|---|---:|---:|---:|
| DOPNet | 311 | 0.601 | 0.631 |
| HorizonNet | 311 | 0.583 | 0.599 |

**Serialized inference failures:**

- HorizonNet `S9hNv5qa7GM/81df67c58e324caca152f29eb64f6993`: `ValueError: predicted floor ray does not intersect floor in front of camera`
- HorizonNet `p5wJjkQkbXX/6d3d71725b3d40429c2e06972e5fa0e8`: `ValueError: A LinearRing must have at least 3 coordinate tuples`

Full auditable distributions: `evaluation/<model>/per_view_iou.csv` and `per_room_iou.csv` under the panorama output directory.

## 3. Building-scale results

Thresholds were selected independently per model using only the frozen dev scenes (1LXtFkjw3qL, 2t7WUuJeko7, JeFG25nYj2p, JmbYfDe2QKZ, YFuZgdQ5vWj, e9zR4mvMWw7, p5wJjkQkbXX, r47D5H71a5s). The held-out scenes were not used in selection. This split is prospective only: all 16 scenes had been observed before it was frozen.

| model / subset | merge IoU | Room P | Room R | Room F1 | matched IoU (n) | Corner .1/.2/.3 F1 | mean room-count error / level |
|---|---:|---:|---:|---:|---:|---:|---:|
| DOPNet / dev | 0.40 | 0.614 | 0.677 | 0.644 | 0.762 (113) | 0.158/0.369/0.498 | +1.13 |
| DOPNet / held-out | 0.40 | 0.571 | 0.608 | 0.589 | 0.794 (96) | 0.201/0.407/0.533 | +0.59 |
| DOPNet / all | 0.40 | 0.594 | 0.643 | 0.617 | 0.776 (209) | 0.179/0.387/0.515 | +0.84 |
| HorizonNet / dev | 0.40 | 0.585 | 0.677 | 0.628 | 0.764 (113) | 0.159/0.381/0.507 | +1.73 |
| HorizonNet / held-out | 0.40 | 0.490 | 0.614 | 0.545 | 0.776 (97) | 0.183/0.402/0.520 | +2.35 |
| HorizonNet / all | 0.40 | 0.537 | 0.646 | 0.587 | 0.769 (210) | 0.171/0.391/0.513 | +2.06 |

### Per building (selected thresholds)

| scene | DOPNet P/R/F1 · IoU | HorizonNet P/R/F1 · IoU | DOPNet pred/GT rooms |
|---|---:|---:|---:|
| 17DRP5sb8fy | 0.429/0.300/0.353 · 0.780 | 0.333/0.300/0.316 · 0.727 | 7/10 |
| 1LXtFkjw3qL | 0.436/0.630/0.515 · 0.725 | 0.459/0.630/0.531 · 0.763 | 39/27 |
| 2t7WUuJeko7 | 1.000/0.833/0.909 · 0.891 | 1.000/0.833/0.909 · 0.860 | 5/6 |
| 5LpN3gDmAk7 | 0.600/0.556/0.577 · 0.786 | 0.306/0.407/0.349 · 0.813 | 25/27 |
| HxpKQynjfin | 0.250/0.125/0.167 · 0.877 | 0.500/0.375/0.429 · 0.610 | 4/8 |
| JeFG25nYj2p | 0.824/0.636/0.718 · 0.734 | 0.722/0.591/0.650 · 0.741 | 17/22 |
| JmbYfDe2QKZ | 0.400/0.444/0.421 · 0.743 | 0.500/0.556/0.526 · 0.726 | 20/18 |
| S9hNv5qa7GM | 0.652/0.833/0.732 · 0.843 | 0.615/0.889/0.727 · 0.843 | 23/18 |
| TbHJrupSAjP | 0.615/0.615/0.615 · 0.802 | 0.531/0.654/0.586 · 0.788 | 26/26 |
| WYY7iVyf5p8 | 0.407/0.524/0.458 · 0.737 | 0.379/0.524/0.440 · 0.734 | 27/21 |
| YFuZgdQ5vWj | 0.579/0.611/0.595 · 0.865 | 0.632/0.667/0.649 · 0.823 | 19/18 |
| e9zR4mvMWw7 | 0.667/0.818/0.735 · 0.779 | 0.613/0.864/0.717 · 0.768 | 27/22 |
| i5noydFURQK | 0.562/0.692/0.621 · 0.821 | 0.562/0.692/0.621 · 0.798 | 16/13 |
| jtcxE69GiFV | 0.650/0.743/0.693 · 0.777 | 0.614/0.771/0.684 · 0.747 | 40/35 |
| p5wJjkQkbXX | 0.698/0.857/0.769 · 0.750 | 0.574/0.771/0.659 · 0.748 | 43/35 |
| r47D5H71a5s | 0.714/0.526/0.606 · 0.702 | 0.625/0.526/0.571 · 0.746 | 14/19 |

### Room count per level (selected thresholds)

| scene | level | GT | DOPNet | HorizonNet |
|---|---:|---:|---:|---:|
| 17DRP5sb8fy | 0 | 10 | 7 | 9 |
| 1LXtFkjw3qL | 0 | 14 | 9 | 11 |
| 1LXtFkjw3qL | 1 | 8 | 22 | 18 |
| 1LXtFkjw3qL | 2 | 5 | 8 | 8 |
| 2t7WUuJeko7 | 0 | 6 | 5 | 5 |
| 5LpN3gDmAk7 | 0 | 14 | 11 | 18 |
| 5LpN3gDmAk7 | 1 | 13 | 14 | 18 |
| HxpKQynjfin | 0 | 8 | 4 | 6 |
| JeFG25nYj2p | 0 | 22 | 17 | 18 |
| JmbYfDe2QKZ | 0 | 11 | 11 | 9 |
| JmbYfDe2QKZ | 1 | 7 | 9 | 11 |
| S9hNv5qa7GM | 0 | 9 | 10 | 12 |
| S9hNv5qa7GM | 1 | 9 | 13 | 14 |
| TbHJrupSAjP | 0 | 5 | 9 | 9 |
| TbHJrupSAjP | 1 | 9 | 7 | 11 |
| TbHJrupSAjP | 2 | 12 | 10 | 12 |
| WYY7iVyf5p8 | 0 | 5 | 4 | 5 |
| WYY7iVyf5p8 | 1 | 5 | 5 | 6 |
| WYY7iVyf5p8 | 2 | 8 | 14 | 14 |
| WYY7iVyf5p8 | 3 | 3 | 4 | 4 |
| YFuZgdQ5vWj | 0 | 6 | 7 | 7 |
| YFuZgdQ5vWj | 1 | 12 | 12 | 12 |
| e9zR4mvMWw7 | 0 | 1 | 2 | 3 |
| e9zR4mvMWw7 | 1 | 10 | 13 | 13 |
| e9zR4mvMWw7 | 2 | 11 | 12 | 15 |
| i5noydFURQK | 0 | 5 | 5 | 5 |
| i5noydFURQK | 1 | 8 | 11 | 11 |
| jtcxE69GiFV | 0 | 20 | 24 | 25 |
| jtcxE69GiFV | 1 | 15 | 16 | 19 |
| p5wJjkQkbXX | 0 | 6 | 8 | 9 |
| p5wJjkQkbXX | 1 | 29 | 35 | 38 |
| r47D5H71a5s | 0 | 19 | 14 | 16 |

The aggregate mean count errors above are pred−GT, including zero-view and stair-view failure effects.

## 4. Merge-threshold sensitivity

Rule: within each z-level, form connected components where pairwise polygon IoU is at least the threshold; retain the component member with highest mean IoU agreement (deterministic pano-ID tie break). No GT room ID, union cleanup, or learned merger is used.

| model | threshold | dev F1 | held-out F1 | all F1 | all P/R | all matched IoU |
|---|---:|---:|---:|---:|---:|---:|
| DOPNet | 0.10 | 0.484 | 0.396 | 0.443 | 0.688/0.326 | 0.764 |
| DOPNet | 0.20 | 0.549 | 0.466 | 0.510 | 0.675/0.409 | 0.762 |
| DOPNet | 0.30 | 0.611 | 0.551 | 0.582 | 0.650/0.526 | 0.770 |
| DOPNet **(selected)** | 0.40 | 0.644 | 0.589 | 0.617 | 0.594/0.643 | 0.776 |
| DOPNet | 0.50 | 0.643 | 0.593 | 0.618 | 0.540/0.723 | 0.772 |
| DOPNet | 0.60 | 0.582 | 0.541 | 0.562 | 0.449/0.751 | 0.771 |
| DOPNet | 0.70 | 0.540 | 0.478 | 0.509 | 0.379/0.775 | 0.779 |
| HorizonNet | 0.10 | 0.458 | 0.391 | 0.427 | 0.667/0.314 | 0.755 |
| HorizonNet | 0.20 | 0.557 | 0.488 | 0.525 | 0.657/0.437 | 0.752 |
| HorizonNet | 0.30 | 0.574 | 0.520 | 0.548 | 0.572/0.526 | 0.762 |
| HorizonNet **(selected)** | 0.40 | 0.628 | 0.545 | 0.587 | 0.537/0.646 | 0.769 |
| HorizonNet | 0.50 | 0.595 | 0.518 | 0.556 | 0.462/0.698 | 0.770 |
| HorizonNet | 0.60 | 0.537 | 0.478 | 0.507 | 0.388/0.732 | 0.769 |
| HorizonNet | 0.70 | 0.485 | 0.434 | 0.460 | 0.329/0.760 | 0.767 |

## 5. Failure analysis

Operational checks use large rooms = top area quartile (≥20.06 m²), corridor = MP3D `h`/hallway, and non-Manhattan = length-weighted residual >10° from the best single orthogonal frame. These labels are diagnostic slices, not tuned filters.

| model | all view IoU (n) | large-area IoU (n) | hallway IoU (n) | non-Manhattan IoU (n) | merge source→output; singleton clusters |
|---|---:|---:|---:|---:|---:|
| DOPNet | 0.597 (1390) | 0.610 (643) | 0.441 (245) | 0.450 (42) | 1477→352; 138 |
| HorizonNet | 0.582 (1388) | 0.593 (643) | 0.422 (245) | 0.404 (42) | 1475→391; 164 |

**Rooms without a viewpoint:** 14 of 325: `1LXtFkjw3qL:R13, 1LXtFkjw3qL:R21, 5LpN3gDmAk7:R10, 5LpN3gDmAk7:R25, JeFG25nYj2p:R21, JeFG25nYj2p:R4, JeFG25nYj2p:R9, JmbYfDe2QKZ:R17, WYY7iVyf5p8:R23, YFuZgdQ5vWj:R18, jtcxE69GiFV:R0, jtcxE69GiFV:R35, p5wJjkQkbXX:R32, r47D5H71a5s:R13`. They are retained in GT. A neighbouring panorama prediction can accidentally overlap one, but that is not a direct observation. Covered-vs-uncovered match counts are in `evaluation/failure_analysis.json`.

The main assembly stress signal is the number of duplicate components and singletons: a high threshold leaves multiple views of one room as false-positive rooms; a low threshold can transitively collapse adjacent rooms. This experiment deliberately stops at that simple merger, as required.

Manhattan VP alignment is part of both official inference paths. Large/open, hallway, and non-Manhattan slices above show where that prior and single-room framing help or hurt; no category-specific repair was added.

## 6. Best / median / worst visualisations

Ranked by selected-threshold DOPNet per-building Room F1 (blue=GT, red=assembled panorama polygons, black dots=viewpoints):

- best: `2t7WUuJeko7`, F1=0.909 — `outputs/eval2d/baselines/panorama_layout_v0_1/visualizations/best_2t7WUuJeko7.png`
- median: `r47D5H71a5s`, F1=0.606 — `outputs/eval2d/baselines/panorama_layout_v0_1/visualizations/median_r47D5H71a5s.png`
- worst: `HxpKQynjfin`, F1=0.167 — `outputs/eval2d/baselines/panorama_layout_v0_1/visualizations/worst_HxpKQynjfin.png`

## 7. Four-way comparison under the same evaluator

All layout numbers below use strict-v3 + official325. Input conditions differ and must be quoted with the score.

| method | condition | Room F1 | matched IoU (n) | Corner .1/.2/.3 | Room+type | Door@.5 | edge-all |
|---|---|---:|---:|---:|---:|---:|---:|
| MULTIFLOOR3D reimpl | full pipeline; predicts levels | 0.602 | 0.785 (193) | 0.195/0.342/0.436 | 0.144 | 0.243 | 0.131 |
| RoomFormer per-floor | oracle `.house` levels + region points | 0.415 | 0.751 (107) | 0.111/0.307/0.437 | 0.093 | 0.244 | 0.161 |
| CAGE per-floor | oracle `.house` levels + region points | 0.633 | 0.779 (199) | 0.194/0.418/0.519 | N/A | N/A | N/A |
| Panorama DOPNet (IoU 0.40) | RGB panoramas + GT poses; z-levels | 0.617 | 0.776 (209) | 0.179/0.387/0.515 | N/A | N/A | N/A |

HorizonNet is retained as the classic panorama reference in Sections 2–4; DOPNet is the spec-designated primary panorama row in the four-way table.

## Reproducibility and deviations

- Official repos/checkpoints: DOPNet MP3D `model_best_mp3d.pkl` (SHA-256 `0dcc7929...38cb`); HorizonNet MP3D `resnet50_rnn__mp3d.pth` (SHA-256 `bdef2e33...0c96`).
- No parameter update, training, fine-tuning, GT-shape scaling, or pose estimation was performed.
- Compatibility-only changes are documented in `src/baselines/panorama/README.md`: device-agnostic DOPNet sampling buffer, no redundant ImageNet initialization, OpenCV contour-list adaptation, and NumPy SVD replacing HorizonNet's one-component sklearn PCA.
- Four official `.house` panorama records remain unassigned rather than being forced into a GT room. All per-view failures are serialized with exception type and message.
