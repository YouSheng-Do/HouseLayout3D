# Experiment Spec — 2D Floorplan Evaluation Suite (HouseLayout3D)

## Why this exists

Our project targets a **navigation-ready 2D floorplan**, not a 3D layout mesh. But the only
metrics we have are HouseLayout3D's 3D ones (Structures / Doors / Windows / Stairs F1 via
Hausdorff surface distance, plus depth Δ5/Δ10). Those measure things we do not need — ceiling
polygon quality affects Structures F1 but is irrelevant to a floorplan — and they do not
measure the things we do need.

Two consequences:
1. We cannot currently answer **"is the floorplan good enough to support an attribute layer?"**
   That is a different question from "did we match the paper's 3D numbers", and it is the one
   that actually gates our research.
2. When we later add navigation attributes (door width, swing, traversability), we will need to
   attribute their errors to floorplan quality. That requires floorplan quality to be measured
   in the same 2D space the attributes live in.

**This spec builds that evaluation suite.** It is not a change to the pipeline — it is a new
measuring instrument. Build it while the full run is in progress.

---

## Grounding: use the canonical literature definitions, do not invent metrics

The Floor-SP → MonteFloor → RoomFormer lineage has a settled protocol. Follow it so our numbers
are comparable to published work.

**Room / Corner / Angle** (MonteFloor / RoomFormer protocol):
- Match each GT room to the best-IoU predicted room; a match is valid if **IoU > 0.5**.
- **Room**: precision / recall / F1 over matched rooms; also report mean **Room IoU**.
- **Corner** (only on matched rooms): a predicted corner is correct if within a distance
  threshold of a GT corner. When several predictions fall near one GT corner, only the closest
  counts as TP; the rest are FP.
- **Angle**: corner is correct **and** the oriented angle difference is **< 5°**.

**Room+type** (RoomFormer's `Room*`): the Room metric with the added constraint that the
predicted room type matches GT.

**Doors / Windows**: correct if L2 distance to the GT element is below a threshold.

**Room++** (Floor-SP): a room counts as correct only if — in addition to passing the room
test — it is **connected to the correct set of rooms as in the ground truth**. This is the
existing academic precedent for connectivity-aware evaluation; we extend it below.

### Thresholds must be in metres, not pixels

The published thresholds (10 px corner, 10 px door) assume a normalised 256×256 frame.
SLIBO-Net (NeurIPS 2023) explicitly criticises this: the metrics are scale-variant, so
**floorplans with more rooms or larger footprints get evaluated more leniently** — which is
exactly the HouseLayout3D regime (building-scale, up to 40 rooms per floor). SLIBO-Net's fix is
real-scale thresholds, e.g. `Corner@0.1` meaning 0.1 m.

HouseLayout3D is in verified real metres (Stage 4: door width median 0.80 m, ceiling 2.05–3.35 m,
room areas 4–30 m², all simultaneously plausible). So use metres.

Report a **threshold sweep**, not a single number:
- Corner / Angle: `@0.1`, `@0.2`, `@0.3` m — `@0.1` is the SLIBO-Net headline
- Doors / Windows: `@0.2`, `@0.5` m

For context, note in the report roughly what the 10 px legacy threshold corresponds to in
metres for a typical scene, so readers can relate our numbers to published 256 px ones. Do not
present that conversion as exact.

---

## Task 1 — Establish the 2D ground truth

We need per-level 2D GT: room polygons, room types, door segments, and the room-to-room
connectivity graph. **Two sources must be merged** — MP3D supplies structure and connectivity,
HouseLayout3D supplies precise door geometry and opening direction.

### 1a. MP3D `.house` (verified against the official `data_organization.md`)

Located in `house_segmentations/xxx.house`. ASCII, one command per line:

```
H name label #images #panoramas #vertices #surfaces #segments #objects
  #categories #regions #portals #levels ...
L level_index #regions label  px py pz  xlo ylo zlo xhi yhi zhi ...
R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height ...
P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi ...
S surface_index region_index 0 label px py pz  nx ny nz ...
V vertex_index surface_index label  px py pz  nx ny nz ...
C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name
```

What this gives us:

- **Room polygons**: per the spec, "the extent of each region is defined by a prism with its
  vertical extent dictated by its height and its horizontal cross-section dictated by the
  **counter-clockwise set of polygon vertices** associated with each surface associated with the
  region." So: `R` → its `S` surfaces → their `V` vertices.
- **Room types**: the `label` field on `R` is a single character from a 31-category vocabulary
  (`a`=bathroom, `b`=bedroom, `c`=closet, `h`=hallway, `k`=kitchen, `l`=living room,
  `s`=stairs, `x`=outdoor, `y`=balcony, `z`=other, `Z`=junk, `-`=no label, …).
- **Levels**: `L` defines levels; each `R` references its `level_index`.

#### ⚠ CORRECTION — portals are empty; connectivity must be derived

An earlier version of this spec asserted that `P` (portal) records give annotated room-to-room
connectivity. **That was wrong and has been verified false**: all 16 buildings report
`#portals = 0` in the `H` header, and every `P` record present is a *panorama* (hash name, not
an integer index), not a portal. Portal computation is optional in MP3D and these houses do not
have it.

The format does define a portal record, but format support ≠ populated data:
```
P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi ...   ← portal (ABSENT)
P name  panorama_index region_index 0  px py pz  0 0 0 0 0                      ← panorama (what exists)
```

**Consequences, stated plainly:**
- Connectivity GT is **derived**, not annotated. Label it as derived everywhere it appears.
- The earlier "58% of GT doors fall on region boundaries" figure stands — portals cannot
  overturn it, because portals do not exist.
- There is **no annotation of door-less traversable connections** (open passages between
  hallways, dining areas). This is a genuine gap in existing indoor datasets and is worth
  reporting as a finding in its own right, since such connections are necessary for a
  navigation graph.

#### Deriving connectivity, and why it is trustworthy

Derive room-to-room edges geometrically from **human-annotated** region polygons (MP3D) plus
**human-annotated** door geometry (HouseLayout3D), using the same probe rule applied to
predictions. This keeps the derivation identical on both sides, so the measured difference
comes from geometry quality — which is exactly what we want to quantify.

This is defensible because the rule was independently validated in the Structured3D project:
on ground-truth geometry it achieved **98.2% interior-door accuracy with ~100% edge precision**,
robust across a wide probe-distance band.

**Required: re-validate the rule on HouseLayout3D before trusting Tier C.** The Structured3D
validation was on synthetic data. Sample ~30 derived edges across buildings, inspect them
visually against the GT geometry, and report the agreement rate and any failure patterns. If
agreement is poor on real scans, say so — that changes how much weight Tier C can carry.

#### Room count reconciliation (open question — resolve before computing Room metrics)

Measured `.house` totals disagree with the paper: **354 regions / 32 levels** vs the paper's
**317 rooms / 33 levels** (doors match exactly at 292). MP3D "region" and the paper's "room"
are evidently not the same unit.

Do not spend long reverse-engineering the paper's count. Instead:
1. Break down all 354 regions by `label` and report the histogram.
2. Test whether excluding non-habitable labels reconciles the numbers — a plausible hypothesis
   is that the 34 annotated staircases are counted separately, since 354 − 34 = 320 ≈ 317.
3. **Then define our own room set explicitly and use it consistently for GT and predictions.**
   Proposed default: all regions except `x` (outdoor), `Z` (junk). Report what `y` (balcony)
   and `z` (other room) contribute and decide based on counts.

**Stairs regions: keep them as nodes, tagged by type.** Do not collapse them into edges. A
stairwell is standing space with real geometry, and collapsing loses it. Exclude them from the
Room-metric denominator if that aids comparability, but keep them in the graph — inter-level
connectivity is then derived separately from stair geometry.

Whatever is chosen, state the definition in the report; the Room P/R/F1 denominator must not
be ambiguous.

### 1b. MP3D `region_segmentations`

Contains "manually specified segment, object instance, and semantic category labels for walls,
floors, ceilings, **doors, windows**, and furniture-sized objects for each region". These are
mesh-face annotations (`face_category` → mpcat40), not polygons — coarser than HouseLayout3D's
CAD annotations. Use as cross-check, not primary door GT.

### 1c. HouseLayout3D annotations

`doors/{scene}.json`: 4 corners + `normal` (opening direction), 292 doors, hand-annotated.
This is the **primary door geometry GT** and the only source of opening direction — MP3D has
none. Also `windows/`, `stairs/`, `structures/`.

### Deliverable

Per building per level:
- room polygons in 2D metres (+ room type from `R` label, + level index)
- door segments in 2D metres (from HouseLayout3D, projected to the floor plane)
- **derived room-to-room connectivity graph**, each edge tagged as door / stairs, and each
  clearly marked as derived rather than annotated

### Sanity checks before trusting the GT
- Parsed counts match the `H` header (`#regions`, `#levels`) — note `#portals` is 0 by design
- Room-count reconciliation resolved and the room-set definition stated (see above)
- Do region polygons tile each level, or leave gaps? (Gaps break the probe rule — this was
  failure mode F2 in the Structured3D project)
- What fraction of HouseLayout3D doors fall on MP3D region boundaries? Compare against the
  earlier 58% figure
- Derivation rule re-validated on HouseLayout3D by visual inspection (~30 sampled edges)
- Are the two coordinate frames (MP3D `.house` vs HouseLayout3D annotations) actually aligned?
  **Verify empirically** — do not assume. Report the check.

---

## Task 2 — Extract the predicted 2D floorplan

Our pipeline already produces a 2D floorplan internally (Stage 4a builds per-level 2D layouts
and runs room segmentation before any 3D extrusion). Extract that directly — **do not derive a
floorplan by slicing the extruded 3D mesh**, which would confound floorplan quality with
extruder quality.

Confirm and state in the report which artefact you used and where it sits in the pipeline
relative to extrusion.

Per level, extract: room polygons (+ predicted type), door segments, and the scene graph edges.

---

## Task 3 — Implement the metric suite

Three tiers. Report all three.

**Tier A — Geometry (comparable to published work)**
- Room P / R / F1 @ IoU>0.5, and mean Room IoU
- Corner P / R / F1 @ 0.1 / 0.2 / 0.3 m
- Angle P / R / F1 (corner correct + angle diff < 5°)
- Room count error per level (predicted vs GT — are we systematically under- or over-segmenting?)

**Tier B — Semantics**
- Room+type P / R / F1 (room match + type match)
- Doors P / R / F1 @ 0.2 / 0.5 m
- Room-type confusion matrix — Stage 4a reported CLIP typing as near-useless (2t7W: 4 of 5
  rooms labelled "bedroom"); quantify that properly rather than leaving it anecdotal

**Tier C — Topology (the tier nobody has evaluated) — REMAINS A PRIMARY TIER**

The GT here is derived, not annotated (see Task 1a). **This does not demote Tier C**, for a
specific reason: the identical derivation rule is applied to both sides, so all measured
difference is attributable to geometry quality. The GT side derives from *human-annotated*
region polygons and door geometry; the prediction side derives from *predicted* geometry. That
is a clean comparison, not two guesses. Report the derived status explicitly and cite the
Structured3D validation (98.2% / ~100% precision on GT geometry) plus the HouseLayout3D
re-validation required in Task 1a.

- **Access graph edge P / R / F1**: nodes = rooms, edges = doors/openings connecting two rooms
- **Room++** (Floor-SP): room correct *and* connected to the correct set of rooms
- Report edges separately for: room↔room, room↔OUTSIDE, and stair edges (inter-level)
- **Exterior doors must be separated from failures.** A door with only one adjacent room is
  usually a correct exterior door, not a miss. Use the marching-probe approach: march the
  empty-side probe outward; if a room appears, it was a thick-wall miss (real failure); if
  nothing appears within ~1.5 m, it is exterior (correct). Report **interior-door accuracy**
  as the headline, with the raw rate alongside.

Note: this reuses methodology developed in the separate Structured3D project
(`s3daccess/score.py`, `s3daccess/derive.py`). Port the *approach*; the code needs adapting
(pixels → metres, different GT source). Keep the implementation self-contained in this repo.

---

## Task 4 — Report

Add `RESULTS_2D.md` with:

1. **The GT source and how it was built** — state plainly that connectivity is *derived*
   (portals are absent), give the room-set definition used, and report the HouseLayout3D
   re-validation of the derivation rule
2. Which pipeline artefact the predicted floorplan came from, and its position relative to extrusion
3. Tier A / B / C tables — per building **and** 16-building mean, with std
4. Threshold sweeps (Corner @0.1/0.2/0.3, Doors @0.2/0.5)
5. **Distribution of quality across buildings**, not just the mean — how many of 16 buildings
   are good? Is the mean carried by a few, or broadly consistent? (We already know pipeline
   variance is ±6 Δ5, so per-building numbers must be read with that in mind.)
6. Visualisations: GT vs predicted floorplan side by side with the access graph overlaid, for
   the best, median, and worst buildings
7. **A direct assessment: is the floorplan good enough to support an attribute layer?** Not
   "did we match the paper" — that is a different question, answered by the existing 3D metrics.

### Rules
- Use the literature definitions above; if you deviate, say so and justify it.
- Metric thresholds in metres. Do not silently use pixel thresholds.
- Report unflattering numbers. Previous stages produced negative results that redirected the
  project usefully; that is the standard.
- Do not tune anything to improve the numbers — this is an instrument, not an optimisation target.
- If the spec is wrong about what the data or pipeline contains, correct it rather than working
  around it.