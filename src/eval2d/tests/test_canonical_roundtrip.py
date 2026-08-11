"""Task D 驗收：canonical round-trip + PKL vs canonical score equivalence（CPU-only）。"""
import glob, json, os, pickle, sys
import numpy as np
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
from canonical_io import SCHEMA_VERSION, load_canonical
from geometry_v2 import to_shapely
from metrics import score_scene, prf

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
CANON = f"{ROOT}/outputs/eval2d/canonical/watershed_v3_pre_report_v0_2"
LEGACY_CANON = f"{ROOT}/outputs/eval2d/canonical/watershed_v3_pre_report"
scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/pred/*.pkl"))

def room_f1(res): return prf(*res["A"]["room"])["f1"]

def by_idx(rooms):
    return {r["idx"]: np.asarray(r.get("metric_geometry", r["poly"]))
            for r in rooms}

fails = []
for S in scenes:
    pred = pickle.load(open(f"{BASE}/pred/{S}.pkl", "rb"))
    gt = pickle.load(open(f"{BASE}/gt/{S}.pkl", "rb"))
    cdict = json.load(open(f"{CANON}/{S}.json"))
    if cdict["schema_version"] != SCHEMA_VERSION:
        fails.append(f"{S} schema {cdict['schema_version']} != {SCHEMA_VERSION}")
    loaded = load_canonical(cdict)
    # room / door / EDGE count 相同（edge 含 room-room + exterior）
    if len(loaded["rooms"]) != len(pred["rooms"]):
        fails.append(f"{S} room count {len(loaded['rooms'])}!={len(pred['rooms'])}")
    if len(loaded["doors"]) != len(pred["doors"]):
        fails.append(f"{S} door count mismatch")
    if len(loaded.get("edges", [])) != len(pred["edges"]):
        fails.append(f"{S} EDGE count {len(loaded.get('edges',[]))}!={len(pred['edges'])} (exterior 掉失?)")
    # room polygon 座標 round-trip（依 ID 對齊）
    po, pc = by_idx(pred["rooms"]), by_idx(loaded["rooms"])
    for k in po:
        if k not in pc or not np.allclose(po[k], pc[k], atol=1e-9):
            fails.append(f"{S} room poly drift id={k}"); break
    # DOOR 座標 round-trip（不只 count）
    do = sorted([tuple(np.round(np.asarray(d["seg"]).ravel(), 6)) for d in pred["doors"]])
    dc = sorted([tuple(np.round(np.asarray(d["seg"]).ravel(), 6)) for d in loaded["doors"]])
    if do != dc:
        fails.append(f"{S} door coord drift")
    if not all(np.isfinite(np.asarray(r["poly"])).all() for r in loaded["rooms"]):
        fails.append(f"{S} non-finite coord")
    if not all(to_shapely(r["geometry"]).is_valid for r in loaded["rooms"]):
        fails.append(f"{S} invalid v0.2 structured geometry")
    # 完整 A/B/C score equivalence（不只 Room F1）
    s_pkl = score_scene(gt, pred); s_can = score_scene(gt, loaded)
    for tier in ("A", "B", "C"):
        for k in set(s_pkl[tier]) | set(s_can[tier]):
            if s_pkl[tier].get(k) != s_can[tier].get(k):
                fails.append(f"{S} {tier}.{k} score mismatch {s_pkl[tier].get(k)} vs {s_can[tier].get(k)}")

# scene order 不影響 aggregate（用 room tp/fp/fn pool）
def agg_room(order):
    t = [0, 0, 0]
    for S in order:
        gt = pickle.load(open(f"{BASE}/gt/{S}.pkl", "rb"))
        cd = load_canonical(json.load(open(f"{CANON}/{S}.json")))
        v = score_scene(gt, cd)["A"]["room"]; t = [t[i]+v[i] for i in range(3)]
    return prf(*t)["f1"]
f_fwd = agg_room(scenes); f_rev = agg_room(scenes[::-1])
if abs(f_fwd - f_rev) > 1e-9:
    fails.append(f"aggregate order-dependent {f_fwd} vs {f_rev}")

# Archived v0.1 remains readable after v0.2 promotion.
legacy_scene = scenes[0]
legacy = load_canonical(json.load(open(f"{LEGACY_CANON}/{legacy_scene}.json")))
if len(legacy["rooms"]) != len(pickle.load(open(f"{BASE}/pred/{legacy_scene}.pkl", "rb"))["rooms"]):
    fails.append("v0.1 backward compatibility room count")

print(f"round-trip + equivalence 檢查 {len(scenes)} 場景")
if fails:
    print("❌ FAIL:"); [print("  ", f) for f in fails]; sys.exit(1)
print(f"✅ ALL PASS：v0.2 16 場景 count/coord/topology、PKL↔canonical score 等價、"
      "aggregate 順序無關、v0.1 backward-compatible")
print(f"   aggregate Room F1（canonical）= {f_fwd:.4f}")
