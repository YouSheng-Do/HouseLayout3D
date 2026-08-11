"""Task D — 匯出 16 份 canonical annotated-floorplan v0.1 JSON（只讀 frozen pred PKL）。"""
import glob, hashlib, json, os, pickle, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_io import build_canonical, load_canonical, SCHEMA_VERSION

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
OUT = f"{ROOT}/outputs/eval2d/canonical/watershed_v3_pre_report"
os.makedirs(OUT, exist_ok=True)

scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/pred/*.pkl"))
files = {}
for S in scenes:
    pred = pickle.load(open(f"{BASE}/pred/{S}.pkl", "rb"))
    cdict = build_canonical(pred)
    # 驗證：JSON 可序列化（無 numpy）、可 round-trip 回讀
    txt = json.dumps(cdict, ensure_ascii=False, indent=1)
    _ = load_canonical(json.loads(txt))
    p = f"{OUT}/{S}.json"; open(p, "w").write(txt)
    files[f"{S}.json"] = hashlib.sha256(txt.encode()).hexdigest()

manifest = {"collection": "canonical_annotated_floorplan_v0.1",
            "schema_version": SCHEMA_VERSION, "baseline_id": "watershed_v3_pre_report",
            "source": "frozen pred PKL (no segmentation rerun)",
            "n_scenes": len(scenes), "scene_ids": scenes, "files_sha256": files,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(manifest, open(f"{OUT}/collection_manifest.json", "w"), ensure_ascii=False, indent=1)
print(f"[Task D] 匯出 {len(scenes)} 份 canonical JSON + collection_manifest → {OUT}")
assert len(scenes) == 16, f"應 16，得 {len(scenes)}"
# 抽驗一份內容
import json as _j
ex = _j.load(open(f"{OUT}/{scenes[0]}.json"))
print(f"範例 {scenes[0]}: schema={ex['schema_version']} levels={len(ex['levels'])} "
      f"rooms={sum(len(l['rooms']) for l in ex['levels'])} doors={sum(len(l['doors']) for l in ex['levels'])}")
print("[Task D 驗收] ✅ 16 JSON + manifest；JSON 無 numpy、可 round-trip")
