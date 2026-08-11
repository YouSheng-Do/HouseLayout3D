"""2D eval suite — Task 1: 建每棟每層的 2D GT（依 docs/eval_2d_metric.md）。

來源合併（論文 §3 已證實）：
  - 房間多邊形＋房型：MP3D region（region_segmentations/regionN.ply ＋ .house R label）
    論文說 rooms 繼承 MP3D per-vertex room ids；我們用 region 為源（354 個），套明講的濾鏡。
  - 門幾何＋開向：HouseLayout3D doors/{scene}.json（唯一有開向；292 門，手標）
  - 連通：portals 全 0（已證），改用 marching-probe 從「人標 region 多邊形＋人標門」幾何推導，
    兩側同規則 → 差異全歸幾何品質。

房間 set 濾鏡（明講、可複現）：排除 label x(outdoor)、Z(junk)；保留 y/z 並回報佔比。
Stairs region(s) 保留為節點、標 type=stairs，但不計入 Room 分母（層間連通另由樓梯幾何推）。

輸出：outputs/eval2d/gt/{scene}.pkl ＋ stdout sanity 報告。
"""
import argparse, glob, json, os, pickle, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import open3d as o3d
from scipy import ndimage

ROOT = "/home/ado/storage/HouseLayout3D"
RES = 0.05
ROOM_EXCLUDE = {"x", "Z"}          # spec 預設濾鏡  # TODO: 依 y/z 佔比再定
LABEL_NAME = {"a":"bathroom","b":"bedroom","c":"closet","d":"dining","e":"entryway","f":"familyroom",
    "g":"garage","h":"hallway","i":"library","j":"laundry","k":"kitchen","l":"living","m":"meeting",
    "n":"lounge","o":"office","p":"porch","r":"rec","s":"stairs","t":"toilet","u":"utility","v":"tv",
    "w":"workout","x":"outdoor","y":"balcony","z":"other","B":"bar","C":"classroom","D":"dining_booth",
    "S":"spa","Z":"junk","-":"none"}


def parse_house(path):
    """回傳 levels[list], regions[list of dict]。R 欄位：idx lvl 0 0 label px py pz xlo ylo zlo ...height"""
    levels, regions = [], []
    hdr = {}
    for line in open(path, errors="ignore"):
        f = line.split()
        if not f:
            continue
        if f[0] == "H":
            # H ... #categories #regions #portals #levels（官方格式欄位 10/11/12）
            hdr = {"n_regions": int(f[10]), "n_portals": int(f[11]), "n_levels": int(f[12])}
        elif f[0] == "L":
            levels.append({"idx": int(f[1])})
        elif f[0] == "R":
            regions.append({"idx": int(f[1]), "level": int(f[2]), "label": f[5],
                            "zlo": float(f[11]), "height": float(f[15]) if len(f) > 15 else 0.0})
    return hdr, levels, regions


def region_polygon(ply_path, origin, shape):
    """region ply → 該 region 在共用格點上的 2D 佔格（bool），供輪廓抽取＋tiling 檢查。"""
    m = o3d.io.read_triangle_mesh(ply_path)
    v = np.asarray(m.vertices)
    if len(v) == 0:
        return None
    j = ((v[:, 0] - origin[0]) / RES).astype(int)
    i = ((v[:, 1] - origin[1]) / RES).astype(int)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    occ = np.zeros(shape, bool)
    occ[i[ok], j[ok]] = True
    occ = ndimage.binary_closing(occ, iterations=2)
    occ = ndimage.binary_fill_holes(occ)
    return occ


def occ_to_poly(occ, origin):
    """佔格 → 最大輪廓多邊形（公尺，RDP 化簡）。"""
    cnts, _ = cv2.findContours(occ.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon=0.10 / RES, closed=True)     # 10cm RDP
    if len(c) < 3:
        return None
    return c.reshape(-1, 2).astype(np.float64) * RES + origin + RES / 2


def build_scene(scene):
    sroot = f"{ROOT}/data/mp3d/v1/scans/{scene}/{scene}"
    hp = f"{sroot}/house_segmentations/{scene}.house"
    hdr, levels, regions = parse_house(hp)

    # 共用格點：涵蓋全棟 region 範圍
    allv = []
    for r in regions:
        p = f"{sroot}/region_segmentations/region{r['idx']}.ply"
        if os.path.exists(p):
            allv.append(np.asarray(o3d.io.read_triangle_mesh(p).vertices)[:, :2])
    allv = np.vstack(allv)
    origin = allv.min(0) - 0.5
    shape = (int((allv[:, 1].max() - origin[1]) / RES) + 20,
             int((allv[:, 0].max() - origin[0]) / RES) + 20)

    # 逐 region → 多邊形 + 佔格（供 tiling）；套濾鏡
    rooms = []
    occ_by_level = defaultdict(lambda: np.zeros(shape, np.int32))
    excl = Counter()
    for r in regions:
        p = f"{sroot}/region_segmentations/region{r['idx']}.ply"
        if not os.path.exists(p):
            continue
        occ = region_polygon(p, origin, shape)
        if occ is None or occ.sum() * RES * RES < 0.5:
            continue
        in_roomset = r["label"] not in ROOM_EXCLUDE
        if not in_roomset:
            excl[r["label"]] += 1
        poly = occ_to_poly(occ, origin)
        if poly is None:
            continue
        occ_by_level[r["level"]][occ] = r["idx"] + 1
        rooms.append({"idx": r["idx"], "level": r["level"], "label": r["label"],
                      "type": LABEL_NAME.get(r["label"], r["label"]),
                      "poly": poly, "area": float(occ.sum() * RES * RES),
                      "in_roomset": in_roomset, "is_stairs": r["label"] == "s",
                      "_occ": occ})

    # 層 floor z（供門分層）
    lvl_z = {}
    for r in regions:
        lvl_z.setdefault(r["level"], []).append(r["zlo"])
    lvl_z = {k: float(np.median(v)) for k, v in lvl_z.items()}

    # HL3D 門 → 2D 段＋開向＋分層
    doors = []
    dp = f"{ROOT}/external/houselayout3d/data/doors/{scene}.json"
    if os.path.exists(dp):
        for d in json.load(open(dp))["doors"]:
            vv = np.array(d["vertices"]); zs = vv[np.argsort(vv[:, 2])][:2]
            seg = zs[:, :2]; zc = float(vv[:, 2].mean())
            lvl = min(lvl_z, key=lambda k: abs(lvl_z[k] - vv[:, 2].min())) if lvl_z else 0
            doors.append({"seg": seg, "normal": d.get("normal"), "z": zc, "level": lvl})

    return {"scene": scene, "hdr": hdr, "origin": origin, "shape": shape,
            "lvl_z": lvl_z, "rooms": rooms, "doors": doors,
            "occ_by_level": {k: v for k, v in occ_by_level.items()}}


def derive_edges(gt, probe=0.30):
    """移植 Structured3D 已驗證規則（point-in-polygon 探針，98.2% on GT）。
    per level：門兩側探針對 room 多邊形做 PIP → success/one_outside(外門)/same_room/…"""
    from access_derive import derive_access_graph, Room, Door, SUCCESS, ONE_OUTSIDE, SAME_ROOM, BOTH_OUTSIDE, OVERLAP
    edges = []; outcomes = Counter()
    by_lvl = defaultdict(lambda: {"rooms": [], "doors": []})
    for r in gt["rooms"]:
        if r["in_roomset"]:                     # 只用 roomset 房（含 stairs 節點）
            by_lvl[r["level"]]["rooms"].append(Room(id=r["idx"], type=r["type"], polygon=r["poly"]))
    for k, d in enumerate(gt["doors"]):
        by_lvl[d["level"]]["doors"].append(Door(id=k, p1=d["seg"][0], p2=d["seg"][1]))
    for lvl, g in by_lvl.items():
        if not g["doors"]:
            continue
        graph, records = derive_access_graph(g["rooms"], g["doors"], d=probe)
        for rec in records:
            outcomes[rec["outcome"]] += 1
            if rec["outcome"] == SUCCESS:
                a, b = rec["probe_a"][0], rec["probe_b"][0]
                edges.append({"rooms": (a, b), "kind": "room-room", "level": lvl})
            elif rec["outcome"] == ONE_OUTSIDE:
                hit = (rec["probe_a"] or rec["probe_b"])[0]
                edges.append({"rooms": (hit, "OUTSIDE"), "kind": "exterior", "level": lvl})
    gt["_outcomes"] = outcomes
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--out", default=f"{ROOT}/outputs/eval2d/gt")
    args = ap.parse_args()
    scenes = args.scenes or sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
        for p in glob.glob(f"{ROOT}/data/mp3d/v1/scans/*/*/house_segmentations"))
    os.makedirs(args.out, exist_ok=True)

    print(f"{'scene':>12s} | {'lvl':>3s} {'reg':>3s} {'roomset':>7s} {'door':>4s} | "
          f"{'框對齊':>6s} {'tiling覆蓋':>9s} {'內門準':>7s} {'succ/外/失敗':>12s}")
    agg = defaultdict(int); tot_edge = Counter()
    for S in scenes:
        gt = build_scene(S)
        edges = derive_edges(gt)
        gt["edges"] = edges
        nlvl = len(gt["lvl_z"]); nreg = len(gt["rooms"])
        nroom = sum(r["in_roomset"] and not r["is_stairs"] for r in gt["rooms"])
        ndoor = len(gt["doors"])
        # 座標框對齊：門中心是否落在 region 範圍內
        if gt["doors"]:
            dctr = np.array([d["seg"].mean(0) for d in gt["doors"]])
            rmin = gt["origin"]; rmax = gt["origin"] + np.array(gt["shape"][::-1]) * RES
            inside = np.mean((dctr >= rmin) & (dctr <= rmax))
            frame = f"{100*inside:.0f}%"
        else:
            frame = "—"
        # tiling：各層 room 佔格佔該層 bbox 比例（漏洞檢查）
        cover = []
        for lvl, lab in gt["occ_by_level"].items():
            filled = (lab > 0)
            if filled.sum():
                ys, xs = np.where(filled)
                bbox = (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)
                cover.append(filled.sum()/max(bbox, 1))
        covs = f"{100*np.mean(cover):.0f}%" if cover else "—"
        # 門結果分類（success=室內連兩房；one_outside=外門；其餘=失敗/歧義）
        oc = gt["_outcomes"]
        succ = oc.get("success", 0); ext = oc.get("one_outside", 0)
        interior_acc = 100 * succ / max(ndoor - ext, 1)   # 論文式：內門準確率（剔外門）
        onb = f"{interior_acc:.0f}%"
        tot_edge += oc
        edstr = f"{succ}/{ext}/{oc.get('same_room',0)+oc.get('both_outside',0)+oc.get('overlap',0)}"
        print(f"{S:>12s} | {nlvl:>3d} {nreg:>3d} {nroom:>7d} {ndoor:>4d} | "
              f"{frame:>6s} {covs:>9s} {onb:>7s} {edstr:>12s}")
        agg["reg"] += nreg; agg["roomset"] += nroom; agg["door"] += ndoor; agg["lvl"] += nlvl
        with open(f"{args.out}/{S}.pkl", "wb") as f:
            pickle.dump({k: v for k, v in gt.items() if k != "occ_by_level"}, f)
    print(f"\n合計: {agg['lvl']} 層 · {agg['reg']} region · {agg['roomset']} roomset房 · {agg['door']} 門")
    print(f"  (論文: 33 層 · 317 房 · 292 門；region 354)")
    succ = tot_edge.get("success", 0); ext = tot_edge.get("one_outside", 0)
    sr = tot_edge.get("same_room", 0); bo = tot_edge.get("both_outside", 0); ov = tot_edge.get("overlap", 0)
    print(f"門結果（PIP 規則）: success(內門連兩房) {succ} · one_outside(外門) {ext} · "
          f"same_room {sr} · both_outside {bo} · overlap {ov}")
    print(f"  → 內門準確率（剔外門）= {100*succ/max(succ+sr+bo+ov,1):.1f}%（S3D on-GT 為 98.2%）")
    print(f"  → 外門佔比 = {100*ext/max(sum(tot_edge.values()),1):.0f}%（S3D ~23%）")


if __name__ == "__main__":
    main()
