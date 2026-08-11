"""2D eval suite — Task 3: Tier A/B/C 指標（依 docs/eval_2d_metric.md）。

[eval2d_v3_strict_levels — 2026-08-11] 延續 v2 的 deterministic threshold-aware
Hungarian matching，並修正 multi-floor correctness：未對齊的 predicted levels 上所有
rooms、doors 與由其幾何推導的 edges 都必須計為 false positives。舊行為可由
``score_scene(..., count_unmatched_pred_levels=False)`` 重現，供 frozen v2 稽核。

[eval2d_v2_hungarian] room/door matching 由 greedy 改為 **deterministic
threshold-aware Hungarian**（linear_sum_assignment）：不受 pred/GT list 排序影響、
不因強迫 assignment 讓有效 pair 被無效 pair 排擠。舊 greedy 版凍存 metrics_v1_greedy.py。

無效多邊形 policy（明確、可測）：`_poly` 以 shapely `buffer(0)` 做 deterministic repair；
若修復後面積為 0 → area=0、不參與匹配（成為 FP/FN），**不 silent drop、不 nondeterministic repair**。

多樓層：GT 與 pred 各層依 floor-z 對齊（堆疊樓層 xy 重疊，必須逐層匹配）。
邊：GT/pred 各自用同一 PIP 規則（access_derive）推導；pred room id 經 IoU 匹配 remap 到 GT 空間。
"""
import argparse, glob, json, os, pickle, sys
from collections import defaultdict, Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.validation import make_valid

EVAL2D_VERSION = "eval2d_v3_strict_levels"

ROOT = "/home/ado/storage/HouseLayout3D"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_derive import derive_access_graph, Room, Door, SUCCESS, ONE_OUTSIDE  # noqa: E402

OUTSIDE = "OUTSIDE"


def _poly(p):
    g = Polygon(np.asarray(p, float))
    if not g.is_valid:
        g = g.buffer(0)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
    return g


def align_levels(gt_z, pred_z, max_dz=1.5):
    """GT 層 ↔ pred 層：依 floor-z 最近貪婪配對（>1.5m 不算同層）。回 {gt_lvl: pred_lvl}。"""
    pairs = sorted((abs(gz - pz), gl, pl) for gl, gz in gt_z.items() for pl, pz in pred_z.items())
    m = {}; ug = set(); up = set()
    for d, gl, pl in pairs:
        if d > max_dz or gl in ug or pl in up:
            continue
        m[gl] = pl; ug.add(gl); up.add(pl)
    return m


def match_rooms_iou(pred_rooms, gt_rooms, thresh=0.5):
    """deterministic 一對一 **Hungarian** IoU>thresh 匹配（公尺 shapely）。
    **permutation-invariant**：先依 stable room ID 排序 → 固定 cost matrix，ties 下反轉輸入仍同結果。
    回 mapping pred_idx->gt_idx, iou_of={pid:(gid,iou)}。"""
    pred_rooms = sorted(pred_rooms, key=lambda r: str(r["idx"]))   # stable-ID → 固定 matrix（tie-safe）
    gt_rooms = sorted(gt_rooms, key=lambda r: str(r["idx"]))
    pids = [r["idx"] for r in pred_rooms]; gids = [r["idx"] for r in gt_rooms]
    if not pids or not gids:
        return {}, {}
    pp = {r["idx"]: _poly(r["poly"]) for r in pred_rooms}
    gp = {r["idx"]: _poly(r["poly"]) for r in gt_rooms}
    IoU = np.zeros((len(pids), len(gids)))
    for i, pid in enumerate(pids):
        pg = pp[pid]
        if pg.area <= 0:
            continue
        for j, gid in enumerate(gids):
            gg = gp[gid]
            if gg.area <= 0 or not pg.intersects(gg):
                continue
            inter = pg.intersection(gg).area
            union = pg.area + gg.area - inter
            IoU[i, j] = inter / union if union > 0 else 0.0
    BIG = 1e6
    cost = np.where(IoU > thresh, -IoU, BIG)      # 有效 pair 給負 cost（優先），無效給大值
    ri, ci = linear_sum_assignment(cost)
    mp = {}; iou_of = {}
    for i, j in zip(ri, ci):
        if IoU[i, j] > thresh:                     # 過門檻才算 match（不強留無效配對）
            mp[pids[i]] = gids[j]; iou_of[pids[i]] = (gids[j], float(IoU[i, j]))
    return mp, iou_of


def corner_angle_prf(pred_poly, gt_poly, thr):
    """Corner（點距≤thr 一對一）＋Angle（點對 ＋ 內角差<5°）的 tp/fp/fn。"""
    P = np.asarray(pred_poly, float); G = np.asarray(gt_poly, float)
    def interior_angles(poly):
        n = len(poly); ang = []
        for i in range(n):
            a = poly[(i-1) % n] - poly[i]; b = poly[(i+1) % n] - poly[i]
            ca = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
            ang.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
        return np.array(ang)
    pa, ga = interior_angles(P), interior_angles(G)
    D = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    order = sorted((D[i, j], i, j) for i in range(len(P)) for j in range(len(G)))
    up = set(); ug = set(); tp_c = tp_a = 0
    for d, i, j in order:
        if d > thr:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp_c += 1
        if abs(pa[i] - ga[j]) < 5.0:
            tp_a += 1
    return (tp_c, len(P)-tp_c, len(G)-tp_c), (tp_a, len(P)-tp_a, len(G)-tp_a)


def match_doors(pred_doors, gt_doors, thr):
    """門：中點 L2≤thr 一對一 **Hungarian**（threshold-aware，permutation-invariant）。回 tp/fp/fn。
    metric 定義不變（沿用中點 L2），只改 matching。"""
    if not pred_doors and not gt_doors:
        return (0, 0, 0)
    if not pred_doors or not gt_doors:
        return (0, len(pred_doors), len(gt_doors))
    # stable 排序（中點座標）→ 固定 matrix，permutation-invariant
    _mk = lambda d: tuple(np.round(np.asarray(d["seg"]).mean(0), 6))
    pred_doors = sorted(pred_doors, key=_mk); gt_doors = sorted(gt_doors, key=_mk)
    P = np.array([np.asarray(d["seg"]).mean(0) for d in pred_doors])
    G = np.array([np.asarray(d["seg"]).mean(0) for d in gt_doors])
    D = np.linalg.norm(P[:, None] - G[None], axis=2)
    BIG = 1e6
    cost = np.where(D <= thr, D, BIG)
    ri, ci = linear_sum_assignment(cost)
    tp = int(sum(D[i, j] <= thr for i, j in zip(ri, ci)))
    return (tp, len(P)-tp, len(G)-tp)


def derive_edges(rooms, doors, d=0.30):
    """rooms/doors dict list → room_room set(frozenset ids), room_outside set(ids)。"""
    R = [Room(id=r["idx"], type=r.get("type", "?"), polygon=r["poly"]) for r in rooms]
    Dd = [Door(id=k, p1=np.asarray(dd["seg"])[0], p2=np.asarray(dd["seg"])[1]) for k, dd in enumerate(doors)]
    rr = set(); ro = set()
    if Dd:
        _, recs = derive_access_graph(R, Dd, d=d)
        for rec in recs:
            if rec["outcome"] == SUCCESS:
                rr.add(frozenset((rec["probe_a"][0], rec["probe_b"][0])))
            elif rec["outcome"] == ONE_OUTSIDE:
                ro.add((rec["probe_a"] or rec["probe_b"])[0])
    return rr, ro


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return {"p": p, "r": r, "f1": 2*p*r/(p+r) if p+r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def score_scene(gt, pred, count_unmatched_pred_levels=True):
    """Score one scene.

    ``count_unmatched_pred_levels=False`` only exists to reproduce the archived
    v2 protocol, which silently ignored predictions on levels that were not
    aligned to a GT level.  New evaluations must keep the default ``True``.
    """
    A = defaultdict(lambda: [0, 0, 0])   # room, corner@.1/.2/.3, angle@.1/.2/.3
    B = defaultdict(lambda: [0, 0, 0])   # room+type, doors@.2/.5
    C = defaultdict(lambda: [0, 0, 0])   # edge room_room/outside/all, room++
    ious = []; rc_err = []
    confusion = Counter()

    lvl_map = align_levels(gt["lvl_z"], pred.get("level_z", {}))
    gt_rooms_by_l = defaultdict(list); pred_rooms_by_l = defaultdict(list)
    for r in gt["rooms"]:
        if r["in_roomset"] and not r.get("is_stairs"):
            gt_rooms_by_l[r["level"]].append(r)
    for r in pred["rooms"]:
        pred_rooms_by_l[r["level"]].append(r)
    gt_doors_by_l = defaultdict(list); pred_doors_by_l = defaultdict(list)
    for d in gt["doors"]:
        gt_doors_by_l[d["level"]].append(d)
    for d in pred["doors"]:
        pred_doors_by_l[d["level"]].append(d)

    for gl, gr in gt_rooms_by_l.items():
        pl = lvl_map.get(gl)
        pr = pred_rooms_by_l.get(pl, []) if pl is not None else []
        mp, iou_of = match_rooms_iou(pr, gr, 0.5)
        tp = len(mp)
        A["room"][0] += tp; A["room"][1] += len(pr)-tp; A["room"][2] += len(gr)-tp
        rc_err.append(len(pr) - len(gr))
        # Room+type & corner/angle on matched pairs
        gtypemap = {r["idx"]: r["type"] for r in gr}
        ptypemap = {r["idx"]: r.get("type", "?") for r in pr}
        gpoly = {r["idx"]: r["poly"] for r in gr}; ppoly = {r["idx"]: r["poly"] for r in pr}
        type_tp = 0
        for pid, gid in mp.items():
            ious.append(iou_of[pid][1])
            gt_t = gtypemap[gid]; pr_t = ptypemap[pid]
            confusion[(gt_t, pr_t)] += 1
            if _type_eq(gt_t, pr_t):
                type_tp += 1
            for ti, thr in enumerate((0.1, 0.2, 0.3)):
                (ctp, cfp, cfn), (atp, afp, afn) = corner_angle_prf(ppoly[pid], gpoly[gid], thr)
                A[f"corner@{thr}"][0] += ctp; A[f"corner@{thr}"][1] += cfp; A[f"corner@{thr}"][2] += cfn
                A[f"angle@{thr}"][0] += atp; A[f"angle@{thr}"][1] += afp; A[f"angle@{thr}"][2] += afn
        B["room+type"][0] += type_tp; B["room+type"][1] += len(pr)-type_tp; B["room+type"][2] += len(gr)-type_tp

        # Tier C edges（該層）
        gt_rr, gt_ro = derive_edges(gr, gt_doors_by_l.get(gl, []))
        pr_rr, pr_ro = derive_edges(pr, pred_doors_by_l.get(pl, []) if pl is not None else [])
        pr_rr = set(frozenset((mp.get(a, f"u{a}"), mp.get(b, f"u{b}"))) for e in pr_rr for a, b in [tuple(e)])
        pr_ro = set(mp.get(a, f"u{a}") for a in pr_ro)
        for a, b in [("room_room", (pr_rr, gt_rr))]:
            tpp = len(b[0] & b[1]); C[a][0] += tpp; C[a][1] += len(b[0]-b[1]); C[a][2] += len(b[1]-b[0])
        po = set((r, OUTSIDE) for r in pr_ro); go = set((r, OUTSIDE) for r in gt_ro)
        C["outside"][0] += len(po & go); C["outside"][1] += len(po-go); C["outside"][2] += len(go-po)
        pall = pr_rr | po; gall = gt_rr | go
        C["all"][0] += len(pall & gall); C["all"][1] += len(pall-gall); C["all"][2] += len(gall-pall)

    # v3 correctness：GT-driven loop above naturally counts unmatched GT levels
    # as FN, but v2 never visited unmatched predicted levels and therefore omitted
    # their FP.  Count every predicted entity on those levels exactly once.
    matched_pred_levels = set(lvl_map.values())
    unmatched_pred_levels = ((set(pred_rooms_by_l) | set(pred_doors_by_l))
                             - matched_pred_levels)
    if count_unmatched_pred_levels:
        for pl in sorted(unmatched_pred_levels, key=str):
            pr = pred_rooms_by_l.get(pl, [])
            pd = pred_doors_by_l.get(pl, [])
            A["room"][1] += len(pr)
            B["room+type"][1] += len(pr)
            if pr:
                rc_err.append(len(pr))

            pr_rr, pr_ro = derive_edges(pr, pd)
            C["room_room"][1] += len(pr_rr)
            C["outside"][1] += len(pr_ro)
            C["all"][1] += len(pr_rr) + len(pr_ro)

    # Doors（逐層，全門；分層對齊）
    for gl, gd in gt_doors_by_l.items():
        pl = lvl_map.get(gl); pd = pred_doors_by_l.get(pl, []) if pl is not None else []
        for thr in (0.2, 0.5):
            tp, fp, fn = match_doors(pd, gd, thr)
            B[f"doors@{thr}"][0] += tp; B[f"doors@{thr}"][1] += fp; B[f"doors@{thr}"][2] += fn

    if count_unmatched_pred_levels:
        for pl in sorted(unmatched_pred_levels, key=str):
            n = len(pred_doors_by_l.get(pl, []))
            for thr in (0.2, 0.5):
                B[f"doors@{thr}"][1] += n

    return {"A": {k: list(v) for k, v in A.items()}, "B": {k: list(v) for k, v in B.items()},
            "C": {k: list(v) for k, v in C.items()}, "iou": ious, "rc_err": rc_err,
            "confusion": dict(confusion),
            "level_diagnostics": {
                "aligned": dict(lvl_map),
                "unmatched_gt": sorted(set(gt["lvl_z"]) - set(lvl_map), key=str),
                "unmatched_pred": sorted(unmatched_pred_levels, key=str),
                "unmatched_pred_counted_as_fp": bool(count_unmatched_pred_levels),
            }}


_TYPE_ALIASES = {"living": "living", "living room": "living", "bedroom": "bedroom", "bed": "bedroom",
    "bathroom": "bathroom", "bath": "bathroom", "toilet": "bathroom", "kitchen": "kitchen",
    "hallway": "hallway", "hall": "hallway", "corridor": "hallway", "dining": "dining",
    "dining room": "dining", "closet": "closet", "office": "office", "study": "office"}
def _type_eq(gt_t, pr_t):
    return _TYPE_ALIASES.get(str(gt_t).lower(), gt_t) == _TYPE_ALIASES.get(str(pr_t).lower(), pr_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=f"{ROOT}/outputs/eval2d/gt")
    ap.add_argument("--pred", default=f"{ROOT}/outputs/eval2d/pred")
    ap.add_argument("--out", default=f"{ROOT}/outputs/eval2d/scores.pkl")
    args = ap.parse_args()
    scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{args.gt}/*.pkl"))
    allres = {}
    for S in scenes:
        gt = pickle.load(open(f"{args.gt}/{S}.pkl", "rb"))
        pp = f"{args.pred}/{S}.pkl"
        if not os.path.exists(pp):
            continue
        pred = pickle.load(open(pp, "rb"))
        allres[S] = score_scene(gt, pred)
    pickle.dump(allres, open(args.out, "wb"))

    def agg(key, sub):
        t = [0, 0, 0]
        for S in allres:
            v = allres[S][key][sub]; t[0] += v[0]; t[1] += v[1]; t[2] += v[2]
        return prf(*t)
    print("==== Tier A — 幾何 ====")
    m = agg("A", "room"); print(f"Room     P {m['p']:.3f} R {m['r']:.3f} F1 {m['f1']:.3f}")
    alliou = [x for S in allres for x in allres[S]["iou"]]
    print(f"mean Room IoU (matched): {np.mean(alliou):.3f}  (n={len(alliou)})")
    for thr in (0.1, 0.2, 0.3):
        c = agg("A", f"corner@{thr}"); a = agg("A", f"angle@{thr}")
        print(f"Corner@{thr}m F1 {c['f1']:.3f} | Angle@{thr}m F1 {a['f1']:.3f}")
    rce = [x for S in allres for x in allres[S]["rc_err"]]
    print(f"房數誤差/層 (pred-GT): 均 {np.mean(rce):+.1f}  |過分割 {sum(x>0 for x in rce)} 層 |欠 {sum(x<0 for x in rce)} 層")
    print("\n==== Tier B — 語意 ====")
    m = agg("B", "room+type"); print(f"Room+type F1 {m['f1']:.3f}")
    for thr in (0.2, 0.5):
        d = agg("B", f"doors@{thr}"); print(f"Doors@{thr}m  P {d['p']:.3f} R {d['r']:.3f} F1 {d['f1']:.3f}")
    print("\n==== Tier C — 拓撲（連通）====")
    for k in ("room_room", "outside", "all"):
        m = agg("C", k); print(f"edge[{k:>9s}] P {m['p']:.3f} R {m['r']:.3f} F1 {m['f1']:.3f}")
    print(f"\n分數存 {args.out}")


if __name__ == "__main__":
    main()
