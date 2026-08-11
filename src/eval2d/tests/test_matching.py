"""eval2d_v2_hungarian regression tests（CPU-only, geometry env）。
執行: CUDA_VISIBLE_DEVICES="" python src/eval2d/tests/test_matching.py
涵蓋 docs/tasks 要求的 11 項：permutation invariance、Hungarian≥greedy、IoU 邊界、
empty、door permutation/boundary、concave、invalid-polygon policy。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import match_rooms_iou, match_doors, match_segments_endpoint
import importlib.util
# v1 greedy（archived）供對照
_spec = importlib.util.spec_from_file_location(
    "metrics_v1_greedy", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics_v1_greedy.py"))
v1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v1)


def R(idx, x0, y0, x1, y1):
    return {"idx": idx, "poly": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)}

def Rpoly(idx, pts):
    return {"idx": idx, "poly": np.array(pts, float)}

def door(seg):
    return {"seg": np.array(seg, float)}

def prf(tp, fp, fn):
    return (tp, fp, fn)

def matched_prf(mp, npred, ngt):
    tp = len(mp); return (tp, npred - tp, ngt - tp)

RESULTS = []
def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"  {'✅' if cond else '❌'} {name}")


# 1) pred 順序反轉 → matching/PRF 不變
def t_perm_pred():
    gt = [R(1, 0, 0, 10, 10), R(2, 20, 0, 30, 10)]
    pr = [R("a", 0, 0, 10, 10), R("b", 20, 0, 30, 10)]
    m1, _ = match_rooms_iou(pr, gt); m2, _ = match_rooms_iou(pr[::-1], gt)
    check("1 pred-order invariance", matched_prf(m1, 2, 2) == matched_prf(m2, 2, 2) and set(m1.items()) == set(m2.items()))

# 2) GT 順序反轉 → 不變
def t_perm_gt():
    gt = [R(1, 0, 0, 10, 10), R(2, 20, 0, 30, 10)]
    pr = [R("a", 0, 0, 10, 10), R("b", 20, 0, 30, 10)]
    m1, _ = match_rooms_iou(pr, gt); m2, _ = match_rooms_iou(pr, gt[::-1])
    check("2 gt-order invariance", set(m1.items()) == set(m2.items()))

# 3) 真 adversarial：greedy 因先選最高 IoU 的 A-gA 而擋掉 B → greedy=1；Hungarian 交換得 2
def t_hungarian_adversarial():
    gA = R(1, 0, 0, 10, 10); gB = R(2, 10, 0, 20, 10)
    A = R("A", 3, 0, 16, 10)    # IoU: gA 0.44, gB 0.35
    B = R("B", -6, 0, 6, 10)    # IoU: gA 0.375, gB 0（僅 gA 可配）
    pr = [A, B]; gt = [gA, gB]; thr = 0.3
    m1, _ = v1.match_rooms_iou(pr, gt, thresh=thr)   # greedy
    m2, _ = match_rooms_iou(pr, gt, thresh=thr)      # hungarian
    print(f"     (thr={thr}: v1 greedy={len(m1)} match, v2 hungarian={len(m2)} match)")
    check("3 adversarial: greedy=1, hungarian=2 (真次優案例)", len(m1) == 1 and len(m2) == 2)

# 3b) exact-tie permutation：完全相同 IoU ties，反轉輸入 → mapping 不變（stable-ID）
def t_exact_tie_permutation():
    gt = [R(1, 0, 0, 10, 10), R(2, 0, 0, 10, 10)]   # 兩個相同 GT
    pr = [R("a", 0, 0, 10, 10), R("b", 0, 0, 10, 10)]  # 兩個相同 pred（全 IoU=1.0 ties）
    m_fwd, _ = match_rooms_iou(pr, gt)
    m_rev, _ = match_rooms_iou(pr[::-1], gt[::-1])
    check("3b exact-tie permutation invariance (mapping identical)", m_fwd == m_rev and len(m_fwd) == 2)

# 4) IoU 邊界（strict >0.5）：=0.5 不配、>0.5 配、<0.5 不配
def t_iou_boundary():
    A = R("a", 0, 0, 1, 1)                       # area 1
    gt_eq = [R(1, 0, 0, 1, 2)]                    # IoU = 1/2 = 0.5 → 不配
    gt_gt = [R(1, 0, 0, 1, 1.9)]                 # 1/1.9 ≈ 0.526 → 配
    gt_lt = [R(1, 0, 0, 1, 2.1)]                 # 1/2.1 ≈ 0.476 → 不配
    m_eq, _ = match_rooms_iou([A], gt_eq); m_gt, _ = match_rooms_iou([A], gt_gt); m_lt, _ = match_rooms_iou([A], gt_lt)
    check("4 IoU boundary (=0.5 no, >0.5 yes, <0.5 no)", len(m_eq) == 0 and len(m_gt) == 1 and len(m_lt) == 0)

# 5) empty GT
def t_empty_gt():
    m, _ = match_rooms_iou([R("a", 0, 0, 1, 1)], [])
    check("5 empty GT", m == {})

# 6) empty pred
def t_empty_pred():
    m, _ = match_rooms_iou([], [R(1, 0, 0, 1, 1)])
    check("6 empty pred", m == {})

# 7) both empty
def t_both_empty():
    m, _ = match_rooms_iou([], [])
    check("7 both empty", m == {})

# 8) door permutation invariance
def t_door_perm():
    gt = [door([[0, 0], [1, 0]]), door([[5, 5], [6, 5]])]
    pr = [door([[0.1, 0], [1.1, 0]]), door([[5, 5], [6, 5]])]
    r1 = match_doors(pr, gt, 0.5); r2 = match_doors(pr[::-1], gt[::-1], 0.5)
    check("8 door permutation invariance", r1 == r2)

def t_window_endpoint_contract():
    gt = [door([[0, 0], [2, 0]])]
    same_reversed = [door([[2, 0], [0, 0]])]
    wrong_length_same_centre = [door([[0.75, 0], [1.25, 0]])]
    check("8b window endpoint orientation invariant",
          match_segments_endpoint(same_reversed, gt, 0.2) == (1, 0, 0))
    check("8c window rejects same-centre wrong length",
          match_segments_endpoint(wrong_length_same_centre, gt, 0.5) == (0, 1, 1))

# 9) door threshold boundary（中點距離恰 = thr → 命中；> thr → 不中）
def t_door_boundary():
    g = [door([[0, 0], [1, 0]])]                 # 中點 (0.5,0)
    p_on = [door([[0, 0.5], [1, 0.5]])]          # 中點 (0.5,0.5)，距 0.5
    p_off = [door([[0, 0.6], [1, 0.6]])]         # 距 0.6
    r_on = match_doors(p_on, g, 0.5); r_off = match_doors(p_off, g, 0.5)
    check("9 door threshold boundary (<=thr hit)", r_on[0] == 1 and r_off[0] == 0)

# 10) concave valid polygon 正常匹配
def t_concave():
    Lshape = Rpoly(1, [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]])   # L 形（凹）
    pred = Rpoly("a", [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]])
    m, iou = match_rooms_iou([pred], [Lshape])
    check("10 concave polygon match (IoU≈1)", len(m) == 1 and iou["a"][1] > 0.99)

# 11) invalid polygon policy：自交多邊形 → buffer(0) deterministic repair；退化→area0→不配、不崩
def t_invalid_policy():
    bowtie = Rpoly("a", [[0, 0], [2, 2], [2, 0], [0, 2]])   # 自交蝴蝶結
    gt = [R(1, 0, 0, 2, 2)]
    try:
        m1, _ = match_rooms_iou([bowtie], gt)
        m2, _ = match_rooms_iou([bowtie], gt)   # deterministic：兩次相同
        ok = (m1 == m2)                          # 不崩、可重現（不論配到與否）
    except Exception as e:
        ok = False; print("     invalid-poly 崩:", e)
    # 退化（零面積線）→ area 0 → 不配、不崩
    degen = {"idx": "d", "poly": np.array([[0, 0], [1, 0], [2, 0]], float)}
    try:
        md, _ = match_rooms_iou([degen], gt); ok = ok and (len(md) == 0)
    except Exception:
        ok = False
    check("11 invalid/degenerate polygon deterministic policy (no silent crash)", ok)


for t in [t_perm_pred, t_perm_gt, t_hungarian_adversarial, t_exact_tie_permutation, t_iou_boundary,
          t_empty_gt, t_empty_pred, t_both_empty, t_door_perm,
          t_window_endpoint_contract, t_door_boundary, t_concave,
          t_invalid_policy]:
    t()
npass = sum(ok for _, ok in RESULTS)
print(f"\n{npass}/{len(RESULTS)} PASS")
sys.exit(0 if npass == len(RESULTS) else 1)
