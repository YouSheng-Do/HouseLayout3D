"""v1 vs v2 vs 論文對照：從 per-scene log 解析 F1/Δτ，出三欄表＋逐棟熱圖。

不重算指標——直接讀 run 時印出的數字（與 aggregate_results.py 同一程式路徑）。
用法: python compare_v1_v2.py [--out-png outputs/mp3d_reports/_v1_v2_heatmap.png]
"""
import argparse, ast, re, os
import numpy as np

ROOT = "/home/ado/storage/HouseLayout3D"
SCENES = ["17DRP5sb8fy","1LXtFkjw3qL","2t7WUuJeko7","5LpN3gDmAk7","HxpKQynjfin",
          "JeFG25nYj2p","JmbYfDe2QKZ","S9hNv5qa7GM","TbHJrupSAjP","WYY7iVyf5p8",
          "YFuZgdQ5vWj","e9zR4mvMWw7","i5noydFURQK","jtcxE69GiFV","p5wJjkQkbXX",
          "r47D5H71a5s"]
PAPER = {"structures": 0.40, "doors": 0.55, "windows": 0.43, "stairs": 0.42,
         "d5": 61.1, "d10": 76.3}


def parse_scene_log(path, scene):
    """抓最後一組 eval 輸出（idempotent 重跑時取最新）。"""
    if not os.path.exists(path):
        return None
    r = {}
    for line in open(path, errors="ignore"):
        m = re.match(r"^(structures|doors|windows|stairs)\s+(\{.*\})\s*$", line)
        if m:
            try:
                r[m.group(1)] = ast.literal_eval(m.group(2))
            except Exception:
                pass
        m = re.search(rf"\[{scene}\] Δ5=([\d.]+)\s+Δ10=([\d.]+)", line)
        if m:
            r["d5"], r["d10"] = float(m.group(1)), float(m.group(2))
    return r if r else None


def collect(logdir):
    out = {}
    for s in SCENES:
        r = parse_scene_log(os.path.join(logdir, f"{s}.log"), s)
        if r:
            out[s] = r
    return out


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-png", default=f"{ROOT}/outputs/mp3d_reports/_v1_v2_heatmap.png")
    args = ap.parse_args()

    v1 = collect(f"{ROOT}/setup/logs/mp3d_run_v1")
    v2 = collect(f"{ROOT}/setup/logs/mp3d_run")

    # ── 逐棟表 ──────────────────────────────────────────────
    print(f"{'scene':>12s} | {'v1 Δ5':>6s} {'v2 Δ5':>6s} {'ΔΔ5':>6s} | "
          f"{'v1 Δ10':>6s} {'v2 Δ10':>6s} | {'v1 St':>5s} {'v2 St':>5s} | "
          f"{'v2 Dr':>5s} {'v2 Wn':>5s} {'v2 St.':>5s}")
    for s in SCENES:
        a, b = v1.get(s, {}), v2.get(s, {})
        d5a, d5b = a.get("d5"), b.get("d5")
        dd = f"{d5b - d5a:+.1f}" if (d5a is not None and d5b is not None) else "—"
        f1 = lambda r, k: (r.get(k) or {}).get("f1@0.5")
        print(f"{s:>12s} | {fmt(d5a,1):>6s} {fmt(d5b,1):>6s} {dd:>6s} | "
              f"{fmt(a.get('d10'),1):>6s} {fmt(b.get('d10'),1):>6s} | "
              f"{fmt(f1(a,'structures')):>5s} {fmt(f1(b,'structures')):>5s} | "
              f"{fmt(f1(b,'doors')):>5s} {fmt(f1(b,'windows')):>5s} {fmt(f1(b,'stairs')):>5s}")

    # ── mean±std 三欄對照 ───────────────────────────────────
    def agg(res, key, sub=None):
        vals = []
        for s in SCENES:
            r = res.get(s, {})
            v = (r.get(key) or {}).get(sub) if sub else r.get(key)
            if v is not None:
                vals.append(v)
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (None, None, 0)

    print("\n===== mean±std：v1 vs v2 vs 論文 Table 2/3 =====")
    rows = [("structures F1", "structures", "f1@0.5", PAPER["structures"]),
            ("doors F1", "doors", "f1@0.5", PAPER["doors"]),
            ("windows F1", "windows", "f1@0.5", PAPER["windows"]),
            ("stairs F1", "stairs", "f1@0.5", PAPER["stairs"]),
            ("Δ5", "d5", None, PAPER["d5"]),
            ("Δ10", "d10", None, PAPER["d10"])]
    for name, key, sub, paper in rows:
        m1, s1, n1 = agg(v1, key, sub)
        m2, s2, n2 = agg(v2, key, sub)
        nd = 1 if key in ("d5", "d10") else 3
        c1 = f"{m1:.{nd}f}±{s1:.{nd}f} (n={n1})" if m1 is not None else "—"
        c2 = f"{m2:.{nd}f}±{s2:.{nd}f} (n={n2})" if m2 is not None else "—"
        print(f"{name:>14s} | v1 {c1:>22s} | v2 {c2:>22s} | 論文 {paper}")

    # ── 熱圖 ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(SCENES, key=lambda s: -(v2.get(s, {}).get("d5") or 0))
    short = [s[:6] for s in order]
    metrics = ["Δ5 v1", "Δ5 v2", "Δ10 v1", "Δ10 v2",
               "St.F1 v1×100", "St.F1 v2×100", "Wn.F1 v2×100", "Dr.F1 v2×100"]
    f1v = lambda r, k: ((r.get(k) or {}).get("f1@0.5") or 0) * 100
    M = np.array([[v1.get(s, {}).get("d5") or 0 for s in order],
                  [v2.get(s, {}).get("d5") or 0 for s in order],
                  [v1.get(s, {}).get("d10") or 0 for s in order],
                  [v2.get(s, {}).get("d10") or 0 for s in order],
                  [f1v(v1.get(s, {}), "structures") for s in order],
                  [f1v(v2.get(s, {}), "structures") for s in order],
                  [f1v(v2.get(s, {}), "windows") for s in order],
                  [f1v(v2.get(s, {}), "doors") for s in order]])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(len(order)), short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(metrics)), metrics, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title("MULTIFLOOR3D repro v1 -> v2 (adaptive K + tight fit tol + crash fixes)"
                 "  |  paper: D5=61 D10=76 St.F1=40", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"\n[heatmap] {args.out_png}")


if __name__ == "__main__":
    main()
