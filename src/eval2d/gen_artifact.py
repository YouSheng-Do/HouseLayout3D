"""生成 current v3＋official MP3D floor-GT 的 2D eval HTML 報告。"""
import base64, json, os, pickle, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import prf

ROOT = "/home/ado/storage/HouseLayout3D"
BASE = f"{ROOT}/outputs/eval2d/baselines/watershed_v3_pre_report"
CANDIDATE = f"{ROOT}/outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1"
AB_DIR = f"{CANDIDATE}/eval_frozen_watershed_v3"
scores = pickle.load(open(f"{AB_DIR}/scores_house_official325.pkl", "rb"))
ab = json.load(open(f"{AB_DIR}/comparison.json"))


def agg(key, sub):
    t = [0, 0, 0]
    for S in scores:
        v = scores[S][key].get(sub, [0, 0, 0]); t = [t[i] + v[i] for i in range(3)]
    return prf(*t)

def f1(S, key, sub): return prf(*scores[S][key].get(sub, [0, 0, 0]))["f1"]

per = {S: f1(S, "A", "room") for S in scores}
order = sorted(per, key=per.get)
median_value = float(np.median(list(per.values())))
worst, med, best = order[0], min(per, key=lambda s: abs(per[s] - median_value)), order[-1]
alliou = [x for S in scores for x in scores[S]["iou"]]
good = sum(v >= 0.5 for v in per.values())

def b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
import glob as _g
_VIZ = f"{BASE}/eval2d_v3_strict_levels/visualizations"
_VIZ_TAG = {"best": "best", "median": "representative", "worst": "worst", "multi_level": "multi_level"}
def _vp(tag): return sorted(_g.glob(f"{_VIZ}/{_VIZ_TAG[tag]}_*.png"))[0]
imgs = {t: b64(_vp(t)) for t in ["best", "median", "worst", "multi_level"]}
_viz_scene = {t: os.path.basename(_vp(t))[len(t)+1:-4] for t in imgs}
_viz_scene["median"] = os.path.basename(_vp("median"))[len("representative")+1:-4]
error_img = b64(f"{_VIZ}/error_analysis_summary.png")
gt_overlap_img = b64(f"{BASE}/eval2d_v2_hungarian/gt_overlap_audit.png")
diagnostics = json.load(open(f"{_VIZ}/error_analysis.json"))
gt_audit = json.load(open(f"{BASE}/eval2d_v2_hungarian/gt_overlap_audit.json"))["summary"]
flag_count = {key: sum(row["flags"][key] for row in diagnostics["per_scene"])
              for key in diagnostics["rules"]}

def band(v):  # F1 → semantic class
    return "good" if v >= 0.5 else ("mid" if v >= 0.25 else "low")

# metric bar rows: (label, unit, f1, extra)
tierA = [("Room F1", "@IoU>0.5 · 幾何", agg("A","room")["f1"], f"P {agg('A','room')['p']:.2f} R {agg('A','room')['r']:.2f}"),
         ("mean Room IoU", "matched 房", np.mean(alliou), f"n={len(alliou)}"),
         ("Corner F1", "@0.1m（SLIBO 主指標）", agg("A","corner@0.1")["f1"],
          f"→0.2m {agg('A','corner@0.2')['f1']:.2f} →0.3m {agg('A','corner@0.3')['f1']:.2f}"),
         ("Angle F1", "@0.1m", agg("A","angle@0.1")["f1"], "角點＋內角<5°")]
tierB = [("Room+type F1", "房匹配＋房型對", agg("B","room+type")["f1"], "CLIP 房型"),
         ("Doors F1", "@0.5m", agg("B","doors@0.5")["f1"], f"@0.2m {agg('B','doors@0.2')['f1']:.2f}")]
tierC = [("edge_all F1", "room↔room ＋ room↔外", agg("C","all")["f1"], f"room-room {agg('C','room_room')['f1']:.2f}"),
         ("edge outside F1", "外門 candidate", agg("C","outside")["f1"], "非獨立 annotated truth")]

def bars(rows):
    h = '<div class="metrics">'
    for lab, unit, v, extra in rows:
        w = min(v, 1.0) * 100
        h += f'''<div class="metric"><div class="mh"><span class="mn">{lab}<span class="u">{unit}</span></span>
        <span class="pill {band(v)}">{v:.3f}</span></div>
        <div class="track"><div class="grid"><i style="left:25%"></i><i style="left:50%"></i><i style="left:75%"></i></div>
        <div class="fill {band(v)}" style="--w:{w:.0f}%"></div></div>
        <div class="mf"><span class="ex">{extra}</span><span class="sc">0 ——— 1.0</span></div></div>'''
    return h + '</div>'

rows_tbl = "".join(
    f'<tr><td>{S}</td><td>{per[S]:.2f}</td><td>{np.mean(scores[S]["iou"] or [0]):.2f}</td>'
    f'<td>{f1(S,"B","doors@0.5"):.2f}</td><td>{f1(S,"C","all"):.2f}</td></tr>'
    for S in sorted(scores, key=lambda s: -per[s]))

conf = Counter()
for S in scores: conf.update(scores[S]["confusion"])
conf_top = "".join(f"<li><span class='ct'>{g}</span> → <span class='cp'>{p}</span> <span class='cn'>×{n}</span></li>"
                   for (g, p), n in conf.most_common(5))

HTML = f'''<style>
:root{{--paper:#eef0f3;--card:#fbfbfc;--card2:#f5f6f8;--ink:#141a20;--ink2:#4c5661;--ink3:#828c96;
--line:#d9dde2;--line2:#e6e9ed;--accent:#0f6e7b;--soft:#0f6e7b1a;--good:#3f7a4b;--mid:#a9700f;--low:#a4392f;
--track:#e2e5e9;--sh:0 1px 2px rgba(20,26,32,.04),0 8px 24px rgba(20,26,32,.05)}}
@media(prefers-color-scheme:dark){{:root{{--paper:#0d1216;--card:#141a20;--card2:#171e25;--ink:#e9edf0;--ink2:#a7b0b9;
--ink3:#6f7982;--line:#242c34;--line2:#1d242b;--accent:#39b9c7;--soft:#39b9c722;--good:#71c081;--mid:#d7a648;
--low:#e07a70;--track:#1e262d;--sh:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35)}}}}
:root[data-theme="light"]{{--paper:#eef0f3;--card:#fbfbfc;--card2:#f5f6f8;--ink:#141a20;--ink2:#4c5661;--ink3:#828c96;
--line:#d9dde2;--line2:#e6e9ed;--accent:#0f6e7b;--soft:#0f6e7b1a;--good:#3f7a4b;--mid:#a9700f;--low:#a4392f;--track:#e2e5e9;--sh:0 1px 2px rgba(20,26,32,.04),0 8px 24px rgba(20,26,32,.05)}}
:root[data-theme="dark"]{{--paper:#0d1216;--card:#141a20;--card2:#171e25;--ink:#e9edf0;--ink2:#a7b0b9;--ink3:#6f7982;
--line:#242c34;--line2:#1d242b;--accent:#39b9c7;--soft:#39b9c722;--good:#71c081;--mid:#d7a648;--low:#e07a70;--track:#1e262d;--sh:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35)}}
*{{box-sizing:border-box}} html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.62}}
.wrap{{max-width:1080px;margin:0 auto;padding:clamp(20px,5vw,60px) clamp(16px,4vw,40px)}}
.mono{{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace}}
.eyebrow{{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);display:flex;gap:.6em;align-items:center}}
.eyebrow .dot{{width:5px;height:5px;border-radius:50%;background:var(--accent)}}
h1{{font-family:ui-monospace,"JetBrains Mono",Menlo,monospace;font-weight:650;font-size:clamp(1.8rem,4.3vw,2.8rem);line-height:1.06;letter-spacing:-.01em;margin:.5rem 0 .1rem;text-wrap:balance}}
.sub{{color:var(--ink2);font-size:1.05rem;max-width:66ch;margin:.4rem 0 0}}
.masthead{{border-bottom:1px solid var(--line);padding-bottom:26px}}
.glance{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:26px}}
.glance>div{{background:var(--card);padding:15px 17px}}
.glance .k{{font-family:ui-monospace,Menlo,monospace;font-size:1.55rem;font-weight:600;letter-spacing:-.02em;line-height:1}}
.glance .l{{color:var(--ink3);font-size:.76rem;margin-top:6px}}
.glance .k.g{{color:var(--good)}} .glance .k.w{{color:var(--mid)}} .glance .k.b{{color:var(--low)}}
section{{margin-top:50px}}
.sec{{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink3);display:flex;align-items:center;gap:.7em;margin-bottom:18px}}
.sec::before{{content:"§";color:var(--accent)}} .sec .r{{flex:1;height:1px;background:var(--line)}}
h2{{font-size:1.4rem;font-weight:640;letter-spacing:-.01em;margin:0 0 .3rem;text-wrap:balance}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(18px,3vw,26px);box-shadow:var(--sh)}}
.verdict{{display:grid;grid-template-columns:auto 1fr;gap:22px}} @media(max-width:620px){{.verdict{{grid-template-columns:1fr}}}}
.verdict .badge{{font-family:ui-monospace,Menlo,monospace;writing-mode:vertical-rl;text-orientation:mixed;letter-spacing:.22em;text-transform:uppercase;font-size:.72rem;color:var(--accent);border-left:2px solid var(--accent);padding-left:12px}}
@media(max-width:620px){{.verdict .badge{{writing-mode:horizontal-tb;border-left:0;border-top:2px solid var(--accent);padding:8px 0 0}}}}
.tiers{{display:flex;flex-direction:column;gap:.6rem;margin-top:.8rem;list-style:none;padding:0}}
.tiers li{{display:grid;grid-template-columns:auto auto 1fr;gap:.7em;align-items:baseline;color:var(--ink2)}}
.tiers .v{{font-family:ui-monospace,monospace;font-weight:700}}
.tiers .yes .m,.tiers .yes .v{{color:var(--good)}} .tiers .no .m,.tiers .no .v{{color:var(--low)}}
.tiers .m{{font-family:ui-monospace,monospace;font-weight:700;width:1.4em}}
.tiers b{{color:var(--ink)}}
.metrics{{display:grid;gap:2px;background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.metric{{background:var(--card);padding:15px clamp(15px,2.5vw,22px)}}
.mh{{display:flex;justify-content:space-between;align-items:baseline;gap:10px}}
.mn{{font-weight:600;font-size:.98rem}} .mn .u{{color:var(--ink3);font-weight:400;font-size:.8em;margin-left:.4em}}
.pill{{font-family:ui-monospace,monospace;font-size:.8rem;font-weight:600;padding:2px 10px;border-radius:100px}}
.pill.good{{color:var(--good);background:color-mix(in srgb,var(--good) 13%,transparent)}}
.pill.mid{{color:var(--mid);background:color-mix(in srgb,var(--mid) 13%,transparent)}}
.pill.low{{color:var(--low);background:color-mix(in srgb,var(--low) 13%,transparent)}}
.track{{position:relative;height:12px;background:var(--track);border-radius:100px;margin:12px 0 8px;overflow:hidden}}
.track .grid i{{position:absolute;top:0;bottom:0;width:1px;background:color-mix(in srgb,var(--ink) 8%,transparent)}}
.fill{{height:100%;border-radius:100px;width:var(--w);animation:g 1s cubic-bezier(.22,.72,.2,1) both}}
.fill.good{{background:linear-gradient(90deg,color-mix(in srgb,var(--good) 80%,#000),var(--good))}}
.fill.mid{{background:linear-gradient(90deg,color-mix(in srgb,var(--mid) 82%,#000),var(--mid))}}
.fill.low{{background:linear-gradient(90deg,color-mix(in srgb,var(--low) 82%,#000),var(--low))}}
@keyframes g{{from{{width:0}}to{{width:var(--w)}}}} @media(prefers-reduced-motion:reduce){{.fill{{animation:none}}}}
.mf{{display:flex;justify-content:space-between;font-family:ui-monospace,monospace;font-size:.78rem;color:var(--ink3)}}
.tier-h{{font-family:ui-monospace,monospace;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin:22px 0 8px}}
.viz{{display:flex;flex-direction:column;gap:22px}}
.vizcard{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--sh)}}
.vizcard .cap{{display:flex;justify-content:space-between;align-items:baseline;padding:13px 18px;border-bottom:1px solid var(--line2);font-family:ui-monospace,monospace;font-size:.82rem;flex-wrap:wrap;gap:8px}}
.vizcard .cap b{{font-size:.95rem}} .vizcard .cap .tag{{padding:2px 9px;border-radius:100px;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}}
.tag.good{{color:var(--good);background:color-mix(in srgb,var(--good) 13%,transparent)}}
.tag.mid{{color:var(--mid);background:color-mix(in srgb,var(--mid) 13%,transparent)}}
.tag.low{{color:var(--low);background:color-mix(in srgb,var(--low) 13%,transparent)}}
.vizcard img{{width:100%;display:block;background:#fff}}
.tbl-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;min-width:520px;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 14px;text-align:right;font-family:ui-monospace,monospace;font-size:.82rem;border-bottom:1px solid var(--line2)}}
th{{color:var(--ink3);font-weight:500;font-size:.7rem;letter-spacing:.03em;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left;color:var(--ink)}} tbody tr:hover{{background:var(--card2)}} tbody tr:last-child td{{border-bottom:0}}
.conf{{list-style:none;padding:0;margin:12px 0 0;display:flex;flex-direction:column;gap:6px;font-family:ui-monospace,monospace;font-size:.9rem}}
.conf .ct{{color:var(--ink)}} .conf .cp{{color:var(--low)}} .conf .cn{{color:var(--ink3)}}
.caveat{{background:var(--card2);border:1px solid var(--line);border-left:3px solid var(--mid);border-radius:8px;padding:14px 18px;margin-top:14px;font-size:.92rem;color:var(--ink2)}}
.caveat b{{color:var(--ink)}} .caveat .mono{{color:var(--accent);font-size:.88em}}
footer{{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);color:var(--ink3);font-size:.84rem}}
footer .mono{{color:var(--ink2)}}
p{{margin:.7rem 0}} strong{{font-weight:640}} .hl{{color:var(--accent);font-weight:600}}
</style>
<div class="wrap">
<header class="masthead">
<div class="eyebrow"><span class="dot"></span> 內部技術檢查點 · 2D Floorplan Evaluation · 2026-08-11</div>
<h1>導航級 floorplan 夠好了嗎？房級有初步訊號，邊界與連通還不行</h1>
<p class="sub">獨立於論文 3D 指標的新量測儀：把 watershed v3 的 frozen floorplan（Stage 4a 擠出前）對官方 MP3D <span class="mono">.house</span> floor polygons，以 strict-level Hungarian protocol 評估。回答一個 3D 指標答不了的問題：<strong>floorplan 精度夠不夠撐 attribute layer？</strong></p>
<div class="glance">
<div><div class="k g mono">{agg('A','room')['f1']:.2f}</div><div class="l">Room F1 @IoU&gt;0.5</div></div>
<div><div class="k g mono">{np.mean(alliou):.2f}</div><div class="l">Room IoU（僅 matched 房 · conditional）</div></div>
<div><div class="k b mono">{agg('A','corner@0.1')['f1']:.2f}</div><div class="l">Corner F1 @0.1m</div></div>
<div><div class="k w mono">{good}/16</div><div class="l">棟 Room F1 ≥ 0.5</div></div>
</div>
</header>

<section><div class="sec">評估結論<span class="r"></span></div>
<div class="card verdict"><div class="badge">Verdict</div><div>
<h2>粗粒度房級有初步訊號，精度／連通／房型不足</h2>
<p style="color:var(--ink2);margin:.4rem 0 0">約六成 GT rooms 能在 IoU&gt;0.5 下配對，成功配對者通常有合理範圍；這是可供後續研究驗證的初步訊號，<b>不代表目前已足以直接掛接下游 attributes</b>。精確角點、房間連通圖與房型分類仍不足。</p>
<ul class="tiers">
<li class="yes"><span class="m">~</span><span class="v">{agg('A','room')['f1']:.2f} / {np.mean(alliou):.2f}</span><span><b>粗粒度房級</b>：Room F1 / matched-房 conditional IoU — 有初步訊號；仍需 downstream validation</span></li>
<li class="no"><span class="m">✗</span><span class="v">{agg('A','corner@0.1')['f1']:.2f}</span><span><b>邊界精度</b>：Corner@10cm — 門寬/貼牆開向這類要精確牆線的屬性不可靠</span></li>
<li class="no"><span class="m">✗</span><span class="v">{agg('C','all')['f1']:.2f}</span><span><b>連通拓撲</b>：access-graph edge — 不足以直接當導航圖</span></li>
<li class="no"><span class="m">✗</span><span class="v">{agg('B','room+type')['f1']:.2f}</span><span><b>房型</b>：Room+type — CLIP 房型近乎失效</span></li>
</ul></div></div></section>

<section><div class="sec">三層指標（16 棟均值 · 帶狀＝0→1 F1）<span class="r"></span></div>
<div class="tier-h">Tier A — 幾何（可比對已發表工作）</div>{bars(tierA)}
<div class="tier-h">Tier B — 語意</div>{bars(tierB)}
<div class="tier-h">Tier C — 拓撲（連通；GT 為 derived，權重下調）</div>{bars(tierC)}
</section>

<section><div class="sec">視覺證據 · GT vs 預測 floorplan（含 room IDs/types/doors/derived edges）<span class="r"></span></div>
<div class="viz">
<div class="vizcard"><div class="cap"><b>最佳 · {best}</b><span class="mono">Room F1 {per[best]:.2f} · IoU {np.mean(scores[best]["iou"] or [0]):.2f}</span><span class="tag good">best</span></div><img alt="best floorplan GT vs pred" src="{imgs['best']}"></div>
<div class="vizcard"><div class="cap"><b>代表場景 · {med}</b><span class="mono">Room F1 {per[med]:.2f} · IoU {np.mean(scores[med]["iou"] or [0]):.2f}</span><span class="tag mid">near median</span></div><img alt="representative floorplan GT vs pred" src="{imgs['median']}"></div>
<div class="vizcard"><div class="cap"><b>最差 · {worst}</b><span class="mono">Room F1 {per[worst]:.2f} · IoU {np.mean(scores[worst]["iou"] or [0]):.2f}</span><span class="tag low">worst</span></div><img alt="worst floorplan GT vs pred" src="{imgs['worst']}"></div>
<div class="vizcard"><div class="cap"><b>多樓層 · {_viz_scene['multi_level']}</b><span class="mono">Room F1 {per[_viz_scene['multi_level']]:.2f} · IoU {np.mean(scores[_viz_scene['multi_level']]["iou"] or [0]):.2f}</span><span class="tag mid">multi-level</span></div><img alt="multi-level floorplan GT vs pred" src="{imgs['multi_level']}"></div>
</div>
<p class="mono" style="color:var(--ink3);font-size:.82rem;margin-top:12px">左＝官方 MP3D `.house` floor polygon GT · 右＝frozen 預測 · 紅段＝門 · 灰線＝推導 access 邊 · 標籤＝房 ID/type。多樓層圖把 unmatched predicted levels 明列；它是 failure evidence，不是成功證據。<b>PRELIMINARY · watershed_v3 local variant · connectivity derived · room types unreliable。</b></p>
</section>

<section><div class="sec">16-scene error analysis<span class="r"></span></div>
<div class="vizcard"><div class="cap"><b>分布、conditional metric 與 failure flags</b><span class="tag mid">diagnostic</span></div>
<img alt="16-scene floorplan error analysis" src="{error_img}"></div>
<div class="caveat"><b>Flags 彼此可重疊，且不是新 benchmark thresholds。</b>
Geometry good {flag_count['geometry_good']}/16；under-segmented {flag_count['under_segmented']}/16；
level mismatch/extra {flag_count['level_mismatch_or_extra']}/16；door/topology weak {flag_count['door_or_topology_weak']}/16。
HxpK 是重要反例：只配到少數 rooms，所以 Room F1 低，但那些少數 matched rooms 的 conditional IoU 很高。圖與旗標均由 current v3／official GT scores 產生。</div>
</section>

<section><div class="sec">逐棟分布（依 Room F1）<span class="r"></span></div>
<div class="tbl-wrap"><table><thead><tr><th>場景</th><th>Room F1</th><th>IoU</th><th>Doors@0.5</th><th>edge_all</th></tr></thead>
<tbody>{rows_tbl}</tbody></table></div>
<p style="color:var(--ink2);margin-top:12px">{good}/16 棟 Room F1 ≥ 0.5，中位 {np.median(list(per.values())):.2f}，範圍 {min(per.values()):.2f}（{worst}）–{max(per.values()):.2f}（{best}）。管線變異 ±6 Δ5 已知，逐棟差異須據此讀。</p>
</section>

<section><div class="sec">房型混淆（GT→pred, top 5）<span class="r"></span></div>
<div class="card"><ul class="conf">{conf_top}</ul>
<p style="color:var(--ink2);margin:14px 0 0">房型 F1 {agg('B','room+type')['f1']:.2f}：多數房被 CLIP 判成單一型，與 Stage 4a 觀察一致。房型需另做分類器，不是這次的 floorplan 品質問題。</p></div></section>

<section><div class="sec">誠實的方法學邊界<span class="r"></span></div>
<div class="caveat"><b>GT geometry 已改為官方 floor polygons。</b> Current GT 直接讀 MP3D `.house` 的 <span class="mono">R→S(F)→ordered V</span>，不再經 rasterization／morphology／RDP。<b>房級 IoU {np.mean(alliou):.2f} 仍是「僅成功 matched 房」的 conditional mean（非全房平均）</b>。</div>
<div class="caveat" style="border-left-color:var(--accent)"><b>Tier C 為 exploratory。</b> MP3D 這 16 棟 <span class="mono">#portals=0</span>，連通 GT 是 point-in-polygon 探針推導，不是 annotated graph。30-case geometry audit：10 room↔room candidates 與 distinct regions 一致；10 one-outside 中 9 個在 1.5m 內沒再碰到 room、1 個於 0.40m 碰到 room（thick-gap candidate）；另有 9 same-region、1 overlapping-region ambiguity。所有類別都不是獨立 truth，因此不再宣稱「79% 可信／accuracy」。</div>
<div class="caveat" style="border-left-color:var(--low)"><b>Evaluator／GT A/B 已完成。</b>
Legacy v2＋raster GT 的 Room F1 <b>0.613</b>；strict v3 對相同 raster GT 為 <b>0.597</b>；strict v3＋官方 polygons 的共同 324-room subset 為 <b>0.603</b>；current explicit 325-room set 為 <b>0.602</b>。GT overlap 從 <b>{100*gt_audit['current_polygon_overlap_ratio']:.2f}%</b> 降至 <b>{100*gt_audit['house_floor_overlap_ratio']:.2f}%</b>。舊 baseline 未被覆寫。</div>
<div class="vizcard" style="margin-top:14px"><div class="cap"><b>GT geometry quality audit</b><span class="tag low">preliminary GT caveat</span></div>
<img alt="GT overlap audit" src="{gt_overlap_img}"></div>
<div class="caveat" style="border-left-color:var(--good)"><b>Validation。</b> 354/354 regions 各有一個 finite、valid、CCW floor polygon；current roomset 為 325 rooms／32 levels，292 doors 完整保留。論文的 317 rooms／33 levels 沒有公開 exact subset manifest，因此不宣稱 current roomset 等同論文 subset。</div>
</section>

<footer>
<p class="mono">evaluator＝eval2d_v3_strict_levels · GT＝mp3d_house_floor_v0_1 · predictions＝frozen watershed_v3_pre_report · scores＝`outputs/eval2d/gt_candidates/mp3d_house_floor_v0_1/eval_frozen_watershed_v3/scores_house_official325.pkl`</p>
<p>往導航級的優先補強（與 3D 結論同源，都指向 floorplan 精度）：①邊界正則化拉角點精度 ②直接門偵測拉連通 recall ③房型分類器。</p>
</footer>
</div>'''

out = f"{ROOT}/docs/eval2d_report.html"
open(out, "w").write(HTML)
print("寫出", out, f"({len(HTML)//1024}KB)")
