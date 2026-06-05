"""
05_scorecard.py — CLIF ICU Ventilator-QI bundle scorecard (landing page).

Builds output/dashboard/scorecard.html: a glanceable, per-unit / per-ISO-week scorecard of
ventilator QI bundle metrics. The scorecard is a *combiner* — it is REGISTRY-DRIVEN:

  * The LPV tile is built in-memory here as a v1 "tile feed" (the same schema every other
    metric vertical emits), from this project's own rollups.
  * Other metrics (ARDS proning, SAT, SBT, mobilization) are their OWN pipelines that emit a
    small, PHI-free `tile_feed_<metric>.json`. Their paths are listed in config.json under
    `scorecard_tiles`; this script reads each, validates it, copies its detail dashboard into
    the bundle, and renders it through ONE shared tile component.
  * A metric with no feed shows a styled "Coming soon..." placeholder, so the scorecard
    always builds even if a sibling project hasn't run.

Contract: plans/02_scorecard_tile_contract.md (authoritative for the feed schema + grain fallback).

LPV tile headline = tidal-volume adherence at <= 8 mL/kg PBW (a realistic QI target), with a
3-segment mini-indicator: Plateau <= 30 · Driving pressure <= 15 · Vt <= 8 in SEVERE respiratory
failure. Lightweight (inline SVG donut + sparkline, no Plotly).

Inputs:  output/02_patient_day_status.parquet, 02_intervals.parquet, 02d_severity.parquet
         + each feed in config `scorecard_tiles` (e.g. ../proning/output/final/tile_feed_proning.json)
Output:  output/dashboard/scorecard.html   (+ copies of each tile's detail dashboard into output/dashboard/)

Run:
    .venv/bin/python code/05_scorecard.py
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]              # bundle root (scorecard/ is one level down)
CFG = json.loads((ROOT / "config.json").read_text())
SITE = CFG.get("site", "Your Site")
DASH_DIR = ROOT / "output" / "dashboard"   # shared shippable bundle (scorecard + per-metric drill-downs)
FEEDS_DIR = ROOT / "feeds"                 # PHI-free feed collection (the per-site consortium submission set)
DASH_DIR.mkdir(parents=True, exist_ok=True)
FEEDS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Named parameters ----
TILE_ORDER = ["lpv", "sat", "sbt", "proning", "mob"]   # slot order; a slot with no feed -> placeholder
PLACEHOLDER_META = {
    "sat":     {"icon": "sat",   "title": "SAT Completion", "subtitle": "Spontaneous Awakening Trials"},
    "sbt":     {"icon": "sbt",   "title": "SBT Completion", "subtitle": "Spontaneous Breathing Trials"},
    "proning": {"icon": "prone", "title": "ARDS Proning",   "subtitle": "Eligible ARDS patients"},
    "mob":     {"icon": "mob",   "title": "Mobilization",   "subtitle": "Target mobilization achieved"},
}
UNIT_ORDER_REST = ["medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
                   "mixed_neuro_icu", "general_icu", "burn_icu"]
UNIT_LABEL = {"__ALL__": "All ICUs", "medical_icu": "Medical ICU",
              "mixed_cardiothoracic_icu": "Cardiothoracic ICU", "surgical_icu": "Surgical ICU",
              "mixed_neuro_icu": "Neuro ICU", "general_icu": "General ICU", "burn_icu": "Burn ICU"}

# Which metrics to load, in slot order. Each is its OWN vertical that emits a v1 feed at
# metrics/<id>/output/final/tile_feed_<id>.json (LPV included -- full symmetry). The combiner only
# collects + renders; it computes nothing metric-specific itself.
METRICS_ENABLED = CFG.get("metrics", TILE_ORDER)

# ----------------------------------------------------------------------------
# 1. Collect each enabled metric's tile feed (stage feed -> feeds/, detail -> dashboard/)
# ----------------------------------------------------------------------------

print("[1] Collecting metric tile feeds ...")
REQUIRED_FEED_KEYS = {"schema_version", "metric_id", "title", "grain", "headline"}


def load_feed(fp):
    # Read + validate one tile_feed_<id>.json (schema_version 1, required keys, PHI-free).
    if not fp.exists():
        return None
    try:
        feed = json.loads(fp.read_text())
    except Exception as e:
        print(f"  [skip] feed unreadable ({e}): {fp}")
        return None
    if feed.get("schema_version") != 1:
        print(f"  [skip] unsupported schema_version={feed.get('schema_version')}: {fp}")
        return None
    missing = REQUIRED_FEED_KEYS - set(feed)
    if missing:
        print(f"  [skip] feed missing keys {missing}: {fp}")
        return None
    assert "hospitalization_id" not in json.dumps(feed) and "patient_id" not in json.dumps(feed), \
        f"PHI substring in feed {fp}"
    return feed


feeds = []
for mid in METRICS_ENABLED:
    fp = ROOT / "metrics" / mid / "output" / "final" / f"tile_feed_{mid}.json"
    feed = load_feed(fp)
    if feed is None:
        print(f"  [skip] {mid}: no feed at {fp} -> placeholder")
        continue
    shutil.copyfile(fp, FEEDS_DIR / f"tile_feed_{mid}.json")   # stage for consortium submission
    href = feed.get("detail_href")
    if href and (fp.parent / href).exists():
        shutil.copyfile(fp.parent / href, DASH_DIR / href)
        print(f"  [feed] {mid}: loaded; copied '{href}' into dashboard/")
    elif href:
        print(f"  [feed] {mid}: loaded; detail '{href}' missing -> dropping link")
        feed["detail_href"] = None
    else:
        print(f"  [feed] {mid}: loaded (no detail link)")
    feeds.append(feed)

feeds_by_id = {f["metric_id"]: f for f in feeds}
print(f"  feeds ready: {[f['metric_id'] for f in feeds]}")

# ----------------------------------------------------------------------------
# 2. Global UI selectors (weeks / months / units). The finest-grained feed (LPV) carries a 'ui'
#    block; otherwise derive from the union of feed cell keys.
# ----------------------------------------------------------------------------

def _derive_ui(feeds):
    wk, mo, seen = set(), set(), []
    for f in feeds:
        for u, per in f.get("headline", {}).get("cells", {}).items():
            if u != "__ALL__" and u not in seen:
                seen.append(u)
            for pk in per:
                if re.fullmatch(r"\d{4}-W\d{2}", pk):
                    wk.add(pk)
                elif re.fullmatch(r"\d{4}-\d{2}", pk):
                    mo.add(pk)
    weeks = sorted(wk)
    months = sorted(mo)
    wl = {w: f"Week {w[-2:].lstrip('0')} · {datetime.strptime(w + '-1', '%G-W%V-%u').strftime('%b %Y')}"
          for w in weeks}
    ml = {m: datetime.strptime(m + "-01", "%Y-%m-%d").strftime("%b %Y") for m in months}
    units = ["__ALL__"] + [u for u in UNIT_ORDER_REST if u in seen]
    return {"weeks": weeks, "week_label": wl, "months": months, "month_label": ml, "units": units}


_ui = (feeds_by_id.get("lpv") or {}).get("ui") or _derive_ui(feeds)
weeks = _ui["weeks"]
week_label = _ui["week_label"]
months = _ui["months"]
month_label = _ui["month_label"]
units = _ui["units"]

# Map each ISO week -> its containing calendar month (the week's Thursday, ISO canonical),
# so a month-only feed (e.g. proning) can resolve a week pick to its month instead of all-time.
week_month = {w: datetime.strptime(w + "-4", "%G-W%V-%u").strftime("%Y-%m") for w in weeks}

# ----------------------------------------------------------------------------
# 3. Tile illustrations / icons (downscale + base64-embed; SVG fallback) + brand logo
# ----------------------------------------------------------------------------

IMG_DIR = ROOT / "assets"
_IMG_FILE = {"lpv": "LPV", "sat": "SAT", "sbt": "SBT", "prone": "Proning", "mob": "Mobilization"}


def _load_tile_img(stem, px=180):
    p = IMG_DIR / f"{stem}.png"
    if not p.exists():
        return None
    try:
        from PIL import Image
        im = Image.open(p).convert("RGBA")
        im.thumbnail((px, px))
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


TILE_IMG = {k: _load_tile_img(v) for k, v in _IMG_FILE.items()}
print(f"[img] tile illustrations embedded: {sum(1 for v in TILE_IMG.values() if v)}/{len(TILE_IMG)}")

LOGO_IMG = _load_tile_img("clif_logo_v2", px=480)
print(f"[img] brand logo embedded: {'yes' if LOGO_IMG else 'no (using SVG fallback)'}")

HEART = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round"><path d="M2 12h4l2-6 4 12 3-9 2 3h5"/></svg>')
ICONS = {
    "lungs": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4v8"/><path d="M9 9c0 4-2 5-2 8a2 2 0 0 0 4 0V9a3 3 0 0 0-2 0Z"/><path d="M15 9c0 4 2 5 2 8a2 2 0 0 1-4 0V9a3 3 0 0 1 2 0Z"/></svg>',
    "lpv": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4v8"/><path d="M9 9c0 4-2 5-2 8a2 2 0 0 0 4 0V9a3 3 0 0 0-2 0Z"/><path d="M15 9c0 4 2 5 2 8a2 2 0 0 1-4 0V9a3 3 0 0 1 2 0Z"/></svg>',
    "sat": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="9" r="4"/><path d="M5 21c1-4 4-6 7-6s6 2 7 6"/><path d="M19 4l1.5 1.5M21 7h-2"/></svg>',
    "sbt": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 12c2-3 4-3 6 0s4 3 6 0 4-3 4-3"/><path d="M4 17h16"/></svg>',
    "prone": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 16h18"/><circle cx="7" cy="12" r="2"/><path d="M9 14h9a2 2 0 0 1 0 4H5"/></svg>',
    "mob": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="13" cy="4" r="2"/><path d="M13 7l-2 5 3 2 1 5M11 12l-4 1M14 9l3 2"/></svg>',
}
# Final icon HTML per key (image if supplied, else inline SVG) — handed to the JS renderer.
ICON_HTML = {k: (f'<img class="ico" src="{TILE_IMG[k]}" alt="">' if TILE_IMG.get(k) else ICONS.get(k, ""))
             for k in set(list(TILE_IMG) + list(ICONS))}

# ----------------------------------------------------------------------------
# 4. Assemble payload
# ----------------------------------------------------------------------------

payload = {
    "site": SITE,
    "feeds": feeds,
    "order": TILE_ORDER,
    "placeholders": PLACEHOLDER_META,
    "iconHtml": ICON_HTML,
    "weeks": weeks, "week_label": week_label, "months": months, "month_label": month_label,
    "week_month": week_month,
    "units": units, "unit_label": UNIT_LABEL,
    "generated": datetime.now().isoformat(timespec="minutes"),
}

# ----------------------------------------------------------------------------
# 5. HTML (CLIF maroon/cream house style; inline SVG donut + sparkline)
# ----------------------------------------------------------------------------

print("[5] Writing dashboard/scorecard.html ...")
latest = weeks[-1]
week_opts = "".join(f'<option value="{w}">{html.escape(week_label[w])}</option>' for w in reversed(weeks))
month_opts = "".join(f'<option value="{m}">{html.escape(month_label[m])}</option>' for m in reversed(months))
unit_sel = "".join(f'<option value="{u}">{html.escape(UNIT_LABEL.get(u, u))}</option>' for u in units)

CSS = """
:root{--maroon:#8a1f2b;--maroon-d:#6f1622;--cream:#f6efe9;--card:#fffdfb;--ink:#3a2c2c;
--muted:#9a8c86;--line:#ece1d9;--good:#2f7d5b;--warn:#b5852a;--bad:#a23b3b;--bar:#efe4dc;}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,-apple-system,'Segoe UI',system-ui,sans-serif;background:var(--cream);
color:var(--ink);}
.wrap{max-width:1500px;margin:0 auto;padding:22px 28px 40px;}
header.top{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:6px;}
.brand{display:flex;align-items:center;gap:9px;font-weight:800;font-size:28px;color:var(--maroon);letter-spacing:.5px;}
.brand img{height:72px;width:auto;display:block;}
.brand svg{width:34px;height:34px}
h1{font-size:23px;font-weight:700;color:var(--maroon-d);margin:0;letter-spacing:.4px;text-transform:uppercase;}
.chips{display:flex;gap:11px;margin-left:auto;align-items:center;flex-wrap:wrap;}
.chip{display:flex;align-items:center;gap:8px;background:var(--card);border:1px solid var(--line);
border-radius:999px;padding:9px 16px;font-size:15px;box-shadow:0 1px 2px rgba(120,30,40,.05);}
.chip b{color:var(--maroon);text-transform:uppercase;font-size:12.5px;letter-spacing:.6px;}
.chip select{border:0;background:transparent;font-size:15px;color:var(--ink);font-weight:600;cursor:pointer;outline:none;}
.subtitle{color:var(--muted);font-size:13px;margin:2px 0 22px;}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:18px;}
@media(max-width:1200px){.grid{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px 18px;
box-shadow:0 3px 10px rgba(120,30,40,.06);display:flex;flex-direction:column;align-items:center;text-align:center;
min-height:370px;position:relative;}
.card.lpv{border-color:#e7c9cd;}
.card.ph{opacity:.62;}
.card .ico{width:40px;height:40px;color:var(--maroon);margin-bottom:6px;}
/* All tile illustrations the same size, above the title. */
.card img.ico{width:108px;height:108px;object-fit:contain;margin:2px 0 6px;}
/* Fixed-height header zone (title + subtitle) so every donut starts at the same y,
   regardless of how many lines the title/subtitle wrap to. Titles/subtitles clamp to 2 lines. */
.card .cardhead{display:flex;flex-direction:column;align-items:center;justify-content:center;
width:100%;height:80px;margin:0 0 6px;}
.card .mname{font-weight:800;font-size:18px;color:var(--maroon-d);letter-spacing:.3px;line-height:1.18;
display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden;}
.card .msub{font-size:11.5px;color:var(--muted);margin:3px 0 0;line-height:1.3;
display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden;}
.badge{display:inline-block;margin-left:6px;font-size:10px;color:var(--maroon);background:#f3e3e5;
border-radius:999px;padding:1px 7px;font-weight:700;letter-spacing:.3px;vertical-align:middle;}
.donut{position:relative;width:128px;height:128px;margin:2px 0 6px;}
.donut svg{width:128px;height:128px;}
.donut .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.donut .pct{font-size:30px;font-weight:800;color:var(--maroon);font-variant-numeric:tabular-nums;line-height:1;}
.donut .plab{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:3px;}
.denom{font-size:12px;color:var(--muted);margin:2px 0 2px;}
.denomsub{font-size:10.5px;color:var(--muted);margin:0 0 8px;}
.goalwrap{width:100%;margin:6px 0 12px;}
.goalbar{height:8px;border-radius:999px;background:var(--bar);overflow:hidden;}
.goalbar>span{display:block;height:100%;background:var(--maroon);border-radius:999px;}
.goaltxt{font-size:10.5px;color:var(--muted);margin-top:4px;display:flex;justify-content:space-between;}
.segs{width:100%;display:flex;flex-direction:column;gap:7px;margin-top:auto;}
.seg{display:flex;align-items:center;gap:8px;font-size:11.5px;}
.seg .sl{flex:0 0 96px;text-align:left;color:var(--ink);}
.seg .sb{flex:1;height:7px;border-radius:999px;background:var(--bar);overflow:hidden;}
.seg .sb>span{display:block;height:100%;border-radius:999px;}
.seg .sv{flex:0 0 38px;text-align:right;font-weight:700;font-variant-numeric:tabular-nums;}
.spark{width:100%;height:34px;margin-top:10px;}
.tilenote{font-size:10px;color:var(--muted);line-height:1.35;margin-top:10px;text-align:left;}
ul.tilenote{margin:10px 0 0;padding-left:14px;}
ul.tilenote li{margin:2.5px 0;}
.bignum{font-size:34px;font-weight:800;color:#b9a59c;font-variant-numeric:tabular-nums;margin:24px 0 4px;}
.phnote{font-size:11px;color:var(--muted);}
.cardlink{position:absolute;bottom:12px;right:14px;font-size:11.5px;color:var(--maroon);text-decoration:none;font-weight:700;}
.cardlink:hover{text-decoration:underline;}
a.cardwrap{text-decoration:none;color:inherit;}
footer{margin-top:26px;color:var(--muted);font-size:11.5px;text-align:center;}
"""

BODY = f"""
<div class="wrap">
<header class="top">
  <div class="brand">{f'<img src="{LOGO_IMG}" alt="CLIF">' if LOGO_IMG else HEART + '<span>CLIF</span>'}</div>
  <h1>ICU Ventilator QI Dashboard — {html.escape(SITE)}</h1>
  <div class="chips">
    <div class="chip"><b>Unit</b><select id="sel-unit">{unit_sel}</select></div>
    <div class="chip"><b>Month</b><select id="sel-month"><option value="all">All time</option><option value="__wk__" hidden>—</option>{month_opts}</select></div>
    <div class="chip"><b>Week</b><select id="sel-week"><option value="all">All</option>{week_opts}</select></div>
  </div>
</header>
<p class="subtitle" id="subtitle"></p>

<div class="grid" id="tiles"></div>

<footer>Data through {{LASTWK}} · Generated {{GEN}} · CLIF (Common Longitudinal ICU data Format) · {html.escape(SITE)}</footer>
</div>
"""

APP_JS = r"""
const P = JSON.parse(document.getElementById('sc-payload').textContent);
const C = 2*Math.PI*56;
const PCT = v => (v==null||isNaN(v)) ? '—' : (v*100).toFixed(0)+'%';
const segColor = r => r==null?'#ccc' : r>=0.8?'#2f7d5b' : r>=0.5?'#b5852a':'#a23b3b';
const esc = s => (s==null?'':String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const rate = c => (c && c.den) ? c.num/c.den : null;
// Note may be a plain string or a '•'-delimited string -> render the latter as bullets.
function noteHtml(n){
  if(!n) return '';
  const parts = String(n).split('•').map(s=>s.trim()).filter(Boolean);
  return parts.length>1
    ? '<ul class="tilenote">'+parts.map(t=>`<li>${esc(t)}</li>`).join('')+'</ul>'
    : `<div class="tilenote">${esc(n)}</div>`;
}

const feedById = {}; P.feeds.forEach(f => feedById[f.metric_id] = f);
let unit = P.units.includes('__ALL__') ? '__ALL__' : P.units[0];
let pType = 'all', pKey = P.weeks[P.weeks.length-1];   // default: all-time

function curPeriodKey(){ return pType==='all' ? 'all' : pKey; }
function periodLabel(){ return pType==='all' ? 'all time' : (pType==='month'?P.month_label[pKey]:P.week_label[pKey]); }

// Grain fallback (contract §4): resolve the (unit, period) this feed can actually answer,
// and a badge so a coarse feed's number is never silently mislabeled.
function resolve(grain){
  let u = unit, ub = '';
  if(unit!=='__ALL__' && !grain.units.includes(unit)){ u='__ALL__'; ub=' · site-wide'; }
  let pk, pb='';
  if(grain.periods.includes(pType)){ pk = curPeriodKey(); }
  else if(pType==='week' && grain.periods.includes('month') && P.week_month[pKey]){
    // week pick, feed has no weekly grain (e.g. proning): show the week's CONTAINING month
    // rather than freezing on all-time — never a noisy <10 single-week denominator.
    const mk = P.week_month[pKey];
    pk = mk; pb = ' · ' + (P.month_label[mk]||mk) + ' · month';
  }
  else { pk='all'; if(curPeriodKey()!=='all') pb=' · all-time'; }
  return {u, pk, badge: ub+pb};
}

function sparkSVG(xs, ys, hiKey){
  const W=240,H=34,pad=2;
  const valid = ys.map((y,i)=>[i,y]).filter(p=>p[1]!=null);
  const X = i => pad + i/Math.max(1,xs.length-1)*(W-2*pad), Y = v => H-pad-v*(H-2*pad);
  const path = valid.map((p,j)=>(j?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  const hi = hiKey!=null ? xs.indexOf(hiKey) : -1;
  const dot = (hi>=0 && ys[hi]!=null) ? `<circle cx="${X(hi).toFixed(1)}" cy="${Y(ys[hi]).toFixed(1)}" r="3" fill="#8a1f2b"/>` : '';
  return `<svg class="spark" viewBox="0 0 240 34" preserveAspectRatio="none"><path d="${path}" fill="none" stroke="#c98a92" stroke-width="1.6"/>${dot}</svg>`;
}

function tileCard(feed){
  const g = feed.grain, R = resolve(g);
  const hc = (feed.headline.cells[R.u]||{})[R.pk] || null;
  const r = rate(hc);
  const fine = g.periods.includes('week') || g.periods.includes('month');

  const donut = `<div class="donut"><svg viewBox="0 0 128 128">`
    + `<circle cx="64" cy="64" r="56" fill="none" stroke="var(--bar)" stroke-width="13"/>`
    + `<circle cx="64" cy="64" r="56" fill="none" stroke="var(--maroon)" stroke-width="13" stroke-linecap="round" `
    + `transform="rotate(-90 64 64)" stroke-dasharray="${(r||0)*C} ${C}"/></svg>`
    + `<div class="val"><span class="pct">${PCT(r)}</span><span class="plab">${esc(feed.headline.label||'')}</span></div></div>`;

  let denom = '';
  if(hc){
    denom = (hc.hrs!=null)
      ? `${hc.hrs.toLocaleString()} patient-hours · ${hc.n.toLocaleString()} patient-days`
      : `${(hc.n!=null?hc.n:hc.den).toLocaleString()} ${esc(feed.headline.n_unit||'')}`;
  }
  const denomsub = feed.headline.den_label ? `<div class="denomsub">${esc(feed.headline.den_label)}</div>` : '';

  let goal = '';
  if(feed.goal!=null){
    const fill = Math.min(100,(r||0)/feed.goal*100);
    goal = `<div class="goalwrap"><div class="goalbar"><span style="width:${fill}%"></span></div>`
      + `<div class="goaltxt"><span>${r==null?'—':PCT(r)+' '+esc(feed.headline.den_label||'')}</span>`
      + `<span>Goal ≥ ${(feed.goal*100).toFixed(0)}%</span></div></div>`;
  }

  let segs = '';
  if(feed.segments && feed.segments.length){
    segs = '<div class="segs">' + feed.segments.slice(0,3).map(s=>{
      const sc = (s.cells[R.u]||{})[R.pk] || null, sv = rate(sc);
      return `<div class="seg"><span class="sl">${esc(s.label)}</span>`
        + `<span class="sb"><span style="width:${(sv||0)*100}%;background:${segColor(sv)}"></span></span>`
        + `<span class="sv">${PCT(sv)}</span></div>`;
    }).join('') + '</div>';
  }

  let spark = '';
  if(fine){
    const hasWk = g.periods.includes('week');
    // month axis when a month is in play (month pick, or a week pick resolved to its month on a
    // weekless feed like proning); otherwise the feed's finest axis (week if it has it, else month).
    const onMonthAxis = (pType==='month') || (pType==='week' && !hasWk);
    const xs = onMonthAxis ? P.months : (hasWk ? P.weeks : P.months);
    const ys = xs.map(b => rate((feed.headline.cells[R.u]||{})[b]));
    const hi = (pType==='all') ? null : (onMonthAxis ? R.pk : pKey);
    spark = sparkSVG(xs, ys, hi);
  }

  const note  = noteHtml(feed.note);
  const badge = R.badge ? `<span class="badge">${esc(R.badge.trim())}</span>` : '';
  const link  = feed.detail_href ? `<span class="cardlink">View details →</span>` : '';
  const inner = `${P.iconHtml[feed.icon]||''}<div class="cardhead"><div class="mname">${esc(feed.title)}</div>`
    + `<div class="msub">${esc(feed.subtitle||'')}${badge}</div></div>${donut}`
    + `<div class="denom">${denom}</div>${denomsub}${goal}${segs}${spark}${note}${link}`;
  const cls = 'card' + (feed.metric_id==='lpv' ? ' lpv' : '');
  return feed.detail_href
    ? `<a class="cardwrap" href="${esc(feed.detail_href)}"><div class="${cls}">${inner}</div></a>`
    : `<div class="${cls}">${inner}</div>`;
}

function placeholderCard(id){
  const m = P.placeholders[id] || {icon:id, title:id, subtitle:''};
  return `<div class="card ph">${P.iconHtml[m.icon]||''}<div class="cardhead"><div class="mname">${esc(m.title)}</div>`
    + `<div class="msub">${esc(m.subtitle||'')}</div></div><div class="bignum">—</div>`
    + `<div class="phnote">Coming soon...</div></div>`;
}

function render(){
  document.getElementById('tiles').innerHTML =
    P.order.map(id => feedById[id] ? tileCard(feedById[id]) : placeholderCard(id)).join('');
  document.getElementById('subtitle').textContent =
    (P.unit_label[unit]||unit)+' · '+periodLabel()+' — ICU ventilator/liberation QI bundle';
}

const selU=document.getElementById('sel-unit'), selM=document.getElementById('sel-month'), selW=document.getElementById('sel-week');
// Month + Week are mutually exclusive. The inactive chip shows a neutral label: Week shows 'All',
// Month shows 'All time' — except while a Week is active, Month shows '—' (the hidden __wk__ option)
// so it never reads as a contradictory 'All time'. Default state is all-time.
selU.value=unit; selW.value='all'; selM.value='all';
selU.onchange = e=>{ unit=e.target.value; render(); };
selM.onchange = e=>{ if(e.target.value!=='all'){ pType='month'; pKey=e.target.value; selW.value='all'; }
                     else { pType='all'; pKey=null; selW.value='all'; } render(); };
selW.onchange = e=>{ if(e.target.value!=='all'){ pType='week'; pKey=e.target.value; selM.value='__wk__'; }
                     else { pType='all'; pKey=null; selM.value='all'; } render(); };
render();
"""

HTML = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    f"<title>CLIF ICU Ventilator QI — {html.escape(SITE)}</title>"
    "<style>@@CSS@@</style></head><body>"
    + BODY.replace("{LASTWK}", html.escape(week_label[latest])).replace("{GEN}", payload["generated"])
    + '<script id="sc-payload" type="application/json">@@PAYLOAD@@</script>'
    + "<script>@@JS@@</script></body></html>"
)
HTML = (HTML.replace("@@CSS@@", CSS)
        .replace("@@PAYLOAD@@", json.dumps(payload, allow_nan=False))
        .replace("@@JS@@", APP_JS))

out_path = DASH_DIR / "scorecard.html"
out_path.write_text(HTML)
print(f"  wrote {out_path}  ({out_path.stat().st_size/1e3:.0f} KB)")

# ----------------------------------------------------------------------------
# 6. Verification
# ----------------------------------------------------------------------------

print("\n[verify]")


def feed_rate(feed, key, u="__ALL__", pk="all"):
    if key == "vt8":
        c = feed["headline"]["cells"][u][pk]
    else:
        c = next(s for s in feed["segments"] if s["key"] == key)["cells"][u][pk]
    return c["num"] / c["den"] if c["den"] else float("nan")


payload_dump = json.dumps(payload)
checks = {
    "tile order has 5 slots": len(TILE_ORDER) == 5,
    "scorecard.html written into dashboard/": out_path.exists() and out_path.parent.name == "dashboard",
    "at least one real feed loaded": len(feeds) >= 1,
    "no hospitalization_id in payload": "hospitalization_id" not in payload_dump,
    "no patient_id in payload": "patient_id" not in payload_dump,
}
for f in feeds:
    checks[f"{f['metric_id']}: detail link resolves"] = (
        f.get("detail_href") is None or (DASH_DIR / f["detail_href"]).exists())
if "lpv" in feeds_by_id:
    lf = feeds_by_id["lpv"]
    checks["lpv Vt headline ~ 83%"] = 0.78 < feed_rate(lf, "vt8") < 0.88
    checks["lpv Plateau ~ 85.8%"] = 0.83 < feed_rate(lf, "plat") < 0.89
    checks["lpv dP ~ 48%"] = 0.44 < feed_rate(lf, "dp") < 0.52
if "proning" in feeds_by_id:
    hc = feeds_by_id["proning"]["headline"]["cells"]["__ALL__"]["all"]
    checks["proning feed: 0 <= num <= den, den>0"] = 0 <= hc["num"] <= hc["den"] and hc["den"] > 0
    print(f"  proning headline: {hc['num']}/{hc['den']} = {hc['num'] / hc['den'] * 100:.1f}% ever proned")

for k, v in checks.items():
    print(f"  [{'ok' if v else 'XX'}] {k}")
if "lpv" in feeds_by_id:
    lf = feeds_by_id["lpv"]
    print(f"\n  LPV all-units/all-time: Vt {feed_rate(lf, 'vt8') * 100:.1f}% · "
          f"Plateau {feed_rate(lf, 'plat') * 100:.1f}% · dP {feed_rate(lf, 'dp') * 100:.1f}% · "
          f"Vt-severe {feed_rate(lf, 'vt8sev') * 100:.1f}%")
_wk = f"weeks: {len(weeks)} ({weeks[0]} -> {weeks[-1]})" if weeks else "weeks: 0"
print(f"  feeds: {[f['metric_id'] for f in feeds]} · {_wk} · units: {len(units)}")
print(f"  bundle dir: {DASH_DIR}  contents: {sorted(p.name for p in DASH_DIR.glob('*.html'))}")
assert all(checks.values()), "VERIFICATION FAILED"
print("\nAll checks passed. Done.")
