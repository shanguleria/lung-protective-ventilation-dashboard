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
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]              # bundle root (scorecard/ is one level down)
CFG = json.loads((ROOT / "config.json").read_text())
SITE = CFG.get("site", "Your Site")
LPV_OUT = ROOT / "metrics" / "lpv" / "output"           # LPV pipeline parquets live here now
DASH_DIR = ROOT / "output" / "dashboard"                # shared shippable bundle (scorecard + drill-downs)
DASH_DIR.mkdir(parents=True, exist_ok=True)

# ---- Named parameters ----
SCORECARD_VT_CUTOFF = 8.0   # headline Vt/kg cutoff for the scorecard tile
LPV_GOAL = 0.90             # target line on the LPV tile
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60

# Tile slot order on the scorecard. A slot with a matching feed (built here or loaded from
# `scorecard_tiles`) renders a real tile; otherwise it falls back to a placeholder.
TILE_ORDER = ["lpv", "sat", "sbt", "proning", "mob"]
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

# ----------------------------------------------------------------------------
# 1. Load + per-(hosp, day) Vt<=8 recompute (status file is default-6)
# ----------------------------------------------------------------------------

print("[1] Loading + computing Vt<=8 per patient-day ...")
status = pd.read_parquet(LPV_OUT / "02_patient_day_status.parquet")
status["hospitalization_id"] = status["hospitalization_id"].astype(str)
status["calendar_day"] = pd.to_datetime(status["calendar_day"]).dt.date

iv = pd.read_parquet(LPV_OUT / "02_intervals.parquet")
iv["hospitalization_id"] = iv["hospitalization_id"].astype(str)
iv["calendar_day"] = pd.to_datetime(iv["calendar_day"]).dt.date
key = ["hospitalization_id", "calendar_day"]
gk = [iv["hospitalization_id"], iv["calendar_day"]]
vt_present = iv["vt_per_pbw"].notna()
vt_assess = iv["duration_min"].where(vt_present, 0.0).groupby(gk).sum()
vt8_in = iv["duration_min"].where(vt_present & (iv["vt_per_pbw"] <= SCORECARD_VT_CUTOFF), 0.0).groupby(gk).sum()
vt = pd.DataFrame({"vt_assess_min": vt_assess, "vt8_in_min": vt8_in}).reset_index()
vt.columns = key + ["vt_assess_min", "vt8_in_min"]

sev = pd.read_parquet(LPV_OUT / "02d_severity.parquet")[["hospitalization_id", "calendar_day", "severity"]]
sev["hospitalization_id"] = sev["hospitalization_id"].astype(str)
sev["calendar_day"] = pd.to_datetime(sev["calendar_day"]).dt.date

day = status[["hospitalization_id", "calendar_day", "assigned_unit", "total_imv_minutes",
              "plat_status", "dp_status"]].merge(vt, on=key, how="left").merge(sev, on=key, how="left")
day[["vt_assess_min", "vt8_in_min"]] = day[["vt_assess_min", "vt8_in_min"]].fillna(0.0)
day["severity"] = day["severity"].fillna("unknown")

# Per-day adherence booleans
day["vt8_ass"] = day["vt_assess_min"] >= MIN_ASSESSABLE_MIN
day["vt8_ad"] = day["vt8_ass"] & ((day["vt8_in_min"] / day["vt_assess_min"].where(day["vt_assess_min"] > 0)) >= ADHERENCE_FRACTION)
day["plat_ass"] = day["plat_status"].isin(["adherent", "non_adherent"])
day["plat_ad"] = day["plat_status"] == "adherent"
day["dp_ass"] = day["dp_status"].isin(["adherent", "non_adherent"])
day["dp_ad"] = day["dp_status"] == "adherent"

# ISO week + calendar month buckets
_dt = pd.to_datetime(day["calendar_day"])
isoc = _dt.dt.isocalendar()
day["week"] = isoc["year"].astype(str) + "-W" + isoc["week"].astype(int).map("{:02d}".format)
day["month"] = _dt.dt.strftime("%Y-%m")

# ----------------------------------------------------------------------------
# 2. Roll up to (unit, period) cells and assemble the in-memory LPV feed
# ----------------------------------------------------------------------------

print("[2] Building the LPV tile feed (per unit × all/month/week) ...")
weeks = sorted(day["week"].unique().tolist())
months = sorted(day["month"].unique().tolist())
units = ["__ALL__"] + [u for u in UNIT_ORDER_REST if u in set(day["assigned_unit"])]
rep = day.groupby("week")["calendar_day"].min()
week_label = {w: f"Week {w[-2:].lstrip('0')} · {pd.Timestamp(rep[w]).strftime('%b %Y')}" for w in weeks}
month_label = {m: pd.Timestamp(m + "-01").strftime("%b %Y") for m in months}


def cell_counts(df: pd.DataFrame) -> dict:
    """(numerator, denominator) per measure + denominator-line counts, for one (unit, period) slice."""
    sevdf = df[df["severity"] == "severe"]
    return {
        "vt8": (int(df["vt8_ad"].sum()), int(df["vt8_ass"].sum())),
        "plat": (int(df["plat_ad"].sum()), int(df["plat_ass"].sum())),
        "dp": (int(df["dp_ad"].sum()), int(df["dp_ass"].sum())),
        "vt8sev": (int(sevdf["vt8_ad"].sum()), int(sevdf["vt8_ass"].sum())),
        "n": int(len(df)),
        "hrs": round(float(df["total_imv_minutes"].sum()) / 60.0),
    }


# raw[unit][period_key] = cell_counts(...)  for period_key in {"all"} ∪ weeks ∪ months
raw = {u: {} for u in units}
raw["__ALL__"]["all"] = cell_counts(day)
for u, gu in day.groupby("assigned_unit"):
    if u in raw:
        raw[u]["all"] = cell_counts(gu)
for bucket in ("week", "month"):
    for b, gb in day.groupby(bucket):
        raw["__ALL__"][b] = cell_counts(gb)
        for u, gu in gb.groupby("assigned_unit"):
            if u in raw:
                raw[u][b] = cell_counts(gu)


def headline_cells() -> dict:
    out = {}
    for u, periods in raw.items():
        out[u] = {pk: {"num": c["vt8"][0], "den": c["vt8"][1], "n": c["n"], "hrs": c["hrs"]}
                  for pk, c in periods.items()}
    return out


def measure_cells(mkey: str) -> dict:
    out = {}
    for u, periods in raw.items():
        out[u] = {pk: {"num": c[mkey][0], "den": c[mkey][1]} for pk, c in periods.items()}
    return out


cut = f"{SCORECARD_VT_CUTOFF:g}"
lpv_feed = {
    "schema_version": 1,
    "metric_id": "lpv",
    "title": "LPV Adherence",
    "subtitle": f"Tidal volume ≤ {cut} mL/kg PBW",
    "icon": "lpv",
    "detail_href": "lpv_dashboard.html",
    "goal": LPV_GOAL,
    "note": None,
    "grain": {"units": units, "periods": ["all", "month", "week"]},
    "headline": {"label": "adherent", "den_label": "of assessable", "n_unit": "patient-days",
                 "cells": headline_cells()},
    "segments": [
        {"key": "plat", "label": "Plateau ≤ 30", "cells": measure_cells("plat")},
        {"key": "dp", "label": "∆P ≤ 15", "cells": measure_cells("dp")},
        {"key": "vt8sev", "label": f"Vt ≤ {cut} · severe", "cells": measure_cells("vt8sev")},
    ],
}

# ----------------------------------------------------------------------------
# 2b. Load external tile feeds (config `scorecard_tiles`) + ship their detail dashboards
# ----------------------------------------------------------------------------

print("[2b] Loading external tile feeds ...")
REQUIRED_FEED_KEYS = {"schema_version", "metric_id", "title", "grain", "headline"}


def load_external_feed(path_str: str):
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        print(f"  [skip] feed not found: {p}")
        return None
    try:
        feed = json.loads(p.read_text())
    except Exception as e:  # malformed JSON shouldn't break the whole scorecard
        print(f"  [skip] feed unreadable ({e}): {p}")
        return None
    if feed.get("schema_version") != 1:
        print(f"  [skip] unsupported schema_version={feed.get('schema_version')}: {p}")
        return None
    missing = REQUIRED_FEED_KEYS - set(feed)
    if missing:
        print(f"  [skip] feed missing keys {missing}: {p}")
        return None
    # PHI guard: a tile feed must be aggregated, never row-level.
    dump = json.dumps(feed)
    assert "hospitalization_id" not in dump and "patient_id" not in dump, f"PHI substring in feed {p}"
    # Ship the detail dashboard alongside the scorecard so the tile's link resolves.
    href = feed.get("detail_href")
    if href:
        src = p.parent / href
        if src.exists():
            shutil.copyfile(src, DASH_DIR / href)
            print(f"  [feed] {feed['metric_id']}: loaded; copied '{href}' into dashboard/")
        else:
            print(f"  [feed] {feed['metric_id']}: loaded; detail '{href}' missing at {src} -> dropping link")
            feed["detail_href"] = None
    else:
        print(f"  [feed] {feed['metric_id']}: loaded (no detail link)")
    return feed


external_feeds = [f for f in (load_external_feed(s) for s in CFG.get("scorecard_tiles", [])) if f]
feeds = [lpv_feed] + external_feeds
feeds_by_id = {f["metric_id"]: f for f in feeds}
print(f"  feeds ready: {[f['metric_id'] for f in feeds]}")

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
    <div class="chip"><b>Month</b><select id="sel-month"><option value="all">All time</option>{month_opts}</select></div>
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

const feedById = {}; P.feeds.forEach(f => feedById[f.metric_id] = f);
let unit = P.units.includes('__ALL__') ? '__ALL__' : P.units[0];
let pType = 'week', pKey = P.weeks[P.weeks.length-1];   // default: latest week

function curPeriodKey(){ return pType==='all' ? 'all' : pKey; }
function periodLabel(){ return pType==='all' ? 'all time' : (pType==='month'?P.month_label[pKey]:P.week_label[pKey]); }

// Grain fallback (contract §4): resolve the (unit, period) this feed can actually answer,
// and a badge so a coarse feed's number is never silently mislabeled.
function resolve(grain){
  let u = unit, ub = '';
  if(unit!=='__ALL__' && !grain.units.includes(unit)){ u='__ALL__'; ub=' · site-wide'; }
  let pk, pb='';
  if(grain.periods.includes(pType)){ pk = curPeriodKey(); }
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
    const useMonth = (pType==='month'), xs = useMonth?P.months:P.weeks;
    const ys = xs.map(b => rate((feed.headline.cells[R.u]||{})[b]));
    spark = sparkSVG(xs, ys, (pType==='all') ? null : pKey);
  }

  const note  = feed.note ? `<div class="tilenote">${esc(feed.note)}</div>` : '';
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
selU.value=unit; selW.value=pKey; selM.value='all';
selU.onchange = e=>{ unit=e.target.value; render(); };
selM.onchange = e=>{ if(e.target.value!=='all'){ pType='month'; pKey=e.target.value; selW.value='all'; }
                     else if(pType==='month'){ pType='all'; pKey=null; } render(); };
selW.onchange = e=>{ if(e.target.value!=='all'){ pType='week'; pKey=e.target.value; selM.value='all'; }
                     else if(pType==='week'){ pType='all'; pKey=null; } render(); };
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
    "Vt<=8 all-units/all-time ~ 83%": 0.78 < feed_rate(lpv_feed, "vt8") < 0.88,
    "Plateau ~ 85.8%": 0.83 < feed_rate(lpv_feed, "plat") < 0.89,
    "∆P ~ 48%": 0.44 < feed_rate(lpv_feed, "dp") < 0.52,
    "Vt<=8 severe computed": lpv_feed["segments"][2]["cells"]["__ALL__"]["all"]["den"] >= 0,
    "LPV feed links to lpv_dashboard.html": lpv_feed["detail_href"] == "lpv_dashboard.html",
    "tile order has 5 slots": len(TILE_ORDER) == 5,
    "scorecard.html written into dashboard/": out_path.exists() and out_path.parent.name == "dashboard",
    "no hospitalization_id in payload": "hospitalization_id" not in payload_dump,
    "no patient_id in payload": "patient_id" not in payload_dump,
}

# Proning (and any external feed) sanity, only if loaded.
if "proning" in feeds_by_id:
    pf = feeds_by_id["proning"]
    hc = pf["headline"]["cells"]["__ALL__"]["all"]
    checks["proning feed: 0 <= num <= den, den>0"] = 0 <= hc["num"] <= hc["den"] and hc["den"] > 0
    checks["proning detail copied into dashboard/"] = (
        pf.get("detail_href") is None or (DASH_DIR / pf["detail_href"]).exists())
    print(f"  proning headline: {hc['num']}/{hc['den']} = {hc['num']/hc['den']*100:.1f}% ever proned")
else:
    print("  (no proning feed loaded — add its path to config 'scorecard_tiles')")

for k, v in checks.items():
    print(f"  [{'ok' if v else 'XX'}] {k}")
print(f"\n  All-units/all-time: Vt≤{cut} {feed_rate(lpv_feed,'vt8')*100:.1f}% · "
      f"Plateau {feed_rate(lpv_feed,'plat')*100:.1f}% · ∆P {feed_rate(lpv_feed,'dp')*100:.1f}% · "
      f"Vt≤{cut} severe {feed_rate(lpv_feed,'vt8sev')*100:.1f}%")
print(f"  feeds: {[f['metric_id'] for f in feeds]} · weeks: {len(weeks)} ({weeks[0]} → {weeks[-1]}) · units: {len(units)}")
print(f"  bundle dir: {DASH_DIR}  contents: {sorted(p.name for p in DASH_DIR.glob('*.html'))}")
assert all(checks.values()), "VERIFICATION FAILED"
print("\nAll checks passed. Done.")
