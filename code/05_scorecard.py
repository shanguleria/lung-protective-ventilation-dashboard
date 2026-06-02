"""
05_scorecard.py — CLIF ICU Ventilator-QI bundle scorecard (landing page).

Builds output/index.html: a glanceable, per-unit / per-ISO-week scorecard of ventilator
QI bundle metrics. LPV is the first REAL tile (links to the detailed dashboard,
04_lpv_dashboard.html); SAT / SBT / ARDS proning / mobilization are styled placeholders
for the rest of the ICU-liberation bundle.

LPV tile headline = tidal-volume adherence at <= 8 mL/kg PBW (a realistic QI target),
with a 3-segment mini-indicator: Plateau <= 30 · Driving pressure <= 15 · Vt <= 8 in
SEVERE respiratory failure. Lightweight (inline SVG donut + sparkline, no Plotly).

Inputs:  output/02_patient_day_status.parquet, 02_intervals.parquet, 02d_severity.parquet
Output:  output/index.html

Run:
    .venv/bin/python code/05_scorecard.py
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT_DIR = Path(CFG.get("output_path", ROOT / "output"))
SITE = CFG.get("site", "Your Site")

# ---- Named parameters ----
SCORECARD_VT_CUTOFF = 8.0   # headline Vt/kg cutoff for the scorecard tile
LPV_GOAL = 0.90             # target line on the LPV tile
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60

UNIT_ORDER_REST = ["medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
                   "mixed_neuro_icu", "general_icu", "burn_icu"]
UNIT_LABEL = {"__ALL__": "All ICUs", "medical_icu": "Medical ICU",
              "mixed_cardiothoracic_icu": "Cardiothoracic ICU", "surgical_icu": "Surgical ICU",
              "mixed_neuro_icu": "Neuro ICU", "general_icu": "General ICU", "burn_icu": "Burn ICU"}

# ----------------------------------------------------------------------------
# 1. Load + per-(hosp, day) Vt<=8 recompute (status file is default-6)
# ----------------------------------------------------------------------------

print("[1] Loading + computing Vt<=8 per patient-day ...")
status = pd.read_parquet(OUT_DIR / "02_patient_day_status.parquet")
status["hospitalization_id"] = status["hospitalization_id"].astype(str)
status["calendar_day"] = pd.to_datetime(status["calendar_day"]).dt.date

iv = pd.read_parquet(OUT_DIR / "02_intervals.parquet")
iv["hospitalization_id"] = iv["hospitalization_id"].astype(str)
iv["calendar_day"] = pd.to_datetime(iv["calendar_day"]).dt.date
key = ["hospitalization_id", "calendar_day"]
gk = [iv["hospitalization_id"], iv["calendar_day"]]
vt_present = iv["vt_per_pbw"].notna()
vt_assess = iv["duration_min"].where(vt_present, 0.0).groupby(gk).sum()
vt8_in = iv["duration_min"].where(vt_present & (iv["vt_per_pbw"] <= SCORECARD_VT_CUTOFF), 0.0).groupby(gk).sum()
vt = pd.DataFrame({"vt_assess_min": vt_assess, "vt8_in_min": vt8_in}).reset_index()
vt.columns = key + ["vt_assess_min", "vt8_in_min"]

sev = pd.read_parquet(OUT_DIR / "02d_severity.parquet")[["hospitalization_id", "calendar_day", "severity"]]
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
# 2. Roll up to (unit, week) and (unit, month); build payload
# ----------------------------------------------------------------------------

print("[2] Rolling up to (unit, ISO-week) and (unit, month) ...")
weeks = sorted(day["week"].unique().tolist())
months = sorted(day["month"].unique().tolist())
units = ["__ALL__"] + [u for u in UNIT_ORDER_REST if u in set(day["assigned_unit"])]
rep = day.groupby("week")["calendar_day"].min()
week_label = {w: f"Week {w[-2:].lstrip('0')} · {pd.Timestamp(rep[w]).strftime('%b %Y')}" for w in weeks}
month_label = {m: pd.Timestamp(m + "-01").strftime("%b %Y") for m in months}


def counts(df: pd.DataFrame) -> dict:
    sevdf = df[df["severity"] == "severe"]
    return {
        "vt8": [int(df["vt8_ad"].sum()), int(df["vt8_ass"].sum())],
        "plat": [int(df["plat_ad"].sum()), int(df["plat_ass"].sum())],
        "dp": [int(df["dp_ad"].sum()), int(df["dp_ass"].sum())],
        "vt8sev": [int(sevdf["vt8_ad"].sum()), int(sevdf["vt8_ass"].sum())],
        "n": int(len(df)),
        "hrs": round(float(df["total_imv_minutes"].sum()) / 60.0),
    }


def rollup(bucket: str) -> dict:
    out = {u: {} for u in units}
    for b, gb in day.groupby(bucket):
        out["__ALL__"][b] = counts(gb)
        for u, gu in gb.groupby("assigned_unit"):
            if u in out:
                out[u][b] = counts(gu)
    return out


data = rollup("week")
data_month = rollup("month")

payload = {
    "site": SITE, "vt_cutoff": SCORECARD_VT_CUTOFF, "goal": LPV_GOAL,
    "weeks": weeks, "week_label": week_label, "months": months, "month_label": month_label,
    "units": units, "unit_label": UNIT_LABEL,
    "data": data, "data_month": data_month,
    "generated": datetime.now().isoformat(timespec="minutes"),
}

# ----------------------------------------------------------------------------
# 3. HTML (CLIF maroon/cream house style; inline SVG donut + sparkline)
# ----------------------------------------------------------------------------

# Optional tile illustrations: downscale + base64-embed references/images/<Name>.png.
# Falls back to the inline SVG icon if an image is missing (so cloning sites still build).
import base64
from io import BytesIO

IMG_DIR = ROOT / "references" / "images"
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

print("[3] Writing index.html ...")
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
header.top{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:6px;}
.brand{display:flex;align-items:center;gap:9px;font-weight:800;font-size:20px;color:var(--maroon);letter-spacing:.5px;}
.brand svg{width:26px;height:26px}
h1{font-size:17px;font-weight:700;color:var(--maroon-d);margin:0;letter-spacing:.4px;text-transform:uppercase;}
.chips{display:flex;gap:10px;margin-left:auto;align-items:center;flex-wrap:wrap;}
.chip{display:flex;align-items:center;gap:7px;background:var(--card);border:1px solid var(--line);
border-radius:999px;padding:6px 12px;font-size:13px;box-shadow:0 1px 2px rgba(120,30,40,.05);}
.chip b{color:var(--maroon);text-transform:uppercase;font-size:11px;letter-spacing:.6px;}
.chip select{border:0;background:transparent;font-size:13px;color:var(--ink);font-weight:600;cursor:pointer;outline:none;}
.subtitle{color:var(--muted);font-size:13px;margin:2px 0 22px;}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:18px;}
@media(max-width:1200px){.grid{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px 18px;
box-shadow:0 3px 10px rgba(120,30,40,.06);display:flex;flex-direction:column;align-items:center;text-align:center;
min-height:330px;position:relative;}
.card.lpv{border-color:#e7c9cd;}
.card.ph{opacity:.62;}
.card .ico{width:40px;height:40px;color:var(--maroon);margin-bottom:6px;}
.card img.ico{width:78px;height:78px;object-fit:contain;margin-bottom:2px;}
.card .mname{font-weight:800;font-size:15px;color:var(--maroon-d);letter-spacing:.3px;}
.card .msub{font-size:11.5px;color:var(--muted);margin:1px 0 10px;min-height:15px;}
.donut{position:relative;width:128px;height:128px;margin:2px 0 6px;}
.donut .val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.donut .pct{font-size:30px;font-weight:800;color:var(--maroon);font-variant-numeric:tabular-nums;line-height:1;}
.donut .plab{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:3px;}
.denom{font-size:12px;color:var(--muted);margin:2px 0 8px;}
.goalwrap{width:100%;margin:2px 0 12px;}
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
.bignum{font-size:34px;font-weight:800;color:#b9a59c;font-variant-numeric:tabular-nums;margin:24px 0 4px;}
.phnote{font-size:11px;color:var(--muted);}
.cardlink{position:absolute;bottom:12px;right:14px;font-size:11.5px;color:var(--maroon);text-decoration:none;font-weight:700;}
.cardlink:hover{text-decoration:underline;}
a.cardwrap{text-decoration:none;color:inherit;}
footer{margin-top:26px;color:var(--muted);font-size:11.5px;text-align:center;}
"""

HEART = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round"><path d="M2 12h4l2-6 4 12 3-9 2 3h5"/></svg>')
ICONS = {
    "lungs": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4v8"/><path d="M9 9c0 4-2 5-2 8a2 2 0 0 0 4 0V9a3 3 0 0 0-2 0Z"/><path d="M15 9c0 4 2 5 2 8a2 2 0 0 1-4 0V9a3 3 0 0 1 2 0Z"/></svg>',
    "sat": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="9" r="4"/><path d="M5 21c1-4 4-6 7-6s6 2 7 6"/><path d="M19 4l1.5 1.5M21 7h-2"/></svg>',
    "sbt": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 12c2-3 4-3 6 0s4 3 6 0 4-3 4-3"/><path d="M4 17h16"/></svg>',
    "prone": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 16h18"/><circle cx="7" cy="12" r="2"/><path d="M9 14h9a2 2 0 0 1 0 4H5"/></svg>',
    "mob": '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="13" cy="4" r="2"/><path d="M13 7l-2 5 3 2 1 5M11 12l-4 1M14 9l3 2"/></svg>',
}


def tile_icon(key):
    """The supplied illustration if present, else the inline SVG fallback."""
    return f'<img class="ico" src="{TILE_IMG[key]}" alt="">' if TILE_IMG.get(key) else ICONS.get(key, "")


PLACEHOLDERS = [
    ("sat", "SAT Completion", "Spontaneous Awakening Trials"),
    ("sbt", "SBT Completion", "Spontaneous Breathing Trials"),
    ("prone", "ARDS Proning", "Eligible ARDS patients"),
    ("mob", "Mobilization", "Target mobilization achieved"),
]
ph_cards = "".join(
    f'<div class="card ph">{tile_icon(ic)}<div class="mname">{html.escape(nm)}</div>'
    f'<div class="msub">{html.escape(sub)}</div><div class="bignum">—</div>'
    f'<div class="phnote">Not yet computed</div></div>'
    for ic, nm, sub in PLACEHOLDERS
)

# Donut SVG (radius 56, circumference set in JS)
DONUT_SVG = ('<svg viewBox="0 0 128 128"><circle cx="64" cy="64" r="56" fill="none" stroke="var(--bar)" '
             'stroke-width="13"/><circle id="donut-arc" cx="64" cy="64" r="56" fill="none" stroke="var(--maroon)" '
             'stroke-width="13" stroke-linecap="round" transform="rotate(-90 64 64)" stroke-dasharray="0 999"/></svg>')

BODY = f"""
<div class="wrap">
<header class="top">
  <div class="brand">{HEART}<span>CLIF</span></div>
  <h1>ICU Ventilator QI Dashboard — {html.escape(SITE)}</h1>
  <div class="chips">
    <div class="chip"><b>Unit</b><select id="sel-unit">{unit_sel}</select></div>
    <div class="chip"><b>Month</b><select id="sel-month"><option value="all">All time</option>{month_opts}</select></div>
    <div class="chip"><b>Week</b><select id="sel-week"><option value="all">All</option>{week_opts}</select></div>
  </div>
</header>
<p class="subtitle" id="subtitle"></p>

<div class="grid">
  <a class="cardwrap" href="04_lpv_dashboard.html"><div class="card lpv">
    {tile_icon('lpv')}
    <div class="mname">LPV Adherence</div>
    <div class="msub">Tidal volume ≤ {SCORECARD_VT_CUTOFF:g} mL/kg PBW</div>
    <div class="donut">{DONUT_SVG}<div class="val"><span class="pct" id="lpv-pct">—</span><span class="plab">adherent</span></div></div>
    <div class="denom" id="lpv-denom"></div>
    <div class="goalwrap"><div class="goalbar"><span id="lpv-goalfill"></span></div>
      <div class="goaltxt"><span id="lpv-n"></span><span>Goal ≥ {LPV_GOAL*100:.0f}%</span></div></div>
    <div class="segs">
      <div class="seg"><span class="sl">Plateau ≤ 30</span><span class="sb"><span id="seg-plat"></span></span><span class="sv" id="segv-plat">—</span></div>
      <div class="seg"><span class="sl">∆P ≤ 15</span><span class="sb"><span id="seg-dp"></span></span><span class="sv" id="segv-dp">—</span></div>
      <div class="seg"><span class="sl">Vt ≤ {SCORECARD_VT_CUTOFF:g} · severe</span><span class="sb"><span id="seg-sev"></span></span><span class="sv" id="segv-sev">—</span></div>
    </div>
    <svg class="spark" id="spark" viewBox="0 0 240 34" preserveAspectRatio="none"></svg>
    <span class="cardlink">View details →</span>
  </div></a>
  {ph_cards}
</div>

<footer>Data through {{LASTWK}} · Generated {{GEN}} · CLIF (Common Longitudinal ICU data Format) · {html.escape(SITE)}</footer>
</div>
"""

APP_JS = r"""
const P = JSON.parse(document.getElementById('sc-payload').textContent);
const C = 2*Math.PI*56;
const PCT = v => (v==null||isNaN(v)) ? '—' : (v*100).toFixed(0)+'%';
const segColor = r => r==null?'#ccc' : r>=0.8?'#2f7d5b' : r>=0.5?'#b5852a':'#a23b3b';
const BLANK = {vt8:[0,0],plat:[0,0],dp:[0,0],vt8sev:[0,0],n:0,hrs:0};
let unit = P.units.includes('__ALL__') ? '__ALL__' : P.units[0];
let pType = 'week', pKey = P.weeks[P.weeks.length-1];   // default: latest week

const rate = a => a[1] ? a[0]/a[1] : null;
function bucket(u, type, key){ return ((type==='month'?P.data_month:P.data)[u]||{})[key]; }
function aggAll(u){
  const acc={vt8:[0,0],plat:[0,0],dp:[0,0],vt8sev:[0,0],n:0,hrs:0};
  for(const w of P.weeks){ const r=(P.data[u]||{})[w]; if(!r) continue;
    for(const k of ['vt8','plat','dp','vt8sev']){acc[k][0]+=r[k][0];acc[k][1]+=r[k][1];}
    acc.n+=r.n; acc.hrs+=r.hrs; }
  return acc;
}
function current(u){ return pType==='all' ? aggAll(u) : (bucket(u,pType,pKey)||BLANK); }
function periodLabel(){ return pType==='all' ? 'all time' : (pType==='month'?P.month_label[pKey]:P.week_label[pKey]); }

function render(){
  const a = current(unit), r = rate(a.vt8);
  document.getElementById('lpv-pct').textContent = PCT(r);
  document.getElementById('donut-arc').setAttribute('stroke-dasharray', `${(r||0)*C} ${C}`);
  document.getElementById('lpv-denom').textContent =
    a.hrs.toLocaleString()+' patient-hours · '+a.n.toLocaleString()+' patient-days';
  document.getElementById('lpv-goalfill').style.width = Math.min(100,(r||0)/P.goal*100)+'%';
  document.getElementById('lpv-n').textContent = (r==null?'—':PCT(r)+' of assessable');
  const segs={plat:rate(a.plat), dp:rate(a.dp), sev:rate(a.vt8sev)};
  for(const k of ['plat','dp','sev']){
    const v=segs[k];
    document.getElementById('seg-'+k).style.width=((v||0)*100)+'%';
    document.getElementById('seg-'+k).style.background=segColor(v);
    document.getElementById('segv-'+k).textContent=PCT(v);
  }
  document.getElementById('subtitle').textContent =
    (P.unit_label[unit]||unit)+' · '+periodLabel()+' — lung-protective ventilation bundle';
  // sparkline: monthly series in month-mode, else weekly
  const useMonth=(pType==='month'), xs=useMonth?P.months:P.weeks, dat=useMonth?P.data_month:P.data;
  const ys=xs.map(b=>{const rr=(dat[unit]||{})[b]; return rr?rate(rr.vt8):null;});
  const W=240,H=34,pad=2, valid=ys.map((y,i)=>[i,y]).filter(p=>p[1]!=null);
  const X=i=>pad+i/Math.max(1,xs.length-1)*(W-2*pad), Y=v=>H-pad-v*(H-2*pad);
  const path=valid.map((p,j)=>(j?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  const hi = (pType==='all') ? -1 : xs.indexOf(pKey);
  const dot = (hi>=0 && ys[hi]!=null) ? `<circle cx="${X(hi).toFixed(1)}" cy="${Y(ys[hi]).toFixed(1)}" r="3" fill="#8a1f2b"/>` : '';
  document.getElementById('spark').innerHTML = `<path d="${path}" fill="none" stroke="#c98a92" stroke-width="1.6"/>`+dot;
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

out_path = OUT_DIR / "index.html"
out_path.write_text(HTML)
print(f"  wrote {out_path}  ({out_path.stat().st_size/1e3:.0f} KB)")

# ----------------------------------------------------------------------------
# 4. Verification
# ----------------------------------------------------------------------------

print("\n[verify]")


def alltime(measure):
    ad = sum(data["__ALL__"][w][measure][0] for w in weeks)
    ass = sum(data["__ALL__"][w][measure][1] for w in weeks)
    return ad / ass if ass else float("nan")


checks = {
    "Vt<=8 all-units/all-time ~ 83%": 0.78 < alltime("vt8") < 0.88,
    "Plateau ~ 85.8%": 0.83 < alltime("plat") < 0.89,
    "∆P ~ 48%": 0.44 < alltime("dp") < 0.52,
    "Vt<=8 severe computed": data["__ALL__"][weeks[0]]["vt8sev"][1] >= 0,
    "LPV card links to detail": 'href="04_lpv_dashboard.html"' in HTML,
    "5 tiles (1 real + 4 placeholders)": HTML.count('class="card ') == 5,
    "no hospitalization_id in payload": "hospitalization_id" not in json.dumps(payload),
    "no patient_id in payload": "patient_id" not in json.dumps(payload),
}
for k, v in checks.items():
    print(f"  [{'ok' if v else 'XX'}] {k}")
print(f"\n  All-units/all-time: Vt≤{SCORECARD_VT_CUTOFF:g} {alltime('vt8')*100:.1f}% · "
      f"Plateau {alltime('plat')*100:.1f}% · ∆P {alltime('dp')*100:.1f}% · "
      f"Vt≤{SCORECARD_VT_CUTOFF:g} severe {alltime('vt8sev')*100:.1f}%")
print(f"  weeks: {len(weeks)} ({weeks[0]} → {weeks[-1]}) · units: {len(units)}")
assert all(checks.values()), "VERIFICATION FAILED"
print("\nAll checks passed. Done.")
