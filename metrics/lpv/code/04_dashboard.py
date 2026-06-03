"""
04_dashboard.py — Single self-contained interactive HTML dashboard for LPV adherence.

Presents the four component-separated measures (Vt/kg headline + slider, Pplat, Pdriving,
Composite) for UChicago, plus per-unit/temporal trends, time-weighted settings histograms,
and a cohort Table 1. Plotly.js is embedded inline so the output is fully portable/offline.

The Vt-cutoff slider indexes the precomputed Vt grid (03_vt_grid_*) — the browser never
recomputes adherence over the interval data. Plateau<=30 & dP<=15 are fixed.

Inputs:  output/03_monthly_unit_summary.parquet, 03_vt_grid_monthly.parquet,
         03_vt_grid_daily_allunits.parquet, 02_intervals.parquet,
         02_patient_day_status.parquet, 02_features_summary.json, 03_aggregate_summary.json
Output:  output/dashboard/lpv_dashboard.html   (+ cached output/_vendor/plotly.min.js)
         (output/dashboard/ is the shippable bundle: scorecard.html + the per-metric drill-downs)

Run:
    .venv/bin/python code/04_dashboard.py
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))
SITE = CFG.get("site", "Your Site")
CLIF_VER = CFG.get("clif_version", "2.x")
VENDOR = OUT_DIR / "_vendor"            # build-time cache only (Plotly is inlined into the HTML)
VENDOR.mkdir(parents=True, exist_ok=True)
DASH_DIR = ROOT / "output" / "dashboard"  # shared shippable bundle (scorecard + all metric drill-downs)
DASH_DIR.mkdir(parents=True, exist_ok=True)

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
UNIT_ORDER_REST = ["medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
                   "mixed_neuro_icu", "general_icu", "burn_icu"]
UNIT_LABEL = {
    "__ALL__": "All ICUs", "medical_icu": "Medical ICU",
    "mixed_cardiothoracic_icu": "Cardiothoracic ICU", "surgical_icu": "Surgical ICU",
    "mixed_neuro_icu": "Neuro ICU", "general_icu": "General ICU", "burn_icu": "Burn ICU",
}
MEASURE_LABEL = {"vt": "Tidal volume (Vt/kg)", "plat": "Plateau ≤ 30",
                 "dp": "Driving pressure ≤ 15", "comp": "Composite (all three)"}


def jsonable(o):
    """Recursively replace NaN/inf with None so JSON has no NaN literals."""
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def med_iqr(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    q = s.quantile([0.25, 0.5, 0.75])
    return f"{q.loc[0.5]:.1f} ({q.loc[0.25]:.1f}, {q.loc[0.75]:.1f})"


def npct(n: int, tot: int) -> str:
    return f"{n:,} ({n/tot*100:.1f}%)" if tot else "0"


# ----------------------------------------------------------------------------
# gtsummary renderer (verbatim from ~/.claude/templates/dashboard_design_guide.md)
# ----------------------------------------------------------------------------

def render_gtsummary_table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""

    def _header_cell(c: str) -> str:
        c = html.escape(str(c), quote=False)
        c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", c)
        return c.replace("\n", "<br>")

    header_row = "".join(f"<th>{_header_cell(c)}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for i, col in enumerate(df.columns):
            val = row[col]
            raw = "" if (pd.isna(val) or val is None) else str(val)
            if i == 0:
                m_bold = re.match(r"^__(.+)__$", raw)
                if m_bold:
                    cell = f"<strong>{html.escape(m_bold.group(1), quote=False)}</strong>"
                elif raw:
                    cell = f'<span style="padding-left: 20px;">{html.escape(raw, quote=False)}</span>'
                else:
                    cell = ""
            else:
                cell = html.escape(raw, quote=False)
            cells.append(f"<td>{cell}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<table class="results-table" border="0">'
            f"<thead><tr>{header_row}</tr></thead>"
            "<tbody>" + "\n".join(body_rows) + "</tbody></table>")


# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------

print("[1] Loading 02/03 outputs ...")
monthly = pd.read_parquet(OUT_DIR / "03_monthly_unit_summary.parquet")
grid = pd.read_parquet(OUT_DIR / "03_vt_grid_monthly.parquet")
daily_grid = pd.read_parquet(OUT_DIR / "03_vt_grid_daily_allunits.parquet")   # site-wide vt/comp daily
daily_sum = pd.read_parquet(OUT_DIR / "03_daily_unit_summary.parquet")        # has __ALL__ daily for plat/dp
wgrid = pd.read_parquet(OUT_DIR / "03_vt_grid_weekly.parquet")                # per-unit vt/comp weekly (all-severity)
wsum = pd.read_parquet(OUT_DIR / "03_weekly_unit_summary.parquet")            # per-unit all-measure weekly
status = pd.read_parquet(OUT_DIR / "02_patient_day_status.parquet")
iv = pd.read_parquet(OUT_DIR / "02_intervals.parquet")
feat = json.loads((OUT_DIR / "02_features_summary.json").read_text())
agg = json.loads((OUT_DIR / "03_aggregate_summary.json").read_text())

# Severity stratifier (severe respiratory failure) — join onto status (Table 1) + intervals (histograms).
SEVS = ["severe", "not_severe", "unknown"]
_sev = pd.read_parquet(OUT_DIR / "02d_severity.parquet")[["hospitalization_id", "calendar_day", "severity"]]
_sev["hospitalization_id"] = _sev["hospitalization_id"].astype(str)
_sev["calendar_day"] = pd.to_datetime(_sev["calendar_day"]).dt.date
for _df in (status, iv):
    _df["hospitalization_id"] = _df["hospitalization_id"].astype(str)
    _df["calendar_day"] = pd.to_datetime(_df["calendar_day"]).dt.date
status = status.merge(_sev, on=["hospitalization_id", "calendar_day"], how="left")
iv = iv.merge(_sev, on=["hospitalization_id", "calendar_day"], how="left")
status["severity"] = status["severity"].fillna("unknown")
iv["severity"] = iv["severity"].fillna("unknown")

VT_GRID = [float(c) for c in agg["params"]["vt_grid"]]
VT_DEFAULT = float(agg["params"]["vt_default"])
PLATEAU_MAX = float(agg["params"]["plateau_max"])
DP_MAX = float(agg["params"]["dp_max"])

months = sorted(monthly["month"].unique().tolist())
units = ["__ALL__"] + [u for u in UNIT_ORDER_REST if u in set(monthly["assigned_unit"])]


def pct_assessable_col(df):
    # for monthly summary (has non_adherent split)
    if "n_non_adherent" in df.columns:
        return (df["n_adherent"] + df["n_non_adherent"]) / df["n_total"]
    return df["n_assessable"] / df["n_total"]


# ----------------------------------------------------------------------------
# 2. Build compact payload
# ----------------------------------------------------------------------------

print("[2] Building payload (per-month counts; period filtering happens in JS) ...")

n_m = len(months)
midx = {mo: i for i, mo in enumerate(months)}
years = sorted({mo[:4] for mo in months})


def counts_array(sub: pd.DataFrame, col: str) -> list:
    """Per-month array of integer counts (0 where the month is absent)."""
    a = [0] * n_m
    for mo, v in zip(sub["month"], sub[col]):
        a[midx[mo]] = int(v)
    return a


# vt/comp counts, split to keep severity tripling bounded (×~1, not ×3):
#   tot[unit][sev]                  — cohort patient-days (same across measure & cutoff)
#   ass[unit][measure][sev]         — assessable (cutoff-independent)
#   ad[unit][measure][cutoff][sev]  — adherent (cutoff-dependent)
tot, ass, ad = {}, {}, {}
for u in units:
    g_u = grid[grid["assigned_unit"] == u]
    tot[u], ass[u], ad[u] = {}, {}, {}
    for s in SEVS:
        tslice = g_u[(g_u["severity"] == s) & (g_u["measure"] == "vt") & (g_u["vt_cutoff"] == VT_DEFAULT)]
        tot[u][s] = counts_array(tslice, "n_total")
    for m in ["vt", "comp"]:
        g_um = g_u[g_u["measure"] == m]
        ass[u][m], ad[u][m] = {}, {}
        for s in SEVS:
            ass[u][m][s] = counts_array(g_um[(g_um["severity"] == s) & (g_um["vt_cutoff"] == VT_DEFAULT)], "n_assessable")
        for c in VT_GRID:
            ck = f"{c:.1f}"
            g_umc = g_um[g_um["vt_cutoff"] == c]
            ad[u][m][ck] = {s: counts_array(g_umc[g_umc["severity"] == s], "n_adherent") for s in SEVS}

# plat/dp: per unit, per severity — monthly counts (cutoff-independent)
stc = {}
for m in ["plat", "dp"]:
    stc[m] = {}
    for u in units:
        g_um = monthly[(monthly["assigned_unit"] == u) & (monthly["measure"] == m)].copy()
        g_um["ass"] = g_um["n_adherent"] + g_um["n_non_adherent"]
        stc[m][u] = {}
        for s in SEVS:
            sub = g_um[g_um["severity"] == s]
            stc[m][u][s] = {"ad": counts_array(sub, "n_adherent"),
                            "ass": counts_array(sub, "ass"),
                            "tot": counts_array(sub, "n_total")}

# Daily site-wide counts (for zooming into a single month → daily granularity).
daily_grid["d"] = pd.to_datetime(daily_grid["calendar_day"]).dt.strftime("%Y-%m-%d")
daily_sum["d"] = pd.to_datetime(daily_sum["calendar_day"]).dt.strftime("%Y-%m-%d")
days = sorted(daily_grid["d"].unique().tolist())
didx = {d: i for i, d in enumerate(days)}

# ISO-week mapping (lean weekly view: site-wide weekly from the daily data, no new payload).
_dd = pd.to_datetime(pd.Series(days))
_iso = _dd.dt.isocalendar()
day_week = (_iso["year"].astype(str) + "-W" + _iso["week"].astype(int).map("{:02d}".format)).tolist()
_wk = pd.DataFrame({"d": days, "wk": day_week})
_rep = _wk.groupby("wk")["d"].min()
week_label = {w: f"Week {int(w[-2:])} · {pd.Timestamp(_rep[w]).strftime('%b %Y')}" for w in _rep.index}
weeks_list = sorted(set(day_week))

# Per-(unit, ISO-week) counts (all-severity) — full weekly view: per-unit bars, etc.
n_w = len(weeks_list)
wkidx = {w: i for i, w in enumerate(weeks_list)}


def warr(sub: pd.DataFrame, col: str) -> list:
    a = [0] * n_w
    for w, v in zip(sub["week"], sub[col]):
        if w in wkidx:
            a[wkidx[w]] = int(v)
    return a


wtot, wass, wad = {}, {}, {}
for u in units:
    gu = wgrid[wgrid["assigned_unit"] == u]
    wtot[u] = warr(gu[(gu["measure"] == "vt") & (gu["vt_cutoff"] == VT_DEFAULT)], "n_total")
    wass[u], wad[u] = {}, {}
    for m in ["vt", "comp"]:
        gm = gu[gu["measure"] == m]
        wass[u][m] = warr(gm[gm["vt_cutoff"] == VT_DEFAULT], "n_assessable")
        wad[u][m] = {f"{c:.1f}": warr(gm[gm["vt_cutoff"] == c], "n_adherent") for c in VT_GRID}
wstc = {}
for m in ["plat", "dp"]:
    wstc[m] = {}
    for u in units:
        s = wsum[(wsum["assigned_unit"] == u) & (wsum["measure"] == m)].copy()
        s["ass"] = s["n_adherent"] + s["n_non_adherent"]
        wstc[m][u] = {"ad": warr(s, "n_adherent"), "ass": warr(s, "ass"), "tot": warr(s, "n_total")}


def darr(sub: pd.DataFrame, col: str) -> list:
    a = [0] * len(days)
    for d, v in zip(sub["d"], sub[col]):
        a[didx[d]] = int(v)
    return a


vtd = {}
for c in VT_GRID:
    ck = f"{c:.1f}"
    vtd[ck] = {}
    for m in ["vt", "comp"]:
        sub = daily_grid[(daily_grid["vt_cutoff"] == c) & (daily_grid["measure"] == m)]
        vtd[ck][m] = {"ad": darr(sub, "n_adherent"), "ass": darr(sub, "n_assessable"), "tot": darr(sub, "n_total")}

std = {}
dall = daily_sum[daily_sum["assigned_unit"] == "__ALL__"].copy()
dall["ass"] = dall["n_adherent"] + dall["n_non_adherent"]
for m in ["plat", "dp"]:
    sub = dall[dall["measure"] == m]
    std[m] = {"ad": darr(sub, "n_adherent"), "ass": darr(sub, "ass"), "tot": darr(sub, "n_total")}

# ----------------------------------------------------------------------------
# 3. Per-month time-weighted histograms from 02_intervals (summable bin counts)
# ----------------------------------------------------------------------------

print("[3] Per-month histograms ...")
HIST_SPEC = {
    "vt_per_pbw": dict(lo=0, hi=15, step=0.25, title="Tidal volume (mL/kg PBW)", thr="slider"),
    "plateau": dict(lo=0, hi=60, step=1.0, title="Plateau pressure (cm H₂O)", thr=PLATEAU_MAX),
    "driving_pressure": dict(lo=0, hi=50, step=1.0, title="Driving pressure ∆P (cm H₂O)", thr=DP_MAX),
    "peep": dict(lo=0, hi=30, step=1.0, title="PEEP (cm H₂O)", thr=None),
    "fio2": dict(lo=0.2, hi=1.0, step=0.05, title="FiO₂", thr=None),
}
iv["month"] = pd.to_datetime(iv["calendar_day"]).dt.strftime("%Y-%m")
histc = {}
for col, spec in HIST_SPEC.items():
    edges = np.arange(spec["lo"], spec["hi"] + spec["step"], spec["step"])
    nb = len(edges) - 1
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    by_sev = {}
    for s in SEVS:
        counts = [[0.0] * nb for _ in range(n_m)]
        # Drop out-of-range values (don't clip) so the tails taper instead of piling into the edge bins.
        sub_s = iv[iv[col].notna() & (iv["severity"] == s)
                   & (iv[col] >= spec["lo"]) & (iv[col] < spec["hi"])]
        for mo, g in sub_s.groupby("month"):
            if mo not in midx:
                continue
            cnt, _ = np.histogram(g[col], bins=edges, weights=g["duration_min"])
            counts[midx[mo]] = [round(float(x), 1) for x in cnt]
        by_sev[s] = counts
    histc[col] = {"centers": centers, "title": spec["title"], "threshold": spec["thr"], "counts": by_sev}

# Weekly histograms (all-severity) — for the full weekly view.
_iso2 = pd.to_datetime(iv["calendar_day"]).dt.isocalendar()
iv["week"] = _iso2["year"].astype(str) + "-W" + _iso2["week"].astype(int).map("{:02d}".format)
whist = {}
for col, spec in HIST_SPEC.items():
    edges = np.arange(spec["lo"], spec["hi"] + spec["step"], spec["step"])
    nb = len(edges) - 1
    counts = [[0.0] * nb for _ in range(n_w)]
    sub = iv[iv[col].notna() & (iv[col] >= spec["lo"]) & (iv[col] < spec["hi"])]
    for w, g in sub.groupby("week"):
        if w in wkidx:
            cnt, _ = np.histogram(g[col], bins=edges, weights=g["duration_min"])
            counts[wkidx[w]] = [round(float(x), 1) for x in cnt]
    whist[col] = counts

# ----------------------------------------------------------------------------
# 4. Per-period Table 1 + headline counts (precomputed; no row-level data shipped)
# ----------------------------------------------------------------------------

print("[4] Per-period Table 1 ...")
status_t = status.copy()
status_t["month"] = pd.to_datetime(status_t["calendar_day"]).dt.strftime("%Y-%m")
status_t["year"] = status_t["month"].str.slice(0, 4)
_isos = pd.to_datetime(status_t["calendar_day"]).dt.isocalendar()
status_t["week"] = _isos["year"].astype(str) + "-W" + _isos["week"].astype(int).map("{:02d}".format)


def build_table1(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "<p class='fig-caption'>No ventilated ICU patient-days in this period.</p>"
    h = sub.groupby("hospitalization_id").agg(
        age=("age_at_admit", "first"), sex=("sex_category", "first"),
        height=("height_cm", "first"), pbw=("pbw_kg", "first"),
        n_days=("calendar_day", "nunique")).reset_index()
    pu = (sub.groupby(["hospitalization_id", "assigned_unit"]).size().reset_index(name="n")
          .sort_values(["hospitalization_id", "n", "assigned_unit"], ascending=[True, False, True])
          .drop_duplicates("hospitalization_id"))
    h = h.merge(pu[["hospitalization_id", "assigned_unit"]], on="hospitalization_id", how="left")
    Nh = len(h)
    rows = [("__Age (years)__", med_iqr(h["age"])),
            ("__Sex__", np.nan),
            ("Male", npct(int((h["sex"] == "Male").sum()), Nh)),
            ("Female", npct(int((h["sex"] == "Female").sum()), Nh)),
            ("__Height (cm)__", med_iqr(h["height"])),
            ("__PBW (kg)__", med_iqr(h["pbw"])),
            ("__IMV ICU patient-days per hospitalization__", med_iqr(h["n_days"])),
            ("__Primary ICU unit__", np.nan)]
    for u in [u for u in UNIT_ORDER_REST if u in set(h["assigned_unit"])]:
        rows.append((UNIT_LABEL.get(u, u), npct(int((h["assigned_unit"] == u).sum()), Nh)))
    return render_gtsummary_table_html(pd.DataFrame(rows, columns=["Characteristic", f"**Overall**\nN = {Nh:,}"]))


def headline(sub: pd.DataFrame) -> dict:
    return {"n_patient_days": int(len(sub)), "n_hosps": int(sub["hospitalization_id"].nunique()),
            "n_patients": int(sub["patient_id"].nunique())}


# Keyed by severity ("all" + strata) × period (all / year / month) — 4 × 92 aggregated tables.
table1, period_headline = {}, {}
for sk in ["all"] + SEVS:
    base = status_t if sk == "all" else status_t[status_t["severity"] == sk]
    table1[sk], period_headline[sk] = {}, {}
    table1[sk]["all"] = build_table1(base); period_headline[sk]["all"] = headline(base)
    for y in years:
        sy = base[base["year"] == y]
        table1[sk][y] = build_table1(sy); period_headline[sk][y] = headline(sy)
    for mo in months:
        sm = base[base["month"] == mo]
        table1[sk][mo] = build_table1(sm); period_headline[sk][mo] = headline(sm)

cohort_headline = {
    **headline(status_t),
    "day_min": str(pd.to_datetime(status["calendar_day"]).min().date()),
    "day_max": str(pd.to_datetime(status["calendar_day"]).max().date()),
}
table1_html = table1["all"]["all"]  # initial render (all-time, all-severity)

# Per-ISO-week Table 1 + headline (all-severity) for the full weekly view.
wtable1, wheadline = {}, {}
for w, sw in status_t.groupby("week"):
    if w in wkidx:
        wtable1[w] = build_table1(sw)
        wheadline[w] = headline(sw)

payload = jsonable({
    "params": {"vt_grid": VT_GRID, "vt_default": VT_DEFAULT, "plateau_max": PLATEAU_MAX,
               "dp_max": DP_MAX, "adherence_fraction": agg["params"]["adherence_fraction"],
               "min_assessable_min": agg["params"]["min_assessable_min"]},
    "months": months, "years": years, "units": units, "days": days, "severity_strata": SEVS,
    "day_week": day_week, "week_label": week_label, "weeks": weeks_list,
    "wtot": wtot, "wass": wass, "wad": wad, "wstc": wstc, "whist": whist,
    "wtable1": wtable1, "wheadline": wheadline,
    "unit_label": UNIT_LABEL, "measure_label": MEASURE_LABEL,
    "cohort_headline": cohort_headline, "period_headline": period_headline,
    "tot": tot, "ass": ass, "ad": ad, "stc": stc, "vtd": vtd, "std": std,
    "histc": histc, "table1": table1,
})

# ----------------------------------------------------------------------------
# 5. Vendor Plotly (cache, else download once)
# ----------------------------------------------------------------------------

plotly_path = VENDOR / "plotly.min.js"
if not plotly_path.exists():
    try:
        # Preferred: offline bundle shipped with the plotly Python package (no network).
        from plotly.offline import get_plotlyjs
        plotly_path.write_text(get_plotlyjs())
        print(f"[5] Vendored Plotly from plotly package → {plotly_path}")
    except Exception:
        print(f"[5] Downloading Plotly once → {plotly_path} ...")
        try:
            urllib.request.urlretrieve(PLOTLY_URL, plotly_path)
        except Exception as e:
            raise SystemExit(f"Could not vendor Plotly (no package, no cache, download failed): {e}\n"
                             f"`pip install plotly` or place plotly.min.js at {plotly_path} and re-run.")
plotly_js = plotly_path.read_text()
print(f"[5] Plotly inlined ({len(plotly_js)/1e6:.1f} MB)")

# CLIF brand logo for the header (icon + wordmark), base64-embedded + CSS-scaled.
# Falls back to the text breadcrumb alone if the PNG is absent (so clone sites still build).
import base64
from io import BytesIO

IMG_DIR = ROOT / "assets"


def _load_logo(stem="clif_logo_v2", px=480):
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


LOGO_IMG = _load_logo()
print(f"[img] brand logo embedded: {'yes' if LOGO_IMG else 'no (breadcrumb only)'}")

# ----------------------------------------------------------------------------
# 6. Assemble HTML
# ----------------------------------------------------------------------------

print("[6] Assembling HTML ...")

CSS = """
:root{--maroon:#8a1f2b;--maroon-d:#6f1622;--cream:#f6efe9;--card:#fffdfb;--ink:#3a2c2c;
--muted:#9a8c86;--line:#ece1d9;--good:#2f7d5b;--warn:#b5852a;--bad:#a23b3b;--bar:#efe4dc;
--rowalt:#faf5f1;}
*{box-sizing:border-box}
body{font-family:Inter,-apple-system,'Segoe UI',system-ui,sans-serif;font-size:14px;
line-height:1.55;color:var(--ink);background:var(--cream);margin:0;}
.wrap{max-width:1600px;margin:0 auto;background:var(--card);padding:0 48px 48px;
box-shadow:0 1px 3px rgba(120,30,40,.08);}
header.sticky{position:sticky;top:0;z-index:50;background:var(--card);border-bottom:1px solid var(--line);
padding:18px 0 14px;margin-bottom:24px;}
.brandrow{display:flex;align-items:center;gap:18px;}
.brandrow img.logo{height:56px;width:auto;display:block;flex:0 0 auto;}
.titleblock{min-width:0;}
.backlink{display:inline-block;font-size:12.5px;color:var(--maroon);text-decoration:none;font-weight:700;margin-bottom:4px;}
.backlink:hover{text-decoration:underline;}
h1{font-size:27px;font-weight:800;letter-spacing:-.3px;color:var(--maroon-d);margin:0 0 4px;}
h2{font-size:19px;font-weight:700;color:var(--maroon-d);border-bottom:1px solid var(--line);
padding-bottom:8px;margin:0 0 18px;}
h3{font-size:15px;font-weight:600;margin:0 0 6px;color:var(--ink);}
.sub{color:var(--muted);font-size:13px;margin:0;}
.slider-bar{display:flex;align-items:center;gap:16px;margin-top:14px;padding:12px 16px;
background:var(--cream);border:1px solid var(--line);border-radius:10px;}
.slider-bar label{font-weight:700;color:var(--maroon);font-size:14px;white-space:nowrap;}
.slider-bar input[type=range]{flex:1;accent-color:var(--maroon);max-width:480px;}
.slider-bar .val{font-variant-numeric:tabular-nums;font-weight:800;font-size:18px;
color:var(--maroon-d);min-width:64px;}
.slider-bar .hint{color:var(--muted);font-size:12px;}
.slider-bar select{padding:7px 11px;border:1px solid var(--line);border-radius:7px;font-size:14px;
background:var(--card);color:var(--ink);font-weight:600;cursor:pointer;}
.slider-bar select:disabled{opacity:.5;}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px;}
.tab{padding:9px 17px;border-radius:999px;border:1px solid var(--line);background:var(--card);
font-size:13.5px;font-weight:600;cursor:pointer;color:var(--ink);}
.tab:hover{background:var(--cream);}
.tab.active{background:var(--maroon);color:#fff;border-color:var(--maroon);
box-shadow:0 1px 3px rgba(138,31,43,.3);}
.panel{display:none;}.panel.active{display:block;}
.section{padding:28px;border:1px solid var(--line);border-radius:14px;background:var(--card);
box-shadow:0 3px 10px rgba(120,30,40,.05);margin-bottom:32px;}
.bignum{display:flex;gap:40px;flex-wrap:wrap;margin:8px 0 20px;}
.bignum .stat .v{font-size:40px;font-weight:800;color:var(--maroon);
font-variant-numeric:tabular-nums;line-height:1;}
.bignum .stat .l{font-size:12px;color:var(--muted);text-transform:uppercase;
letter-spacing:.5px;margin-top:6px;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:8px 0 18px;}
.card{border:1px solid var(--line);border-radius:14px;padding:18px;background:var(--card);}
.card .name{font-size:13px;font-weight:700;color:var(--maroon-d);}
.card .rate{font-size:30px;font-weight:800;color:var(--maroon);font-variant-numeric:tabular-nums;margin:6px 0 2px;}
.card .meta{font-size:12px;color:var(--muted);}
.card.fixed .rate{color:var(--muted);}
.controls{display:flex;gap:18px;align-items:center;margin-bottom:14px;flex-wrap:wrap;}
.controls select{padding:7px 11px;border:1px solid var(--line);border-radius:7px;font-size:14px;
background:var(--card);color:var(--ink);font-weight:600;}
.controls label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.fig-caption{font-size:13px;color:var(--muted);margin:6px 0 0;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px;}
.results-table{border-collapse:collapse;width:auto;font-size:13px;margin-top:8px;}
.results-table thead th{background:var(--cream);border-bottom:2px solid var(--maroon-d);
padding:9px 12px;text-align:left;font-weight:700;color:var(--maroon-d);}
.results-table tbody td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;}
.results-table tbody tr:nth-child(even){background:var(--rowalt);}
.narr{background:var(--cream);border:1px solid var(--line);border-left:3px solid var(--maroon);
border-radius:10px;padding:14px 18px;font-size:13px;color:var(--ink);margin-bottom:18px;}
.ts{color:var(--muted);font-size:11px;margin-top:28px;}
"""

APP_JS = r"""
const P = JSON.parse(document.getElementById('payload').textContent);
const COLORS = {"__ALL__":"#8a1f2b","medical_icu":"#0072B2","mixed_cardiothoracic_icu":"#E69F00",
"surgical_icu":"#009E73","mixed_neuro_icu":"#CC79A7","general_icu":"#882255","burn_icu":"#44AA99"};
const MNAME = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const PCT = v => (v==null||isNaN(v)) ? '—' : (v*100).toFixed(1)+'%';
const CK = c => Number(c).toFixed(1);
const FONT = {family:"Inter, system-ui, sans-serif", size:12, color:"#3a2c2c"};
const baseLayout = extra => Object.assign({font:FONT, margin:{l:54,r:18,t:28,b:44},
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", hovermode:"x unified",
  yaxis:{tickformat:".0%", rangemode:"tozero", gridcolor:"#ece1d9"},
  xaxis:{gridcolor:"#f6efe9"}, legend:{font:{size:11}}}, extra||{});
const CFG = {displayModeBar:false, responsive:true};
let state = {cutoff:P.params.vt_default, trendMeasure:"vt", year:"all", month:"all", week:"all", severity:"all"};
let active = "p-vt";
function sevList(){ return state.severity==="all" ? P.severity_strata : [state.severity]; }
// ISO-week → day-index map (lean weekly = site-wide, from the daily data).
const WEEKDAYS = {}; (P.day_week||[]).forEach((wk,i)=>{ (WEEKDAYS[wk]=WEEKDAYS[wk]||[]).push(i); });
function isWeek(){ return state.year!=="all" && state.week!=="all"; }
function widx(){ return P.weeks.indexOf(state.week); }
// Per-(unit, week) rate from the weekly count arrays (all-severity, slider-aware for vt/comp).
function mRateW(measure, unit){
  const i=widx(); if(i<0) return rates(0,0,0);
  if(measure==="vt"||measure==="comp"){
    const ck=CK(state.cutoff);
    return rates(P.wad[unit][measure][ck][i], P.wass[unit][measure][i], P.wtot[unit][i]);
  }
  const o=P.wstc[measure][unit]; return rates(o.ad[i], o.ass[i], o.tot[i]);
}
function weekN(){ const i=widx(); return i<0 ? 0 : P.wtot["__ALL__"][i]; }
const SEV_LABEL = {all:"all severity", severe:"severe resp failure", not_severe:"not severe", unknown:"unknown (no usable O₂)"};

// ---------- Period helpers ----------
function periodKey(){
  if(state.year==="all") return "all";
  if(state.month==="all") return state.year;
  return state.year+"-"+state.month;
}
function periodLabel(){
  if(state.year==="all") return "all time";
  if(isWeek()) return (P.week_label[state.week]||state.week);
  if(state.month==="all") return state.year;
  return MNAME[parseInt(state.month,10)-1]+" "+state.year;
}
function periodIdxs(){
  if(state.year==="all") return P.months.map((_,i)=>i);
  if(state.month==="all") return P.months.map((m,i)=>m.startsWith(state.year)?i:-1).filter(i=>i>=0);
  const i=P.months.indexOf(state.year+"-"+state.month); return i>=0?[i]:[];
}
function periodSpan(idxs){ return idxs.length ? [Math.min(...idxs), Math.max(...idxs)] : null; }

// ---------- Count summing -> rates (over period months × selected severity strata) ----------
function rates(ad, ass, tot){ return {ar:ass? ad/ass:null, pct:tot? ass/tot:null, crude:tot? ad/tot:null}; }
function mRate(measure, unit, idxs){
  const sevs=sevList(); let ad=0,ass=0,tot=0;
  if(measure==="vt"||measure==="comp"){
    const ck=CK(state.cutoff);
    for(const s of sevs){
      const adA=P.ad[unit][measure][ck][s], asA=P.ass[unit][measure][s], toA=P.tot[unit][s];
      for(const i of idxs){ ad+=adA[i]; ass+=asA[i]; tot+=toA[i]; }
    }
  } else {
    for(const s of sevs){ const o=P.stc[measure][unit][s]; for(const i of idxs){ ad+=o.ad[i]; ass+=o.ass[i]; tot+=o.tot[i]; } }
  }
  return rates(ad,ass,tot);
}
// Full monthly series for a measure/unit (sum severity per month; trends keep the whole timeline)
function trendSeries(measure, unit){
  const sevs=sevList(), n=P.months.length, y=new Array(n), isVt=(measure==="vt"||measure==="comp"), ck=CK(state.cutoff);
  for(let i=0;i<n;i++){
    let ad=0,ass=0;
    if(isVt){ for(const s of sevs){ ad+=P.ad[unit][measure][ck][s][i]; ass+=P.ass[unit][measure][s][i]; } }
    else { for(const s of sevs){ const o=P.stc[measure][unit][s]; ad+=o.ad[i]; ass+=o.ass[i]; } }
    y[i] = ass ? ad/ass : null;
  }
  return y;
}
function unitTraces(measure){
  return P.units.map(u=>({
    x:P.months, y:trendSeries(measure,u), name:P.unit_label[u]||u, mode:"lines",
    line:{color:COLORS[u]||"#888", width:(u==="__ALL__")?3:1.3},
    opacity:(u==="__ALL__")?1:0.85, connectgaps:false
  }));
}
function monthAfter(key){ const p=key.split('-'); const d=new Date(Date.UTC(+p[0], +p[1], 1)); return d.toISOString().slice(0,10); }
function isYear(){ return state.year!=="all" && state.month==="all"; }
function isMonth(){ return state.year!=="all" && state.month!=="all"; }

// Daily site-wide series (for the single-month zoom).
let DAYIDX = null;
function dayIdx(){ if(!DAYIDX){ DAYIDX={}; P.days.forEach((d,i)=>DAYIDX[d]=i); } return DAYIDX; }
function dailySeries(measure, dayKeys){
  const ck=CK(state.cutoff);
  const o = (measure==="vt"||measure==="comp") ? P.vtd[ck][measure] : P.std[measure];
  const di=dayIdx();
  return dayKeys.map(d=>{ const i=di[d]; return (i!=null && o.ass[i]) ? o.ad[i]/o.ass[i] : null; });
}
// Build trend traces + x-axis range depending on the selected period granularity.
// allOnly = just the combined "All ICUs" line (used by the Vt headline tab).
function dayPlus(s){ const d=new Date(s+'T00:00:00Z'); d.setUTCDate(d.getUTCDate()+1); return d.toISOString().slice(0,10); }
function buildTrend(measure, allOnly){
  if(isWeek()){
    const dk=(WEEKDAYS[state.week]||[]).map(i=>P.days[i]);
    return {traces:[{x:dk, y:dailySeries(measure,dk), name:"All ICUs (daily)", mode:"lines+markers",
                     line:{color:"#6f1622",width:2}, marker:{size:6,color:"#8a1f2b"}, connectgaps:false}],
            xrange: dk.length?[dk[0], dayPlus(dk[dk.length-1])]:null, daily:true};
  }
  if(isMonth()){
    const mk=state.year+"-"+state.month;
    const dk=P.days.filter(d=>d.slice(0,7)===mk);
    return {traces:[{x:dk, y:dailySeries(measure,dk), name:"All ICUs (daily)", mode:"lines+markers",
                     line:{color:"#6f1622",width:2}, marker:{size:5,color:"#8a1f2b"}, connectgaps:false}],
            xrange:[mk+"-01", monthAfter(mk)], daily:true};
  }
  const traces = allOnly
    ? [{x:P.months, y:trendSeries(measure,"__ALL__"), name:"All ICUs", mode:"lines",
        line:{color:COLORS["__ALL__"],width:2.5}, connectgaps:false}]
    : unitTraces(measure);
  const xrange = isYear() ? [state.year+"-01-01", monthAfter(state.year+"-12")] : null;
  return {traces, xrange, daily:false};
}

// ---------- Header / table / big-number text ----------
function setText(){
  const idxs=periodIdxs();
  document.getElementById('cutoff-readout').textContent = Number(state.cutoff).toFixed(1);
  const R = m => isWeek() ? mRateW(m,"__ALL__") : mRate(m,"__ALL__",idxs);
  const vt=R("vt"), comp=R("comp"), plat=R("plat"), dp=R("dp");
  document.getElementById('vt-big').textContent = PCT(vt.ar);
  document.getElementById('vt-pct').textContent = PCT(vt.pct);
  document.getElementById('vt-crude').textContent = PCT(vt.crude);
  document.getElementById('cb-vt-rate').textContent = PCT(vt.ar);
  document.getElementById('cb-vt-pct').textContent = "assessable on "+PCT(vt.pct);
  document.getElementById('cb-plat-rate').textContent = PCT(plat.ar);
  document.getElementById('cb-plat-pct').textContent = "assessable on "+PCT(plat.pct);
  document.getElementById('cb-dp-rate').textContent = PCT(dp.ar);
  document.getElementById('cb-dp-pct').textContent = "assessable on "+PCT(dp.pct);
  document.getElementById('cb-comp-rate').textContent = PCT(comp.ar);
  document.getElementById('cb-comp-pct').textContent = "assessable on "+PCT(comp.pct);
}
function setHeaderAndTable(){
  const k=periodKey(), sv=state.severity, lbl=periodLabel();
  document.getElementById('period-readout').textContent = lbl;
  if(isWeek()){
    const ph=P.wheadline[state.week]||P.cohort_headline;
    document.getElementById('cohort-line').textContent =
      ph.n_patient_days.toLocaleString()+" IMV-on-ICU patient-days · "+
      ph.n_hosps.toLocaleString()+" hospitalizations · "+ph.n_patients.toLocaleString()+" patients ("+lbl+")";
    document.getElementById('table1-box').innerHTML = P.wtable1[state.week] || P.table1["all"]["all"];
    document.getElementById('t1-period').textContent = " — "+lbl;
    return;
  }
  const sevTxt = (sv==="all") ? "" : " · "+SEV_LABEL[sv];
  const ph=(P.period_headline[sv]||P.period_headline["all"])[k] || P.cohort_headline;
  document.getElementById('cohort-line').textContent =
    ph.n_patient_days.toLocaleString()+" IMV-on-ICU patient-days · "+
    ph.n_hosps.toLocaleString()+" hospitalizations · "+ph.n_patients.toLocaleString()+" patients ("+lbl+sevTxt+")";
  document.getElementById('table1-box').innerHTML = (P.table1[sv]||P.table1["all"])[k] || P.table1["all"]["all"];
  document.getElementById('t1-period').textContent = " — "+lbl+sevTxt;
}

// ---------- Per-panel draw (panel visible when called) ----------
function drawVt(){
  const t=buildTrend("vt", true);   // headline: combined All-ICUs curve only
  Plotly.react('vt-trend', t.traces, baseLayout({showlegend:false, xaxis:{range:t.xrange, gridcolor:"#f6efe9",
    title:t.daily?{text:"Daily — "+periodLabel(),font:{size:11}}:undefined}}), CFG);
}
function drawComp(){
  const idxs=periodIdxs(), ms=["vt","plat","dp","comp"];
  const rs=ms.map(m=> (isWeek()?mRateW(m,"__ALL__"):mRate(m,"__ALL__",idxs)).ar);
  Plotly.react('cb-bar', [{x:ms.map(m=>P.measure_label[m]), y:rs, type:"bar",
    marker:{color:["#8a1f2b","#9a8c86","#9a8c86","#8a1f2b"]}, text:rs.map(PCT),
    textposition:"outside", cliponaxis:false}],
    baseLayout({margin:{l:54,r:18,t:10,b:60}}), CFG);
}
function drawTrends(){
  const m=state.trendMeasure, idxs=periodIdxs(), isVt=(m==="vt"||m==="comp");
  // Trend = per-unit MONTHLY lines, x-axis zoomed to the selected period's year (full for all-time).
  const xr = (state.year==="all") ? null : [state.year+"-01-01", monthAfter(state.year+"-12")];
  Plotly.react('tr-trend', unitTraces(m), baseLayout({xaxis:{range:xr, gridcolor:"#f6efe9"}}), CFG);
  // Bar = the exact selected period (month / week / year / all).
  const us=P.units.filter(u=>u!=="__ALL__");
  const rate=u=> (isWeek()?mRateW(m,u):mRate(m,u,idxs)).ar;
  Plotly.react('tr-bar', [{x:us.map(u=>P.unit_label[u]||u), y:us.map(rate), type:"bar",
    marker:{color:us.map(u=>COLORS[u]||"#888")}, text:us.map(u=>PCT(rate(u))),
    textposition:"outside", cliponaxis:false}],
    baseLayout({margin:{l:54,r:18,t:10,b:90}}), CFG);
  document.getElementById('tr-trend-title').textContent =
    "Per-unit monthly adherence" + (state.year==="all" ? " · all time" : " · "+state.year);
  document.getElementById('tr-bar-title').textContent = "By unit · " + periodLabel();
  document.getElementById('tr-note').textContent =
    (isVt ? ("Vt cutoff "+Number(state.cutoff).toFixed(1)+" mL/kg") : "Fixed threshold")
    + (state.severity!=="all" ? " · "+SEV_LABEL[state.severity] : "");
}
function drawDist(){
  const idxs=periodIdxs(), sevs=sevList(), wk=isWeek(), wi=widx();
  ["vt_per_pbw","plateau","driving_pressure","peep","fio2"].forEach(col=>{
    const h=P.histc[col], n=h.centers.length;
    let tot=0; const agg=new Array(n).fill(0);
    if(wk){ const row=(P.whist[col]||[])[wi]||[]; for(let b=0;b<n;b++) agg[b]=row[b]||0; }
    else { for(const s of sevs){ const C=h.counts[s];
      for(const i of idxs){ const row=C[i]; for(let b=0;b<n;b++){agg[b]+=row[b];} } } }
    for(const v of agg) tot+=v;
    const frac=agg.map(v=> tot? v/tot : 0);
    const thr=(h.threshold==="slider")?state.cutoff:h.threshold;
    const shapes=(thr==null)?[]:[{type:"line",x0:thr,x1:thr,yref:"paper",y0:0,y1:1,line:{color:"#b5852a",width:2,dash:"dash"}}];
    Plotly.react("hist-"+col, [{x:h.centers,y:frac,type:"bar",marker:{color:"#8a1f2b"},
      hovertemplate:"%{x}: %{y:.1%}<extra></extra>"}],
      baseLayout({margin:{l:54,r:14,t:24,b:42}, shapes:shapes,
        xaxis:{title:{text:h.title,font:{size:11}},gridcolor:"#f6efe9"},
        yaxis:{tickformat:".0%",gridcolor:"#ece1d9"}}), CFG);
  });
}
const DRAW = {"p-vt":drawVt, "p-comp":drawComp, "p-trend":drawTrends, "p-dist":drawDist};

function showPanel(id, writeHash){
  if(!DRAW[id]) id="p-vt";
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active', x.dataset.tab===id));
  document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active', x.id===id));
  active = id;
  // Only reflect the panel in the URL hash on an actual tab CLICK (which happens post-load,
  // so replaceState there never triggers a scroll). Writing the hash during init would add a
  // #panel fragment that the browser "scroll-to-fragment"s at the load event — landing ~285px
  // down, with the header/tabs pushed under the sticky bar. That was the real bug.
  if(writeHash){ try{ history.replaceState(null, '', '#'+id); }catch(e){} }
  window.scrollTo(0, 0);
  DRAW[id]();
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>showPanel(t.dataset.tab, true));

// ---------- Controls ----------
document.getElementById('vt-slider').oninput = e => {
  state.cutoff = parseFloat(e.target.value); setText(); DRAW[active]();
};
document.getElementById('tr-measure').onchange = e => {
  state.trendMeasure = e.target.value; if(active==="p-trend") drawTrends();
};
const yearSel=document.getElementById('sel-year'), monthSel=document.getElementById('sel-month'),
      weekSel=document.getElementById('sel-week'), sevSel=document.getElementById('sel-severity');
function populateWeeks(yr){
  let opts='<option value="all">All weeks</option>';
  if(yr!=="all"){ for(const w of P.weeks){ if(w.startsWith(yr+"-W")) opts+=`<option value="${w}">${P.week_label[w]||w}</option>`; } }
  weekSel.innerHTML=opts;
}
function applyPeriod(){
  state.year=yearSel.value;
  monthSel.disabled = weekSel.disabled = (state.year==="all");
  if(state.year==="all"){ monthSel.value="all"; weekSel.value="all"; }
  state.month=monthSel.value; state.week=weekSel.value;
  sevSel.disabled = isWeek();                          // weekly = all-severity (daily data isn't split)
  if(isWeek()){ sevSel.value="all"; state.severity="all"; }
  setText(); setHeaderAndTable(); DRAW[active]();
}
yearSel.onchange = ()=>{ monthSel.value="all"; weekSel.value="all"; populateWeeks(yearSel.value); applyPeriod(); };
monthSel.onchange = ()=>{ if(monthSel.value!=="all") weekSel.value="all"; applyPeriod(); };   // month & week mutually exclusive
weekSel.onchange  = ()=>{ if(weekSel.value!=="all") monthSel.value="all"; applyPeriod(); };
sevSel.onchange   = ()=>{ state.severity=sevSel.value; setText(); setHeaderAndTable(); DRAW[active](); };

// ---------- Init ----------
if(history.scrollRestoration){ history.scrollRestoration = 'manual'; }
monthSel.disabled = weekSel.disabled = true;
setText(); setHeaderAndTable();
var initId=(location.hash||"#p-vt").slice(1);
// Strip any incoming fragment BEFORE load completes so the browser's load-time
// "scroll to fragment" can't fire (e.g. arriving at lpv_dashboard.html#p-comp).
try{ history.replaceState(null, '', location.pathname+location.search); }catch(e){}
showPanel(initId, false);          // select + draw, but do NOT write the hash on init
// Backstop: re-assert the top after the browser's own load / anchor / bfcache scrolling,
// which can run after init. setTimeout(0) lands after the load event's synchronous scroll.
function toTop(){ window.scrollTo(0, 0); }
requestAnimationFrame(toTop);
window.addEventListener('load', ()=>{ toTop(); requestAnimationFrame(toTop); setTimeout(toTop, 0); });
window.addEventListener('pageshow', ()=>{ toTop(); requestAnimationFrame(toTop); });
"""

# Big-number / card / hist DOM built server-side, charts filled by JS
hist_divs = "".join(
    f'<div><h3>{html.escape(HIST_SPEC[c]["title"])}</h3><div id="hist-{c}" style="height:260px"></div></div>'
    for c in ["vt_per_pbw", "plateau", "driving_pressure", "peep", "fio2"]
)
unit_opts = "".join(f'<option value="{m}">{html.escape(MEASURE_LABEL[m])}</option>' for m in ["vt", "plat", "dp", "comp"])
year_opts = '<option value="all">All years</option>' + "".join(f'<option value="{y}">{y}</option>' for y in years)
_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_opts = '<option value="all">All months</option>' + "".join(f'<option value="{i:02d}">{_MON[i-1]}</option>' for i in range(1, 13))
sev_opts = ('<option value="all">All severity</option>'
            '<option value="severe">Severe resp failure</option>'
            '<option value="not_severe">Not severe</option>'
            '<option value="unknown">Unknown (no O₂)</option>')
ch = cohort_headline

BODY = f"""
<div class="wrap">
<header class="sticky">
  <div class="brandrow">
    {f'<img class="logo" src="{LOGO_IMG}" alt="CLIF">' if LOGO_IMG else ''}
    <div class="titleblock">
      <a class="backlink" href="scorecard.html">← CLIF ICU Ventilator QI Bundle</a>
      <h1>Lung-Protective Ventilation Adherence — {SITE}</h1>
      <p class="sub" id="cohort-line">{ch['n_patient_days']:,} IMV-on-ICU patient-days · {ch['n_hosps']:,} hospitalizations · {ch['n_patients']:,} patients (all time)</p>
      <p class="sub" style="margin-top:1px">{ch['day_min']} → {ch['day_max']} · Descriptive; component-separated, each measure on its own denominator.</p>
    </div>
  </div>
  <div class="slider-bar">
    <label>Tidal volume cutoff</label>
    <input type="range" id="vt-slider" min="{VT_GRID[0]}" max="{VT_GRID[-1]}" step="0.5" value="{VT_DEFAULT}">
    <span class="val"><span id="cutoff-readout">{VT_DEFAULT:.1f}</span> mL/kg</span>
    <span class="hint">Plateau ≤ {PLATEAU_MAX:.0f} &amp; ∆P ≤ {DP_MAX:.0f} fixed</span>
    <span style="flex:1 1 24px"></span>
    <label>Severity</label>
    <select id="sel-severity" title="Severe respiratory failure = P/F<300 or S/F<315 (SpO₂≤97%) with PEEP>5">{sev_opts}</select>
    <label>Period</label>
    <select id="sel-year">{year_opts}</select>
    <select id="sel-month">{month_opts}</select>
    <select id="sel-week"><option value="all">All weeks</option></select>
    <span class="val" style="font-size:13px;min-width:90px">📅 <span id="period-readout">all time</span></span>
  </div>
</header>

<div class="tabs">
  <button class="tab active" data-tab="p-vt">Tidal Volume</button>
  <button class="tab" data-tab="p-comp">Component breakdown</button>
  <button class="tab" data-tab="p-trend">By unit &amp; over time</button>
  <button class="tab" data-tab="p-dist">Distributions &amp; cohort</button>
</div>

<div id="p-vt" class="panel active"><div class="section">
  <h2>Tidal Volume</h2>
  <div class="bignum">
    <div class="stat"><div class="v" id="vt-big">—</div><div class="l">Vt assessable adherence</div></div>
    <div class="stat"><div class="v" id="vt-pct">—</div><div class="l">% of patient-days assessable</div></div>
    <div class="stat"><div class="v" id="vt-crude">—</div><div class="l">crude adherence</div></div>
  </div>
  <p class="fig-caption">Among Vt-assessable patient-days (Vt + PBW present, mode-eligible IMV), the % with ≥80% of assessable time at Vt/kg ≤ the chosen cutoff. Move the slider above.</p>
  <div id="vt-trend" style="height:420px"></div>
  <p class="fig-caption">Monthly Vt assessable-adherence, all ICUs combined. (Per-unit lines are on the "By unit &amp; over time" tab.)</p>
</div></div>

<div id="p-comp" class="panel"><div class="section">
  <h2>Component breakdown</h2>
  <div class="narr">Separating the bundle shows what the composite hides: <strong>tidal volume</strong> is the low, tunable lever; <strong>plateau</strong> is rarely the clinical problem (≈86% pass when measured) — its drag on the composite is a documentation gap; <strong>driving pressure</strong> is the real pressure limiter (≈48%).</div>
  <div class="cards">
    <div class="card"><div class="name">Tidal volume (Vt/kg)</div><div class="rate" id="cb-vt-rate">—</div><div class="meta" id="cb-vt-pct"></div></div>
    <div class="card fixed"><div class="name">Plateau ≤ 30</div><div class="rate" id="cb-plat-rate">—</div><div class="meta" id="cb-plat-pct"></div></div>
    <div class="card fixed"><div class="name">Driving pressure ≤ 15</div><div class="rate" id="cb-dp-rate">—</div><div class="meta" id="cb-dp-pct"></div></div>
    <div class="card"><div class="name">Composite (all three)</div><div class="rate" id="cb-comp-rate">—</div><div class="meta" id="cb-comp-pct"></div></div>
  </div>
  <div id="cb-bar" style="height:360px"></div>
  <p class="fig-caption">Assessable adherence by measure. Vt and composite move with the slider; plateau and ∆P are fixed.</p>
</div></div>

<div id="p-trend" class="panel"><div class="section">
  <h2>By ICU unit &amp; over time</h2>
  <div class="controls">
    <span><label>Measure</label> <select id="tr-measure">{unit_opts}</select></span>
    <span class="fig-caption" id="tr-note"></span>
  </div>
  <div class="grid2">
    <div><h3 id="tr-trend-title">Per-unit monthly adherence</h3><div id="tr-trend" style="height:380px"></div></div>
    <div><h3 id="tr-bar-title">By unit</h3><div id="tr-bar" style="height:380px"></div></div>
  </div>
</div></div>

<div id="p-dist" class="panel">
  <div class="section">
    <h2>Settings distributions <span class="fig-caption" style="font-weight:400">(time-weighted by minutes on IMV)</span></h2>
    <div class="grid3">{hist_divs}</div>
    <p class="fig-caption">Gold dashed line = threshold. Tidal-volume line follows the slider; plateau (30) and ∆P (15) are fixed.</p>
  </div>
  <div class="section">
    <h2>Cohort — Table 1<span id="t1-period" style="font-weight:400;color:#9a8c86"></span></h2>
    <p class="fig-caption">Hospitalizations with ≥1 ventilated ICU patient-day in the selected period.</p>
    <div id="table1-box">{table1_html}</div>
  </div>
</div>

<p class="ts">Generated {{GENERATED}} · LPV adherence pipeline (lpv) · {SITE} CLIF v{CLIF_VER}</p>
</div>
"""

HTML = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    f"<title>LPV Adherence — {SITE}</title>"
    "<style>@@CSS@@</style>"
    "<script>@@PLOTLY@@</script>"
    "</head><body>"
    + BODY.replace("{GENERATED}", datetime.now().isoformat(timespec="minutes"))
    + '<script id="payload" type="application/json">@@PAYLOAD@@</script>'
    + "<script>@@APPJS@@</script>"
    "</body></html>"
)
HTML = (HTML.replace("@@CSS@@", CSS)
        .replace("@@PLOTLY@@", plotly_js)
        .replace("@@PAYLOAD@@", json.dumps(payload, allow_nan=False))
        .replace("@@APPJS@@", APP_JS))

out_html = DASH_DIR / "lpv_dashboard.html"
out_html.write_text(HTML)
size_mb = out_html.stat().st_size / 1e6
print(f"\nWrote {out_html}  ({size_mb:.1f} MB)")

# ----------------------------------------------------------------------------
# 7. Verification
# ----------------------------------------------------------------------------

print("\n[verify]")
text = out_html.read_text()
m = re.search(r'<script id="payload"[^>]*>(.*?)</script>', text, re.S)
pl = json.loads(m.group(1))
c6 = f"{VT_DEFAULT:.1f}"


def alltime_rate(measure, cutoff_key=c6):
    """Reproduce 02_features assessable_rate: sum all months × all severity strata for __ALL__."""
    ad = ass = 0
    for s in pl["severity_strata"]:
        if measure in ("vt", "comp"):
            ad += sum(pl["ad"]["__ALL__"][measure][cutoff_key][s])
            ass += sum(pl["ass"]["__ALL__"][measure][s])
        else:
            o = pl["stc"][measure]["__ALL__"][s]
            ad += sum(o["ad"]); ass += sum(o["ass"])
    return ad / ass if ass else float("nan")


checks = {
    "vt@6 == 02_features (24.62%)": abs(alltime_rate("vt") - feat["per_measure"]["vt"]["assessable_rate"]) < 1e-6,
    "comp@6 == 02_features (11.28%)": abs(alltime_rate("comp") - feat["per_measure"]["comp"]["assessable_rate"]) < 1e-6,
    "plat == 02_features (85.98%)": abs(alltime_rate("plat") - feat["per_measure"]["plat"]["assessable_rate"]) < 1e-6,
    "dp == 02_features (48.24%)": abs(alltime_rate("dp") - feat["per_measure"]["dp"]["assessable_rate"]) < 1e-6,
    "slider endpoint vt@10 > 0.90": alltime_rate("vt", "10.0") > 0.90,
    "period selectors present": 'id="sel-year"' in text and 'id="sel-month"' in text,
    "severity selector present": 'id="sel-severity"' in text,
    "table1 nested by severity×period": (all(s in pl["table1"] for s in ["all"] + pl["severity_strata"])
                                         and "all" in pl["table1"]["all"]
                                         and pl["years"][0] in pl["table1"]["all"]
                                         and pl["months"][0] in pl["table1"]["all"]),
    "per-month hist counts present (per stratum)":
        all(len(pl["histc"][c]["counts"]["severe"]) == len(pl["months"]) for c in pl["histc"]),
    "daily site-wide data present": (len(pl["days"]) > 2000 and "6.0" in pl["vtd"] and "plat" in pl["std"]),
    "weekly full data present": (len(pl["weeks"]) > 100 and len(pl["day_week"]) == len(pl["days"])
                                 and 'id="sel-week"' in text and "__ALL__" in pl["wtot"]
                                 and "vt_per_pbw" in pl["whist"] and len(pl["wtable1"]) > 100),
    "plotly inlined": "Plotly" in text[:300000],
    "slider input present": 'id="vt-slider"' in text,
    "4 tab panels": text.count('class="panel') == 4,
    "no NaN literal in payload": ":NaN" not in m.group(1) and "undefined" not in m.group(1),
    "no hospitalization_id in payload": "hospitalization_id" not in m.group(1),
    "no patient_id in payload": "patient_id" not in m.group(1),
}
# Structural: the 4 panels must be siblings (equal div-nesting depth), not nested.
_body = text[text.find('<div id="p-vt"'):text.find('<script id="payload"')]
_depth, _at = 0, {}
for _t in re.finditer(r'<div\b|</div>|id="(p-(?:vt|comp|trend|dist))"', _body):
    s = _t.group(0)
    if s == "<div":
        _depth += 1
    elif s == "</div>":
        _depth -= 1
    else:
        _at[_t.group(1)] = _depth
checks["4 panels are siblings (not nested)"] = len(set(_at.values())) == 1 and len(_at) == 4
# Daily (all-severity) counts over a month must equal that month's monthly count summed over severity.
_mo = pl["months"][len(pl["months"]) // 2]
_mi = pl["months"].index(_mo)
_didx = [i for i, d in enumerate(pl["days"]) if d[:7] == _mo]
_daily_sum = sum(pl["vtd"]["6.0"]["vt"]["ad"][i] for i in _didx)
_monthly = sum(pl["ad"]["__ALL__"]["vt"]["6.0"][s][_mi] for s in pl["severity_strata"])
checks[f"daily≡monthly counts ({_mo})"] = (_daily_sum == _monthly)


def sev_rate(measure, sev, ck=c6):
    ad = sum(pl["ad"]["__ALL__"][measure][ck][sev]); ass = sum(pl["ass"]["__ALL__"][measure][sev])
    return ad / ass if ass else float("nan")


checks["severe != not_severe (vt@6)"] = abs(sev_rate("vt", "severe") - sev_rate("vt", "not_severe")) > 1e-6
for k, v in checks.items():
    print(f"  [{'ok' if v else 'XX'}] {k}")
print(f"\n  Vt slider (site-wide assessable): " +
      " ".join(f"{c:g}:{alltime_rate('vt', f'{c:.1f}')*100:.0f}%" for c in [4.0, 6.0, 8.0, 10.0]))
print(f"  Vt@6 by severity: severe {sev_rate('vt','severe')*100:.1f}% · "
      f"not_severe {sev_rate('vt','not_severe')*100:.1f}% · unknown {sev_rate('vt','unknown')*100:.1f}%")
print(f"  Per-period Table 1: {len(pl['table1']['all'])} periods × {len(pl['table1'])} severity keys")
assert all(checks.values()), "VERIFICATION FAILED"
print("\nAll checks passed. Done.")

