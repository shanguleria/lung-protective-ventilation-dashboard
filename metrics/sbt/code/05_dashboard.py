"""Render the SBT delivery QI dashboard (self-contained HTML).

CLIF maroon-cream house style (~/.claude/templates/dashboard_design_guide.md; lpv
scorecard/dashboard are the brand reference). One self-contained file: logo and
figures are base64-embedded so it ships as a single HTML for any site.

Components:
    - Brand header (logo lockup) + reactive headline donut (unit × period filters).
    - SBT-delivered-rate-over-time trend (reacts to filters).
    - SBT delivery by ICU unit (side by side).
    - Cohort flow funnel (vent-ICU days → non-trach → eligible → SBT).
    - Table 1 — eligible patients, ever-SBT vs never (gtsummary renderer).
    - Eligibility / data-quality caveat (amber info box).

Inputs (from 04_metrics.py / 03 / 01):
    output/intermediate/metrics_patient_day_level.parquet
    output/intermediate/metrics_slices.parquet
    output/intermediate/sbt_diag.json
    output/final/metrics_site_summary.csv

Output:
    output/final/sbt_dashboard.html
    output/final/graphs/cohort_consort.png/.svg
"""

from __future__ import annotations

import base64
import html
import importlib.util
import json as _json
import logging
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = PROJECT_ROOT / "code"
log = logging.getLogger("sbt.dashboard")

MAROON, MAROON_D, CREAM = "#8a1f2b", "#6f1622", "#f6efe9"
CARD, INK, MUTED, LINE, BAR = "#fffdfb", "#3a2c2c", "#9a8c86", "#ece1d9", "#efe4dc"
GOOD, WARN, BAD = "#2f7d5b", "#b5852a", "#a23b3b"

CATEGORICAL_DISPLAY = {
    "admission_type_category": {
        "ed": "Emergency dept.", "osh": "Outside-hospital transfer",
        "direct": "Direct admission", "facility": "Facility transfer",
    },
    "sex_category": {"male": "Male", "female": "Female"},
}
UNIT_LABELS = {
    "__ALL__": "All ICUs", "medical_icu": "Medical ICU",
    "mixed_cardiothoracic_icu": "Cardiothoracic ICU", "surgical_icu": "Surgical ICU",
    "mixed_neuro_icu": "Neuro ICU", "general_icu": "General ICU", "burn_icu": "Burn ICU",
    "unknown": "Unknown unit",
}
GRAN_LABELS = {"all": "All-time", "month": "Monthly", "week": "Weekly"}


def _period_label(key: str) -> str:
    import datetime as _dt
    try:
        if "-W" in key:
            y, w = key.split("-W")
            d = _dt.date.fromisocalendar(int(y), int(w), 1)
            return f"Week {int(w)} · {d.strftime('%b %Y')}"
        return _dt.datetime.strptime(key + "-01", "%Y-%m-%d").strftime("%b %Y")
    except Exception:
        return key


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sbt_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# --- embedding ---
def _load_logo(px: int = 480):
    for p in (BUNDLE_ROOT / "assets" / "clif_logo_v2.png",
              PROJECT_ROOT / "references" / "images" / "clif_logo_v2.png"):
        if p.exists():
            try:
                from PIL import Image
                im = Image.open(p).convert("RGBA"); im.thumbnail((px, px))
                buf = BytesIO(); im.save(buf, format="PNG", optimize=True)
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                return None
    return None


def _fig_to_uri(fig) -> str:
    import matplotlib.pyplot as plt
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=CARD)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --- gtsummary renderer (verbatim from the design guide) ---
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
            f"<thead><tr>{header_row}</tr></thead><tbody>" + "\n".join(body_rows) + "</tbody></table>")


# --- Table 1: eligible patients, ever-SBT vs never ---
def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_med(s) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "—" if s.empty else f"{s.median():.1f} ({s.quantile(.25):.1f}, {s.quantile(.75):.1f})"


def _fmt_np(n, d) -> str:
    return f"{n:,} ({100*n/d:.1f}%)" if d else "—"


def _display(col, val) -> str:
    if pd.isna(val):
        return "Unknown"
    raw = str(val)
    return CATEGORICAL_DISPLAY.get(col, {}).get(raw.lower(), raw if raw else "Unknown")


def build_patient_table(obs: pd.DataFrame) -> pd.DataFrame:
    elig = obs[obs["eligible"]].copy()
    if "patient_id" not in elig.columns:
        return pd.DataFrame()
    elig["__sbt_day"] = elig["sbt_delivered"].astype(bool)
    agg = {"__sbt_day": "any", "icu_day": "count"}
    for c in ("age_at_admission", "sex_category", "race_category", "ethnicity_category",
              "admission_type_category", "discharge_category"):
        if c in elig.columns:
            agg[c] = "first"
    pt = elig.groupby("patient_id").agg(agg).rename(
        columns={"__sbt_day": "ever_sbt", "icu_day": "n_eligible_days"}).reset_index()
    if "discharge_category" in pt.columns:
        pt["in_hospital_mortality"] = pt["discharge_category"].astype("string").str.lower().eq("expired")
    return pt


def build_table1(pt: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    groups = {"sbt": pt[pt["ever_sbt"]], "no": pt[~pt["ever_sbt"]]}
    n_all, n_y, n_n = len(pt), len(groups["sbt"]), len(groups["no"])
    cols = ["**Characteristic**", f"**Overall**\nN = {n_all:,}",
            f"**Ever SBT**\nN = {n_y:,}", f"**Never SBT**\nN = {n_n:,}", "**p-value**"]
    rows = []

    def add_cont(label, col):
        if col not in pt.columns:
            return
        a = pd.to_numeric(groups["sbt"][col], errors="coerce").dropna()
        b = pd.to_numeric(groups["no"][col], errors="coerce").dropna()
        p = stats.kruskal(a, b).pvalue if len(a) and len(b) else np.nan
        rows.append([f"__{label}__", _fmt_med(pt[col]), _fmt_med(groups["sbt"][col]),
                     _fmt_med(groups["no"][col]), _fmt_p(p)])

    def add_binary(label, col):
        if col not in pt.columns:
            return
        a = groups["sbt"][col].astype(bool); b = groups["no"][col].astype(bool)
        ct = np.array([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
        try:
            p = stats.chi2_contingency(ct)[1]
        except ValueError:
            p = np.nan
        rows.append([f"__{label}__", _fmt_np(int(pt[col].sum()), n_all),
                     _fmt_np(int(a.sum()), n_y), _fmt_np(int(b.sum()), n_n), _fmt_p(p)])

    def add_cat(label, col):
        if col not in pt.columns:
            return
        da = pt[col].map(lambda v: _display(col, v))
        dy = groups["sbt"][col].map(lambda v: _display(col, v))
        dn = groups["no"][col].map(lambda v: _display(col, v))
        levels = sorted(da.dropna().unique())
        ct = np.array([[(dy == lv).sum() for lv in levels], [(dn == lv).sum() for lv in levels]])
        try:
            p = stats.chi2_contingency(ct)[1] if ct.shape[1] > 1 and ct.sum() else np.nan
        except ValueError:
            p = np.nan
        rows.append([f"__{label}__", np.nan, np.nan, np.nan, _fmt_p(p)])
        for lv in levels:
            rows.append([lv, _fmt_np(int((da == lv).sum()), n_all),
                         _fmt_np(int((dy == lv).sum()), n_y), _fmt_np(int((dn == lv).sum()), n_n), np.nan])

    add_cont("Age (years)", "age_at_admission")
    add_cat("Sex", "sex_category")
    add_cat("Race", "race_category")
    add_cat("Ethnicity", "ethnicity_category")
    add_cat("Admission type", "admission_type_category")
    add_cont("Eligible vent-days / patient", "n_eligible_days")
    add_binary("In-hospital mortality", "in_hospital_mortality")
    return pd.DataFrame(rows, columns=cols)


# --- criterion-mask histogram → embedded JS (the exclusion-toggle engine reads this) ---
def build_masks_js(masks: pd.DataFrame) -> dict:
    """Nested {unit:{granularity:{period:[[mask,count],...]}}} — the exclusion-toggle engine
    sums these live in JS per the active toggles. PHI-free (day counts per criterion mask)."""
    out: dict = {}
    for r in masks.itertuples(index=False):
        (out.setdefault(r.unit, {}).setdefault(r.granularity, {})
            .setdefault(r.period, []).append([int(r.mask), int(r.count)]))
    return out


def build_duration_payload(durs: pd.DataFrame, obs: pd.DataFrame) -> tuple[dict, dict]:
    """Compact, PHI-free arrays for the duration panel, binned live in JS.

    episodes = per controlled→support transition (the SBT duration, minutes).
    spont    = per on-spontaneous day (total minutes on a support mode that day).
    Each record carries duration + unit/month/week INDICES so JS filters by the same
    Unit/Time/Period selectors. No patient/encounter ids.
    """
    epi = durs.copy()
    if not epi.empty:
        epi["icu_day"] = epi["icu_day"].astype(str)
        epi["mon"] = epi["icu_day"].str.slice(0, 7)
        d = pd.to_datetime(epi["icu_day"], errors="coerce")
        iso = d.dt.isocalendar()
        epi["wk"] = (iso["year"].astype("Int64").astype(str) + "-W"
                     + iso["week"].astype("Int64").astype(str).str.zfill(2))
        epi["unit"] = epi["unit"].astype("string").fillna("unknown").replace("", "unknown")
        epi["val"] = pd.to_numeric(epi["dur_min"], errors="coerce").fillna(0.0)
    sp = obs.loc[obs["on_spontaneous"], ["unit", "period_month", "period_week", "spont_minutes"]].copy()
    sp = sp.rename(columns={"period_month": "mon", "period_week": "wk", "spont_minutes": "val"})
    sp["unit"] = sp["unit"].astype("string").fillna("unknown").replace("", "unknown")
    sp["val"] = pd.to_numeric(sp["val"], errors="coerce").fillna(0.0)

    def _vals(s):
        return sorted(set(s.dropna().tolist()))
    units = _vals(pd.concat([epi.get("unit", pd.Series(dtype=str)), sp["unit"]]))
    months = _vals(pd.concat([epi.get("mon", pd.Series(dtype=str)), sp["mon"]]))
    weeks = _vals(pd.concat([epi.get("wk", pd.Series(dtype=str)), sp["wk"]]))
    uidx = {u: i for i, u in enumerate(units)}
    midx = {m: i for i, m in enumerate(months)}
    widx = {w: i for i, w in enumerate(weeks)}

    def pack(df):
        if df.empty:
            return {"dur": [], "u": [], "m": [], "w": []}
        return {"dur": [int(round(x)) for x in df["val"].tolist()],
                "u": [uidx[u] for u in df["unit"].tolist()],
                "m": [midx.get(m, -1) for m in df["mon"].tolist()],
                "w": [widx.get(w, -1) for w in df["wk"].tolist()]}

    DUR = {"episodes": pack(epi if not epi.empty else pd.DataFrame(columns=["val", "unit", "mon", "wk"])),
           "spont": pack(sp)}
    DURCFG = {"units": units, "months": months, "weeks": weeks}
    return DUR, DURCFG


FILTER_JS = r"""
(function(){
  const $ = id => document.getElementById(id);
  const min = CFG.smallCellMin;
  const state = {unit: "__ALL__", gran: "all", period: "__all__", unitDim: "type",
                 tog: {exTrach:false, exParal:false, req12h:false, reqOxy:false, reqVaso:false,
                       reqTrans:false, reqDur:false, reqPeep:false}};
  const unitSel = $("f-unit"), periodSel = $("f-period"), periodWrap = $("f-period-wrap");
  const plabel = p => (CFG.periodLabels && CFG.periodLabels[p]) || p;
  // "Group ICUs by" dimension: location_type (default) vs specific unit (location_name). Both
  // grains are in MASKS; this just picks which unit list the by-unit panel + Unit dropdown use.
  const groupSel = $("f-group");
  function unitsForDim(){ return state.unitDim === "name" ? (CFG.nameOrder || ["__ALL__"]) : CFG.unitOrder; }
  function dimNoun(){ return state.unitDim === "name" ? "specific unit" : "ICU unit"; }
  function rebuildUnitOptions(){
    const list = unitsForDim();
    unitSel.innerHTML = list.map(u => '<option value="' + u + '">' + (CFG.unitLabels[u] || u) + '</option>').join('');
    state.unit = "__ALL__"; unitSel.value = "__ALL__";
  }

  // ---- Exclusion-toggle engine (plan 04): num/den computed LIVE from the per-day
  //      criterion-mask histogram (MASKS) given the active toggles (state.tog). ----
  const BIT = {}; MASK_BITS.forEach((n, i) => BIT[n] = i);
  const has = (m, name) => (m >> BIT[name]) & 1;
  const T = () => state.tog;

  function maskPassesDen(m){
    const t = T();
    if (t.exTrach && has(m, 'db_trach')) return false;
    if (t.exParal && has(m, 'db_paralytic')) return false;
    if (t.req12h && !has(m, 'db_accrued12')) return false;
    if (t.reqOxy && t.reqVaso){ if (!has(m, 'db_stable_both')) return false; }
    else if (t.reqOxy){ if (!has(m, 'db_stable_oxy')) return false; }
    else if (t.reqVaso){ if (!has(m, 'db_stable_vaso')) return false; }
    // "Require transition" also restricts the DENOMINATOR: a day already parked on a
    // spontaneous mode with no controlled→support transition is not a transition candidate,
    // so it leaves both numerator and denominator (it is not a missed SBT). (num ⊆ den safe:
    // any numerator day has nb_t, so it is never dropped here.)
    if (t.reqTrans && has(m, 'on_spontaneous') && !has(m, 'nb_t')) return false;
    return true;
  }
  function numBitName(){
    const t = T(), a = (t.reqTrans ? 't' : '') + (t.reqDur ? 'd' : '') + (t.reqPeep ? 'p' : '');
    return {'':'on_spontaneous','t':'nb_t','d':'nb_d','p':'nb_p',
            'td':'nb_td','tp':'nb_tp','dp':'nb_dp','tdp':'nb_tdp'}[a];
  }
  function maskArr(unit, gran, period){ return ((MASKS[unit]||{})[gran]||{})[period] || null; }
  function aggFor(unit, gran, period){
    const arr = maskArr(unit, gran, period);
    if (!arr) return null;
    const nb = numBitName(); let vent=0, den=0, num=0;
    for (let i=0;i<arr.length;i++){ const m=arr[i][0], c=arr[i][1]; vent+=c;
      if (maskPassesDen(m)){ den+=c; if (has(m, nb)) num+=c; } }
    return {vent:vent, den:den, num:num};
  }
  const denVal = c => c ? c.den : 0;
  const numVal = c => c ? c.num : 0;
  const fracOf = c => { const d = denVal(c); return d ? numVal(c)/d : null; };

  function numLabel(){
    const t = T(), p = [];
    if (t.reqTrans) p.push("transition"); if (t.reqDur) p.push("≥2 min"); if (t.reqPeep) p.push("low-PEEP");
    return p.length ? "SBT — " + p.join(" · ") : "On a spontaneous mode";
  }
  function denLabel(){
    const t = T(), p = [];
    if (t.exTrach) p.push("non-trach"); if (t.exParal) p.push("non-paralytic");
    if (t.req12h) p.push("≥12h controlled");
    if (t.reqOxy) p.push("stable O₂"); if (t.reqVaso) p.push("NEE≤0.2");
    if (t.reqTrans) p.push("transition candidates");
    return p.length ? "Vent-ICU days · " + p.join(", ") : "All vent-ICU days";
  }

  const DC = 2 * Math.PI * 52;
  function drawDonut(frac, small){
    const arc = $("donut-arc"), txt = $("donut-pct");
    if (!arc) return;
    const f = (frac == null) ? 0 : Math.max(0, Math.min(1, frac));
    arc.setAttribute("stroke-dasharray", (f*DC).toFixed(1) + " " + DC.toFixed(1));
    arc.setAttribute("stroke", small ? "#d8c7c0" : "#8a1f2b");
    txt.setAttribute("fill", small ? "#b39a93" : "#8a1f2b");
    txt.textContent = (frac == null) ? "—" : (100*f).toFixed(0) + "%";
  }

  function periodsFor(unit, gran){
    if (gran === "all") return [];
    return Object.keys((MASKS[unit] || {})[gran] || {}).sort();
  }
  function fillPeriods(){
    const ps = periodsFor(state.unit, state.gran);
    if (!ps.length){ periodWrap.style.display = "none"; return; }
    periodWrap.style.display = "";
    let opts = '<option value="__all__">All periods</option>';
    for (const p of ps) opts += '<option value="' + p + '">' + plabel(p) + '</option>';
    periodSel.innerHTML = opts;
    if (!ps.includes(state.period)) state.period = "__all__";
    periodSel.value = state.period;
  }
  function resolveGP(){
    const useAll = (state.gran === "all" || state.period === "__all__");
    return {g: useAll ? "all" : state.gran, p: useAll ? "all" : state.period};
  }
  function cellFor(unit){ const r = resolveGP(); return aggFor(unit, r.g, r.p); }
  const cell = () => cellFor(state.unit);
  function pct(x, dp){ return x == null ? "—" : (100*x).toFixed(dp == null ? 0 : dp) + "%"; }

  function render(){
    syncToggleButtons();
    const c = cell();
    $("donut-cap").textContent = numLabel();
    $("hd-lab").textContent = denLabel();
    const ctx = CFG.unitLabels[state.unit] + " · " +
      (state.gran === "all" || state.period === "__all__" ? "all time" : plabel(state.period));
    if (!c || !denVal(c)){
      $("hd-elig").textContent = "—";
      $("hd-sub").textContent = "no days · " + ctx;
      $("ptline").textContent = "";
      drawDonut(null, false);
      $("smallnote").style.display = "none"; drawWaterfall(null); drawTrend(); drawUnits(); drawDurations(); return;
    }
    const small = denVal(c) < min;
    const frac = fracOf(c);
    $("hd-elig").textContent = denVal(c).toLocaleString();
    $("hd-sub").textContent = numVal(c).toLocaleString() + " " + numLabel().toLowerCase()
      + " (" + pct(frac) + ") · " + ctx;
    $("ptline").textContent = "";
    drawDonut(frac, small);
    $("smallnote").style.display = small ? "block" : "none";
    drawWaterfall(c); drawTrend(); drawUnits(); drawDurations();
  }
  function syncToggleButtons(){
    document.querySelectorAll("#f-den button, #f-num button").forEach(b => {
      b.classList.toggle("on", !!state.tog[b.dataset.tog]);
    });
  }

  // ---- duration histogram + percentile table (per-trial; per-day for spontaneous) ----
  const DBUCK = [0, 5, 15, 30, 60, 120, 240, 480, Infinity];
  const DBLAB = ["<5m", "5–15m", "15–30m", "30–60m", "1–2h", "2–4h", "4–8h", ">8h"];
  function durSpec(){
    const t = T();
    // No transition required -> the broadest "on a spontaneous mode" view shows time-on-support
    // per day. Once a transition is required, show per-trial transition-episode durations
    // (these are arm-qualified/low-PEEP by construction); the >=2 min floor follows reqDur.
    if (!t.reqTrans) return {ds: DUR.spont, minDur: 0, unit: "days",
      label: "Time on a Spontaneous Mode (per day)"};
    return {ds: DUR.episodes, minDur: t.reqDur ? 2 : 0, unit: "trials",
      label: "SBT Duration" + (t.reqDur ? " — ≥2 min" : "") + " (per qualifying trial)"};
  }
  function durValues(){
    const sp = durSpec(), ds = sp.ds;
    const selU = state.unit === "__ALL__" ? -1 : DURCFG.units.indexOf(state.unit);
    let mode = 0, sel = -1;
    if (!(state.gran === "all" || state.period === "__all__")){
      if (state.gran === "month"){ mode = 1; sel = DURCFG.months.indexOf(state.period); }
      else if (state.gran === "week"){ mode = 2; sel = DURCFG.weeks.indexOf(state.period); }
    }
    const D = ds.dur, U = ds.u, M = ds.m, W = ds.w, out = [];
    for (let i = 0; i < D.length; i++){
      if (D[i] < sp.minDur) continue;
      if (selU >= 0 && U[i] !== selU) continue;
      if (mode === 1 && M[i] !== sel) continue;
      if (mode === 2 && W[i] !== sel) continue;
      out.push(D[i]);
    }
    return out;
  }
  function fmtDur(m){ return m == null ? "—" : (m < 60 ? m.toFixed(0) + " min" : (m/60).toFixed(1) + " h"); }
  function qtile(sorted, q){
    if (!sorted.length) return null;
    return sorted[Math.min(sorted.length - 1, Math.round(q * (sorted.length - 1)))];
  }
  function drawDurations(){
    const sp = durSpec(), vals = durValues();
    const when = (state.gran === "all" || state.period === "__all__") ? "all time" : plabel(state.period);
    $("durTitle").textContent = sp.label + " · " + CFG.unitLabels[state.unit] + " · " + when;
    const host = $("durHist"), tbl = $("durTable");
    if (!vals.length){
      host.innerHTML = '<div class="muted">No ' + sp.unit + ' in this slice.</div>'; tbl.innerHTML = ""; return;
    }
    const n = vals.length;
    const counts = new Array(DBLAB.length).fill(0);
    for (const v of vals){
      let b = DBUCK.length - 2;
      for (let j = 0; j < DBUCK.length - 1; j++){ if (v < DBUCK[j+1]){ b = j; break; } }
      counts[b]++;
    }
    const maxc = Math.max.apply(null, counts) || 1;
    const padL = 34, padB = 30, padT = 16, padR = 10, bw = 70, ih = 168;
    const W = padL + padR + DBLAB.length * bw, H = padT + ih + padB;
    let svg = '<line x1="'+padL+'" y1="'+padT+'" x2="'+padL+'" y2="'+(padT+ih)+'" stroke="#ece1d9"/>'
            + '<line x1="'+padL+'" y1="'+(padT+ih)+'" x2="'+(W-padR)+'" y2="'+(padT+ih)+'" stroke="#ece1d9"/>';
    counts.forEach((c, i) => {
      const x = padL + i*bw + bw*0.16, w = bw*0.68;
      const h = ih*(c/maxc), y = padT + ih - h, pc = 100*c/n;
      const over = (i === DBLAB.length - 1);
      svg += '<g><title>'+DBLAB[i]+': '+c.toLocaleString()+' ('+pc.toFixed(1)+'%)</title>';
      svg += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" fill="'+(over?"#7d8a86":"#8a1f2b")+'" rx="2"/></g>';
      if (c > 0) svg += '<text x="'+(x+w/2)+'" y="'+(y-3)+'" font-size="9.5" text-anchor="middle" fill="#6b5d57">'+pc.toFixed(0)+'%</text>';
      svg += '<text x="'+(x+w/2)+'" y="'+(padT+ih+13)+'" font-size="9.5" text-anchor="middle" fill="#6b5d57">'+DBLAB[i]+'</text>';
    });
    svg += '<text x="'+(padL-5)+'" y="'+(padT+5)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+maxc.toLocaleString()+'</text>'
         + '<text x="'+(padL-5)+'" y="'+(padT+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" style="max-width:100%">'+svg+'</svg>';
    const s = vals.slice().sort((a,b) => a - b);
    const qs = [["p10",0.10],["p25",0.25],["Median",0.50],["p75",0.75],["p90",0.90]];
    let head = '<tr><th>'+sp.unit+' (n)</th>', body = '<tr><td>'+n.toLocaleString()+'</td>';
    for (const [lab, q] of qs){ head += '<th>'+lab+'</th>'; body += '<td>'+fmtDur(qtile(s, q))+'</td>'; }
    tbl.innerHTML = '<table class="dur-table">'+head+'</tr>'+body+'</tr></table>';
  }

  // ---- Exclusion waterfall: all vent-ICU days peeled by each ACTIVE denominator toggle
  //      (catalogue order) -> final denominator -> numerator (active trial-quality criteria) ----
  function drawWaterfall(c){
    const host = $("waterfall"), note = $("waterfallNote");
    const r = resolveGP(), arr = maskArr(state.unit, r.g, r.p);
    if (!arr){ host.innerHTML = ''; note.textContent = ''; return; }
    const t = T();
    const steps = [];
    if (t.exTrach) steps.push({lab:"Exclude tracheostomy", pred:m=>!has(m,'db_trach')});
    if (t.exParal) steps.push({lab:"Exclude continuous paralytic", pred:m=>!has(m,'db_paralytic')});
    if (t.req12h)  steps.push({lab:"Require ≥12 h controlled vent", pred:m=>!!has(m,'db_accrued12')});
    if (t.reqOxy || t.reqVaso){
      const both = t.reqOxy && t.reqVaso;
      const lab = both ? "Require stable window (O₂ & NEE≤0.2)"
                : t.reqOxy ? "Require stable oxygenation (O₂)"
                : "Require low vasopressors (NEE≤0.2)";
      const bit = both ? 'db_stable_both' : t.reqOxy ? 'db_stable_oxy' : 'db_stable_vaso';
      steps.push({lab:lab, pred:m=>!!has(m,bit)});
    }
    if (t.reqTrans) steps.push({lab:"Drop parked-on-spontaneous (no transition)",
                                pred:m=>!(has(m,'on_spontaneous') && !has(m,'nb_t'))});
    let vent = 0; for (const e of arr) vent += e[1];
    const rows = [{lab:"All vent-ICU days", n:vent, removed:0}];
    const preds = []; let prev = vent;
    for (const s of steps){
      preds.push(s.pred);
      let rem = 0; for (const e of arr){ if (preds.every(p=>p(e[0]))) rem += e[1]; }
      rows.push({lab:s.lab, n:rem, removed:prev-rem}); prev = rem;
    }
    const den = prev, nb = numBitName();
    let num = 0; for (const e of arr){ if (preds.every(p=>p(e[0])) && has(e[0],nb)) num += e[1]; }

    // render: shrinking horizontal bars (funnel) + a final numerator split of the denominator
    const W = 780, rowH = 34, padL = 312, barMax = W - padL - 78, top = 4;
    const H = top + (rows.length + 1) * rowH + 6;
    let svg = "";
    rows.forEach((rw, i) => {
      const y = top + i*rowH, w = vent ? Math.max(1, (rw.n/vent)*barMax) : 0;
      const isStart = i === 0;
      svg += '<text x="'+(padL-10)+'" y="'+(y+rowH/2+3)+'" font-size="11.5" text-anchor="end" fill="'
           + (isStart?'#3a2c2c':'#6b5d57')+'"'+(isStart?' font-weight="700"':'')+'>'+rw.lab+'</text>';
      svg += '<rect x="'+padL+'" y="'+(y+6)+'" width="'+w+'" height="'+(rowH-14)+'" fill="'
           + (isStart?'#b9a59d':'#9a8c86')+'" rx="3"><title>'+rw.lab+': '+rw.n.toLocaleString()+'</title></rect>';
      svg += '<text x="'+(padL+w+8)+'" y="'+(y+rowH/2+3)+'" font-size="11" fill="#6b5d57">'+rw.n.toLocaleString()
           + (rw.removed>0 ? ' <tspan fill="#b06a4f">(−'+rw.removed.toLocaleString()+')</tspan>' : '')+'</text>';
    });
    // numerator row: filled num within the denominator width
    const yN = top + rows.length*rowH;
    const denW = vent ? Math.max(1,(den/vent)*barMax) : 0, numW = den ? (num/den)*denW : 0;
    const rate = den ? num/den : null;
    svg += '<text x="'+(padL-10)+'" y="'+(yN+rowH/2+3)+'" font-size="11.5" text-anchor="end" fill="#8a1f2b" font-weight="700">'
         + numLabel()+'</text>';
    svg += '<rect x="'+padL+'" y="'+(yN+6)+'" width="'+denW+'" height="'+(rowH-14)+'" fill="#efe4dc" rx="3"/>';
    svg += '<rect x="'+padL+'" y="'+(yN+6)+'" width="'+numW+'" height="'+(rowH-14)+'" fill="#8a1f2b" rx="3">'
         + '<title>'+numLabel()+': '+num.toLocaleString()+' / '+den.toLocaleString()+'</title></rect>';
    svg += '<text x="'+(padL+denW+8)+'" y="'+(yN+rowH/2+3)+'" font-size="11" fill="#8a1f2b" font-weight="700">'
         + num.toLocaleString()+' / '+den.toLocaleString()+'  ('+pct(rate)+')</text>';
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" style="max-width:100%">'+svg+'</svg>';

    const off = !steps.length;
    note.innerHTML = off
      ? ('No candidate-day filters active — the denominator is <b>all '+vent.toLocaleString()
         +'</b> vent-ICU days. Turn on filters above to peel the denominator down to a stricter cohort; '
         +'each step shows the days it removes. The bottom bar is the numerator: <b>'+num.toLocaleString()
         +' ('+pct(rate)+')</b> '+numLabel().toLowerCase()+'.')
      : ('Each active filter removes the days shown in <span style="color:#b06a4f">(−n)</span>; the '
         +'denominator shrinks from <b>'+vent.toLocaleString()+'</b> to <b>'+den.toLocaleString()+'</b>. '
         +'Of those, <b>'+num.toLocaleString()+' ('+pct(rate)+')</b> met the active numerator criteria ('
         + numLabel().toLowerCase()+').');
  }

  function drawUnits(){
    const host = $("units");
    const when = (state.gran === "all" || state.period === "__all__") ? "all time" : plabel(state.period);
    $("unitsTitle").textContent = numLabel() + " rate (÷ " + denLabel().toLowerCase() + ") by " + dimNoun() + " · " + when;
    const rows = [];
    for (const u of unitsForDim()){
      const c = cellFor(u); if (!c || !denVal(c)) continue;
      rows.push({u: u, label: CFG.unitLabels[u] || u, rate: numVal(c)/denVal(c), num: numVal(c), den: denVal(c)});
    }
    if (!rows.length){ host.innerHTML = '<div class="muted">No days in this period.</div>'; return; }
    const ref = rows.find(r => r.u === "__ALL__");
    let units = rows.filter(r => r.u !== "__ALL__").sort((a,b) => b.rate - a.rate);
    const ordered = (ref ? [ref] : []).concat(units);
    const rowH = 26, padL = 172, padR = 140, barMax = 360, top = 8;
    const W = padL + barMax + padR, H = top + ordered.length*rowH + 6;
    let svg = "";
    ordered.forEach((r, i) => {
      const y = top + i*rowH, small = r.den < min, isAll = (r.u === "__ALL__");
      const w = Math.max(2, r.rate*barMax);
      const fill = small ? "#e2d3cc" : (isAll ? "#6f1622" : "#8a1f2b");
      svg += '<text x="'+(padL-8)+'" y="'+(y+rowH/2+4)+'" font-size="11.5" text-anchor="end" fill="'+(isAll?"#6f1622":"#3a2c2c")+'"'+(isAll?' font-weight="700"':'')+'>'+r.label+'</text>';
      svg += '<rect x="'+padL+'" y="'+(y+4)+'" width="'+barMax+'" height="'+(rowH-10)+'" fill="#efe4dc" rx="3"/>';
      svg += '<g><title>'+r.label+'\n'+r.num+'/'+r.den+' = '+(100*r.rate).toFixed(1)+'%'+(small?'  — n small':'')+'</title>';
      svg += '<rect x="'+padL+'" y="'+(y+4)+'" width="'+w+'" height="'+(rowH-10)+'" fill="'+fill+'" rx="3"/></g>';
      svg += '<text x="'+(padL+barMax+8)+'" y="'+(y+rowH/2+4)+'" font-size="11" fill="#9a8c86">'+(100*r.rate).toFixed(0)+'%  ('+r.den.toLocaleString()+')</text>';
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" style="max-width:100%">'+svg+'</svg>';
  }

  function drawTrend(){
    const tg = state.gran === "all" ? "month" : state.gran;
    const series = (MASKS[state.unit] || {})[tg] || {};
    const keys = Object.keys(series).sort();
    const Tg = tg.charAt(0).toUpperCase() + tg.slice(1);
    $("trendTitle").textContent = numLabel() + " rate by " + Tg + " · " + CFG.unitLabels[state.unit];
    const host = $("trend");
    if (!keys.length){ host.innerHTML = '<div class="muted">No periods in this slice.</div>'; return; }
    const cellAt = k => aggFor(state.unit, tg, k);
    const slot = keys.length > 40 ? 15 : (keys.length > 15 ? 34 : 56);
    const pad = {l:36, r:12, t:14, b:48}, ih = 150;
    const W = pad.l + pad.r + keys.length*slot, H = pad.t + ih + pad.b;
    let maxr = 0.05; for (const k of keys){ const d = cellAt(k), dn = denVal(d); if (dn) maxr = Math.max(maxr, numVal(d)/dn); }
    const top = Math.max(0.1, Math.ceil(maxr*100/10)*10/100);
    const lblStep = Math.ceil(keys.length/24);
    let svg = '<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<line x1="'+pad.l+'" y1="'+(pad.t+ih)+'" x2="'+(W-pad.r)+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+4)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+(100*top).toFixed(0)+'%</text>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    keys.forEach((k, i) => {
      const d = cellAt(k), dn = denVal(d), r = dn ? numVal(d)/dn : 0;
      const x = pad.l + i*slot + slot*0.16, w = slot*0.68;
      const yT = pad.t + ih*(1 - r/top), hT = ih*(r/top);
      const dim = dn < min, sel = (k === state.period);
      const cBar = dim ? "#e2d3cc" : "#8a1f2b";
      svg += '<g><title>' + k + "\n" + numVal(d) + "/" + dn + " (" + (100*r).toFixed(0) + "%)" +
             (dim ? "  — n small" : "") + '</title>';
      svg += '<rect x="'+x+'" y="'+yT+'" width="'+w+'" height="'+hT+'" fill="'+cBar+'"' + (sel ? ' stroke="#3a2c2c" stroke-width="1.5"' : '') + '/></g>';
      if (i % lblStep === 0){
        const lab = tg === "month" ? k.slice(2) : k.replace(/^\d{4}-/, "");
        const cx = x + w/2;
        svg += '<text x="'+cx+'" y="'+(H-pad.b+12)+'" font-size="8.5" text-anchor="end" fill="#9a8c86" transform="rotate(-35 '+cx+' '+(H-pad.b+12)+')">'+lab+'</text>';
      }
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'" width="'+W+'" style="max-width:none">'+svg+'</svg>';
  }

  unitSel.onchange = () => { state.unit = unitSel.value; fillPeriods(); render(); };
  if (groupSel) groupSel.onchange = () => {
    state.unitDim = groupSel.value; rebuildUnitOptions(); fillPeriods(); render();
  };
  periodSel.onchange = () => { state.period = periodSel.value; render(); };
  document.querySelectorAll("#f-gran button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#f-gran button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); state.gran = b.dataset.g; state.period = "__all__"; fillPeriods(); render();
  });
  // exclusion toggles: each button flips state.tog[key] (on/off); engine recomputes live.
  document.querySelectorAll("#f-den button, #f-num button").forEach(b => b.onclick = () => {
    const k = b.dataset.tog; state.tog[k] = !state.tog[k]; render();
  });
  const rb = $("f-reset"); if (rb) rb.onclick = () => {
    Object.keys(state.tog).forEach(k => state.tog[k] = false); render();
  };
  fillPeriods(); render();
})();
"""


# Exclusion-toggle catalogue (plan 04). Each: (key, label, definition, effect). Keys map
# to state.tog in the JS engine; the same list drives the toggle buttons + the on-screen
# catalogue panel so what each toggle removes (num/den/both) is always documented.
EXCLUSION_CATALOGUE = {
    "den": [
        ("exTrach", "Exclude tracheostomy",
         "Remove vent-days on which the patient was tracheostomized (a different liberation path — trach-collar, not a vent SBT).", "den + num"),
        ("exParal", "Exclude continuous paralytic",
         "Remove vent-days with a continuous neuromuscular-blocker infusion (no respiratory drive → not a candidate).", "den + num"),
        ("req12h", "Require ≥12h controlled",
         "Keep only days with ≥12 h of controlled ventilation accrued before the day (SBT meaningful only after sustained controlled vent).", "den + num"),
        ("reqOxy", "Require stable oxygenation",
         "Keep only days with a ≥2 h window of FiO₂ ≤ 0.50, PEEP ≤ 8, SpO₂ ≥ 88% (safe to attempt weaning).", "den + num"),
        ("reqVaso", "Require low vasopressors (NEE≤0.2)",
         "Keep only days with a ≥2 h window of norepinephrine-equivalent ≤ 0.2 mcg/kg/min — low-dose pressors are allowed, only days above 0.2 are excluded (hemodynamic stability; not universally applied).", "den + num"),
        ("reqTrans", "Require controlled→support transition",
         "Changes the question to transitions specifically: count the day only if a controlled→support "
         "transition occurred, AND drop days already parked on a spontaneous mode with no transition from the "
         "denominator too (they are not transition candidates, not missed SBTs).", "den + num"),
    ],
    "num": [
        ("reqDur", "Require sustained ≥2 min",
         "Count the day only if a support episode lasted ≥ 2 minutes (not a momentary blip).", "num"),
        ("reqPeep", "Require low PEEP on support",
         "Count the day only if a support episode met PEEP ≤ 8 (pressure-support) / ≤ 5 (CPAP) — a genuine weaning trial.", "num"),
    ],
}


def build_catalogue_panel() -> str:
    """Static on-screen table documenting every toggle and what it removes (num/den/both)."""
    def rows(items, cls):
        out = ""
        for k, lab, defn, eff in items:
            out += (f'<tr><td class="catlab">{html.escape(lab)}</td>'
                    f'<td>{html.escape(defn)}</td>'
                    f'<td class="cateff {cls}">{html.escape(eff)}</td></tr>')
        return out
    return (
        '<div class="section catalogue"><h2>What each toggle does</h2>'
        '<div class="fig-caption">One rate. With every toggle <b>off</b> it is the broadest SBT lens — '
        '<b>any spontaneous-mode presence</b> ÷ <b>all vent-ICU days</b>. Each toggle below applies one '
        'exclusion, stating whether it removes days from the <b>denominator</b> (candidate days; carries the '
        'numerator with them), or only disqualifies the <b>numerator</b> attempt. All eight on = the '
        'by-the-book strict SBT rate.</div>'
        '<table class="cattable"><thead><tr><th>Toggle</th><th>Effect on a vent-ICU day</th>'
        '<th>Removes from</th></tr></thead><tbody>'
        '<tr class="catsub"><td colspan="3">Candidate-day filters — denominator</td></tr>'
        + rows(EXCLUSION_CATALOGUE["den"], "eff-den")
        + '<tr class="catsub"><td colspan="3">Trial-quality filters — numerator</td></tr>'
        + rows(EXCLUSION_CATALOGUE["num"], "eff-num")
        + '</tbody></table></div>')


def build_controls(slices: pd.DataFrame) -> str:
    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    units = [u for u in UNIT_LABELS if u in set(typ["unit"])]
    opts = "".join(f'<option value="{html.escape(u)}">{html.escape(UNIT_LABELS[u])}</option>' for u in units)
    name_units = sorted(set(slices.loc[slices["dim"] == "name", "unit"])) if "dim" in slices.columns else []
    splits = "dim" in slices.columns and slices[slices["dim"] == "name"].groupby("parent")["unit"].nunique().gt(1).any()
    group_ctl = ('<label class="ctl">Group ICUs by<select id="f-group">'
                 '<option value="type">ICU type</option>'
                 f'<option value="name">Specific unit ({len(name_units)})</option>'
                 '</select></label>') if (name_units and splits) else ""
    gran_btns = "".join('<button data-g="{g}"{on}>{lab}</button>'.format(
        g=g, on=' class="on"' if g == "all" else "", lab=html.escape(GRAN_LABELS[g]))
        for g in ("all", "month", "week"))

    def seg(group_id, items, default):
        btns = "".join('<button data-v="{v}"{on}>{lab}</button>'.format(
            v=v, on=' class="on"' if v == default else "", lab=html.escape(lab)) for v, lab in items)
        return f'<div class="seg" id="{group_id}">{btns}</div>'

    def togs(items):
        return "".join('<button data-tog="{k}" title="{tt}">{lab}</button>'.format(
            k=k, lab=html.escape(lab), tt=html.escape(tt)) for k, lab, tt, _ in items)
    den_t = togs(EXCLUSION_CATALOGUE["den"])
    num_t = togs(EXCLUSION_CATALOGUE["num"])
    return ('<div class="controls">'
            + group_ctl
            + f'<label class="ctl">Unit<select id="f-unit">{opts}</select></label>'
            f'<div class="ctl">Time<div class="seg" id="f-gran">{gran_btns}</div></div>'
            '<label class="ctl" id="f-period-wrap" style="display:none">Period<select id="f-period"></select></label>'
            '</div>'
            '<div class="controls toggles">'
            f'<div class="ctl tgroup"><span class="tglab">Candidate-day filters '
            '<em>(denominator)</em></span>'
            f'<div class="seg toggleseg" id="f-den">{den_t}</div></div>'
            f'<div class="ctl tgroup"><span class="tglab">Trial-quality filters '
            '<em>(numerator)</em></span>'
            f'<div class="seg toggleseg" id="f-num">{num_t}</div></div>'
            '<button class="resetbtn" id="f-reset" title="Turn all toggles off (broadest view)">Reset</button>'
            '</div>')


def build_html(ctx) -> str:
    brand = (f'<img src="{ctx["logo_uri"]}" alt="CLIF">' if ctx["logo_uri"]
             else '<span style="font-size:28px;font-weight:800;color:#8a1f2b">CLIF</span>')

    css = f"""
:root{{--maroon:{MAROON};--maroon-d:{MAROON_D};--cream:{CREAM};--card:{CARD};--ink:{INK};
--muted:{MUTED};--line:{LINE};--bar:{BAR};--good:{GOOD};--warn:{WARN};--bad:{BAD};}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:Inter,-apple-system,'Segoe UI',system-ui,sans-serif;
background:var(--cream);color:var(--ink);font-size:14px;line-height:1.55;}}
.wrap{{max-width:1180px;margin:0 auto;padding:30px 40px 56px;background:var(--card);
box-shadow:0 3px 16px rgba(120,30,40,.06);}}
header.top{{display:flex;align-items:center;gap:18px;border-bottom:1px solid var(--line);
padding-bottom:18px;margin-bottom:8px;}}
header.top img{{height:72px;width:auto;display:block;flex:0 0 auto;}}
.backlink{{display:inline-block;font-size:12.5px;color:var(--maroon);text-decoration:none;font-weight:700;margin-bottom:4px;}}
.backlink:hover{{text-decoration:underline;}}
h1{{font-size:27px;font-weight:800;color:var(--maroon-d);margin:0;letter-spacing:-.3px;}}
.sub{{color:var(--muted);font-size:13px;margin-top:3px;}}
h2{{font-size:19px;font-weight:700;color:var(--maroon-d);border-bottom:1px solid var(--line);
padding-bottom:6px;margin:0 0 16px;}}
.section{{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:26px 28px;margin:0 0 34px;box-shadow:0 3px 10px rgba(120,30,40,.05);}}
.headline-card{{display:flex;align-items:center;justify-content:center;gap:42px;flex-wrap:wrap;
background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 34px;
margin:24px 0 34px;box-shadow:0 3px 10px rgba(120,30,40,.05);}}
.donut-wrap{{display:flex;flex-direction:column;align-items:center;gap:8px;}}
.donut-wrap text{{font-variant-numeric:tabular-nums;}}
.donut-cap{{font-size:13.5px;font-weight:700;color:var(--ink);}}
#donut-arc{{transition:stroke-dasharray .35s ease;}}
.hd-text{{display:flex;flex-direction:column;gap:3px;min-width:220px;}}
.hd-big{{font-size:44px;font-weight:800;color:var(--maroon);line-height:1.02;font-variant-numeric:tabular-nums;}}
.hd-lab{{font-size:14px;font-weight:700;color:var(--ink);}}
.hd-sub{{font-size:12.5px;color:var(--muted);margin-top:4px;}}
.hd-pt{{font-size:12px;color:var(--maroon-d);margin-top:7px;font-weight:600;}}
.decomp .legend{{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px;font-size:12px;color:var(--ink);}}
.decomp .legend span{{display:inline-flex;align-items:center;gap:6px;}}
.decomp .legend i{{width:12px;height:12px;border-radius:3px;display:inline-block;}}
.decompNote{{font-size:13.5px;color:var(--ink);margin-top:12px;line-height:1.65;
background:var(--cream);border:1px solid var(--line);border-radius:10px;padding:12px 16px;}}
.decompNote b{{color:var(--maroon-d);}}
.decompWhyHd{{font-size:15px;font-weight:600;color:var(--maroon-d);margin:26px 0 4px;}}
.dur-wrap{{display:flex;flex-wrap:wrap;align-items:flex-start;gap:24px;}}
.dur-table{{border-collapse:collapse;margin:8px 0 2px;font-size:13px;}}
.dur-table th{{background:var(--cream);color:var(--maroon-d);font-weight:700;padding:7px 15px;
border-bottom:2px solid var(--maroon-d);text-align:center;}}
.dur-table td{{padding:7px 15px;border-bottom:1px solid var(--line);text-align:center;
font-variant-numeric:tabular-nums;}}
.controls{{display:flex;flex-wrap:wrap;align-items:flex-end;gap:18px;margin:22px 0 4px;}}
.ctl{{display:flex;flex-direction:column;gap:5px;font-size:11px;font-weight:700;
color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}}
.ctl select{{font-size:13px;font-weight:600;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:9px;padding:7px 10px;min-width:150px;
font-family:inherit;text-transform:none;letter-spacing:0;}}
.seg{{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;}}
.seg button{{font:inherit;font-size:13px;font-weight:600;border:0;background:var(--card);
color:var(--ink);padding:7px 13px;cursor:pointer;border-left:1px solid var(--line);
text-transform:none;letter-spacing:0;}}
.seg button:first-child{{border-left:0;}}
.seg button.on{{background:var(--maroon);color:#fff;}}
.controls.toggles{{align-items:flex-start;gap:24px;margin:4px 0 8px;padding:14px 0 4px;
border-top:1px dashed var(--line);}}
.tgroup{{gap:7px;}}
.tglab{{font-size:11px;font-weight:800;color:var(--maroon-d);}}
.tglab em{{font-weight:600;font-style:normal;color:var(--muted);text-transform:none;}}
.toggleseg{{flex-wrap:wrap;border:0;gap:7px;}}
.toggleseg button{{border:1px solid var(--line);border-radius:8px;background:var(--card);
color:var(--ink);padding:7px 11px;font-size:12.5px;}}
.toggleseg button.on{{background:var(--maroon);color:#fff;border-color:var(--maroon);}}
.resetbtn{{align-self:flex-end;font:inherit;font-size:12px;font-weight:700;color:var(--muted);
background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer;}}
.resetbtn:hover{{color:var(--maroon-d);border-color:var(--maroon);}}
.cattable{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px;}}
.cattable th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
color:var(--muted);border-bottom:1px solid var(--line);padding:6px 10px;}}
.cattable td{{padding:7px 10px;border-bottom:1px solid #f0e7e1;vertical-align:top;}}
.cattable .catsub td{{background:var(--cream);font-weight:800;color:var(--maroon-d);
font-size:11px;text-transform:uppercase;letter-spacing:.04em;}}
.cattable .catlab{{font-weight:700;color:var(--ink);white-space:nowrap;}}
.cateff{{font-weight:700;white-space:nowrap;font-size:11.5px;}}
.cateff.eff-den{{color:#2f6f7d;}} .cateff.eff-num{{color:#8a1f2b;}}
#waterfall{{overflow-x:auto;margin-top:4px;}}
.smallnote{{display:none;font-size:11.5px;color:var(--warn);margin:-22px 0 26px;}}
.trend-wrap{{overflow-x:auto;padding-bottom:4px;}}
.muted{{color:var(--muted);font-size:13px;}}
.fig{{text-align:center;margin:6px 0;}}
.fig img{{max-width:100%;height:auto;border-radius:8px;}}
.fig-caption{{font-size:13px;color:var(--muted);margin-top:8px;text-align:left;}}
.amber{{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:10px;
padding:14px 18px;font-size:13px;margin:0 0 22px;}}
.amber b{{color:#7a3a0a;}}
.amber ul{{margin:9px 0 6px;padding-left:20px;}}
.amber li{{margin:5px 0;line-height:1.55;}}
table.results-table{{border-collapse:collapse;width:auto;font-size:13px;margin-top:10px;}}
table.results-table th{{background:var(--cream);color:var(--maroon-d);text-align:left;
padding:9px 12px;border-bottom:2px solid var(--maroon-d);font-weight:700;}}
table.results-table td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top;}}
table.results-table tbody tr:nth-child(even){{background:#faf5f1;}}
footer{{margin-top:30px;color:var(--muted);font-size:11.5px;text-align:center;
border-top:1px solid var(--line);padding-top:14px;}}
"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SBT Delivery QI — {html.escape(ctx['site'])}</title><style>{css}</style></head><body>
<div class="wrap">
  <header class="top">{brand}
    <div><a class="backlink" href="scorecard.html">← CLIF ICU Ventilator QI Bundle</a>
    <h1>Spontaneous Breathing Trial — Quality-of-Care</h1>
    <div class="sub">Daily controlled→support breathing-trial delivery · {html.escape(ctx['site'])} ·
    generated {html.escape(ctx['generated'])}</div></div>
  </header>

  {ctx['controls']}
  <div class="headline-card">
    <div class="donut-wrap">
    <svg viewBox="0 0 120 120" width="150" height="150" role="img" aria-label="SBT delivery donut">
    <circle cx="60" cy="60" r="52" fill="none" stroke="var(--bar)" stroke-width="13"/>
    <circle id="donut-arc" cx="60" cy="60" r="52" fill="none" stroke="var(--maroon)"
    stroke-width="13" stroke-linecap="round" transform="rotate(-90 60 60)"
    stroke-dasharray="{ctx['frac0_dash']}"/>
    <text id="donut-pct" x="60" y="60" text-anchor="middle" dominant-baseline="central"
    font-size="30" font-weight="800" fill="var(--maroon)"
    font-family="Inter,system-ui,sans-serif">{ctx['frac0_pct']}</text>
    </svg><div class="donut-cap" id="donut-cap">On a spontaneous mode</div></div>
    <div class="hd-text">
    <div class="hd-big" id="hd-elig">{ctx['n_elig']:,}</div>
    <div class="hd-lab" id="hd-lab">All vent-ICU days</div>
    <div class="hd-sub" id="hd-sub">{ctx['hd_sub0']}</div>
    <div class="hd-pt" id="ptline"></div>
    </div></div>
  {ctx['smallnote']}

  {ctx['catalogue']}

  <div class="section"><h2>Exclusion Waterfall — Where the Vent-ICU Days Go</h2>
    <div class="fig-caption">All vent-ICU days in the selected slice, peeled by each <b>active candidate-day
    filter</b> (top → bottom, in catalogue order) down to the final <b>denominator</b>; the bottom maroon
    bar is the <b>numerator</b> (days meeting the active trial-quality filters) within it — its fill is the
    headline rate. With no toggles active the denominator is all vent-ICU days. Reacts to every toggle and
    to Unit/Time/Period.</div>
    <div id="waterfall"></div>
    <div class="decompNote" id="waterfallNote"></div>
  </div>

  {ctx['caveat']}

  <div class="section"><h2>Rate Over Time</h2>
    <div class="fig-caption" id="trendTitle"></div>
    <div class="trend-wrap">{ctx['trend']}</div>
    <div class="fig-caption">Each bar = the live rate (under the active toggles) for one period in the
    selected unit. Bars are grayed when the period has fewer than the small-cell threshold of denominator
    days. Toggles and unit/granularity all update this; pick a Period to drill the headline to one bucket.</div>
  </div>

  <div class="section"><h2>Rate by ICU Unit</h2>
    <div class="fig-caption" id="unitsTitle"></div>
    <div class="trend-wrap" id="units"></div>
    <div class="fig-caption">Every ICU unit side by side for the time period selected above (the Unit
    filter does not affect this panel). The maroon <b>All ICUs</b> bar is the site-wide reference;
    units are ordered by rate. Each bar shows the selected rate with the denominator-day count in
    parentheses; bars are grayed below the small-cell threshold.</div>
  </div>

  <div class="section"><h2>How Long Are the Trials?</h2>
    <div class="fig-caption" id="durTitle"></div>
    <div class="dur-wrap"><div class="trend-wrap" id="durHist"></div><div id="durTable"></div></div>
    <div class="fig-caption">When <b>require transition</b> is on: distribution of <b>SBT durations</b>
    (per qualifying controlled→support trial), with the ≥2 min floor following that toggle. When it is off
    (broadest): <b>time on a spontaneous mode per day</b>. Reacts to Unit / Time / Period and the numerator
    toggles. The gray <b>&gt;8 h</b> bin is largely
    sustained support ventilation rather than a discrete trial — a true SBT ends in extubation or a return
    to a controlled mode. Where charting is hourly a brief trial may be missed entirely (a lower bound).</div>
  </div>

  <div class="section"><h2>Table 1 — Eligible Patients, Ever-SBT vs Never (n = {ctx['table_n']:,})
    <span style="font-size:12px;font-weight:600;color:var(--muted)">· site-wide · all time</span></h2>
    <div class="fig-caption">Patients with ≥1 eligible vent-ICU day, stratified by whether an SBT was
    ever delivered. Continuous: median (Q1, Q3), Kruskal–Wallis. Categorical: n (%), χ².
    Patient-level secondary framing — the headline metric is day-level.</div>
    {ctx['table1']}
  </div>

  <footer>CLIF consortium · multi-site federated QI · SBT vertical · row-level data never leaves the
  site — only counts and rates are shared.</footer>
</div>
{ctx['script']}
</body></html>"""


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "05_dashboard.log", mode="w")])
    cfg = cohort_mod.load_config()
    site = cfg.get("site", "unknown")
    inter, final = cohort_mod.INTERMEDIATE_DIR, cohort_mod.FINAL_DIR

    summary = pd.read_csv(final / "metrics_site_summary.csv")
    obs = pd.read_parquet(inter / "metrics_patient_day_level.parquet")
    slices = pd.read_parquet(inter / "metrics_slices.parquet")
    masks = pd.read_parquet(inter / "metrics_masks.parquet")
    mask_bits = _json.loads((inter / "metrics_masks_bits.json").read_text())
    durs = (pd.read_parquet(inter / "sbt_durations.parquet")
            if (inter / "sbt_durations.parquet").exists()
            else pd.DataFrame(columns=["unit", "icu_day", "dur_min", "arm"]))
    diag = {}
    if (inter / "sbt_diag.json").exists():
        diag = _json.loads((inter / "sbt_diag.json").read_text())

    def s(metric):
        return summary.loc[summary["metric"] == metric].iloc[0]

    n_vent = int(s("vent_icu_days")["numerator"])
    n_nontrach = int(s("nontrach_days")["numerator"])
    n_elig = int(s("eligible_days")["numerator"])
    n_sbt = int(s("sbt_delivered")["numerator"])
    n_notassess = int(s("not_assessable_days")["numerator"])
    n_paralytic = int(s("excluded_paralytic_days")["numerator"])
    pts_sbt = int(s("patients_ever_sbt")["numerator"]); pts_elig = int(s("patients_ever_sbt")["denominator"])
    generated = str(s("vent_icu_days")["generated"])
    small_cell_min = int(cfg.get("reporting", {}).get("small_cell_min_den", 10))

    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    present_units = set(typ["unit"].unique())
    unit_order = [u for u in UNIT_LABELS if u in present_units and u != "unknown"]
    # Specific-unit (location_name) dimension: name keys ordered under their parent type.
    name_rows = slices[slices["dim"] == "name"] if "dim" in slices.columns else slices.iloc[0:0]
    name_parent = name_rows.drop_duplicates("unit").set_index("unit")["parent"].to_dict()
    _canon = [u for u in UNIT_LABELS if u not in ("__ALL__", "unknown")]
    name_units = sorted(name_parent, key=lambda n: (_canon.index(name_parent[n])
                        if name_parent[n] in _canon else 99, n))
    name_order = ["__ALL__"] + name_units
    LABELS = cfg.get("unit_labels", {}) or {}
    unit_labels = {u: UNIT_LABELS.get(u, u) for u in slices["unit"].unique()}
    unit_labels.update({n: LABELS.get(n, n) for n in name_units})
    unit_labels.update({u: LABELS[u] for u in unit_order if u in LABELS})
    period_labels = {p: _period_label(p) for p in
                     slices.loc[slices["granularity"].isin(["month", "week"]), "period"].unique()}
    cfg_js = {"smallCellMin": small_cell_min,
              "unitLabels": unit_labels,
              "unitOrder": unit_order, "nameOrder": name_order,
              "periodLabels": period_labels}
    masks_js = build_masks_js(masks)
    dur_payload, dur_cfg = build_duration_payload(durs, obs)
    script_html = ("<script>\nconst MASKS = " + _json.dumps(masks_js, ensure_ascii=False)
                   + ";\nconst MASK_BITS = " + _json.dumps(mask_bits, ensure_ascii=False)
                   + ";\nconst CFG = " + _json.dumps(cfg_js, ensure_ascii=False)
                   + ";\nconst DUR = " + _json.dumps(dur_payload, ensure_ascii=False)
                   + ";\nconst DURCFG = " + _json.dumps(dur_cfg, ensure_ascii=False)
                   + ";\n" + FILTER_JS + "\n</script>")

    import math
    _C = 2 * math.pi * 52
    # Initial donut = the BROADEST default (all toggles off): on_spontaneous / all vent-days.
    n_spont_all = int(obs["on_spontaneous"].sum())
    _frac0 = n_spont_all / max(n_vent, 1)
    # All-toggles-ON endpoint: numerator = strict SBT (n_sbt); denominator = eligible days that
    # are transition candidates (drop eligible days parked on a spontaneous mode with no transition,
    # per the "require transition" den+num rule). Differs from the legacy all-eligible rate (n_sbt/n_elig).
    _allon_den = int((obs["eligible"] & ~(obs["on_spontaneous"] & ~obs["nb_t"])).sum())

    native = diag.get("pct_native_support_rows")
    native_li = (
        f'{native:.0f}% of support-mode readings here are charted at native (sub-hourly) resolution'
        if native is not None else
        'where ventilator settings are charted only hourly a brief trial can be missed')
    pts_pct = 100 * pts_sbt / max(pts_elig, 1)
    caveat = (
        '<div class="amber"><b>Definitions &amp; data quality.</b> One rate, broadest by default; each toggle '
        'above applies one exclusion (see <em>What each toggle does</em>). Endpoints on this cohort: all '
        f'toggles <b>off</b> = {n_spont_all:,} / {n_vent:,} = {100*_frac0:.1f}% (any spontaneous-mode '
        f'presence ÷ all vent-ICU days); all <b>on</b> = strict SBT among transition candidates, {n_sbt:,} / '
        f'{_allon_den:,} = {100*n_sbt/max(_allon_den,1):.1f}%. (The numerator is the strict-SBT transition; '
        f'the scorecard tile reports this same {n_sbt:,} numerator over transition-candidate days. The legacy '
        f'all-eligible rate {n_sbt:,} / {n_elig:,} = {100*n_sbt/max(n_elig,1):.1f}% keeps parked-on-spontaneous '
        f'days in the denominator.)'
        '<ul>'
        '<li><b>Candidate-day filters (denominator):</b> tracheostomy, continuous paralytic, ≥12 h controlled '
        'accrued, ≥2 h stable oxygenation (FiO₂≤0.50/PEEP≤8/SpO₂≥88), ≥2 h low vasopressors (NEE≤0.2; '
        'low-dose allowed, only days above 0.2 excluded). '
        'Removing a day removes its attempt from the numerator too (a numerator day is always a denominator '
        'day). Vasopressor is a separate toggle from oxygenation because it is not shared across institutions.</li>'
        '<li><b>Trial-quality filters (numerator):</b> require a controlled→support transition, require it '
        'sustained ≥ 2 min, require low PEEP on support (≤8 PS / ≤5 CPAP). The latter two only change whether '
        'an attempt counts; <b>require transition also trims the denominator</b> — a day already parked on a '
        'spontaneous mode with no transition is not a transition candidate, so it leaves both sides (it is not '
        'a missed SBT).</li>'
        f'<li><b>Lower bound:</b> {native_li} — a brief trial charted only hourly can be missed, so any rate '
        'with “require transition” active is a lower bound. CPAP pressure is read from PEEP (CLIF has no '
        'dedicated CPAP column).</li>'
        f'<li><b>Stability assessability:</b> when a stability toggle is on, a day with no qualifying ≥2 h '
        'window (including ~{n_notassess:,} days where the signals were too sparse to assess) is excluded. '
        'Norepinephrine-equivalents use standard published conversion factors (config-driven).</li>'
        '</ul></div>'
    )
    smallnote = (f'<div class="smallnote" id="smallnote">† Rate grayed: this slice has fewer than '
                 f'{small_cell_min} eligible days — interpret with caution.</div>')

    logo_uri = _load_logo()
    pt = build_patient_table(obs)
    table1 = build_table1(pt) if not pt.empty else pd.DataFrame()
    table1_html = render_gtsummary_table_html(table1)

    ctx = {
        "logo_uri": logo_uri, "site": site, "generated": generated,
        "controls": build_controls(slices), "smallnote": smallnote, "caveat": caveat,
        "catalogue": build_catalogue_panel(),
        "trend": '<div id="trend"></div>',
        "table1": table1_html, "table_n": len(pt), "script": script_html,
        "n_elig": n_vent,
        "frac0_dash": f"{_frac0*_C:.1f} {_C:.1f}",
        "frac0_pct": f"{100*_frac0:.0f}%",
        "hd_sub0": f"{n_spont_all:,} on a spontaneous mode ({100*_frac0:.0f}%) · All ICUs · all time",
    }
    out_path = final / "sbt_dashboard.html"
    out_path.write_text(build_html(ctx), encoding="utf-8")

    log.info("logo embedded: %s | filters: %d units × {all,month,week}; %d slice cells; small-cell min=%d",
             "yes" if logo_uri else "no", slices["unit"].nunique(), len(slices), small_cell_min)
    log.info("funnel: vent %d → non-trach %d → eligible %d → SBT %d", n_vent, n_nontrach, n_elig, n_sbt)
    log.info("Table 1 patients: %d", len(pt))
    log.info("wrote: %s (%.0f KB)", out_path.relative_to(PROJECT_ROOT), out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
