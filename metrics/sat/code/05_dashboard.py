"""Render the SAT adherence QI dashboard (self-contained HTML).

CLIF maroon-cream house style (~/.claude/templates/dashboard_design_guide.md;
lpv scorecard/dashboard are the brand reference). One self-contained file: logo
and figures are base64-embedded so it ships as a single HTML for any site.

Components:
    - Brand header (logo lockup) + reactive metric cards (unit × period filters).
    - SAT-rate-over-time trend (reacts to filters).
    - Cohort flow funnel (vent-ICU days → eligible → SAT → resumed).
    - Kress et al. 2000 dose-resumption section (ratio distribution + ≤half-dose).
    - Table 1 — eligible patients, ever-SAT vs never (gtsummary renderer).
    - Eligibility / documentation caveat (amber info box).

Inputs (from 04_metrics.py / 01):
    output/intermediate/metrics_patient_day_level.parquet
    output/intermediate/metrics_slices.parquet
    output/intermediate/kress_resumption.parquet
    output/final/metrics_site_summary.csv
    output/final/kress_summary.csv

Output:
    output/final/sat_dashboard.html
    output/final/graphs/cohort_consort.png/.svg, kress_resumption.png
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
CODE_DIR = PROJECT_ROOT / "code"
log = logging.getLogger("sat.dashboard")

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
    """Friendly period label, matching the lpv house style:
    month 'YYYY-MM' -> 'Jul 2023'; ISO week 'YYYY-Www' -> 'Week 42 · Oct 2023'."""
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
    spec = importlib.util.spec_from_file_location("sat_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# --- embedding ---
def _load_logo(p: Path, px: int = 480):
    if not p.exists():
        return None
    try:
        from PIL import Image
        im = Image.open(p).convert("RGBA"); im.thumbnail((px, px))
        buf = BytesIO(); im.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
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


# --- Table 1: eligible patients, ever-SAT vs never ---
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
    """Collapse eligible patient-days to one row per patient for Table 1."""
    elig = obs[obs["eligible"]].copy()
    if "patient_id" not in elig.columns:
        return pd.DataFrame()
    elig["__sat_day"] = elig["sat_performed"].astype(bool)
    agg = {"__sat_day": "any", "icu_day": "count"}
    for c in ("age_at_admission", "sex_category", "race_category", "ethnicity_category",
              "admission_type_category", "discharge_category"):
        if c in elig.columns:
            agg[c] = "first"
    pt = elig.groupby("patient_id").agg(agg).rename(
        columns={"__sat_day": "ever_sat", "icu_day": "n_eligible_days"}).reset_index()
    if "discharge_category" in pt.columns:
        pt["in_hospital_mortality"] = pt["discharge_category"].astype("string").str.lower().eq("expired")
    return pt


def build_table1(pt: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    groups = {"sat": pt[pt["ever_sat"]], "no": pt[~pt["ever_sat"]]}
    n_all, n_y, n_n = len(pt), len(groups["sat"]), len(groups["no"])
    cols = ["**Characteristic**", f"**Overall**\nN = {n_all:,}",
            f"**Ever SAT**\nN = {n_y:,}", f"**Never SAT**\nN = {n_n:,}", "**p-value**"]
    rows = []

    def add_cont(label, col):
        if col not in pt.columns:
            return
        a = pd.to_numeric(groups["sat"][col], errors="coerce").dropna()
        b = pd.to_numeric(groups["no"][col], errors="coerce").dropna()
        p = stats.kruskal(a, b).pvalue if len(a) and len(b) else np.nan
        rows.append([f"__{label}__", _fmt_med(pt[col]), _fmt_med(groups["sat"][col]),
                     _fmt_med(groups["no"][col]), _fmt_p(p)])

    def add_binary(label, col):
        if col not in pt.columns:
            return
        a = groups["sat"][col].astype(bool); b = groups["no"][col].astype(bool)
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
        dy = groups["sat"][col].map(lambda v: _display(col, v))
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
    add_cont("Eligible vent-sedation days / patient", "n_eligible_days")
    add_binary("In-hospital mortality", "in_hospital_mortality")
    return pd.DataFrame(rows, columns=cols)


# --- figures ---


def make_kress_svg(kress: pd.DataFrame, half: float) -> str | None:
    """Inline-SVG histogram of the resumed/pre-hold dose ratio, drawn in the same
    visual language as the trend + per-unit bar charts (maroon bars, thin
    --line axes, muted labels, hover titles) rather than matplotlib. Site-wide /
    all-time, so it is static (not filter-reactive)."""
    import math
    r = (kress["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
         if not kress.empty else pd.Series([], dtype=float))
    if r.empty:
        return None
    edges = [i / 10 for i in range(0, 21)]          # 0.0 .. 2.0 in 0.1 bins
    clipped = r.clip(upper=1.9999)                   # >=2x folds into the last bin
    counts = [int(((clipped >= edges[i]) & (clipped < edges[i + 1])).sum()) for i in range(20)]
    median = float(r.median())

    PADL, PADR, PADT, PADB = 48, 18, 34, 42
    PLOTW, PLOTH = 648, 228
    W, H = PADL + PLOTW + PADR, PADT + PLOTH + PADB
    bw = PLOTW / 20
    raw = max(counts) or 1
    mag = 10 ** math.floor(math.log10(raw)); step = mag / 2
    ytop = math.ceil(raw / step) * step

    def xpx(v): return PADL + (v / 2.0) * PLOTW
    def ypx(c): return PADT + PLOTH - (c / ytop) * PLOTH

    p = []
    # axes
    p.append(f'<line x1="{PADL}" y1="{PADT}" x2="{PADL}" y2="{PADT+PLOTH}" stroke="#ece1d9"/>')
    p.append(f'<line x1="{PADL}" y1="{PADT+PLOTH}" x2="{PADL+PLOTW}" y2="{PADT+PLOTH}" stroke="#ece1d9"/>')
    p.append(f'<text x="{PADL-6}" y="{PADT+4}" font-size="9" text-anchor="end" fill="#9a8c86">{int(ytop):,}</text>')
    p.append(f'<text x="{PADL-6}" y="{PADT+PLOTH}" font-size="9" text-anchor="end" fill="#9a8c86">0</text>')
    # bars (maroon, small gap, like the trend bars)
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        x, y = xpx(edges[i]) + 1.2, ypx(c)
        w, h = bw - 2.4, PADT + PLOTH - ypx(c)
        p.append(f'<g><title>{edges[i]:.1f}–{edges[i+1]:.1f}×: {c:,} resumptions</title>'
                 f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#8a1f2b" rx="1.5"/></g>')
    # reference lines: Kress half-dose target (gold dashed) + same-dose (muted dotted)
    x05, x10 = xpx(half), xpx(1.0)
    p.append(f'<line x1="{x05:.1f}" y1="{PADT}" x2="{x05:.1f}" y2="{PADT+PLOTH}" stroke="#b5852a" stroke-width="2" stroke-dasharray="6 4"/>')
    p.append(f'<line x1="{x10:.1f}" y1="{PADT}" x2="{x10:.1f}" y2="{PADT+PLOTH}" stroke="#9a8c86" stroke-width="1.2" stroke-dasharray="2 3"/>')
    # top-margin labels (clear of the bars)
    p.append(f'<text x="{PADL+4}" y="16" font-size="10.5" font-weight="700" fill="#6f1622">median {median:.2f}×</text>')
    p.append(f'<text x="{x05:.1f}" y="16" font-size="10.5" font-weight="700" text-anchor="middle" fill="#b5852a">Kress target ≤{half:g}×</text>')
    p.append(f'<text x="{x10:.1f}" y="16" font-size="10" text-anchor="middle" fill="#9a8c86">same dose</text>')
    # x-axis ticks + label
    for v, lab in [(0, "0"), (0.5, "0.5×"), (1.0, "1×"), (1.5, "1.5×"), (2.0, "≥2×")]:
        p.append(f'<text x="{xpx(v):.1f}" y="{PADT+PLOTH+16}" font-size="9.5" text-anchor="middle" fill="#9a8c86">{lab}</text>')
    p.append(f'<text x="{PADL+PLOTW/2:.1f}" y="{H-4}" font-size="10.5" text-anchor="middle" fill="#9a8c86">'
             f'Resumed ÷ pre-hold dose (per drug; ≥2× clipped)</text>')

    return (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%" '
            f'role="img" aria-label="Kress dose-resumption distribution">' + "".join(p) + '</svg>')


# --- slices → embedded JS ---
def build_slices_js(slices: pd.DataFrame) -> dict:
    out: dict = {}
    for r in slices.itertuples(index=False):
        cell = {"vent": int(r.n_vent_days), "elig": int(r.n_eligible),
                "sat": int(r.n_sat), "resumed": int(r.n_resumed),
                "notresumed": int(r.n_notresumed), "extubated": int(r.n_extubated)}
        out.setdefault(r.unit, {}).setdefault(r.granularity, {})[r.period] = cell
    return out


def build_duration_payload(durs: pd.DataFrame) -> tuple[dict, dict]:
    """Compact, PHI-free arrays for the SAT off-sedation duration panel (binned live in JS).
    One record per qualifying hold: off-minutes + unit/month/week INDICES + resumed flag."""
    d = durs.copy()
    if not d.empty:
        d["icu_day"] = d["icu_day"].astype(str)
        d["mon"] = d["icu_day"].str.slice(0, 7)
        iso = pd.to_datetime(d["icu_day"], errors="coerce").dt.isocalendar()
        d["wk"] = (iso["year"].astype("Int64").astype(str) + "-W"
                   + iso["week"].astype("Int64").astype(str).str.zfill(2))
        d["unit"] = d["unit"].astype("string").fillna("unknown").replace("", "unknown")
        d["off_min"] = pd.to_numeric(d["off_min"], errors="coerce").fillna(0.0)
        d["resumed"] = d["resumed"].astype(bool)
    units = sorted(d["unit"].dropna().unique().tolist()) if not d.empty else []
    months = sorted(d["mon"].dropna().unique().tolist()) if not d.empty else []
    weeks = sorted(d["wk"].dropna().unique().tolist()) if not d.empty else []
    uidx = {u: i for i, u in enumerate(units)}
    midx = {m: i for i, m in enumerate(months)}
    widx = {w: i for i, w in enumerate(weeks)}
    if d.empty:
        DUR = {"dur": [], "u": [], "m": [], "w": [], "res": []}
    else:
        DUR = {"dur": [int(round(x)) for x in d["off_min"].tolist()],
               "u": [uidx[u] for u in d["unit"].tolist()],
               "m": [midx.get(m, -1) for m in d["mon"].tolist()],
               "w": [widx.get(w, -1) for w in d["wk"].tolist()],
               "res": [1 if x else 0 for x in d["resumed"].tolist()]}
    return DUR, {"units": units, "months": months, "weeks": weeks}


def build_eligibility_panel(med_sets: dict, hold_min: float, exclude_paralytic: bool) -> str:
    """Static, config-driven catalogue of the SAT eligibility/numerator criteria — explicit so a
    future iteration can turn it into a toggle matrix (à la SBT)."""
    def fmt(s):
        return ", ".join(sorted(s)) if s else "—"
    rows = [
        ("__sub__", "Denominator (eligibility) — who is an SAT opportunity"),
        ("Continuous SAT-relevant sedation", "einc", "include",
         f"The day has ≥1 active continuous sedative/analgesic infusion: {fmt(med_sets['sat_relevant'])}."),
        ("Dexmedetomidine allowed", "enc", "allowed",
         f"Dexmedetomidine ({fmt(med_sets['dex'])}) may continue — it does not block eligibility and is "
         "not in the held set."),
        ("No continuous paralytic", "eexc", "exclude",
         f"Days on a continuous neuromuscular blocker ({fmt(med_sets['paralytic'])}) are excluded — a "
         "paralyzed patient is not an SAT candidate."
         + ("" if exclude_paralytic else " (currently DISABLED in config).")),
        ("Safety-screen exclusions", "enc", "not codable",
         "Active seizures, alcohol withdrawal, myocardial ischemia, and raised ICP would exclude a patient "
         "clinically but are not reliably encodable in CLIF — so this is crude eligibility, not full "
         "safety-screen-passed eligibility."),
        ("__sub__", "Numerator — what counts as an SAT performed"),
        (f"All sedation held to 0 for ≥{hold_min:.0f} min", "einc", "numerator",
         f"An interval where ALL SAT-relevant infusions are simultaneously at rate 0 for ≥{hold_min:.0f} "
         "min while the patient stays ventilated and sedation was running earlier that day. "
         "Dexmedetomidine running is ignored."),
    ]
    body = ""
    for r in rows:
        if r[0] == "__sub__":
            body += f'<tr class="esub"><td colspan="3">{html.escape(r[1])}</td></tr>'
        else:
            lab, cls, role, defn = r
            body += (f'<tr><td class="elab">{html.escape(lab)}</td>'
                     f'<td>{html.escape(defn)}</td>'
                     f'<td class="erole {cls}">{html.escape(role)}</td></tr>')
    return ('<div class="section"><h2>Eligibility Criteria</h2>'
            '<div class="fig-caption">The exact inclusion / exclusion rules behind the denominator and '
            'numerator, stated explicitly and driven by <code>config.json</code>. The per-day flags they '
            'rest on (on SAT-relevant sedation · on paralytic · on dexmedetomidine) are already persisted '
            'per patient-day, so these can become an interactive <b>toggle matrix</b> like the SBT '
            'dashboard in a future iteration.</div>'
            '<table class="elig-table"><thead><tr><th>Criterion</th><th>Definition</th><th>Role</th></tr>'
            '</thead><tbody>' + body + '</tbody></table></div>')


FILTER_JS = r"""
(function(){
  const $ = id => document.getElementById(id);
  const min = CFG.smallCellMin;
  const state = {unit: "__ALL__", gran: "all", period: "__all__", unitDim: "type"};
  const unitSel = $("f-unit"), periodSel = $("f-period"), periodWrap = $("f-period-wrap");
  const plabel = p => (CFG.periodLabels && CFG.periodLabels[p]) || p;
  // "Group ICUs by" dimension: location_type (default) vs specific unit (location_name).
  // Both grains are pre-aggregated into SLICES; this just picks which unit list the by-unit
  // panel + Unit dropdown use.
  const groupSel = $("f-group");
  function unitsForDim(){ return state.unitDim === "name" ? (CFG.nameOrder || ["__ALL__"]) : CFG.unitOrder; }
  function dimNoun(){ return state.unitDim === "name" ? "Specific Unit" : "ICU Unit"; }
  function rebuildUnitOptions(){
    const list = unitsForDim();
    unitSel.innerHTML = list.map(u => '<option value="' + u + '">' + (CFG.unitLabels[u] || u) + '</option>').join('');
    state.unit = "__ALL__"; unitSel.value = "__ALL__";
  }

  // Reactive headline donut (fills to the SAT-performed rate; r=52).
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
    return Object.keys((SLICES[unit] || {})[gran] || {}).sort();
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
  function cell(){
    const u = SLICES[state.unit] || {};
    if (state.gran === "all" || state.period === "__all__") return (u.all || {}).all || null;
    return (u[state.gran] || {})[state.period] || null;
  }
  function pct(x, dp){ return x == null ? "—" : (100*x).toFixed(dp == null ? 0 : dp) + "%"; }
  function setBig(el, txt, small){ el.innerHTML = txt; el.classList.toggle("dim", !!small); }

  function render(){
    const c = cell();
    const ctx = CFG.unitLabels[state.unit] + " · " +
      (state.gran === "all" || state.period === "__all__" ? "all time" : plabel(state.period));
    if (!c){
      $("hd-elig").textContent = "—";
      $("hd-sub").textContent = "no eligible days · " + ctx;
      drawDonut(null, false);
      $("smallnote").style.display = "none"; drawTrend(); drawUnits(); drawOutcome(); drawDurations(); return;
    }
    const small = c.elig < min;
    const frac = c.elig ? c.sat/c.elig : null;
    $("hd-elig").textContent = c.elig.toLocaleString();
    $("hd-sub").textContent = c.sat.toLocaleString() + " received a SAT (" + pct(frac) + ") · " + ctx;
    drawDonut(frac, small);
    $("smallnote").style.display = small ? "block" : "none";
    drawTrend(); drawUnits(); drawOutcome(); drawDurations();
  }

  // The cell for a GIVEN unit at the current time selection (mirrors cell()).
  function cellFor(unit){
    const u = SLICES[unit] || {};
    if (state.gran === "all" || state.period === "__all__") return (u.all || {}).all || null;
    return (u[state.gran] || {})[state.period] || null;
  }

  // All ICU units side by side: one horizontal SAT-rate bar per unit for the
  // currently selected time period. Independent of the Unit selector.
  function drawUnits(){
    const host = $("units");
    const when = (state.gran === "all" || state.period === "__all__") ? "all time" : plabel(state.period);
    $("unitsTitle").textContent = "SAT-Performed Rate by " + dimNoun() + " · " + when;
    const rows = [];
    for (const u of unitsForDim()){
      const c = cellFor(u);
      if (!c || !c.elig) continue;
      rows.push({u: u, label: CFG.unitLabels[u] || u, rate: c.sat/c.elig, sat: c.sat, elig: c.elig});
    }
    if (!rows.length){ host.innerHTML = '<div class="muted">No eligible days in this period.</div>'; return; }
    // All-units reference first (maroon), then ICU units sorted by rate desc.
    const ref = rows.find(r => r.u === "__ALL__");
    let units = rows.filter(r => r.u !== "__ALL__").sort((a,b) => b.rate - a.rate);
    const ordered = (ref ? [ref] : []).concat(units);
    const rowH = 26, padL = 172, padR = 140, barMax = 360, top = 8;
    const W = padL + barMax + padR, H = top + ordered.length*rowH + 6;
    let svg = "";
    ordered.forEach((r, i) => {
      const y = top + i*rowH, small = r.elig < min, isAll = (r.u === "__ALL__");
      const w = Math.max(2, r.rate*barMax);
      const fill = small ? "#e2d3cc" : (isAll ? "#6f1622" : "#8a1f2b");
      svg += '<text x="'+(padL-8)+'" y="'+(y+rowH/2+4)+'" font-size="11.5" text-anchor="end" fill="'+(isAll?"#6f1622":"#3a2c2c")+'"'+(isAll?' font-weight="700"':'')+'>'+r.label+'</text>';
      svg += '<rect x="'+padL+'" y="'+(y+4)+'" width="'+barMax+'" height="'+(rowH-10)+'" fill="#efe4dc" rx="3"/>';
      svg += '<g><title>'+r.label+'\n'+r.sat+'/'+r.elig+' = '+(100*r.rate).toFixed(1)+'%'+(small?'  — n small':'')+'</title>';
      svg += '<rect x="'+padL+'" y="'+(y+4)+'" width="'+w+'" height="'+(rowH-10)+'" fill="'+fill+'" rx="3"/></g>';
      svg += '<text x="'+(padL+barMax+8)+'" y="'+(y+rowH/2+4)+'" font-size="11" fill="#9a8c86">'+(100*r.rate).toFixed(0)+'%  ('+r.elig.toLocaleString()+')</text>';
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" style="max-width:100%">'+svg+'</svg>';
  }

  function drawTrend(){
    const tg = state.gran === "all" ? "month" : state.gran;
    const series = (SLICES[state.unit] || {})[tg] || {};
    const keys = Object.keys(series).sort();
    const Tg = tg.charAt(0).toUpperCase() + tg.slice(1);
    $("trendTitle").textContent = "SAT-Performed Rate by " + Tg + " · " + CFG.unitLabels[state.unit];
    const host = $("trend");
    if (!keys.length){ host.innerHTML = '<div class="muted">No periods in this slice.</div>'; return; }
    const slot = keys.length > 40 ? 15 : (keys.length > 15 ? 34 : 56);
    const pad = {l:36, r:12, t:14, b:48}, ih = 150;
    const W = pad.l + pad.r + keys.length*slot, H = pad.t + ih + pad.b;
    let maxr = 0.05; for (const k of keys){ const d = series[k]; if (d.elig) maxr = Math.max(maxr, d.sat/d.elig); }
    const top = Math.max(0.1, Math.ceil(maxr*100/10)*10/100);
    const lblStep = Math.ceil(keys.length/24);
    let svg = '<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<line x1="'+pad.l+'" y1="'+(pad.t+ih)+'" x2="'+(W-pad.r)+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+4)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+(100*top).toFixed(0)+'%</text>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    keys.forEach((k, i) => {
      const d = series[k], r = d.elig ? d.sat/d.elig : 0;
      const x = pad.l + i*slot + slot*0.16, w = slot*0.68;
      const yT = pad.t + ih*(1 - r/top), hT = ih*(r/top);
      const dim = d.elig < min, sel = (k === state.period);
      const cBar = dim ? "#e2d3cc" : "#8a1f2b";
      svg += '<g><title>' + k + "\n" + d.sat + "/" + d.elig + " SAT (" + (100*r).toFixed(0) + "%)" +
             (dim ? "  — n small" : "") + '</title>';
      svg += '<rect x="'+x+'" y="'+yT+'" width="'+w+'" height="'+hT+'" fill="'+cBar+'"' + (sel ? ' stroke="#3a2c2c" stroke-width="1.5"' : '') + '/></g>';
      if (i % lblStep === 0){
        const lab = tg === "month" ? k.slice(2) : k.replace(/^\d{4}-/, "");
        const cx = x + w/2;
        svg += '<text x="'+cx+'" y="'+(H-pad.b+12)+'" font-size="8.5" text-anchor="end" fill="#9a8c86" transform="rotate(35 '+cx+' '+(H-pad.b+12)+')">'+lab+'</text>';
      }
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'" width="'+W+'" style="max-width:none">'+svg+'</svg>';
  }

  // ---- SAT outcome: what happened after the SAT (resumed / not resumed / extubated same day) ----
  function drawOutcome(){
    const c = cell();
    const host = $("outcome"), note = $("outcomeNote");
    if (!c || !c.sat){
      host.innerHTML = '<div class="muted">No SATs in this slice.</div>'; note.innerHTML = ''; drawExtubTrend(); return;
    }
    const sat = c.sat;
    const segs = [
      {n: c.resumed,    col: "#b5852a", lab: "Resumed sedation"},
      {n: c.notresumed, col: "#7d8a86", lab: "Not resumed that day"},
      {n: c.extubated,  col: "#8a1f2b", lab: "Off IMV (extubated) by end of day"}
    ];
    const W = 580, rowH = 32, padL = 224, barMax = W - padL - 86, top = 2;
    const H = top + segs.length*rowH + 2;
    let svg = "";
    segs.forEach((s, i) => {
      const y = top + i*rowH, r = sat ? s.n/sat : 0, w = Math.max(1, r*barMax);
      svg += '<text x="'+(padL-10)+'" y="'+(y+rowH/2+3)+'" font-size="11.5" text-anchor="end" fill="#3a2c2c">'+s.lab+'</text>';
      svg += '<rect x="'+padL+'" y="'+(y+5)+'" width="'+barMax+'" height="'+(rowH-12)+'" fill="#efe4dc" rx="3"/>';
      svg += '<rect x="'+padL+'" y="'+(y+5)+'" width="'+w+'" height="'+(rowH-12)+'" fill="'+s.col+'" rx="3">'
           + '<title>'+s.lab+': '+s.n.toLocaleString()+' / '+sat.toLocaleString()+'</title></rect>';
      svg += '<text x="'+(padL+barMax+8)+'" y="'+(y+rowH/2+3)+'" font-size="11" fill="#6b5d57">'+pct(r)+'  ('+s.n.toLocaleString()+')</text>';
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" style="max-width:100%">'+svg+'</svg>';
    note.innerHTML = 'Of <b>'+sat.toLocaleString()+'</b> SATs in this slice, <b>'+pct(sat?c.extubated/sat:null)+'</b> ended '
      + 'with the patient <b>off invasive ventilation (extubated) by the end of the SAT day</b> (alive). The sedation '
      + 'side: '+pct(sat?c.resumed/sat:null)+' restarted sedation, '+pct(sat?c.notresumed/sat:null)+' did not that day '
      + '(these two are complementary; same-day extubation overlaps both).';
    drawExtubTrend();
  }
  function drawExtubTrend(){
    const tg = state.gran === "all" ? "month" : state.gran;
    const series = (SLICES[state.unit] || {})[tg] || {};
    const keys = Object.keys(series).sort();
    const Tg = tg.charAt(0).toUpperCase() + tg.slice(1);
    $("extubTrendTitle").textContent = "Same-day extubation rate (of SATs) by " + Tg + " · " + CFG.unitLabels[state.unit];
    const host = $("extubTrend");
    if (!keys.length){ host.innerHTML = '<div class="muted">No periods in this slice.</div>'; return; }
    const slot = keys.length > 40 ? 15 : (keys.length > 15 ? 34 : 56);
    const pad = {l:36, r:12, t:14, b:48}, ih = 150;
    const W = pad.l + pad.r + keys.length*slot, H = pad.t + ih + pad.b;
    let maxr = 0.05; for (const k of keys){ const d = series[k]; if (d.sat) maxr = Math.max(maxr, d.extubated/d.sat); }
    const top = Math.max(0.1, Math.ceil(maxr*100/10)*10/100);
    const lblStep = Math.ceil(keys.length/24);
    let svg = '<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<line x1="'+pad.l+'" y1="'+(pad.t+ih)+'" x2="'+(W-pad.r)+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+4)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+(100*top).toFixed(0)+'%</text>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    keys.forEach((k, i) => {
      const d = series[k], r = d.sat ? d.extubated/d.sat : 0;
      const x = pad.l + i*slot + slot*0.16, w = slot*0.68;
      const yT = pad.t + ih*(1 - r/top), hT = ih*(r/top);
      const dim = d.sat < min, sel = (k === state.period);
      const cBar = dim ? "#e2d3cc" : "#8a1f2b";
      svg += '<g><title>' + k + "\n" + d.extubated + "/" + d.sat + " extubated (" + (100*r).toFixed(0) + "%)" +
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

  // ---- SAT off-sedation hold-duration histogram + percentile table ----
  const DBUCK = [0, 60, 120, 240, 480, 720, Infinity];
  const DBLAB = ["<1h", "1–2h", "2–4h", "4–8h", "8–12h", ">12h"];
  function durValues(){
    const selU = state.unit === "__ALL__" ? -1 : DURCFG.units.indexOf(state.unit);
    let mode = 0, sel = -1;
    if (!(state.gran === "all" || state.period === "__all__")){
      if (state.gran === "month"){ mode = 1; sel = DURCFG.months.indexOf(state.period); }
      else if (state.gran === "week"){ mode = 2; sel = DURCFG.weeks.indexOf(state.period); }
    }
    const D = DUR.dur, U = DUR.u, M = DUR.m, W = DUR.w, R = DUR.res, out = []; let nres = 0;
    for (let i = 0; i < D.length; i++){
      if (selU >= 0 && U[i] !== selU) continue;
      if (mode === 1 && M[i] !== sel) continue;
      if (mode === 2 && W[i] !== sel) continue;
      out.push(D[i]); if (R[i]) nres++;
    }
    return {vals: out, nres: nres};
  }
  function fmtDur(m){ return m == null ? "—" : (m < 60 ? m.toFixed(0) + " min" : (m/60).toFixed(1) + " h"); }
  function qtile(sorted, q){
    if (!sorted.length) return null;
    return sorted[Math.min(sorted.length - 1, Math.round(q * (sorted.length - 1)))];
  }
  function drawDurations(){
    const dv = durValues(), vals = dv.vals;
    const when = (state.gran === "all" || state.period === "__all__") ? "all time" : plabel(state.period);
    $("durTitle").textContent = "SAT off-sedation hold duration · " + CFG.unitLabels[state.unit] + " · " + when;
    const host = $("durHist"), tbl = $("durTable"), note = $("durNote");
    if (!vals.length){
      host.innerHTML = '<div class="muted">No SAT holds in this slice.</div>'; tbl.innerHTML = ""; note.innerHTML = ""; return;
    }
    const n = vals.length;
    const counts = new Array(DBLAB.length).fill(0);
    for (const v of vals){
      let b = DBUCK.length - 2;
      for (let j = 0; j < DBUCK.length - 1; j++){ if (v < DBUCK[j+1]){ b = j; break; } }
      counts[b]++;
    }
    const maxc = Math.max.apply(null, counts) || 1;
    const padL = 34, padB = 30, padT = 16, padR = 10, bw = 78, ih = 168;
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
    let head = '<tr><th>holds (n)</th>', body = '<tr><td>'+n.toLocaleString()+'</td>';
    for (const [lab,q] of qs){ head += '<th>'+lab+'</th>'; body += '<td>'+fmtDur(qtile(s,q))+'</td>'; }
    tbl.innerHTML = '<table class="dur-table">'+head+'</tr>'+body+'</tr></table>';
    note.innerHTML = '<b>'+dv.nres.toLocaleString()+' ('+pct(n?dv.nres/n:null)+')</b> of these holds were followed by '
      + 'sedation being restarted; the remaining '+(n-dv.nres).toLocaleString()+' left the patient off continuous '
      + 'sedation (a successful interruption needing no further sedation — often pre-extubation).';
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
  fillPeriods(); render();
})();
"""


def _card_slot(cid, label, big0, sub0):
    return (f'<div class="mcard"><div class="big" id="{cid}big">{big0}</div>'
            f'<div class="mlab">{html.escape(label)}</div>'
            f'<div class="msub" id="{cid}sub">{html.escape(sub0)}</div></div>')


def build_controls(slices: pd.DataFrame) -> str:
    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    units = [u for u in UNIT_LABELS if u in set(typ["unit"])]
    opts = "".join(f'<option value="{html.escape(u)}">{html.escape(UNIT_LABELS[u])}</option>' for u in units)
    name_units = sorted(set(slices.loc[slices["dim"] == "name", "unit"])) if "dim" in slices.columns else []
    # "Group ICUs by" toggle — only when ≥1 location_type splits into multiple specific units.
    splits = "dim" in slices.columns and slices[slices["dim"] == "name"].groupby("parent")["unit"].nunique().gt(1).any()
    group_ctl = ('<label class="ctl">Group ICUs by<select id="f-group">'
                 '<option value="type">ICU type</option>'
                 f'<option value="name">Specific unit ({len(name_units)})</option>'
                 '</select></label>') if (name_units and splits) else ""
    gran_btns = "".join('<button data-g="{g}"{on}>{lab}</button>'.format(
        g=g, on=' class="on"' if g == "all" else "", lab=html.escape(GRAN_LABELS[g]))
        for g in ("all", "month", "week"))
    return ('<div class="controls">'
            + group_ctl
            + f'<label class="ctl">Unit<select id="f-unit">{opts}</select></label>'
            f'<div class="ctl">Time<div class="seg" id="f-gran">{gran_btns}</div></div>'
            '<label class="ctl" id="f-period-wrap" style="display:none">Period<select id="f-period"></select></label>'
            '</div>')


def build_kress_table(kress_sum: pd.DataFrame) -> str:
    # "All drugs" total row first, then individual drugs sorted by n (desc).
    ks = kress_sum.copy()
    ks["_isall"] = ks["drug"] == "__ALL__"
    ks = ks.sort_values(["_isall", "n_resumptions"], ascending=[False, False])
    rows = []
    for r in ks.itertuples(index=False):
        if not r.n_resumptions:
            continue
        name = "All drugs" if r.drug == "__ALL__" else str(r.drug).capitalize()
        med = f"{r.median_ratio:.2f}× ({r.q1_ratio:.2f}–{r.q3_ratio:.2f})" if r.median_ratio is not None else "—"
        rows.append([f"__{name}__" if r.drug == "__ALL__" else name,
                     f"{int(r.n_resumptions):,}", med,
                     f"{r.pct_at_or_below_half:.0f}%" if r.pct_at_or_below_half is not None else "—"])
    df = pd.DataFrame(rows, columns=["**Drug**", "**Resumptions (n)**",
                                     "**Resumed÷prior, median (IQR)**", "**≤half-dose**"])
    return render_gtsummary_table_html(df)


def build_html(ctx) -> str:
    brand = (f'<img src="{ctx["logo_uri"]}" alt="CLIF">' if ctx["logo_uri"]
             else '<span style="font-size:28px;font-weight:800;color:#8a1f2b">CLIF</span>')
    kress_block = (
        f'<div class="section"><h2>Sedation Resumed After a SAT — the Kress (2000) Half-Dose Benchmark '
        f'<span style="font-size:12px;font-weight:600;color:var(--muted)">· site-wide · all time</span></h2>'
        f'<div class="fig-caption">Among SATs that <b>resumed</b> sedation (the interruption was not '
        f'tolerated, n = {ctx["kress_n"]:,} drug-level resumptions), the ratio of the restarted infusion '
        f'rate to the pre-hold rate, per drug. Kress et al. recommended restarting at <b>half</b> the '
        f'prior dose; the dashed line marks that target. Successful SATs that never restarted sedation '
        f'are excluded from this denominator.</div>'
        f'<div class="trend-wrap">{ctx["kress_svg"]}</div>'
        f'{ctx["kress_table"]}</div>'
    ) if ctx["kress_svg"] else (
        '<div class="section"><h2>Sedation Resumed After a SAT — the Kress (2000) Half-Dose Benchmark</h2>'
        '<div class="amber">No resumed SATs with a comparable pre/post dose were found.</div></div>')

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
.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:24px 0 34px;}}
.mcard{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 16px;
text-align:center;box-shadow:0 3px 10px rgba(120,30,40,.05);}}
.mcard .big{{font-size:32px;font-weight:800;color:var(--maroon);font-variant-numeric:tabular-nums;line-height:1.05;}}
.mcard .big.dim{{color:var(--muted);}}
.mcard .mlab{{font-size:13px;font-weight:700;color:var(--ink);margin-top:5px;}}
.mcard .msub{{font-size:11.5px;color:var(--muted);margin-top:3px;min-height:28px;}}
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
.smallnote{{display:none;font-size:11.5px;color:var(--warn);margin:-22px 0 26px;}}
.trend-wrap{{overflow-x:auto;padding-bottom:4px;}}
.muted{{color:var(--muted);font-size:13px;}}
.dur-wrap{{display:flex;flex-wrap:wrap;align-items:flex-start;gap:24px;}}
.dur-table{{border-collapse:collapse;margin:8px 0 2px;font-size:13px;}}
.dur-table th{{background:var(--cream);color:var(--maroon-d);font-weight:700;padding:7px 15px;
border-bottom:2px solid var(--maroon-d);text-align:center;}}
.dur-table td{{padding:7px 15px;border-bottom:1px solid var(--line);text-align:center;
font-variant-numeric:tabular-nums;}}
.durNote{{font-size:13.5px;color:var(--ink);margin-top:12px;line-height:1.65;
background:var(--cream);border:1px solid var(--line);border-radius:10px;padding:12px 16px;}}
.durNote b{{color:var(--maroon-d);}}
.elig-table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px;}}
.elig-table th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
color:var(--muted);border-bottom:1px solid var(--line);padding:6px 10px;}}
.elig-table td{{padding:7px 10px;border-bottom:1px solid #f0e7e1;vertical-align:top;}}
.elig-table .esub td{{background:var(--cream);font-weight:800;color:var(--maroon-d);
font-size:11px;text-transform:uppercase;letter-spacing:.04em;}}
.elig-table .elab{{font-weight:700;color:var(--ink);white-space:nowrap;}}
.erole{{font-weight:700;white-space:nowrap;font-size:11.5px;}}
.erole.einc{{color:#2f6f7d;}} .erole.eexc{{color:#8a1f2b;}} .erole.enc{{color:#9a8c86;}}
.fig{{text-align:center;margin:6px 0;}}
.fig img{{max-width:100%;height:auto;border-radius:8px;}}
.fig-caption{{font-size:13px;color:var(--muted);margin-top:8px;text-align:left;}}
.amber{{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:10px;
padding:14px 18px;font-size:13px;margin:0 0 22px;}}
.amber b{{color:#7a3a0a;}}
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
<title>SAT Adherence QI — {html.escape(ctx['site'])}</title><style>{css}</style></head><body>
<div class="wrap">
  <header class="top">{brand}
    <div><a class="backlink" href="scorecard.html">← CLIF ICU Ventilator QI Bundle</a>
    <h1>Spontaneous Awakening Trial — Quality-of-Care</h1>
    <div class="sub">Daily sedation-interruption adherence · {html.escape(ctx['site'])} ·
    generated {html.escape(ctx['generated'])}</div></div>
  </header>

  {ctx['controls']}
  {ctx['cards']}
  {ctx['smallnote']}

  {ctx['caveat']}

  <div class="section"><h2>SAT-Performed Rate Over Time</h2>
    <div class="fig-caption" id="trendTitle"></div>
    <div class="trend-wrap">{ctx['trend']}</div>
    <div class="fig-caption">Each bar = the SAT-performed rate (of eligible vent-sedation days) for one
    period in the selected unit. Bars are grayed when the period has fewer than the small-cell
    threshold of eligible days. Use the controls to switch unit/granularity; pick a Period to drill the
    cards to one bucket.</div>
  </div>

  <div class="section"><h2>SAT Adherence by ICU Unit</h2>
    <div class="fig-caption" id="unitsTitle"></div>
    <div class="trend-wrap" id="units"></div>
    <div class="fig-caption">Every ICU unit side by side for the time period selected above (the Unit
    filter does not affect this panel). The maroon <b>All ICUs</b> bar is the site-wide reference;
    units are ordered by rate. Each bar shows the SAT-performed rate with the eligible-day count in
    parentheses; bars are grayed below the small-cell threshold.</div>
  </div>

  <div class="section"><h2>What Happens After a SAT?</h2>
    <div class="fig-caption">Of the SATs delivered in the selected slice: the share that <b>resumed sedation</b>,
    the share <b>not resumed</b> that day, and the share that ended with the patient <b>off invasive
    ventilation (extubated) by the end of the SAT day</b>. Reacts to Unit / Time / Period.</div>
    <div id="outcome"></div>
    <div class="durNote" id="outcomeNote"></div>
    <div class="fig-caption" id="extubTrendTitle" style="margin-top:20px"></div>
    <div class="trend-wrap" id="extubTrend"></div>
    <div class="fig-caption"><b>Extubated by end of day</b> = not on an invasive-vent device at the next
    midnight after the SAT day, alive — built on the pure-IMV device timeline, so transfer out of the ICU
    while still intubated and death on the vent do <b>not</b> count, and reintubation before midnight keeps
    the patient “on.” Where charting simply stops, end-of-day status is a bound.</div>
  </div>

  <div class="section"><h2>How Long Are the SATs?</h2>
    <div class="fig-caption" id="durTitle"></div>
    <div class="dur-wrap"><div class="trend-wrap" id="durHist"></div><div id="durTable"></div></div>
    <div class="durNote" id="durNote"></div>
    <div class="fig-caption">Distribution of the <b>off-sedation hold duration</b> for every qualifying
    SAT (all SAT-relevant infusions at rate 0) in the selected slice. Reacts to Unit / Time / Period.
    Long holds (&gt;12 h) are sustained sedation-free time, often pre-extubation; a brief hold charted
    only hourly can be missed, so short durations are a lower bound.</div>
  </div>

  {kress_block}

  {ctx['eligibility']}

  <div class="section"><h2>Table 1 — Eligible Patients, Ever-SAT vs Never (n = {ctx['table_n']:,})
    <span style="font-size:12px;font-weight:600;color:var(--muted)">· site-wide · all time</span></h2>
    <div class="fig-caption">Patients with ≥1 eligible vent-sedation day, stratified by whether a SAT
    was ever performed. Continuous: median (Q1, Q3), Kruskal–Wallis. Categorical: n (%), χ².
    Patient-level secondary framing — the headline metric is day-level.</div>
    {ctx['table1']}
  </div>

  <footer>CLIF consortium · multi-site federated QI · SAT vertical · row-level data never leaves the
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
    half = float(cfg["sat_observation"].get("kress_half_dose_threshold", 0.5))
    hold_min = float(cfg["sat_observation"].get("hold_min_minutes", 30))

    summary = pd.read_csv(final / "metrics_site_summary.csv")
    kress_sum = pd.read_csv(final / "kress_summary.csv")
    obs = pd.read_parquet(inter / "metrics_patient_day_level.parquet")
    slices = pd.read_parquet(inter / "metrics_slices.parquet")
    kress = pd.read_parquet(inter / "kress_resumption.parquet")
    durs = (pd.read_parquet(inter / "sat_durations.parquet")
            if (inter / "sat_durations.parquet").exists()
            else pd.DataFrame(columns=["unit", "icu_day", "off_min", "resumed"]))

    def s(metric):
        return summary.loc[summary["metric"] == metric].iloc[0]

    n_vent = int(s("vent_icu_days")["numerator"])
    n_elig = int(s("eligible_days")["numerator"])
    n_sat = int(s("sat_performed")["numerator"])
    n_resumed = int(s("sat_resumed")["numerator"])
    pts_sat = int(s("patients_ever_sat")["numerator"]); pts_elig = int(s("patients_ever_sat")["denominator"])
    generated = str(s("vent_icu_days")["generated"])
    small_cell_min = int(cfg.get("reporting", {}).get("small_cell_min_den", 10))

    slices_js = build_slices_js(slices)
    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    present_units = set(typ["unit"].unique())
    # Order for the side-by-side panel: __ALL__ reference first, then named ICU
    # units in canonical order (drop "unknown" — it's folded into __ALL__).
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
    unit_labels.update({n: LABELS.get(n, n) for n in name_units})         # specific-unit display (raw code or config)
    unit_labels.update({u: LABELS[u] for u in unit_order if u in LABELS})  # optional type-label override
    period_labels = {p: _period_label(p) for p in
                     slices.loc[slices["granularity"].isin(["month", "week"]), "period"].unique()}
    cfg_js = {"smallCellMin": small_cell_min,
              "unitLabels": unit_labels,
              "unitOrder": unit_order, "nameOrder": name_order,
              "periodLabels": period_labels}
    dur_payload, dur_cfg = build_duration_payload(durs)
    script_html = ("<script>\nconst SLICES = " + _json.dumps(slices_js, ensure_ascii=False)
                   + ";\nconst CFG = " + _json.dumps(cfg_js, ensure_ascii=False)
                   + ";\nconst DUR = " + _json.dumps(dur_payload, ensure_ascii=False)
                   + ";\nconst DURCFG = " + _json.dumps(dur_cfg, ensure_ascii=False)
                   + ";\n" + FILTER_JS + "\n</script>")

    import math
    _C = 2 * math.pi * 52                       # donut ring circumference (r=52)
    _frac0 = n_sat / max(n_elig, 1)             # seed = all-ICU / all-time rate
    cards = (
        '<div class="headline-card">'
        '<div class="donut-wrap">'
        '<svg viewBox="0 0 120 120" width="150" height="150" role="img" aria-label="SAT adherence donut">'
        '<circle cx="60" cy="60" r="52" fill="none" stroke="var(--bar)" stroke-width="13"/>'
        '<circle id="donut-arc" cx="60" cy="60" r="52" fill="none" stroke="var(--maroon)" '
        'stroke-width="13" stroke-linecap="round" transform="rotate(-90 60 60)" '
        f'stroke-dasharray="{_frac0*_C:.1f} {_C:.1f}"/>'
        '<text id="donut-pct" x="60" y="60" text-anchor="middle" dominant-baseline="central" '
        'font-size="30" font-weight="800" fill="var(--maroon)" '
        f'font-family="Inter,system-ui,sans-serif">{100*_frac0:.0f}%</text>'
        '</svg><div class="donut-cap">SAT Performed</div></div>'
        '<div class="hd-text">'
        f'<div class="hd-big" id="hd-elig">{n_elig:,}</div>'
        '<div class="hd-lab">Eligible Vent-Sedation Days</div>'
        f'<div class="hd-sub" id="hd-sub">{n_sat:,} received a SAT ({100*_frac0:.0f}%) · All ICUs · all time</div>'
        '</div></div>'
    )
    smallnote = (f'<div class="smallnote" id="smallnote">† Rate grayed: this slice has fewer than '
                 f'{small_cell_min} eligible days — interpret with caution.</div>')

    caveat = (
        '<div class="amber"><b>Eligibility &amp; documentation.</b> SAT delivery is derived from '
        'continuous-infusion records: a day counts as <b>SAT performed</b> when all SAT-relevant '
        f'infusions (propofol, midazolam, fentanyl &amp; other benzo/opioid infusions) are held to '
        f'rate 0 for ≥{hold_min:.0f} min while ventilated — dexmedetomidine may continue. Holds are '
        'directly observable at this site (explicit dose-0 rows and start/stop markers are charted), so '
        'this is a real documented rate, not a coverage bound. A hold counts <b>whether or not</b> '
        'sedation is later restarted: a successful interruption that leaves the patient needing no '
        'further continuous sedation is still a SAT (indeed a better outcome). <b>Eligibility is '
        'crude</b>: CLIF cannot '
        'encode all SAT safety-screen exclusions (active seizures, alcohol withdrawal, myocardial '
        'ischemia, raised ICP); continuous paralytic days are excluded. The discrete SAT-assessment '
        'field is charted on a negligible fraction of days, so it is not used as the primary signal. '
        f'Patient-level: {pts_sat:,} of {pts_elig:,} eligible patients '
        f'({100*pts_sat/max(pts_elig,1):.0f}%) ever received a SAT.</div>'
    )

    logo_uri = _load_logo(PROJECT_ROOT / "references" / "images" / "clif_logo_v2.png")
    med_sets = cohort_mod.sat_med_sets(cfg)
    exclude_paralytic = bool(cfg.get("sat_eligibility", {}).get("exclude_paralytic_days", True))
    eligibility_html = build_eligibility_panel(med_sets, hold_min, exclude_paralytic)
    kress_svg = make_kress_svg(kress, half)
    kress_n = int(kress["ratio"].replace([np.inf, -np.inf], np.nan).dropna().shape[0]) if not kress.empty else 0
    kress_table = build_kress_table(kress_sum) if kress_n else ""

    pt = build_patient_table(obs)
    table1 = build_table1(pt) if not pt.empty else pd.DataFrame()
    table1_html = render_gtsummary_table_html(table1)

    ctx = {
        "logo_uri": logo_uri, "site": site, "generated": generated,
        "controls": build_controls(slices), "cards": cards, "smallnote": smallnote,
        "caveat": caveat, "trend": '<div id="trend"></div>', "eligibility": eligibility_html,
        "kress_svg": kress_svg, "kress_table": kress_table, "kress_n": kress_n,
        "table1": table1_html, "table_n": len(pt), "script": script_html,
    }
    out_path = final / "sat_dashboard.html"
    out_path.write_text(build_html(ctx), encoding="utf-8")

    log.info("logo embedded: %s | filters: %d units × {all,month,week}; %d slice cells; small-cell min=%d",
             "yes" if logo_uri else "no", slices["unit"].nunique(), len(slices), small_cell_min)
    log.info("funnel: vent %d → eligible %d → SAT %d → resumed %d", n_vent, n_elig, n_sat, n_resumed)
    log.info("Kress drug-level resumptions: %d | Table 1 patients: %d", kress_n, len(pt))
    log.info("wrote: %s (%.0f KB)", out_path.relative_to(PROJECT_ROOT), out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
