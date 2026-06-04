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


# --- figures ---
def make_consort(counts: dict, graphs_dir: Path) -> str:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    elig = max(counts["eligible"], 1)
    stages = [
        ("Ventilated-ICU Patient-Days", counts["vent"], None),
        ("Non-Tracheostomized Days", counts["nontrach"],
         f"{counts['vent']-counts['nontrach']:,} tracheostomized (excluded)"),
        ("Eligible SBT-Opportunity Days", counts["eligible"],
         f"{counts['nontrach']-counts['eligible']:,} <12h controlled or no stable window"),
        ("SBT Delivered", counts["sbt"],
         f"{counts['eligible']-counts['sbt']:,} no controlled→support transition"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); fig.patch.set_facecolor(CARD)
    box_w, box_h, cx = 5.4, 1.15, 3.0
    ys = np.linspace(8.8, 0.9, len(stages))
    for i, (label, n, excl) in enumerate(stages):
        y = ys[i]
        ax.add_patch(FancyBboxPatch((cx - box_w/2, y - box_h/2), box_w, box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.3,
                    edgecolor=MAROON_D, facecolor=CREAM))
        ax.text(cx, y + 0.16, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color=MAROON_D)
        pct = f"  ({100*n/elig:.1f}% of eligible)" if label == "SBT Delivered" else ""
        ax.text(cx, y - 0.26, f"n = {n:,}{pct}", ha="center", va="center", fontsize=11, color=INK)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((cx, y - box_h/2), (cx, ys[i+1] + box_h/2),
                        arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=MUTED))
        if excl:
            ax.annotate(excl, xy=(cx, (ys[i-1] + y) / 2 if i else y),
                        xytext=(cx + box_w/2 + 0.2, (ys[i-1] + y) / 2 if i else y),
                        ha="left", va="center", fontsize=8.0, color=MUTED)
    fig.tight_layout(); graphs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(graphs_dir / "cohort_consort.png", dpi=150, bbox_inches="tight", facecolor=CARD)
    fig.savefig(graphs_dir / "cohort_consort.svg", bbox_inches="tight", facecolor=CARD)
    return _fig_to_uri(fig)


# --- slices → embedded JS ---
def build_slices_js(slices: pd.DataFrame) -> dict:
    out: dict = {}
    for r in slices.itertuples(index=False):
        cell = {"vent": int(r.n_vent_days), "nontrach": int(r.n_nontrach),
                "elig": int(r.n_eligible), "sbt": int(r.n_sbt),
                "notassess": int(r.n_not_assessable)}
        out.setdefault(r.unit, {}).setdefault(r.granularity, {})[r.period] = cell
    return out


FILTER_JS = r"""
(function(){
  const $ = id => document.getElementById(id);
  const min = CFG.smallCellMin;
  const state = {unit: "__ALL__", gran: "all", period: "__all__"};
  const unitSel = $("f-unit"), periodSel = $("f-period"), periodWrap = $("f-period-wrap");
  const plabel = p => (CFG.periodLabels && CFG.periodLabels[p]) || p;

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

  function render(){
    const c = cell();
    const ctx = CFG.unitLabels[state.unit] + " · " +
      (state.gran === "all" || state.period === "__all__" ? "all time" : plabel(state.period));
    if (!c){
      $("hd-elig").textContent = "—";
      $("hd-sub").textContent = "no eligible days · " + ctx;
      drawDonut(null, false);
      $("smallnote").style.display = "none"; drawTrend(); drawUnits(); return;
    }
    const small = c.elig < min;
    const frac = c.elig ? c.sbt/c.elig : null;
    $("hd-elig").textContent = c.elig.toLocaleString();
    $("hd-sub").textContent = c.sbt.toLocaleString() + " had an SBT delivered (" + pct(frac) + ") · " + ctx;
    drawDonut(frac, small);
    $("smallnote").style.display = small ? "block" : "none";
    drawTrend(); drawUnits();
  }

  function cellFor(unit){
    const u = SLICES[unit] || {};
    if (state.gran === "all" || state.period === "__all__") return (u.all || {}).all || null;
    return (u[state.gran] || {})[state.period] || null;
  }

  function drawUnits(){
    const host = $("units");
    const when = (state.gran === "all" || state.period === "__all__") ? "all time" : plabel(state.period);
    $("unitsTitle").textContent = "SBT-Delivered Rate by ICU Unit · " + when;
    const rows = [];
    for (const u of CFG.unitOrder){
      const c = cellFor(u);
      if (!c || !c.elig) continue;
      rows.push({u: u, label: CFG.unitLabels[u] || u, rate: c.sbt/c.elig, sbt: c.sbt, elig: c.elig});
    }
    if (!rows.length){ host.innerHTML = '<div class="muted">No eligible days in this period.</div>'; return; }
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
      svg += '<g><title>'+r.label+'\n'+r.sbt+'/'+r.elig+' = '+(100*r.rate).toFixed(1)+'%'+(small?'  — n small':'')+'</title>';
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
    $("trendTitle").textContent = "SBT-Delivered Rate by " + Tg + " · " + CFG.unitLabels[state.unit];
    const host = $("trend");
    if (!keys.length){ host.innerHTML = '<div class="muted">No periods in this slice.</div>'; return; }
    const slot = keys.length > 40 ? 15 : (keys.length > 15 ? 34 : 56);
    const pad = {l:36, r:12, t:14, b:48}, ih = 150;
    const W = pad.l + pad.r + keys.length*slot, H = pad.t + ih + pad.b;
    let maxr = 0.05; for (const k of keys){ const d = series[k]; if (d.elig) maxr = Math.max(maxr, d.sbt/d.elig); }
    const top = Math.max(0.1, Math.ceil(maxr*100/10)*10/100);
    const lblStep = Math.ceil(keys.length/24);
    let svg = '<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<line x1="'+pad.l+'" y1="'+(pad.t+ih)+'" x2="'+(W-pad.r)+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+4)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+(100*top).toFixed(0)+'%</text>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    keys.forEach((k, i) => {
      const d = series[k], r = d.elig ? d.sbt/d.elig : 0;
      const x = pad.l + i*slot + slot*0.16, w = slot*0.68;
      const yT = pad.t + ih*(1 - r/top), hT = ih*(r/top);
      const dim = d.elig < min, sel = (k === state.period);
      const cBar = dim ? "#e2d3cc" : "#8a1f2b";
      svg += '<g><title>' + k + "\n" + d.sbt + "/" + d.elig + " SBT (" + (100*r).toFixed(0) + "%)" +
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

  unitSel.onchange = () => { state.unit = unitSel.value; fillPeriods(); render(); };
  periodSel.onchange = () => { state.period = periodSel.value; render(); };
  document.querySelectorAll("#f-gran button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#f-gran button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); state.gran = b.dataset.g; state.period = "__all__"; fillPeriods(); render();
  });
  fillPeriods(); render();
})();
"""


def build_controls(slices: pd.DataFrame) -> str:
    units = [u for u in UNIT_LABELS if u in set(slices["unit"])]
    opts = "".join(f'<option value="{html.escape(u)}">{html.escape(UNIT_LABELS[u])}</option>' for u in units)
    gran_btns = "".join('<button data-g="{g}"{on}>{lab}</button>'.format(
        g=g, on=' class="on"' if g == "all" else "", lab=html.escape(GRAN_LABELS[g]))
        for g in ("all", "month", "week"))
    return ('<div class="controls">'
            f'<label class="ctl">Unit<select id="f-unit">{opts}</select></label>'
            f'<div class="ctl">Time<div class="seg" id="f-gran">{gran_btns}</div></div>'
            '<label class="ctl" id="f-period-wrap" style="display:none">Period<select id="f-period"></select></label>'
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
<title>SBT Delivery QI — {html.escape(ctx['site'])}</title><style>{css}</style></head><body>
<div class="wrap">
  <header class="top">{brand}
    <div><a class="backlink" href="scorecard.html">← CLIF ICU Ventilator QI Bundle</a>
    <h1>Spontaneous Breathing Trial — Quality-of-Care</h1>
    <div class="sub">Daily controlled→support breathing-trial delivery (Jain et al.) · {html.escape(ctx['site'])} ·
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
    </svg><div class="donut-cap">SBT Delivered</div></div>
    <div class="hd-text">
    <div class="hd-big" id="hd-elig">{ctx['n_elig']:,}</div>
    <div class="hd-lab">Eligible Vent-ICU Days</div>
    <div class="hd-sub" id="hd-sub">{ctx['hd_sub0']}</div>
    </div></div>
  {ctx['smallnote']}

  {ctx['caveat']}

  <div class="section"><h2>SBT-Delivered Rate Over Time</h2>
    <div class="fig-caption" id="trendTitle"></div>
    <div class="trend-wrap">{ctx['trend']}</div>
    <div class="fig-caption">Each bar = the SBT-delivered rate (of eligible vent-ICU days) for one
    period in the selected unit. Bars are grayed when the period has fewer than the small-cell
    threshold of eligible days. Use the controls to switch unit/granularity; pick a Period to drill the
    headline to one bucket.</div>
  </div>

  <div class="section"><h2>SBT Delivery by ICU Unit</h2>
    <div class="fig-caption" id="unitsTitle"></div>
    <div class="trend-wrap" id="units"></div>
    <div class="fig-caption">Every ICU unit side by side for the time period selected above (the Unit
    filter does not affect this panel). The maroon <b>All ICUs</b> bar is the site-wide reference;
    units are ordered by rate. Each bar shows the SBT-delivered rate with the eligible-day count in
    parentheses; bars are grayed below the small-cell threshold.</div>
  </div>

  <div class="section"><h2>Cohort Flow</h2>
    <div class="fig"><img src="{ctx['consort_uri']}" alt="cohort funnel"></div>
    <div class="fig-caption">From ventilated-ICU patient-days to non-tracheostomized days, eligible
    SBT-opportunity days (≥12h controlled + ≥2h stable window), and days with a controlled→support
    transition. The SBT-delivered percentage is of the eligible denominator.</div>
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
    pts_sbt = int(s("patients_ever_sbt")["numerator"]); pts_elig = int(s("patients_ever_sbt")["denominator"])
    generated = str(s("vent_icu_days")["generated"])
    small_cell_min = int(cfg.get("reporting", {}).get("small_cell_min_den", 10))

    slices_js = build_slices_js(slices)
    present_units = set(slices["unit"].unique())
    unit_order = [u for u in UNIT_LABELS if u in present_units and u != "unknown"]
    period_labels = {p: _period_label(p) for p in
                     slices.loc[slices["granularity"].isin(["month", "week"]), "period"].unique()}
    cfg_js = {"smallCellMin": small_cell_min,
              "unitLabels": {u: UNIT_LABELS.get(u, u) for u in slices["unit"].unique()},
              "unitOrder": unit_order,
              "periodLabels": period_labels}
    script_html = ("<script>\nconst SLICES = " + _json.dumps(slices_js, ensure_ascii=False)
                   + ";\nconst CFG = " + _json.dumps(cfg_js, ensure_ascii=False)
                   + ";\n" + FILTER_JS + "\n</script>")

    import math
    _C = 2 * math.pi * 52
    _frac0 = n_sbt / max(n_elig, 1)

    native = diag.get("pct_native_support_rows")
    native_clause = (f"At this site {native:.0f}% of support-mode readings are native-resolution; "
                     f"a brief trial charted only hourly can be missed, so the rate is a lower bound. "
                     if native is not None else
                     "Where ventilator settings are charted only hourly a brief trial can be missed, "
                     "so the rate is a lower bound. ")
    caveat = (
        '<div class="amber"><b>Eligibility &amp; data quality (Jain et al.).</b> A ventilated-ICU day '
        'is <b>eligible</b> when ≥12h of controlled ventilation has accrued <em>and</em> a ≥2h window '
        'holds FiO2 ≤ 0.50, PEEP ≤ 8, SpO2 ≥ 88%, and norepinephrine-equivalent ≤ 0.2 mcg/kg/min. An '
        '<b>SBT</b> is counted when the patient <b>transitions</b> from a controlled mode to '
        'pressure-support/CPAP (PEEP ≤ 8, or CPAP ≤ 5) for ≥ 2 min — transition-only, so a patient '
        'already parked on support all day is not counted. <b>Tracheostomized days are excluded</b> '
        f'from numerator and denominator ({n_vent-n_nontrach:,} of {n_vent:,} vent-ICU days). '
        f'{native_clause}CPAP pressure is read from PEEP (CLIF has no dedicated CPAP column). Stability '
        f'could not be assessed on {n_notassess:,} days (reported separately, excluded from the rate). '
        f'Norepinephrine-equivalents use standard published conversion factors (config-driven). '
        f'Patient-level: {pts_sbt:,} of {pts_elig:,} eligible patients '
        f'({100*pts_sbt/max(pts_elig,1):.0f}%) ever had an SBT.</div>'
    )
    smallnote = (f'<div class="smallnote" id="smallnote">† Rate grayed: this slice has fewer than '
                 f'{small_cell_min} eligible days — interpret with caution.</div>')

    logo_uri = _load_logo()
    consort_uri = make_consort(
        {"vent": n_vent, "nontrach": n_nontrach, "eligible": n_elig, "sbt": n_sbt}, final / "graphs")

    pt = build_patient_table(obs)
    table1 = build_table1(pt) if not pt.empty else pd.DataFrame()
    table1_html = render_gtsummary_table_html(table1)

    ctx = {
        "logo_uri": logo_uri, "site": site, "generated": generated,
        "controls": build_controls(slices), "smallnote": smallnote, "caveat": caveat,
        "trend": '<div id="trend"></div>', "consort_uri": consort_uri,
        "table1": table1_html, "table_n": len(pt), "script": script_html,
        "n_elig": n_elig,
        "frac0_dash": f"{_frac0*_C:.1f} {_C:.1f}",
        "frac0_pct": f"{100*_frac0:.0f}%",
        "hd_sub0": f"{n_sbt:,} had an SBT delivered ({100*_frac0:.0f}%) · All ICUs · all time",
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
