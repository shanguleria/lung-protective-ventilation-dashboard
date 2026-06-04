"""Render the proning QI dashboard (self-contained HTML).

CLIF maroon-cream house style (see ~/.claude/templates/dashboard_design_guide.md
and lpv/code/05_scorecard.py). Single self-contained file: logo + figures are
base64-embedded so the dashboard ships as one HTML for any consortium site.

Components:
    - Brand header (logo lockup) + headline metric cards.
    - CONSORT funnel (matplotlib → standalone PNG/SVG + embedded).
    - Time-to-prone cumulative-incidence figure (descriptive; 7-day horizon).
    - Table 1 — proned vs not-proned within the PROSEVA-eligible cohort,
      via the verbatim gtsummary renderer (every cell html.escape'd).
    - Position-table coverage caveat (amber info box).

Inputs:
    output/intermediate/metrics_patient_level.parquet   (per eligible patient)
    output/final/metrics_site_summary.csv               (counts + rates)
    output/final/cohort_flow.csv                         (CONSORT counts)

Output:
    output/final/proning_dashboard.html
    output/final/graphs/cohort_consort.png / .svg
"""

from __future__ import annotations

import base64
import html
import importlib.util
import logging
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
log = logging.getLogger("proning.dashboard")

# --- palette (CLIF maroon-cream) -------------------------------------------
MAROON, MAROON_D, CREAM = "#8a1f2b", "#6f1622", "#f6efe9"
CARD, INK, MUTED, LINE, BAR = "#fffdfb", "#3a2c2c", "#9a8c86", "#ece1d9", "#efe4dc"
GOOD, WARN, BAD = "#2f7d5b", "#b5852a", "#a23b3b"

CDF_HORIZONS_H = [24, 48, 72, 168]

# Site-stored categorical values → display labels (data left untouched on disk).
CATEGORICAL_DISPLAY = {
    "admission_type_category": {
        "ed": "Emergency dept.", "osh": "Outside-hospital transfer",
        "direct": "Direct admission", "facility": "Facility transfer",
    },
    "sex_category": {"male": "Male", "female": "Female"},
}

# Unit slug → display label (dashboard filter; ordered for the dropdown).
UNIT_LABELS = {
    "__ALL__": "All ICUs",
    "medical_icu": "Medical ICU",
    "mixed_cardiothoracic_icu": "Cardiothoracic ICU",
    "surgical_icu": "Surgical ICU",
    "mixed_neuro_icu": "Neuro ICU",
    "general_icu": "General ICU",
    "burn_icu": "Burn ICU",
    "unknown": "Unknown unit",
}
GRAN_LABELS = {"all": "All-time", "year": "Yearly", "month": "Monthly", "week": "Weekly"}


def _period_label(key: str) -> str:
    """month 'YYYY-MM' -> 'Jul 2023'; ISO week 'YYYY-Www' -> 'Week 42 · Oct 2023';
    year 'YYYY' -> unchanged."""
    import datetime as _dt
    try:
        if "-W" in key:
            y, w = key.split("-W")
            d = _dt.date.fromisocalendar(int(y), int(w), 1)
            return f"Week {int(w)} · {d.strftime('%b %Y')}"
        if "-" in key:                       # month YYYY-MM
            return _dt.datetime.strptime(key + "-01", "%Y-%m-%d").strftime("%b %Y")
        return key                            # year YYYY already friendly
    except Exception:
        return key


def _load_cohort_module():
    path = CODE_DIR / "01_build_cohort.py"
    spec = importlib.util.spec_from_file_location("proning_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Logo / figure embedding
# ---------------------------------------------------------------------------
def _load_logo(p: Path, px: int = 480):
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


def _fig_to_uri(fig) -> str:
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=CARD)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# gtsummary renderer (verbatim from the dashboard design guide)
# ---------------------------------------------------------------------------
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
    return (
        '<table class="results-table" border="0">'
        f"<thead><tr>{header_row}</tr></thead>"
        "<tbody>" + "\n".join(body_rows) + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Table 1 builder — proned vs not-proned within the eligible cohort
# ---------------------------------------------------------------------------
def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_med(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "—"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}, {s.quantile(.75):.1f})"


def _fmt_np(n: int, d: int) -> str:
    return f"{n:,} ({100*n/d:.1f}%)" if d else "—"


def _display(col: str, val) -> str:
    if pd.isna(val):
        return "Unknown"
    raw = str(val)
    mp = CATEGORICAL_DISPLAY.get(col, {})
    return mp.get(raw.lower(), raw if raw else "Unknown")


def build_table1(pl: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    groups = {"proned": pl[pl["any_prone"]], "not": pl[~pl["any_prone"]]}
    n_all, n_pr, n_no = len(pl), len(groups["proned"]), len(groups["not"])
    cols = [
        "**Characteristic**",
        f"**Overall**\nN = {n_all:,}",
        f"**Proned**\nN = {n_pr:,}",
        f"**Not proned**\nN = {n_no:,}",
        "**p-value**",
    ]
    rows = []

    def add_continuous(label, col):
        s_all, s_pr, s_no = pl[col], groups["proned"][col], groups["not"][col]
        a = pd.to_numeric(s_pr, errors="coerce").dropna()
        b = pd.to_numeric(s_no, errors="coerce").dropna()
        p = stats.kruskal(a, b).pvalue if len(a) and len(b) else np.nan
        rows.append([f"__{label}__", _fmt_med(s_all), _fmt_med(s_pr), _fmt_med(s_no), _fmt_p(p)])

    def add_binary(label, mask_col):
        a = groups["proned"][mask_col].astype(bool)
        b = groups["not"][mask_col].astype(bool)
        ct = np.array([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
        try:
            p = stats.chi2_contingency(ct)[1]
        except ValueError:
            p = np.nan
        rows.append([f"__{label}__",
                     _fmt_np(int(pl[mask_col].sum()), n_all),
                     _fmt_np(int(a.sum()), n_pr),
                     _fmt_np(int(b.sum()), n_no), _fmt_p(p)])

    def add_categorical(label, col):
        disp_all = pl[col].map(lambda v: _display(col, v))
        disp_pr = groups["proned"][col].map(lambda v: _display(col, v))
        disp_no = groups["not"][col].map(lambda v: _display(col, v))
        levels = sorted(disp_all.dropna().unique())
        ct = np.array([[ (disp_pr == lv).sum() for lv in levels],
                       [ (disp_no == lv).sum() for lv in levels]])
        try:
            p = stats.chi2_contingency(ct)[1] if ct.shape[1] > 1 and ct.sum() else np.nan
        except ValueError:
            p = np.nan
        rows.append([f"__{label}__", np.nan, np.nan, np.nan, _fmt_p(p)])
        for lv in levels:
            rows.append([lv,
                         _fmt_np(int((disp_all == lv).sum()), n_all),
                         _fmt_np(int((disp_pr == lv).sum()), n_pr),
                         _fmt_np(int((disp_no == lv).sum()), n_no), np.nan])

    add_continuous("Age (years)", "age_at_admission")
    add_categorical("Sex", "sex_category")
    add_categorical("Race", "race_category")
    add_categorical("Ethnicity", "ethnicity_category")
    add_categorical("Admission type", "admission_type_category")
    add_continuous("P/F ratio at T₀ (mmHg)", "pf_at_t0")
    add_continuous("FiO₂ at T₀ (fraction)", "fio2_at_t0")
    add_continuous("PEEP at T₀ (cmH₂O)", "peep_at_t0")
    add_binary("In-hospital mortality", "in_hospital_mortality")

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_consort(counts: dict, graphs_dir: Path) -> str:
    """Minimalist vertical CONSORT funnel (CRRT house style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ("ARDS cohort", counts["ards"], None),
        ("PROSEVA-strict eligible", counts["eligible"],
         f"{counts['ards']-counts['eligible']:,} did not reach PROSEVA-strict severity"),
        ("Ever proned", counts["proned"],
         f"{counts['eligible']-counts['proned']:,} no documented prone session*"),
        ("Adherent prone ≥16 h", counts["adherent"],
         f"{counts['proned']-counts['adherent']:,} longest session <16 h"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.patch.set_facecolor(CARD)

    box_w, box_h = 5.0, 1.25
    cx = 3.1
    ys = np.linspace(8.6, 1.1, len(stages))
    for i, (label, n, excl) in enumerate(stages):
        y = ys[i]
        ax.add_patch(FancyBboxPatch(
            (cx - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.3, edgecolor=MAROON_D, facecolor=CREAM))
        ax.text(cx, y + 0.18, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=MAROON_D)
        pct = f"  ({100*n/counts['eligible']:.1f}% of eligible)" if label not in ("ARDS cohort", "PROSEVA-strict eligible") else ""
        ax.text(cx, y - 0.26, f"n = {n:,}{pct}", ha="center", va="center",
                fontsize=11, color=INK)
        if i < len(stages) - 1:
            y_next = ys[i+1]
            ax.add_patch(FancyArrowPatch(
                (cx, y - box_h/2), (cx, y_next + box_h/2),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=MUTED))
        if excl and i > 0:
            # Place the exclusion note beside the arrow that drops INTO this stage.
            ax.annotate(excl, xy=(cx, (ys[i-1] + y) / 2),
                        xytext=(cx + box_w/2 + 0.25, (ys[i-1] + y) / 2),
                        ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.text(cx - box_w/2, 0.15,
            "* position table at this site charts only proning episodes",
            ha="left", va="center", fontsize=7.5, color=MUTED, style="italic")

    fig.tight_layout()
    graphs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(graphs_dir / "cohort_consort.png", dpi=150, bbox_inches="tight", facecolor=CARD)
    fig.savefig(graphs_dir / "cohort_consort.svg", bbox_inches="tight", facecolor=CARD)
    return _fig_to_uri(fig)


def make_ttp_cdf(pl: pd.DataFrame, n_eligible: int) -> str:
    """Cumulative incidence of first prone vs hours since T_eligible (7-day horizon)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ttp = pl["time_to_prone_hours"].dropna().sort_values().to_numpy()
    grid = np.linspace(0, 168, 400)
    # event counted at horizon h if first prone occurred at or before h (negatives → 0)
    cum = np.array([(ttp <= h).sum() for h in grid]) / n_eligible * 100.0

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.plot(grid, cum, color=MAROON, linewidth=2.2)
    ax.fill_between(grid, 0, cum, color=MAROON, alpha=0.07)
    for h in CDF_HORIZONS_H:
        y = (ttp <= h).sum() / n_eligible * 100.0
        ax.plot([h], [y], "o", color=MAROON_D, ms=5)
        ax.annotate(f"{y:.1f}%", (h, y), textcoords="offset points", xytext=(4, 6),
                    fontsize=9, color=MAROON_D, fontweight="bold")
    ax.set_xlim(0, 168); ax.set_ylim(0, max(20, cum.max() * 1.25))
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("Hours since T_eligible", fontsize=11, color=INK)
    ax.set_ylabel("Cumulative % of eligible proned", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    fig.tight_layout()
    return _fig_to_uri(fig)


# ---------------------------------------------------------------------------
# Sliced metrics → embedded JS (unit × granularity × period)
# ---------------------------------------------------------------------------
def build_slices_js(slices: pd.DataFrame) -> dict:
    """Nest the slice table into SLICES[unit][granularity][period] = {counts}.
    Counts only — no ids/dates — so the embed stays PHI-free."""
    out: dict = {}
    def num(v):
        return None if pd.isna(v) else round(float(v), 1)
    for r in slices.itertuples(index=False):
        cell = {
            "den": int(r.n_eligible), "proned": int(r.n_ever_proned),
            "adherent": int(r.n_adherent), "documented": int(r.n_documented),
            "ttp_median": num(r.ttp_median_h), "ttp_q1": num(r.ttp_q1_h), "ttp_q3": num(r.ttp_q3_h),
        }
        out.setdefault(r.unit, {}).setdefault(r.granularity, {})[r.period] = cell
    return out


# Reactive filter + trend logic. Plain string (no f-string) so braces are literal;
# it reads two Python-injected globals: SLICES and CFG.
FILTER_JS = r"""
(function(){
  const $ = id => document.getElementById(id);
  const min = CFG.smallCellMin;
  const state = {unit: "__ALL__", gran: "all", period: "__all__"};

  const unitSel = $("f-unit"), periodSel = $("f-period"), periodWrap = $("f-period-wrap");
  const plabel = p => (CFG.periodLabels && CFG.periodLabels[p]) || p;

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
      ["c1","c2","c3","c4"].forEach(k => { setBig($(k+"big"), "—", false); $(k+"sub").textContent = ""; });
      $("c1sub").textContent = ctx; $("c2sub").textContent = "no eligible patients";
      $("smallnote").style.display = "none"; drawTrend(); return;
    }
    const small = c.den < min;
    setBig($("c1big"), c.den.toLocaleString(), false); $("c1sub").textContent = ctx;
    setBig($("c2big"), pct(c.den ? c.proned/c.den : null), small);
    $("c2sub").textContent = c.proned.toLocaleString() + " of " + c.den.toLocaleString() + " eligible";
    const lb = c.den ? c.adherent/c.den : null, ub = c.documented ? c.adherent/c.documented : null;
    let b3 = "—";
    if (lb != null) b3 = (100*lb).toFixed(0) + (ub != null ? "–" + (100*ub).toFixed(0) : "") + "%";
    setBig($("c3big"), b3, small);
    $("c3sub").textContent = "all-eligible " + c.adherent + "/" + c.den +
      " → charted " + c.adherent + "/" + c.documented;
    setBig($("c4big"), c.ttp_median == null ? "—" : Math.round(c.ttp_median) + " h", small);
    $("c4sub").textContent = c.ttp_median == null ? "no proned patients" :
      "IQR " + Math.round(c.ttp_q1) + "–" + Math.round(c.ttp_q3) + " h, among proned";
    $("smallnote").style.display = small ? "block" : "none";
    drawTrend();
  }

  function drawTrend(){
    const tg = state.gran === "all" ? "year" : state.gran;
    const series = (SLICES[state.unit] || {})[tg] || {};
    const keys = Object.keys(series).sort();
    $("trendTitle").textContent = "Ever-proned & adherent rate by " +
      ({year:"year", month:"month", week:"week"}[tg]) + " · " + CFG.unitLabels[state.unit];
    const host = $("trend");
    if (!keys.length){ host.innerHTML = '<div class="muted">No periods in this slice.</div>'; return; }
    const slot = keys.length > 40 ? 15 : (keys.length > 15 ? 34 : 56);
    const pad = {l:36, r:12, t:14, b:48}, ih = 150;
    const W = pad.l + pad.r + keys.length*slot, H = pad.t + ih + pad.b;
    let maxr = 0.05; for (const k of keys){ const d = series[k]; if (d.den) maxr = Math.max(maxr, d.proned/d.den); }
    const top = Math.max(0.1, Math.ceil(maxr*100/10)*10/100);
    const lblStep = Math.ceil(keys.length/24);
    let svg = '<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<line x1="'+pad.l+'" y1="'+(pad.t+ih)+'" x2="'+(W-pad.r)+'" y2="'+(pad.t+ih)+'" stroke="#ece1d9"/>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+4)+'" font-size="9" text-anchor="end" fill="#9a8c86">'+(100*top).toFixed(0)+'%</text>' +
              '<text x="'+(pad.l-6)+'" y="'+(pad.t+ih)+'" font-size="9" text-anchor="end" fill="#9a8c86">0</text>';
    keys.forEach((k, i) => {
      const d = series[k], r = d.den ? d.proned/d.den : 0, ar = d.den ? d.adherent/d.den : 0;
      const x = pad.l + i*slot + slot*0.16, w = slot*0.68;
      const yT = pad.t + ih*(1 - r/top), hT = ih*(r/top);
      const yA = pad.t + ih*(1 - ar/top), hA = ih*(ar/top);
      const dim = d.den < min, sel = (k === state.period);
      const cL = dim ? "#e2d3cc" : "#e9c2c7", cD = dim ? "#b39a93" : "#8a1f2b";
      svg += '<g><title>' + k + "\n" + d.proned + "/" + d.den + " proned (" + (100*r).toFixed(0) + "%)\n" +
             d.adherent + "/" + d.den + " adherent" + (dim ? "  — n small" : "") + '</title>';
      svg += '<rect x="'+x+'" y="'+yT+'" width="'+w+'" height="'+hT+'" fill="'+cL+'"' + (sel ? ' stroke="#3a2c2c" stroke-width="1.5"' : '') + '/>';
      svg += '<rect x="'+x+'" y="'+yA+'" width="'+w+'" height="'+hA+'" fill="'+cD+'"/></g>';
      if (i % lblStep === 0){
        const lab = tg === "month" ? k.slice(2) : (tg === "week" ? k.replace(/^\d{4}-/, "") : k);
        const cx = x + w/2;
        svg += '<text x="'+cx+'" y="'+(H-pad.b+12)+'" font-size="8.5" text-anchor="end" fill="#9a8c86" transform="rotate(35 '+cx+' '+(H-pad.b+12)+')">'+lab+'</text>';
      }
    });
    host.innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'" width="'+W+'" style="max-width:none">'+svg+'</svg>';
  }

  // wire controls
  unitSel.onchange = () => { state.unit = unitSel.value; fillPeriods(); render(); };
  periodSel.onchange = () => { state.period = periodSel.value; render(); };
  document.querySelectorAll("#f-gran button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#f-gran button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); state.gran = b.dataset.g; state.period = "__all__"; fillPeriods(); render();
  });
  fillPeriods(); render();
})();
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
def _card(big, label, sub):
    return (f'<div class="mcard"><div class="big">{big}</div>'
            f'<div class="mlab">{html.escape(label)}</div>'
            f'<div class="msub">{html.escape(sub)}</div></div>')


def _card_slot(cid, label, big0, sub0):
    """Card with an id'd big/sub so the filter JS can rewrite it; seeded with the
    all-units/all-time values so it reads correctly even before JS runs."""
    return (f'<div class="mcard"><div class="big" id="{cid}big">{big0}</div>'
            f'<div class="mlab">{html.escape(label)}</div>'
            f'<div class="msub" id="{cid}sub">{html.escape(sub0)}</div></div>')


def build_controls(slices: pd.DataFrame) -> str:
    units = [u for u in UNIT_LABELS if u in set(slices["unit"])]
    opts = "".join(
        f'<option value="{html.escape(u)}">{html.escape(UNIT_LABELS[u])}</option>' for u in units)
    gran_btns = "".join(
        '<button data-g="{g}"{on}>{lab}</button>'.format(
            g=g, on=' class="on"' if g == "all" else "", lab=html.escape(GRAN_LABELS[g]))
        for g in ("all", "year", "month", "week"))
    return (
        '<div class="controls">'
        f'<label class="ctl">Unit<select id="f-unit">{opts}</select></label>'
        f'<div class="ctl">Time<div class="seg" id="f-gran">{gran_btns}</div></div>'
        '<label class="ctl" id="f-period-wrap" style="display:none">Period<select id="f-period"></select></label>'
        '</div>'
    )


def build_html(logo_uri, controls_html, cards_html, smallnote_html, trend_html, caveat,
               consort_uri, ttp_uri, table1_html, table_n, site, generated, script_html) -> str:
    brand = (f'<img src="{logo_uri}" alt="CLIF">' if logo_uri
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
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0 34px;}}
.mcard{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 16px;
text-align:center;box-shadow:0 3px 10px rgba(120,30,40,.05);}}
.mcard .big{{font-size:32px;font-weight:800;color:var(--maroon);font-variant-numeric:tabular-nums;line-height:1.05;}}
.mcard .big.dim{{color:var(--muted);}}
.mcard .mlab{{font-size:13px;font-weight:700;color:var(--ink);margin-top:5px;}}
.mcard .msub{{font-size:11.5px;color:var(--muted);margin-top:3px;min-height:28px;}}
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
table.results-table{{border-collapse:collapse;width:auto;font-size:13px;margin-top:6px;}}
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
<title>ARDS Proning QI — {html.escape(site)}</title><style>{css}</style></head><body>
<div class="wrap">
  <header class="top">{brand}
    <div><a class="backlink" href="scorecard.html">← CLIF ICU Ventilator QI Bundle</a>
    <h1>ARDS Proning — Quality-of-Care</h1>
    <div class="sub">PROSEVA-strict proning eligibility &amp; adherence · {html.escape(site)} ·
    generated {html.escape(generated)}</div></div>
  </header>

  {controls_html}
  <div class="cards">{cards_html}</div>
  {smallnote_html}

  {caveat}

  <div class="section"><h2>Proning rate over time</h2>
    <div class="fig-caption" id="trendTitle"></div>
    <div class="trend-wrap">{trend_html}</div>
    <div class="fig-caption">Each bar is one period for the selected unit. Full bar = ever-proned
    rate; dark inner = adherent-≥16&nbsp;h rate (lower bound, of all eligible). Bars are grayed when
    the period has fewer than the small-cell threshold of eligible patients. Use the controls above
    to switch unit and granularity; pick a Period to drill the cards to a single bucket.</div>
  </div>

  <div class="section"><h2>Cohort flow</h2>
    <div class="fig"><img src="{consort_uri}" alt="CONSORT funnel"></div>
    <div class="fig-caption">From the ARDS cohort to PROSEVA-strict eligibility, any documented
    prone session, and PROSEVA-adherent (≥16 h) proning. Percentages are of the eligible
    denominator.</div>
  </div>

  <div class="section"><h2>Time from eligibility to first prone <span style="font-size:12px;font-weight:600;color:var(--muted)">· site-wide · all time</span></h2>
    <div class="fig"><img src="{ttp_uri}" alt="time-to-prone cumulative incidence"></div>
    <div class="fig-caption">Cumulative incidence of the first documented prone session over the
    7&nbsp;days after T_eligible, as a fraction of all PROSEVA-eligible patients (descriptive;
    patients never proned are event-free, not censored). Markers at 24 / 48 / 72 / 168 h.</div>
  </div>

  <div class="section"><h2>Table 1 — Baseline characteristics (eligible n = {table_n:,}) <span style="font-size:12px;font-weight:600;color:var(--muted)">· site-wide · all time</span></h2>
    <div class="fig-caption">PROSEVA-eligible patients, stratified by whether a prone session was
    ever documented. Continuous variables: median (Q1, Q3), Kruskal–Wallis. Categorical: n (%),
    χ². "Not proned" includes patients with no position record (see coverage note above).</div>
    {table1_html}
  </div>

  <footer>CLIF consortium · multi-site federated QI · proning vertical · row-level data never
  leaves the site — only counts and rates are shared.</footer>
</div>
{script_html}
</body></html>"""


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cohort_mod.LOGS_DIR / "05_dashboard.log", mode="w"),
        ],
    )
    cfg = cohort_mod.load_config(cohort_mod.CONFIG_PATH)
    site = cfg.get("site", "unknown")
    inter, final = cohort_mod.INTERMEDIATE_DIR, cohort_mod.FINAL_DIR

    pl_path = inter / "metrics_patient_level.parquet"
    summary_path = final / "metrics_site_summary.csv"
    if not pl_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run code/04_metrics.py first (metrics outputs missing).")
    pl = pd.read_parquet(pl_path)
    summary = pd.read_csv(summary_path)

    def s(metric):
        return summary.loc[summary["metric"] == metric].iloc[0]

    n_ards = int(s("ards_cohort")["numerator"])
    n_eligible = int(s("proseva_eligible")["numerator"])
    n_proned = int(s("ever_proned")["numerator"])
    n_adherent = int(s("adherent_all_eligible")["numerator"])
    n_documented = int(s("position_data_present")["numerator"])
    ttp_median = s("time_to_prone_median_h")["rate"]
    ttp_q1 = s("time_to_prone_q1_h")["rate"]
    ttp_q3 = s("time_to_prone_q3_h")["rate"]
    generated = str(s("ards_cohort")["generated"])

    counts = {"ards": n_ards, "eligible": n_eligible, "proned": n_proned, "adherent": n_adherent}

    # ---- sliced metrics → filters + trend (unit × granularity × period) ----
    slices_path = inter / "metrics_slices.parquet"
    if not slices_path.exists():
        raise FileNotFoundError("Run code/04_metrics.py first (metrics_slices.parquet missing).")
    slices = pd.read_parquet(slices_path)
    small_cell_min = int(cfg.get("reporting", {}).get("small_cell_min_den", 10))

    slices_js = build_slices_js(slices)
    period_labels = {p: _period_label(p) for p in
                     slices.loc[slices["granularity"].isin(["year", "month", "week"]), "period"].unique()}
    cfg_js = {
        "smallCellMin": small_cell_min,
        "unitLabels": {u: UNIT_LABELS.get(u, u) for u in slices["unit"].unique()},
        "granLabels": GRAN_LABELS,
        "periodLabels": period_labels,
    }
    import json as _json
    script_html = (
        "<script>\nconst SLICES = " + _json.dumps(slices_js, ensure_ascii=False)
        + ";\nconst CFG = " + _json.dumps(cfg_js, ensure_ascii=False)
        + ";\n" + FILTER_JS + "\n</script>"
    )
    controls_html = build_controls(slices)

    # Cards seeded with the all-units/all-time values; JS rewrites them on filter change.
    cards = "".join([
        _card_slot("c1", "PROSEVA-eligible", f"{n_eligible:,}", "All ICUs · all time"),
        _card_slot("c2", "Ever proned", f"{100*n_proned/n_eligible:.0f}%",
                   f"{n_proned:,} of {n_eligible:,} eligible"),
        _card_slot("c3", "Adherent ≥16 h",
                   f"{100*n_adherent/n_eligible:.0f}–{100*n_adherent/n_documented:.0f}%",
                   f"all-eligible {n_adherent:,}/{n_eligible:,} → charted {n_adherent:,}/{n_documented:,}"),
        _card_slot("c4", "Median time to prone", f"{ttp_median:.0f} h",
                   f"IQR {ttp_q1:.0f}–{ttp_q3:.0f} h, among proned"),
    ])
    smallnote_html = (
        f'<div class="smallnote" id="smallnote">† Rate grayed: this slice has fewer than '
        f'{small_cell_min} eligible patients — interpret with caution.</div>'
    )
    trend_html = '<div id="trend"></div>'

    caveat = (
        '<div class="amber"><b>Data-coverage caveat.</b> At this site the CLIF '
        f'<code>position</code> table appears to chart only proning episodes, not routine supine: '
        f'only {n_documented:,} of {n_eligible:,} eligible patients ({100*n_documented/n_eligible:.0f}%) '
        'have any position record, and every one of them was proned. The adherence rate is therefore '
        f'reported as a <b>bound</b>: <b>{100*n_adherent/n_eligible:.1f}%</b> if patients with no '
        f'position data are counted as not adherent (lower bound), up to <b>{100*n_adherent/n_documented:.1f}%</b> '
        'among the charted subset only (upper bound).</div>'
    )

    logo_uri = _load_logo(PROJECT_ROOT / "references" / "images" / "clif_logo_v2.png")
    consort_uri = make_consort(counts, final / "graphs")
    ttp_uri = make_ttp_cdf(pl, n_eligible)
    table1 = build_table1(pl)
    table1_html = render_gtsummary_table_html(table1)

    out = build_html(logo_uri, controls_html, cards, smallnote_html, trend_html, caveat,
                     consort_uri, ttp_uri, table1_html, n_eligible, site, generated, script_html)
    out_path = final / "proning_dashboard.html"
    out_path.write_text(out, encoding="utf-8")

    log.info("logo embedded: %s", "yes" if logo_uri else "no (fallback)")
    log.info("filters: %d units × {all,year,month,week}; %d slice cells embedded; small-cell min=%d",
             slices["unit"].nunique(), len(slices), small_cell_min)
    log.info("CONSORT: ARDS %d → eligible %d → proned %d → adherent %d",
             n_ards, n_eligible, n_proned, n_adherent)
    log.info("Table 1 rows: %d", len(table1))
    log.info("wrote: %s (%.0f KB)", out_path.relative_to(PROJECT_ROOT), out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
