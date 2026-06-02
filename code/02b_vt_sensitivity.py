"""
02b_vt_sensitivity.py — How do the Vt-driven measures change as the Vt/kg cutoff moves?

Pure downstream recompute from `output/02_intervals.parquet` (all mode-eligible IMV
interval-pieces with nullable component values + duration_min). Holds the two "less
negotiable" components fixed (plateau <= 30, driving pressure <= 15) and sweeps the Vt
cutoff for two measures, each on its OWN denominator:

  - Vt component : among Vt-assessable patient-days (Vt+PBW present), what % are
                   Vt-adherent (>=80% of assessable time at Vt/kg <= cutoff)?
  - Composite    : among composite-assessable patient-days (all three present), what %
                   are bundle-adherent (Vt<=cutoff AND plateau<=30 AND dP<=15)?

This is the exact engine a dashboard Vt-cutoff slider would use.

Outputs:
  output/02b_vt_sensitivity.csv   — both sweeps across the cutoff grid
  output/02b_vt_sensitivity.json  — grid + per-unit 6-vs-8 (both measures)
  output/figs/vt_sensitivity.png  — adherence vs Vt cutoff (Vt component + composite)

Run:
    .venv/bin/python code/02b_vt_sensitivity.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT_DIR = Path(CFG.get("output_path", ROOT / "output"))
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLATEAU_MAX, DP_MAX = 30.0, 15.0
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60
VT_GRID = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0]

key = ["hospitalization_id", "calendar_day"]

iv = pd.read_parquet(OUT_DIR / "02_intervals.parquet")
status = pd.read_parquet(OUT_DIR / "02_patient_day_status.parquet")
N_TOTAL = len(status)

# Presence masks (null component value => not present for that piece)
iv["vt_present"] = iv["vt_per_pbw"].notna()
iv["comp_present"] = iv["vt_per_pbw"].notna() & iv["driving_pressure"].notna() & iv["plateau"].notna()
# Fixed (non-negotiable) pass for the composite
iv["fixed_ok"] = (iv["plateau"] <= PLATEAU_MAX) & (iv["driving_pressure"] <= DP_MAX)

gk = [iv["hospitalization_id"], iv["calendar_day"]]
vt_assess = iv["duration_min"].where(iv["vt_present"], 0.0).groupby(gk).sum()
comp_assess = iv["duration_min"].where(iv["comp_present"], 0.0).groupby(gk).sum()
vt_days = vt_assess[vt_assess >= MIN_ASSESSABLE_MIN]
comp_days = comp_assess[comp_assess >= MIN_ASSESSABLE_MIN]
print(f"cohort patient-days: {N_TOTAL:,}")
print(f"  Vt-assessable days:        {len(vt_days):,}  ({len(vt_days)/N_TOTAL*100:.1f}%)")
print(f"  composite-assessable days: {len(comp_days):,}  ({len(comp_days)/N_TOTAL*100:.1f}%)")


def rate_at(cutoff, present_col, pass_mask, assess, denom_days):
    in_mask = iv[present_col] & pass_mask
    bmin = iv["duration_min"].where(in_mask, 0.0).groupby(gk).sum()
    frac = (bmin / assess).reindex(denom_days.index)
    n_adher = int((frac >= ADHERENCE_FRACTION).sum())
    return n_adher


rows = []
for c in VT_GRID:
    vt_adh = rate_at(c, "vt_present", iv["vt_per_pbw"] <= c, vt_assess, vt_days)
    comp_adh = rate_at(c, "comp_present", iv["fixed_ok"] & (iv["vt_per_pbw"] <= c), comp_assess, comp_days)
    rows.append({
        "vt_cutoff": c,
        "vt_n_adherent": vt_adh,
        "vt_assessable_rate": vt_adh / len(vt_days),
        "vt_crude_rate": vt_adh / N_TOTAL,
        "comp_n_adherent": comp_adh,
        "comp_assessable_rate": comp_adh / len(comp_days),
        "comp_crude_rate": comp_adh / N_TOTAL,
    })
grid = pd.DataFrame(rows)
grid.to_csv(OUT_DIR / "02b_vt_sensitivity.csv", index=False)

print("\nVt-cutoff sweep (plateau<=30 & dP<=15 fixed):")
print(f"  {'Vt/kg':>6} | {'Vt-component':>22} | {'Composite':>22}")
print(f"  {'':>6} | {'assess-rate':>12} {'crude':>9} | {'assess-rate':>12} {'crude':>9}")
for _, r in grid.iterrows():
    print(f"  {r['vt_cutoff']:>6.1f} | {r['vt_assessable_rate']*100:>11.1f}% {r['vt_crude_rate']*100:>8.1f}% "
          f"| {r['comp_assessable_rate']*100:>11.1f}% {r['comp_crude_rate']*100:>8.1f}%")

# Validation against 02_features defaults (Vt<=6: 24.6%; composite<=6: 11.3%)
r6 = grid.loc[grid["vt_cutoff"] == 6.0].iloc[0]
print(f"\n[validate] cutoff 6.0 reproduces 02_features: "
      f"Vt-component {r6['vt_assessable_rate']*100:.1f}% (exp 24.6%), "
      f"composite {r6['comp_assessable_rate']*100:.1f}% (exp 11.3%)")

# Per-unit 6-vs-8, both measures
iv_u = iv  # already carries assigned_unit
guk = [iv_u["assigned_unit"], iv_u["hospitalization_id"], iv_u["calendar_day"]]

def per_unit(cutoff, present_col, pass_mask):
    amin = iv_u["duration_min"].where(iv_u[present_col], 0.0).groupby(guk).sum()
    bmin = iv_u["duration_min"].where(iv_u[present_col] & pass_mask, 0.0).groupby(guk).sum()
    df = pd.DataFrame({"a": amin, "b": bmin})
    df = df[df["a"] >= MIN_ASSESSABLE_MIN]
    df["adh"] = (df["b"] / df["a"]) >= ADHERENCE_FRACTION
    return {str(u): float(g["adh"].mean()) for u, g in df.groupby(level=0)}

cmp = {
    "vt_vt6": per_unit(6.0, "vt_present", iv["vt_per_pbw"] <= 6.0),
    "vt_vt8": per_unit(8.0, "vt_present", iv["vt_per_pbw"] <= 8.0),
}
print("\nPer-unit Vt-component assessable rate: Vt<=6 -> Vt<=8")
for u in sorted(cmp["vt_vt6"]):
    a6, a8 = cmp["vt_vt6"][u] * 100, cmp["vt_vt8"][u] * 100
    print(f"  {u:>26s}: {a6:5.1f}% -> {a8:5.1f}%  (+{a8-a6:.1f} pts)")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(grid["vt_cutoff"], grid["vt_assessable_rate"] * 100, "-o", color="#2c3e50", label="Vt component (own denominator)")
ax.plot(grid["vt_cutoff"], grid["comp_assessable_rate"] * 100, "-o", color="#c0392b", label="Composite (all three)")
for cval in (6.0, 8.0):
    ax.axvline(cval, ls="--", lw=1, color="#7f8c8d", alpha=0.6)
ax.set_xlabel("Tidal volume cutoff (mL/kg PBW)  —  plateau≤30 & ∆P≤15 fixed")
ax.set_ylabel("Assessable adherence (%)")
ax.set_title(f"Adherence vs tidal-volume cutoff ({CFG.get('site', 'site')}, IMV-on-ICU patient-days)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "vt_sensitivity.png", dpi=130)
plt.close(fig)

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "fixed": {"plateau_max": PLATEAU_MAX, "dp_max": DP_MAX,
              "adherence_fraction": ADHERENCE_FRACTION, "min_assessable_min": MIN_ASSESSABLE_MIN},
    "n_total_patient_days": N_TOTAL,
    "n_vt_assessable": len(vt_days), "n_comp_assessable": len(comp_days),
    "grid": grid.to_dict("records"),
    "per_unit_vt_component_6_vs_8": cmp,
}
(OUT_DIR / "02b_vt_sensitivity.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '02b_vt_sensitivity.csv'}")
print(f"Wrote {OUT_DIR / '02b_vt_sensitivity.json'}")
print(f"Wrote {FIG_DIR / 'vt_sensitivity.png'}")
print("Done.")
