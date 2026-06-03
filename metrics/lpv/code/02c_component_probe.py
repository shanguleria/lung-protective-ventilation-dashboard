"""
02c_component_probe.py — Quantify component-separated LPV adherence BEFORE deciding
whether to rebuild 02_features.py.

Mirrors 02_features.py's load + carry-forward + interval-split logic, but instead of
a single all-three composite it computes each component on ITS OWN denominator:

  - Vt/kg PBW  : assessable when mode-eligible IMV interval has Vt(obs->set) + PBW present
  - Pplat <=30 : assessable when mode-eligible IMV interval has plateau present
  - Pdriving<=15: assessable when mode-eligible IMV interval has plateau + PEEP present
  - Composite  : assessable when all three present (== current 02_features definition)

Same time-weighted rule per measure: a patient-day is component-assessable if it has
>= 60 min of that component's assessable time, and component-adherent if >= 80% of that
time passes the threshold.

Probe only — prints aggregated magnitudes + writes output/02c_component_probe.json.
No persisted patient-level artifacts. (If we proceed, this logic folds into 02_features.py.)

Run:
    .venv/bin/python code/02c_component_probe.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import RespiratorySupport

ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))

# Params (mirror 02_features.py)
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60
CF_FAST = pd.Timedelta(hours=2)
CF_SLOW = pd.Timedelta(hours=6)
MAX_GAP = pd.Timedelta(hours=24)
TRAIL_CLIP = pd.Timedelta(hours=1)
ELIGIBLE_MODES = {"Assist Control-Volume Control", "Pressure-Regulated Volume Control", "SIMV", "Pressure Control"}
PLATEAU_MAX, DP_MAX = 30.0, 15.0
VT_CUTOFFS = [6.0, 8.0]


def tlimited_ffill(df, col, window):
    grp = df["hospitalization_id"]
    val_ff = df.groupby(grp, sort=False)[col].ffill()
    ts_src = df["recorded_dttm"].where(df[col].notna())
    ts_ff = ts_src.groupby(grp, sort=False).ffill()
    return val_ff.where((df["recorded_dttm"] - ts_ff) <= window)


def to_central(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)


# ----- Load (mirror 02_features.py Step 0) -----
print("[0] cohort + respiratory_support ...")
cohort = pd.read_parquet(OUT_DIR / "01_cohort_patient_days.parquet")
cohort["hospitalization_id"] = cohort["hospitalization_id"].astype(str)
cohort["calendar_day"] = pd.to_datetime(cohort["calendar_day"]).dt.date
cohort_ids = cohort["hospitalization_id"].unique().tolist()
N_TOTAL = len(cohort)
pbw_map = cohort.dropna(subset=["pbw_kg"]).groupby("hospitalization_id")["pbw_kg"].first().reset_index()
unit_map = cohort[["hospitalization_id", "calendar_day", "assigned_unit"]]

rs_tbl = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": cohort_ids},
    columns=["hospitalization_id", "recorded_dttm", "device_category", "mode_category",
             "tidal_volume_obs", "tidal_volume_set", "plateau_pressure_obs", "peep_obs", "peep_set"],
)
from clifpy.utils.outlier_handler import apply_outlier_handling
apply_outlier_handling(rs_tbl)
rs = rs_tbl.df
rs["hospitalization_id"] = rs["hospitalization_id"].astype(str)
rs["recorded_dttm"] = to_central(rs["recorded_dttm"])
rs = rs.dropna(subset=["recorded_dttm"]).sort_values(["hospitalization_id", "recorded_dttm"]).reset_index(drop=True)

# ----- Carry-forward (Step A) -----
print("[A] carry-forward ...")
tv_eff = tlimited_ffill(rs, "tidal_volume_obs", CF_FAST).fillna(tlimited_ffill(rs, "tidal_volume_set", CF_FAST))
peep_eff = tlimited_ffill(rs, "peep_obs", CF_FAST).fillna(tlimited_ffill(rs, "peep_set", CF_FAST))
rs["plateau_eff"] = tlimited_ffill(rs, "plateau_pressure_obs", CF_SLOW)
rs["mode_eff"] = tlimited_ffill(rs, "mode_category", CF_SLOW)
rs["tv_eff"], rs["peep_eff"] = tv_eff, peep_eff

# ----- Intervals + day split (Step B) -----
print("[B] intervals + day split ...")
grp = rs["hospitalization_id"]
next_dttm = rs.groupby(grp, sort=False)["recorded_dttm"].shift(-1)
start = rs["recorded_dttm"]
gap = next_dttm - start
end = next_dttm.where(next_dttm.notna() & (gap <= MAX_GAP), start + TRAIL_CLIP)
local_naive = start.dt.tz_localize(None)
next_mid = (local_naive.dt.normalize() + pd.Timedelta(days=1)).dt.tz_localize(TZ)
start_day = local_naive.dt.normalize().dt.date

cols = ["hospitalization_id", "device_category", "mode_eff", "tv_eff", "plateau_eff", "peep_eff"]
p1 = pd.DataFrame({c: rs[c].values for c in cols})
p1["calendar_day"] = start_day.values
p1["duration_min"] = (end.where(end <= next_mid, next_mid) - start).dt.total_seconds().values / 60.0
mask2 = (end > next_mid).values
next_day = (local_naive.dt.normalize() + pd.Timedelta(days=1)).dt.date
p2 = pd.DataFrame({c: rs[c].values[mask2] for c in cols})
p2["calendar_day"] = next_day.values[mask2]
p2["duration_min"] = (end - next_mid).dt.total_seconds().values[mask2] / 60.0
pieces = pd.concat([p1, p2], ignore_index=True)
pieces = pieces[pieces["duration_min"] > 0].merge(pbw_map, on="hospitalization_id", how="left")

# ----- Per-component present/pass flags (mode-eligible IMV only) -----
print("[C] component flags ...")
elig = (pieces["device_category"] == "IMV") & pieces["mode_eff"].isin(ELIGIBLE_MODES)
pieces["vt_per_pbw"] = pieces["tv_eff"] / pieces["pbw_kg"]
pieces["dp"] = pieces["plateau_eff"] - pieces["peep_eff"]

vt_present = elig & pieces["tv_eff"].notna() & pieces["pbw_kg"].notna()
plat_present = elig & pieces["plateau_eff"].notna()
dp_present = elig & pieces["plateau_eff"].notna() & pieces["peep_eff"].notna()
comp_present = vt_present & plat_present & pieces["peep_eff"].notna()

d = pieces["duration_min"]
key = ["hospitalization_id", "calendar_day"]


def rollup(present_mask, pass_mask):
    """Per (hosp,day): assessable minutes and in-bundle minutes for one component."""
    a = d.where(present_mask, 0.0).groupby([pieces[k] for k in key]).sum()
    b = d.where(present_mask & pass_mask, 0.0).groupby([pieces[k] for k in key]).sum()
    return pd.DataFrame({"assess_min": a, "in_min": b})


def summarize(roll, label):
    r = roll[roll["assess_min"] >= MIN_ASSESSABLE_MIN].copy()
    n_assess = len(r)
    r["adher"] = (r["in_min"] / r["assess_min"]) >= ADHERENCE_FRACTION
    n_adher = int(r["adher"].sum())
    return {
        "measure": label,
        "n_assessable": n_assess,
        "pct_assessable": n_assess / N_TOTAL,
        "n_adherent": n_adher,
        "assessable_rate": (n_adher / n_assess) if n_assess else float("nan"),
        "crude_rate": n_adher / N_TOTAL,
    }


results = []
plat_pass = pieces["plateau_eff"] <= PLATEAU_MAX
dp_pass = pieces["dp"] <= DP_MAX
results.append(summarize(rollup(plat_present, plat_pass), "Pplat<=30"))
results.append(summarize(rollup(dp_present, dp_pass), "Pdriving<=15"))
for c in VT_CUTOFFS:
    results.append(summarize(rollup(vt_present, pieces["vt_per_pbw"] <= c), f"Vt/kg<={c:g}"))
    comp_pass = (pieces["vt_per_pbw"] <= c) & plat_pass & dp_pass
    results.append(summarize(rollup(comp_present, comp_pass), f"Composite(Vt<={c:g})"))

# ----- Report -----
print(f"\nAll cohort patient-days: {N_TOTAL:,}\n")
print(f"{'Measure':>20} {'%assessable':>12} {'assessable_n':>13} {'assess-rate':>12} {'crude':>8}")
for r in results:
    print(f"{r['measure']:>20} {r['pct_assessable']*100:>11.1f}% {r['n_assessable']:>13,} "
          f"{r['assessable_rate']*100:>11.1f}% {r['crude_rate']*100:>7.1f}%")

# Per-unit Vt(<=6) and Pplat assessable% to show the denominator gain
print("\nPer-unit %assessable: Vt vs Pplat (shows Vt's larger denominator)")
pu = pieces.merge(unit_map, on=key, how="left")
du = pu["duration_min"]
for unit, idx in pu.groupby("assigned_unit").groups.items():
    sub = pu.loc[idx]
    sd = sub["duration_min"]
    vt_a = sd.where((sub["device_category"] == "IMV") & sub["mode_eff"].isin(ELIGIBLE_MODES)
                    & sub["tv_eff"].notna() & sub["pbw_kg"].notna(), 0.0).groupby([sub["hospitalization_id"], sub["calendar_day"]]).sum()
    pl_a = sd.where((sub["device_category"] == "IMV") & sub["mode_eff"].isin(ELIGIBLE_MODES)
                    & sub["plateau_eff"].notna(), 0.0).groupby([sub["hospitalization_id"], sub["calendar_day"]]).sum()
    n_u = len(unit_map[unit_map["assigned_unit"] == unit])
    print(f"  {str(unit):>26}: Vt {int((vt_a>=60).sum())/n_u*100:5.1f}%   Pplat {int((pl_a>=60).sum())/n_u*100:5.1f}%   (n={n_u:,})")

summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "n_total_patient_days": N_TOTAL, "results": results}
(OUT_DIR / "02c_component_probe.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '02c_component_probe.json'}")
print("Done.")
