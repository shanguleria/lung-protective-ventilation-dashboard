"""
02_features.py — Component-separated, time-weighted LPV adherence per patient-day.

Reads the cohort skeleton (01_cohort_patient_days.parquet) and the raw
respiratory_support timeline, reconstructs effective ventilator settings via
time-limited carry-forward, splits the timeline into per-calendar-day interval
pieces, and classifies each of FOUR measures — each on ITS OWN denominator:

  - Vt/kg PBW <= cutoff   (assessable when Vt+PBW present; cutoff is the dashboard slider, default 6)
  - Pplat <= 30           (assessable when plateau present)
  - Pdriving <= 15        (assessable when plateau + PEEP present)   [dP = plateau - PEEP]
  - Composite             (assessable when all three present; adherent when all three pass)

Separating the components frees the densely-charted Vt measure from the sparse
plateau measure's denominator (see code/02c_component_probe.py for the rationale:
Vt-assessable ~77% of patient-days vs composite ~58%; plateau is ~86% adherent when
measured but charted on only ~60% of days; driving pressure is the real pressure
limiter at ~48%).

Per-measure rule (time-weighted): a patient-day is {measure}-assessable if it has
>= MIN_ASSESSABLE_MIN of that measure's assessable IMV time, and {measure}-adherent if
>= ADHERENCE_FRACTION of that time is in-bundle. Else {measure}-not_assessable.

Outputs:
  output/02_patient_day_status.parquet  — one row per cohort (hosp, day), wide: per-measure minutes + status
  output/02_intervals.parquet           — one row per mode-eligible IMV interval-piece (nullable component
                                           values + duration_min) — engine for the Vt slider & distributions
  output/02_features_summary.json        — per-measure diagnostics + invariants

Vt cutoff is a SLIDER: the patient-day file stores Vt/composite status at VT_MAX_DEFAULT,
while 02_intervals.parquet stores raw vt_per_pbw so 03_aggregate / the dashboard can
recompute at any cutoff (plateau<=30 & dP<=15 stay fixed — "less negotiable").

Run:
    .venv/bin/python code/02_features.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import RespiratorySupport

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
OUT_DIR = Path(CFG.get("output_path", ROOT / "output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Named parameters ----
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60
LONG_SPAN_DAYS = 200

CF_FAST = pd.Timedelta(hours=2)    # Vt, PEEP, FiO2
CF_SLOW = pd.Timedelta(hours=6)    # plateau, mode, tracheostomy
MAX_GAP = pd.Timedelta(hours=24)
TRAIL_CLIP = pd.Timedelta(hours=1)

ELIGIBLE_MODES = {
    "Assist Control-Volume Control",
    "Pressure-Regulated Volume Control",
    "SIMV",
    "Pressure Control",
}
VT_MAX_DEFAULT = 6.0   # default Vt/kg cutoff for stored status (dashboard slider overrides)
PLATEAU_MAX = 30.0     # fixed
DP_MAX = 15.0          # fixed

FALLBACK_RANGES = {
    "tidal_volume_obs": (100.0, 3000.0), "tidal_volume_set": (100.0, 3000.0),
    "plateau_pressure_obs": (0.0, 100.0), "peep_obs": (0.0, 50.0),
    "peep_set": (0.0, 30.0), "fio2_set": (0.21, 1.0),
}

MEASURES = ["vt", "plat", "dp", "comp"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def tlimited_ffill(df: pd.DataFrame, col: str, window: pd.Timedelta) -> pd.Series:
    """Forward-fill `col` within each hospitalization, only if the last non-null
    observation is within `window`. Assumes df sorted by hosp, recorded_dttm."""
    grp = df["hospitalization_id"]
    val_ff = df.groupby(grp, sort=False)[col].ffill()
    ts_src = df["recorded_dttm"].where(df[col].notna())
    ts_ff = ts_src.groupby(grp, sort=False).ffill()
    return val_ff.where((df["recorded_dttm"] - ts_ff) <= window)


def to_central(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)


def status_from(assess_min: pd.Series, in_min: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per-measure bundle_fraction + status (adherent / non_adherent / not_assessable)."""
    frac = np.where(assess_min > 0, in_min / assess_min.where(assess_min > 0, np.nan), np.nan)
    frac = pd.Series(frac, index=assess_min.index)
    status = np.where(
        assess_min < MIN_ASSESSABLE_MIN, "not_assessable",
        np.where(frac >= ADHERENCE_FRACTION, "adherent", "non_adherent"),
    )
    return frac, pd.Series(status, index=assess_min.index)


# ----------------------------------------------------------------------------
# Step 0 — Load cohort skeleton + respiratory_support
# ----------------------------------------------------------------------------

print("[0] Loading cohort skeleton ...")
cohort = pd.read_parquet(OUT_DIR / "01_cohort_patient_days.parquet")
cohort["hospitalization_id"] = cohort["hospitalization_id"].astype(str)
cohort["calendar_day"] = pd.to_datetime(cohort["calendar_day"]).dt.date
cohort_hosp_ids = cohort["hospitalization_id"].unique().tolist()
print(f"  cohort patient-days: {len(cohort):,}  hosps: {len(cohort_hosp_ids):,}")

pbw_map = (cohort.dropna(subset=["pbw_kg"]).groupby("hospitalization_id")["pbw_kg"].first()
           .reset_index())
span = (cohort.groupby("hospitalization_id")["calendar_day"]
        .agg(lambda s: (max(s) - min(s)).days + 1)
        .rename("encounter_span_days").reset_index())
unit_map = cohort[["hospitalization_id", "calendar_day", "assigned_unit"]]

print("[0] Loading respiratory_support (all device categories, cohort hosps) ...")
rs_tbl = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": cohort_hosp_ids},
    columns=[
        "hospitalization_id", "recorded_dttm", "device_category", "mode_category",
        "tracheostomy", "tidal_volume_obs", "tidal_volume_set",
        "plateau_pressure_obs", "peep_obs", "peep_set", "fio2_set",
    ],
)

print("[0] Applying outlier handling ...")
try:
    from clifpy.utils.outlier_handler import apply_outlier_handling
    apply_outlier_handling(rs_tbl)
    rs = rs_tbl.df
    print("  used clifpy apply_outlier_handling")
except Exception as e:  # pragma: no cover
    print(f"  clifpy helper unavailable ({e}); using manual ranges")
    rs = rs_tbl.df
    for col, (lo, hi) in FALLBACK_RANGES.items():
        if col in rs.columns:
            v = pd.to_numeric(rs[col], errors="coerce")
            rs[col] = v.where((v >= lo) & (v <= hi))

rs["hospitalization_id"] = rs["hospitalization_id"].astype(str)
rs["recorded_dttm"] = to_central(rs["recorded_dttm"])
rs = rs.dropna(subset=["recorded_dttm"]).sort_values(
    ["hospitalization_id", "recorded_dttm"]).reset_index(drop=True)
print(f"  respiratory_support rows: {len(rs):,}")

# ----------------------------------------------------------------------------
# Step A — Time-limited carry-forward
# ----------------------------------------------------------------------------

print("[A] Time-limited carry-forward ...")
tv_obs_eff = tlimited_ffill(rs, "tidal_volume_obs", CF_FAST)
tv_set_eff = tlimited_ffill(rs, "tidal_volume_set", CF_FAST)
peep_obs_eff = tlimited_ffill(rs, "peep_obs", CF_FAST)
peep_set_eff = tlimited_ffill(rs, "peep_set", CF_FAST)
rs["fio2_eff"] = tlimited_ffill(rs, "fio2_set", CF_FAST)
rs["plateau_eff"] = tlimited_ffill(rs, "plateau_pressure_obs", CF_SLOW)
rs["mode_eff"] = tlimited_ffill(rs, "mode_category", CF_SLOW)
rs["tv_eff"] = tv_obs_eff.fillna(tv_set_eff)
rs["peep_eff"] = peep_obs_eff.fillna(peep_set_eff)

# ----------------------------------------------------------------------------
# Step B — Interval construction + calendar-day split (<=2 pieces; DST-safe)
# ----------------------------------------------------------------------------

print("[B] Building intervals + splitting on calendar-day boundary ...")
grp = rs["hospitalization_id"]
next_dttm = rs.groupby(grp, sort=False)["recorded_dttm"].shift(-1)
start = rs["recorded_dttm"]
gap = next_dttm - start
end = next_dttm.where(next_dttm.notna() & (gap <= MAX_GAP), start + TRAIL_CLIP)

local_naive = start.dt.tz_localize(None)
next_mid = (local_naive.dt.normalize() + pd.Timedelta(days=1)).dt.tz_localize(TZ)
start_day = local_naive.dt.normalize().dt.date
next_day = (local_naive.dt.normalize() + pd.Timedelta(days=1)).dt.date

carry = ["hospitalization_id", "device_category", "mode_eff",
         "tv_eff", "plateau_eff", "peep_eff", "fio2_eff"]
p1 = pd.DataFrame({c: rs[c].values for c in carry})
p1["calendar_day"] = start_day.values
p1["duration_min"] = (end.where(end <= next_mid, next_mid) - start).dt.total_seconds().values / 60.0
mask2 = (end > next_mid).values
p2 = pd.DataFrame({c: rs[c].values[mask2] for c in carry})
p2["calendar_day"] = next_day.values[mask2]
p2["duration_min"] = (end - next_mid).dt.total_seconds().values[mask2] / 60.0
pieces = pd.concat([p1, p2], ignore_index=True)
pieces = pieces[pieces["duration_min"] > 0].merge(pbw_map, on="hospitalization_id", how="left")
print(f"  interval-pieces: {len(pieces):,}  (midnight crossings: {int(mask2.sum()):,})")

# ----------------------------------------------------------------------------
# Step C — Per-component present/pass flags (mode-eligible IMV only)
# ----------------------------------------------------------------------------

print("[C] Component present/pass flags ...")
pieces["is_imv"] = pieces["device_category"] == "IMV"
pieces["mode_eligible"] = pieces["mode_eff"].isin(ELIGIBLE_MODES)
elig = pieces["is_imv"] & pieces["mode_eligible"]

pieces["vt_per_pbw"] = pieces["tv_eff"] / pieces["pbw_kg"]
pieces["driving_pressure"] = pieces["plateau_eff"] - pieces["peep_eff"]

# present masks (each component's own denominator)
vt_present = elig & pieces["tv_eff"].notna() & pieces["pbw_kg"].notna()
plat_present = elig & pieces["plateau_eff"].notna()
dp_present = elig & pieces["plateau_eff"].notna() & pieces["peep_eff"].notna()
comp_present = vt_present & dp_present  # Vt + plateau + PEEP all present

# pass masks
vt_pass = pieces["vt_per_pbw"] <= VT_MAX_DEFAULT
plat_pass = pieces["plateau_eff"] <= PLATEAU_MAX
dp_pass = pieces["driving_pressure"] <= DP_MAX
comp_pass = vt_pass & plat_pass & dp_pass

present = {"vt": vt_present, "plat": plat_present, "dp": dp_present, "comp": comp_present}
passing = {"vt": vt_pass, "plat": plat_pass, "dp": dp_pass, "comp": comp_pass}

# ----------------------------------------------------------------------------
# Step D — Roll up each measure to (hosp, calendar_day)
# ----------------------------------------------------------------------------

print("[D] Rolling up per measure ...")
key = ["hospitalization_id", "calendar_day"]
d = pieces["duration_min"]
gkeys = [pieces["hospitalization_id"], pieces["calendar_day"]]

roll = pd.DataFrame({
    "total_imv_minutes": d.where(pieces["is_imv"], 0.0).groupby(gkeys).sum(),
    "mode_eligible_minutes": d.where(elig, 0.0).groupby(gkeys).sum(),
})
for m in MEASURES:
    roll[f"{m}_assessable_min"] = d.where(present[m], 0.0).groupby(gkeys).sum()
    roll[f"{m}_in_min"] = d.where(present[m] & passing[m], 0.0).groupby(gkeys).sum()
roll = roll.reset_index()
roll.columns = key + list(roll.columns[len(key):])

# ----------------------------------------------------------------------------
# Step E — Join to cohort skeleton, classify per-measure status, persist
# ----------------------------------------------------------------------------

print("[E] Joining to cohort skeleton + per-measure status ...")
out = cohort.merge(roll, on=key, how="left")
min_cols = (["total_imv_minutes", "mode_eligible_minutes"]
            + [f"{m}_{s}" for m in MEASURES for s in ("assessable_min", "in_min")])
out[min_cols] = out[min_cols].fillna(0.0)

for m in MEASURES:
    frac, st = status_from(out[f"{m}_assessable_min"], out[f"{m}_in_min"])
    out[f"{m}_bundle_fraction"] = frac
    out[f"{m}_status"] = st

out = out.merge(span, on="hospitalization_id", how="left")
out["long_span_flag"] = out["encounter_span_days"] > LONG_SPAN_DAYS

id_cols = ["hospitalization_id", "patient_id", "calendar_day", "assigned_unit",
           "sex_category", "age_at_admit", "height_cm", "pbw_kg",
           "encounter_span_days", "long_span_flag",
           "total_imv_minutes", "mode_eligible_minutes"]
measure_cols = [f"{m}_{s}" for m in MEASURES
                for s in ("assessable_min", "in_min", "bundle_fraction", "status")]
out = out[id_cols + measure_cols].sort_values(["calendar_day", "hospitalization_id"]).reset_index(drop=True)

out_path = OUT_DIR / "02_patient_day_status.parquet"
out.to_parquet(out_path, index=False)
print(f"  wrote {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

# Artifact #2 — all mode-eligible IMV interval-pieces, restricted to cohort (hosp, day).
# Nullable component values: null vt_per_pbw => Vt not assessable that piece, etc.
print("[E] Building interval artifact (Vt-slider / distribution engine) ...")
iv = pieces.loc[elig, key + ["duration_min", "vt_per_pbw", "plateau_eff",
                             "driving_pressure", "peep_eff", "fio2_eff"]].copy()
iv = iv.rename(columns={"plateau_eff": "plateau", "peep_eff": "peep", "fio2_eff": "fio2"})
# Nullable component values already encode presence: vt_per_pbw is null iff Vt or PBW
# absent; plateau null iff plateau absent; driving_pressure null iff plateau or PEEP absent.
iv = iv.merge(unit_map, on=key, how="inner")
iv_path = OUT_DIR / "02_intervals.parquet"
iv.to_parquet(iv_path, index=False)
print(f"  wrote {iv_path}  ({len(iv):,} rows)")

# Clean up the superseded single-composite interval file if present.
old_iv = OUT_DIR / "02_assessable_intervals.parquet"
if old_iv.exists():
    old_iv.unlink()
    print(f"  removed superseded {old_iv.name}")

# ----------------------------------------------------------------------------
# Diagnostics + summary JSON
# ----------------------------------------------------------------------------

N = len(out)
print(f"\n[diag] Per-measure (default Vt cutoff = {VT_MAX_DEFAULT:g}; plateau<=30 & dP<=15 fixed):")
print(f"  {'measure':>10} {'%assessable':>12} {'assess-rate':>12} {'crude':>8}")
labels = {"vt": f"Vt/kg<={VT_MAX_DEFAULT:g}", "plat": "Pplat<=30", "dp": "Pdriving<=15", "comp": "Composite"}
per_measure = {}
for m in MEASURES:
    a = int((out[f"{m}_status"] == "adherent").sum())
    n = int((out[f"{m}_status"] == "non_adherent").sum())
    na = int((out[f"{m}_status"] == "not_assessable").sum())
    assessable = a + n
    ar = a / assessable if assessable else float("nan")
    cr = a / N
    pa = assessable / N
    print(f"  {labels[m]:>10} {pa*100:>11.1f}% {ar*100:>11.1f}% {cr*100:>7.1f}%")
    per_measure[m] = {"label": labels[m], "n_adherent": a, "n_non_adherent": n,
                      "n_not_assessable": na, "pct_assessable": pa,
                      "assessable_rate": ar, "crude_rate": cr}

# Invariants
print("\n[diag] Invariant checks:")
chain_ok = True
for m in MEASURES:
    chain_ok &= bool((out[f"{m}_in_min"] <= out[f"{m}_assessable_min"] + 1e-6).all())
    chain_ok &= bool((out[f"{m}_assessable_min"] <= out["mode_eligible_minutes"] + 1e-6).all())
chain_ok &= bool((out["mode_eligible_minutes"] <= out["total_imv_minutes"] + 1e-6).all())
# composite assessable <= each component assessable (composite is the strictest present-set)
comp_subset_ok = bool(
    (out["comp_assessable_min"] <= out["vt_assessable_min"] + 1e-6).all()
    and (out["comp_assessable_min"] <= out["dp_assessable_min"] + 1e-6).all()
)
per_day = pieces.groupby(key)["duration_min"].sum()
max_day = float(per_day.max())
DAY_CAP = 1500.0  # 25h DST fall-back day
frac_ok = all(bool(out[f"{m}_bundle_fraction"].dropna().between(0, 1).all()) for m in MEASURES)
iv_sum = float(iv["duration_min"].sum())
mode_sum = float(out["mode_eligible_minutes"].sum())
iv_consistent = abs(iv_sum - mode_sum) <= max(1.0, 1e-4 * mode_sum)
n_flag = int(out["long_span_flag"].sum())
print(f"    minute chain (in<=assess<=mode<=total, all measures): {chain_ok}")
print(f"    composite assessable <= vt & dp assessable: {comp_subset_ok}")
print(f"    max minutes/day {max_day:.1f} (<=1500, 25h DST day): {max_day <= DAY_CAP + 1e-6}")
print(f"    all bundle_fractions in [0,1]: {frac_ok}")
print(f"    interval Σduration ({iv_sum:,.0f}) == mode_eligible Σ ({mode_sum:,.0f}): {iv_consistent}")
print(f"    long_span_flag patient-days: {n_flag:,} "
      f"(hosps: {out.loc[out['long_span_flag'],'hospitalization_id'].nunique()})")

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "params": {
        "adherence_fraction": ADHERENCE_FRACTION, "min_assessable_min": MIN_ASSESSABLE_MIN,
        "vt_max_default": VT_MAX_DEFAULT, "plateau_max": PLATEAU_MAX, "dp_max": DP_MAX,
        "cf_fast_hours": 2, "cf_slow_hours": 6, "max_gap_hours": 24,
        "eligible_modes": sorted(ELIGIBLE_MODES), "long_span_days": LONG_SPAN_DAYS,
    },
    "n_patient_days": N,
    "per_measure": per_measure,
    "n_intervals": int(len(iv)),
    "invariants": {
        "minute_chain_ok": chain_ok, "composite_subset_ok": comp_subset_ok,
        "max_minutes_per_day": max_day, "max_minutes_per_day_ok": bool(max_day <= DAY_CAP + 1e-6),
        "bundle_fraction_in_unit_interval": frac_ok,
        "interval_vs_mode_eligible_consistent": bool(iv_consistent),
    },
    "n_patient_days_long_span_flag": n_flag,
}
(OUT_DIR / "02_features_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '02_features_summary.json'}")
print("Done.")
