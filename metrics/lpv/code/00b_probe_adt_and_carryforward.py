"""
00b_probe_adt_and_carryforward.py — Second probe needed before building the
patient-day adherence pipeline.

Answers:
  A. What ICU unit categories exist at UChicago (`adt.location_type`)?
     Needed for the by-unit faceting in the dashboard.
  B. Plateau-pressure measurement cadence on IMV: distribution of
     time-between-consecutive-plateau-readings within a hospitalization.
     Determines a reasonable plateau carry-forward window for the
     time-weighted adherence rule.

DATA SAFETY: aggregated only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import Adt, RespiratorySupport, Hospitalization, Patient


ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))


# ----------------------------------------------------------------------------
# Identify IMV hospitalization_ids (reuse the same definition as 00_probe)
# ----------------------------------------------------------------------------

pt_df = Patient.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df
hosp_df = Hospitalization.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df

admit_col = next(c for c in ("admission_dttm", "admit_dttm") if c in hosp_df.columns)
hosp_age = hosp_df[["hospitalization_id", "patient_id", admit_col]].merge(
    pt_df[["patient_id", "birth_date"]], on="patient_id", how="left"
)
hosp_age["age"] = (
    (pd.to_datetime(hosp_age[admit_col]) - pd.to_datetime(hosp_age["birth_date"])).dt.days / 365.25
)
adult_ids = hosp_age.loc[hosp_age["age"] >= 18, "hospitalization_id"].astype(str).unique().tolist()


# ----------------------------------------------------------------------------
# A. ADT: location categories during IMV hosps
# ----------------------------------------------------------------------------

print("\n=== A. ADT location categories ===")
adt_df = Adt.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": adult_ids},
).df
print(f"  adt rows (adult cohort): {len(adt_df):,}")
print(f"  columns: {sorted(adt_df.columns.tolist())}")

print("\n  location_category counts:")
print(adt_df["location_category"].fillna("__missing__").value_counts().to_dict())

if "location_type" in adt_df.columns:
    print("\n  location_type (ICU rows only):")
    icu_mask = adt_df["location_category"].str.lower().fillna("") == "icu"
    print(adt_df.loc[icu_mask, "location_type"].fillna("__missing__").value_counts().to_dict())
else:
    print("  (no location_type column)")

if "hospital_type" in adt_df.columns:
    print("\n  hospital_type:")
    print(adt_df["hospital_type"].fillna("__missing__").value_counts().to_dict())


# ----------------------------------------------------------------------------
# B. Plateau measurement cadence
# ----------------------------------------------------------------------------

print("\n=== B. Plateau measurement cadence (hours between consecutive readings) ===")
resp = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": adult_ids, "device_category": ["IMV"]},
    columns=["hospitalization_id", "recorded_dttm", "device_category", "plateau_pressure_obs"],
).df

plat = resp.dropna(subset=["plateau_pressure_obs"]).copy()
plat["recorded_dttm"] = pd.to_datetime(plat["recorded_dttm"])
plat = plat.sort_values(["hospitalization_id", "recorded_dttm"])

# Gap between consecutive plateau readings within the same hospitalization
plat["prev_dttm"] = plat.groupby("hospitalization_id")["recorded_dttm"].shift()
plat["gap_h"] = (plat["recorded_dttm"] - plat["prev_dttm"]).dt.total_seconds() / 3600.0
gaps = plat["gap_h"].dropna()
gaps_within_24h = gaps[gaps <= 24]

print(f"  hospitalizations with >=1 plateau: {plat['hospitalization_id'].nunique():,}")
print(f"  total plateau readings: {len(plat):,}")
print(f"  gaps observed (n): {len(gaps):,}")
if len(gaps):
    q = gaps.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"  gap-hours distribution: "
          f"p10={q.loc[0.10]:.2f}, p25={q.loc[0.25]:.2f}, p50={q.loc[0.50]:.2f}, "
          f"p75={q.loc[0.75]:.2f}, p90={q.loc[0.90]:.2f}, p95={q.loc[0.95]:.2f}, p99={q.loc[0.99]:.2f}")
    for h in (2, 4, 6, 8, 12, 24):
        pct = (gaps <= h).mean() * 100
        print(f"  % of gaps <= {h}h: {pct:.1f}%")

# Per-hospitalization median gap (clinical: how often does a typical pt get plateau?)
median_gap_per_hosp = plat.groupby("hospitalization_id")["gap_h"].median().dropna()
if len(median_gap_per_hosp):
    q = median_gap_per_hosp.quantile([0.25, 0.50, 0.75])
    print(f"\n  per-hosp median plateau gap (h): "
          f"p25={q.loc[0.25]:.2f}, p50={q.loc[0.50]:.2f}, p75={q.loc[0.75]:.2f}")

# Per-day plateau count: among (hosp, calendar day) cells with any IMV row, how many had >=1 plateau?
resp["recorded_dttm"] = pd.to_datetime(resp["recorded_dttm"])
resp["day"] = resp["recorded_dttm"].dt.tz_convert(TZ).dt.date if resp["recorded_dttm"].dt.tz is not None else resp["recorded_dttm"].dt.date
per_day = resp.groupby(["hospitalization_id", "day"]).agg(
    n_rows=("recorded_dttm", "size"),
    n_plat=("plateau_pressure_obs", lambda s: s.notna().sum()),
).reset_index()
print(f"\n  IMV patient-days: {len(per_day):,}")
print(f"  % patient-days with >=1 plateau reading: {(per_day['n_plat'] > 0).mean() * 100:.1f}%")
print(f"  % patient-days with >=2 plateau readings: {(per_day['n_plat'] >= 2).mean() * 100:.1f}%")
print(f"  plateau-per-day distribution (among days with any): "
      f"{pd.Series(per_day.loc[per_day['n_plat'] > 0, 'n_plat']).describe().to_dict()}")

print("\nDone.")
