"""
01_cohort.py — Build the patient-day cohort for the LPV adherence dashboard.

For each adult hospitalization with any IMV row at UChicago, emit one row per
calendar day (US/Central) where the patient was both:
  (a) on IMV at some point, AND
  (b) inside an ICU stay (adt.location_category == 'icu') at some point.

Attach: assigned ICU unit (most IMV-rows that day), sex, age_at_admit,
representative height (median per hospitalization), PBW (Devine).

Quantify and report ward-IMV exclusions (IMV without overlapping ICU).

Output: output/01_cohort_patient_days.parquet

Run:
    .venv/bin/python code/01_cohort.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import Patient, Hospitalization, Adt, RespiratorySupport, Vitals


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

# Constants
HEIGHT_CM_MIN = 100.0   # exclude clearly impossible heights
HEIGHT_CM_MAX = 230.0


# ----------------------------------------------------------------------------
# 1. Patient + hospitalization — define adult cohort
# ----------------------------------------------------------------------------

print("[1] Loading patient + hospitalization ...")
pt_df = Patient.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df
hosp_df = Hospitalization.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df

admit_col = next(
    c for c in ("admission_dttm", "admit_dttm", "hospital_admission_dttm") if c in hosp_df.columns
)

hosp_age = hosp_df[["hospitalization_id", "patient_id", admit_col]].merge(
    pt_df[["patient_id", "birth_date", "sex_category"]], on="patient_id", how="left"
)
hosp_age[admit_col] = pd.to_datetime(hosp_age[admit_col])
hosp_age["birth_date"] = pd.to_datetime(hosp_age["birth_date"])
hosp_age["age_at_admit"] = (hosp_age[admit_col] - hosp_age["birth_date"]).dt.days / 365.25
hosp_age["hospitalization_id"] = hosp_age["hospitalization_id"].astype(str)

adult_mask = hosp_age["age_at_admit"] >= 18
adult_hosp = hosp_age.loc[adult_mask, ["hospitalization_id", "patient_id", "sex_category", "age_at_admit"]].copy()
adult_ids = adult_hosp["hospitalization_id"].tolist()
print(f"  adult hospitalizations: {len(adult_hosp):,}")


# ----------------------------------------------------------------------------
# 2. IMV rows for adult cohort
# ----------------------------------------------------------------------------

print("[2] Loading respiratory_support (IMV only) ...")
imv = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": adult_ids, "device_category": ["IMV"]},
    columns=["hospitalization_id", "recorded_dttm", "device_category"],
).df
imv["hospitalization_id"] = imv["hospitalization_id"].astype(str)
imv["recorded_dttm"] = pd.to_datetime(imv["recorded_dttm"])
print(f"  IMV rows: {len(imv):,}")

# Calendar day in US/Central
if imv["recorded_dttm"].dt.tz is None:
    imv["recorded_dttm"] = imv["recorded_dttm"].dt.tz_localize(TZ)
else:
    imv["recorded_dttm"] = imv["recorded_dttm"].dt.tz_convert(TZ)
imv["calendar_day"] = imv["recorded_dttm"].dt.date


# ----------------------------------------------------------------------------
# 3. ADT — ICU stays for adult cohort
# ----------------------------------------------------------------------------

print("[3] Loading adt (ICU stays) ...")
adt = Adt.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": adult_ids},
).df
adt["hospitalization_id"] = adt["hospitalization_id"].astype(str)
adt["in_dttm"] = pd.to_datetime(adt["in_dttm"])
adt["out_dttm"] = pd.to_datetime(adt["out_dttm"])

# Restrict to ICU stays
icu = adt.loc[adt["location_category"].str.lower().fillna("") == "icu",
              ["hospitalization_id", "location_type", "in_dttm", "out_dttm"]].copy()
icu["location_type"] = icu["location_type"].fillna("__unknown_icu_type__")
print(f"  ICU stays: {len(icu):,} across {icu['hospitalization_id'].nunique():,} hosps")


# ----------------------------------------------------------------------------
# 4. Attribute each IMV row to an ICU stay (if any)
# ----------------------------------------------------------------------------

print("[4] Attributing IMV rows to ICU stays via merge_asof ...")

# Sort for merge_asof — must be sorted by the `on` column globally
imv_sorted = imv.sort_values("recorded_dttm").reset_index(drop=True)
icu_sorted = icu.sort_values("in_dttm").reset_index(drop=True)

# Backward-merge: for each IMV row, take the most recent ICU stay that started before recorded_dttm
attr = pd.merge_asof(
    imv_sorted,
    icu_sorted,
    by="hospitalization_id",
    left_on="recorded_dttm",
    right_on="in_dttm",
    direction="backward",
)

# Keep only rows where IMV time falls within the stay (recorded_dttm < out_dttm)
attr["in_icu"] = attr["recorded_dttm"] < attr["out_dttm"]
n_imv_total = len(attr)
n_imv_in_icu = int(attr["in_icu"].sum())
n_imv_ward = n_imv_total - n_imv_in_icu
print(f"  IMV rows attributed to an ICU stay: {n_imv_in_icu:,} ({n_imv_in_icu / n_imv_total * 100:.1f}%)")
print(f"  IMV rows with no ICU overlap (ward-IMV): {n_imv_ward:,} ({n_imv_ward / n_imv_total * 100:.1f}%)")

# Drop ward-IMV rows (excluded from cohort per design)
attr_icu = attr.loc[attr["in_icu"]].copy()

# Reset location_type to nan for ward-IMV (we excluded them) — not used now
# Build per-day-per-unit counts for unit assignment
attr_icu["calendar_day"] = attr_icu["recorded_dttm"].dt.date
day_unit = (
    attr_icu.groupby(["hospitalization_id", "calendar_day", "location_type"])
    .size()
    .reset_index(name="n_imv_rows_in_unit")
)
# Pick top unit per (hosp, day) — ties broken alphabetically (sort ascending on location_type, descending on count, then keep first)
day_unit = day_unit.sort_values(
    ["hospitalization_id", "calendar_day", "n_imv_rows_in_unit", "location_type"],
    ascending=[True, True, False, True],
)
top_unit = day_unit.drop_duplicates(subset=["hospitalization_id", "calendar_day"], keep="first")
top_unit = top_unit.rename(columns={"location_type": "assigned_unit"})

# Per (hosp, day): total IMV rows on ICU
day_totals = (
    attr_icu.groupby(["hospitalization_id", "calendar_day"])
    .size()
    .reset_index(name="n_imv_rows")
)


# ----------------------------------------------------------------------------
# 5. Patient-day cohort = (hosp, day) with both IMV-on-ICU AND >=1 ICU stay
# ----------------------------------------------------------------------------

patient_days = day_totals.merge(
    top_unit[["hospitalization_id", "calendar_day", "assigned_unit"]],
    on=["hospitalization_id", "calendar_day"],
    how="left",
)
print(f"\n[5] Patient-days in ICU+IMV cohort: {len(patient_days):,}")
print(f"  unique hospitalizations: {patient_days['hospitalization_id'].nunique():,}")


# ----------------------------------------------------------------------------
# 6. Height (vitals.height_cm) — median per hospitalization
# ----------------------------------------------------------------------------

print("[6] Loading vitals.height_cm ...")
imv_hosps_with_pdays = patient_days["hospitalization_id"].unique().tolist()
ht = Vitals.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": imv_hosps_with_pdays, "vital_category": ["height_cm"]},
    columns=["hospitalization_id", "vital_category", "vital_value"],
).df
ht["hospitalization_id"] = ht["hospitalization_id"].astype(str)
ht["vital_value"] = pd.to_numeric(ht["vital_value"], errors="coerce")

# Outlier handling
ht.loc[(ht["vital_value"] < HEIGHT_CM_MIN) | (ht["vital_value"] > HEIGHT_CM_MAX), "vital_value"] = np.nan
print(f"  height_cm rows after outlier filtering: {ht['vital_value'].notna().sum():,}")

# Per-hospitalization median height
ht_per_hosp = ht.groupby("hospitalization_id")["vital_value"].median().rename("height_cm_hosp").reset_index()

# Fallback: per-patient median across all hospitalizations for that patient
hosp_to_pt = adult_hosp[["hospitalization_id", "patient_id"]]
ht_per_hosp_pt = ht_per_hosp.merge(hosp_to_pt, on="hospitalization_id", how="right")
ht_per_pt = ht_per_hosp_pt.groupby("patient_id")["height_cm_hosp"].median().rename("height_cm_pt").reset_index()


# ----------------------------------------------------------------------------
# 7. Attach demographics + PBW
# ----------------------------------------------------------------------------

print("[7] Attaching demographics + PBW ...")
patient_days = patient_days.merge(adult_hosp, on="hospitalization_id", how="left")
patient_days = patient_days.merge(ht_per_hosp, on="hospitalization_id", how="left")
patient_days = patient_days.merge(ht_per_pt, on="patient_id", how="left")
patient_days["height_cm"] = patient_days["height_cm_hosp"].fillna(patient_days["height_cm_pt"])
patient_days = patient_days.drop(columns=["height_cm_hosp", "height_cm_pt"])

# PBW (Devine / ARDSnet)
height_in = patient_days["height_cm"] / 2.54
sex = patient_days["sex_category"]
pbw = pd.Series(np.nan, index=patient_days.index, dtype="float64")
pbw.loc[sex == "Male"] = 50.0 + 2.3 * (height_in.loc[sex == "Male"] - 60.0)
pbw.loc[sex == "Female"] = 45.5 + 2.3 * (height_in.loc[sex == "Female"] - 60.0)
# Reject implausibly small PBW (very short patients) — keep NaN
pbw.loc[pbw < 25] = np.nan
patient_days["pbw_kg"] = pbw


# ----------------------------------------------------------------------------
# 8. Diagnostics
# ----------------------------------------------------------------------------

print("\n[8] Cohort diagnostics")
print(f"  patient-days: {len(patient_days):,}")
print(f"  hospitalizations: {patient_days['hospitalization_id'].nunique():,}")
print(f"  patients:        {patient_days['patient_id'].nunique():,}")
print(f"  calendar_day range: {patient_days['calendar_day'].min()} -> {patient_days['calendar_day'].max()}")

print(f"\n  assigned_unit distribution (patient-days):")
for k, v in patient_days["assigned_unit"].fillna("__none__").value_counts().items():
    print(f"    {k:>28s}: {v:>8,}  ({v / len(patient_days) * 100:.2f}%)")

print(f"\n  sex_category distribution (patient-days):")
for k, v in patient_days["sex_category"].fillna("__missing__").value_counts().items():
    print(f"    {k:>28s}: {v:>8,}  ({v / len(patient_days) * 100:.2f}%)")

print(f"\n  height_cm: non-null on {patient_days['height_cm'].notna().sum():,} "
      f"({patient_days['height_cm'].notna().mean() * 100:.2f}%) of patient-days")
print(f"  pbw_kg:    non-null on {patient_days['pbw_kg'].notna().sum():,} "
      f"({patient_days['pbw_kg'].notna().mean() * 100:.2f}%) of patient-days")
if patient_days["pbw_kg"].notna().any():
    q = patient_days["pbw_kg"].quantile([0.01, 0.50, 0.99])
    print(f"  pbw_kg distribution: p01={q.loc[0.01]:.1f}, p50={q.loc[0.50]:.1f}, p99={q.loc[0.99]:.1f}")

# Ward-IMV exclusion accounting (in hospitalizations and in IMV rows)
hosps_with_any_imv = set(imv["hospitalization_id"].unique())
hosps_in_cohort = set(patient_days["hospitalization_id"].unique())
hosps_excluded = hosps_with_any_imv - hosps_in_cohort
print(f"\n  IMV hosps total: {len(hosps_with_any_imv):,}")
print(f"  IMV hosps that landed in cohort: {len(hosps_in_cohort):,}")
print(f"  IMV hosps fully excluded (no ICU overlap any day): {len(hosps_excluded):,}")


# ----------------------------------------------------------------------------
# 9. Save
# ----------------------------------------------------------------------------

# Final column order
patient_days = patient_days[[
    "hospitalization_id", "patient_id", "calendar_day", "assigned_unit",
    "sex_category", "age_at_admit", "height_cm", "pbw_kg", "n_imv_rows",
]].sort_values(["calendar_day", "hospitalization_id"]).reset_index(drop=True)

out_path = OUT_DIR / "01_cohort_patient_days.parquet"
patient_days.to_parquet(out_path, index=False)
print(f"\nWrote {out_path}  ({len(patient_days):,} rows)")

# Summary JSON for the dashboard / readme
summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "n_patient_days": int(len(patient_days)),
    "n_hospitalizations": int(patient_days["hospitalization_id"].nunique()),
    "n_patients": int(patient_days["patient_id"].nunique()),
    "calendar_day_min": str(patient_days["calendar_day"].min()),
    "calendar_day_max": str(patient_days["calendar_day"].max()),
    "assigned_unit_counts": {
        str(k): int(v) for k, v in patient_days["assigned_unit"].fillna("__none__").value_counts().items()
    },
    "sex_counts": {
        str(k): int(v) for k, v in patient_days["sex_category"].fillna("__missing__").value_counts().items()
    },
    "height_pct_non_null": float(patient_days["height_cm"].notna().mean() * 100),
    "pbw_pct_non_null": float(patient_days["pbw_kg"].notna().mean() * 100),
    "imv_hosps_total": len(hosps_with_any_imv),
    "imv_hosps_in_cohort": len(hosps_in_cohort),
    "imv_hosps_excluded_no_icu_overlap": len(hosps_excluded),
    "n_imv_rows_in_icu": n_imv_in_icu,
    "n_imv_rows_ward": n_imv_ward,
}
(OUT_DIR / "01_cohort_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote {OUT_DIR / '01_cohort_summary.json'}")

print("\nDone.")
