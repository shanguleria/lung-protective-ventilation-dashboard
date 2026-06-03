"""
01b_cohort_assessment.py — Inspect the patient-day cohort BEFORE building 02_features.py.

Reads the derived patient-day skeleton (output/01_cohort_patient_days.parquet) and
runs a battery of data-safe, aggregated-only checks:

  1. Headline reconciliation (vs 01_cohort_summary.json)
  2. Temporal volume — monthly + yearly patient-day counts (COVID surge, data gaps)
  3. Per-unit volume over time (unit x year matrix)
  4. Patient-day-per-hospitalization distribution (long-stay tails)
  5. The fully-excluded ward-only-IMV hospitalizations — admit type, LOS, IMV mode mix
  6. PBW = NaN patient-days — decomposed into unknown-sex vs no-height vs PBW<25 reject
  7. Fallback-height hospitalizations — had no in-hosp height; quantify share

Outputs (aggregated only — no raw patient rows ever printed):
  output/01b_cohort_assessment.md     — human report
  output/01b_cohort_assessment.json   — machine summary
  output/figs/monthly_patient_days.png
  output/figs/monthly_patient_days_by_unit.png

Run:
    .venv/bin/python code/01b_cohort_assessment.py
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

from clifpy.tables import Hospitalization, RespiratorySupport, Vitals

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_CM_MIN = 100.0
HEIGHT_CM_MAX = 230.0

report: list[str] = []
J: dict = {"generated_at": datetime.now().isoformat(timespec="seconds")}


def section(title: str) -> None:
    report.append(f"\n## {title}\n")
    print(f"\n=== {title} ===")


def line(s: str = "") -> None:
    report.append(s)
    print(s)


# ----------------------------------------------------------------------------
# Load cohort skeleton
# ----------------------------------------------------------------------------

pd_path = OUT_DIR / "01_cohort_patient_days.parquet"
cohort = pd.read_parquet(pd_path)
cohort["calendar_day"] = pd.to_datetime(cohort["calendar_day"])
cohort["year"] = cohort["calendar_day"].dt.year
cohort["month"] = cohort["calendar_day"].dt.to_period("M")

report.append("# LPV Cohort Assessment\n")
report.append(f"_Generated {J['generated_at']} from `{pd_path.name}`._\n")

# ----------------------------------------------------------------------------
# 1. Headline reconciliation
# ----------------------------------------------------------------------------

section("1. Headline reconciliation")
n_pd = len(cohort)
n_hosp = cohort["hospitalization_id"].nunique()
n_pt = cohort["patient_id"].nunique()
line(f"- patient-days: **{n_pd:,}**")
line(f"- hospitalizations: **{n_hosp:,}**")
line(f"- patients: **{n_pt:,}**")
line(f"- calendar_day range: {cohort['calendar_day'].min().date()} → {cohort['calendar_day'].max().date()}")
line(f"- mean patient-days / hosp: {n_pd / n_hosp:.2f}")

summ_path = OUT_DIR / "01_cohort_summary.json"
if summ_path.exists():
    prior = json.loads(summ_path.read_text())
    match = (prior.get("n_patient_days") == n_pd and prior.get("n_hospitalizations") == n_hosp)
    line(f"- matches `01_cohort_summary.json`: **{match}** "
         f"(summary says {prior.get('n_patient_days'):,} pd / {prior.get('n_hospitalizations'):,} hosp)")
J["headline"] = {"n_patient_days": n_pd, "n_hospitalizations": n_hosp, "n_patients": n_pt,
                 "day_min": str(cohort["calendar_day"].min().date()),
                 "day_max": str(cohort["calendar_day"].max().date())}

# ----------------------------------------------------------------------------
# 2. Temporal volume
# ----------------------------------------------------------------------------

section("2. Temporal volume (patient-days)")
by_year = cohort.groupby("year").agg(
    patient_days=("hospitalization_id", "size"),
    hosps=("hospitalization_id", "nunique"),
).reset_index()
line("| Year | Patient-days | Hosps |")
line("|---|---:|---:|")
for _, r in by_year.iterrows():
    line(f"| {int(r['year'])} | {int(r['patient_days']):,} | {int(r['hosps']):,} |")
J["by_year"] = by_year.assign(year=by_year["year"].astype(int)).to_dict("records")

monthly = cohort.groupby("month").size()
# Detect months with zero patient-days inside the observed span (data gaps)
full_idx = pd.period_range(cohort["month"].min(), cohort["month"].max(), freq="M")
monthly_full = monthly.reindex(full_idx, fill_value=0)
zero_months = [str(p) for p, v in monthly_full.items() if v == 0]
line(f"\n- monthly patient-day range: {int(monthly_full.min()):,} → {int(monthly_full.max()):,}")
line(f"- median monthly volume: {int(monthly_full.median()):,}")
line(f"- months with ZERO patient-days inside span: {len(zero_months)}"
     + (f" → {zero_months}" if zero_months else ""))
# Largest month-over-month swings (possible data artifacts)
mom = monthly_full.astype(float)
mom_pct = mom.pct_change()
big = mom_pct.abs().sort_values(ascending=False).head(5)
line("- largest month-over-month swings (possible artifacts):")
for p, v in big.items():
    line(f"    - {p}: {v*100:+.0f}%  ({int(monthly_full.loc[p]):,} pd)")
J["months_zero"] = zero_months
J["monthly_min"] = int(monthly_full.min())
J["monthly_max"] = int(monthly_full.max())
J["monthly_median"] = float(monthly_full.median())

# Plot monthly
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly_full.index.to_timestamp(), monthly_full.values, lw=1.4, color="#2c3e50")
ax.fill_between(monthly_full.index.to_timestamp(), monthly_full.values, alpha=0.12, color="#2c3e50")
ax.set_title(f"Monthly IMV-on-ICU patient-days ({CFG.get('site', 'site')})")
ax.set_ylabel("patient-days / month")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "monthly_patient_days.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Per-unit volume over time
# ----------------------------------------------------------------------------

section("3. Per-unit volume")
unit_year = cohort.assign(unit=cohort["assigned_unit"].fillna("__none__")) \
    .pivot_table(index="unit", columns="year", values="hospitalization_id", aggfunc="size", fill_value=0)
unit_year["TOTAL"] = unit_year.sum(axis=1)
unit_year = unit_year.sort_values("TOTAL", ascending=False)
years = [c for c in unit_year.columns if c != "TOTAL"]
line("| Unit | " + " | ".join(str(y) for y in years) + " | TOTAL |")
line("|---|" + "---:|" * (len(years) + 1))
for unit, r in unit_year.iterrows():
    line(f"| {unit} | " + " | ".join(f"{int(r[y]):,}" for y in years) + f" | {int(r['TOTAL']):,} |")
J["unit_year"] = {str(u): {str(y): int(unit_year.loc[u, y]) for y in unit_year.columns}
                  for u in unit_year.index}

# Per-unit monthly plot
fig, ax = plt.subplots(figsize=(12, 5))
top_units = unit_year.index[:6]
for unit in top_units:
    if unit == "__none__":
        continue
    s = cohort[cohort["assigned_unit"] == unit].groupby("month").size().reindex(full_idx, fill_value=0)
    ax.plot(s.index.to_timestamp(), s.values, lw=1.1, label=unit)
ax.set_title("Monthly IMV-on-ICU patient-days by ICU unit")
ax.set_ylabel("patient-days / month")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "monthly_patient_days_by_unit.png", dpi=130)
plt.close(fig)

n_unit_missing = int(cohort["assigned_unit"].isna().sum())
line(f"\n- patient-days with no assigned_unit: {n_unit_missing:,} ({n_unit_missing/n_pd*100:.2f}%)")

# ----------------------------------------------------------------------------
# 4. Patient-days per hospitalization (long-stay tails)
# ----------------------------------------------------------------------------

section("4. Patient-days per hospitalization")
pdays_per_hosp = cohort.groupby("hospitalization_id").size()
q = pdays_per_hosp.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
line("| Quantile | Patient-days |")
line("|---|---:|")
for k, v in q.items():
    line(f"| p{int(k*100)} | {v:.0f} |")
line(f"\n- mean: {pdays_per_hosp.mean():.2f}")
for thr in (7, 14, 30, 90, 180, 365):
    n = int((pdays_per_hosp > thr).sum())
    share = int(pdays_per_hosp[pdays_per_hosp > thr].sum())
    line(f"- hosps with > {thr} patient-days: {n:,} ({n/n_hosp*100:.1f}% of hosps), "
         f"contributing {share:,} pd ({share/n_pd*100:.1f}% of all patient-days)")
J["pdays_per_hosp"] = {f"p{int(k*100)}": float(v) for k, v in q.items()}
J["pdays_per_hosp"]["mean"] = float(pdays_per_hosp.mean())

# Extreme-LOS audit: calendar span vs patient-day count for the longest hosps.
# A span >> the IMV/ICU clinical max (~6mo) signals a reused/merged hospitalization_id.
span = cohort.groupby("hospitalization_id")["calendar_day"].agg(lambda s: (s.max() - s.min()).days + 1)
units_per_hosp = cohort.groupby("hospitalization_id")["assigned_unit"].nunique()
extreme = pdays_per_hosp[pdays_per_hosp > 180].index
line(f"\n- hospitalizations with calendar span > 365 days (implausible for one encounter): "
     f"**{int((span > 365).sum())}**")
if len(extreme):
    line("- the extreme long-stay tail (>180 patient-days):")
    line("  | span (days) | patient-days | distinct units |")
    line("  |---:|---:|---:|")
    for hid in span.loc[extreme].sort_values(ascending=False).index:
        line(f"  | {int(span.loc[hid])} | {int(pdays_per_hosp.loc[hid])} | {int(units_per_hosp.loc[hid])} |")
J["pdays_per_hosp"]["n_span_over_365d"] = int((span > 365).sum())
J["pdays_per_hosp"]["max_span_days"] = int(span.max())

# ----------------------------------------------------------------------------
# 5. Excluded ward-only-IMV hospitalizations
# ----------------------------------------------------------------------------

section("5. Excluded ward-only-IMV hospitalizations")
# Re-derive the set: adult IMV hosps that did NOT make the cohort.
print("  loading hospitalization + patient for adult IMV universe ...")
from clifpy.tables import Patient

pt_df = Patient.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df
hosp_df = Hospitalization.from_file(DATA_DIR, filetype=FILETYPE, timezone=TZ).df
admit_col = next(c for c in ("admission_dttm", "admit_dttm", "hospital_admission_dttm") if c in hosp_df.columns)
disch_col = next((c for c in ("discharge_dttm", "hospital_discharge_dttm") if c in hosp_df.columns), None)

h = hosp_df.copy()
h["hospitalization_id"] = h["hospitalization_id"].astype(str)
h[admit_col] = pd.to_datetime(h[admit_col])
h = h.merge(pt_df[["patient_id", "birth_date"]], on="patient_id", how="left")
h["birth_date"] = pd.to_datetime(h["birth_date"])
h["age_at_admit"] = (h[admit_col] - h["birth_date"]).dt.days / 365.25
adult_ids = h.loc[h["age_at_admit"] >= 18, "hospitalization_id"].tolist()

print("  loading IMV rows (mode_category) for adult universe ...")
imv = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": adult_ids, "device_category": ["IMV"]},
    columns=["hospitalization_id", "recorded_dttm", "device_category", "mode_category"],
).df
imv["hospitalization_id"] = imv["hospitalization_id"].astype(str)

imv_hosps = set(imv["hospitalization_id"].unique())
cohort_hosps = set(cohort["hospitalization_id"].unique())
excluded = imv_hosps - cohort_hosps
line(f"- adult IMV hospitalizations total: {len(imv_hosps):,}")
line(f"- in cohort (≥1 ICU-overlapping IMV day): {len(cohort_hosps):,}")
line(f"- **fully excluded (ward-only IMV): {len(excluded):,}** ({len(excluded)/len(imv_hosps)*100:.1f}%)")
J["excluded"] = {"adult_imv_hosps": len(imv_hosps), "in_cohort": len(cohort_hosps),
                 "excluded_ward_only": len(excluded)}

if excluded:
    exc = h[h["hospitalization_id"].isin(excluded)].copy()
    # Admit type mix
    admit_type_col = next((c for c in ("admission_type_category", "admit_type_category",
                                       "admission_type_name", "admission_category") if c in exc.columns), None)
    if admit_type_col:
        line(f"\n- admit type (`{admit_type_col}`) mix of excluded hosps:")
        vc = exc[admit_type_col].fillna("__missing__").value_counts()
        for k, v in vc.head(10).items():
            line(f"    - {k}: {v:,} ({v/len(exc)*100:.1f}%)")
        J["excluded"]["admit_type"] = {str(k): int(v) for k, v in vc.items()}
    # LOS
    if disch_col:
        exc[disch_col] = pd.to_datetime(exc[disch_col])
        los = (exc[disch_col] - exc[admit_col]).dt.total_seconds() / 86400.0
        los = los[(los >= 0) & (los < 400)]
        if len(los):
            lq = los.quantile([0.25, 0.5, 0.75, 0.9])
            line(f"\n- length-of-stay (days) of excluded hosps: "
                 f"p25={lq.loc[0.25]:.1f}, median={lq.loc[0.5]:.1f}, p75={lq.loc[0.75]:.1f}, p90={lq.loc[0.9]:.1f}")
            J["excluded"]["los_median_days"] = float(lq.loc[0.5])
    # IMV mode mix + IMV-row-count per excluded hosp
    imv_exc = imv[imv["hospitalization_id"].isin(excluded)]
    rows_per = imv_exc.groupby("hospitalization_id").size()
    line(f"\n- IMV rows per excluded hosp: median={rows_per.median():.0f}, "
         f"p90={rows_per.quantile(0.9):.0f}, max={rows_per.max():.0f}")
    line(f"- excluded hosps with only 1 IMV row (likely transient/intubation event): "
         f"{int((rows_per == 1).sum()):,} ({(rows_per==1).mean()*100:.1f}%)")
    line(f"\n- IMV `mode_category` mix among excluded hosps' rows:")
    vc = imv_exc["mode_category"].fillna("__missing__").value_counts()
    for k, v in vc.head(10).items():
        line(f"    - {k}: {v:,} ({v/len(imv_exc)*100:.1f}%)")
    J["excluded"]["imv_rows_median"] = float(rows_per.median())
    J["excluded"]["pct_single_imv_row"] = float((rows_per == 1).mean() * 100)

# ----------------------------------------------------------------------------
# 6. PBW = NaN decomposition
# ----------------------------------------------------------------------------

section("6. PBW = NaN patient-days")
nan_pbw = cohort[cohort["pbw_kg"].isna()]
line(f"- patient-days with PBW = NaN: {len(nan_pbw):,} ({len(nan_pbw)/n_pd*100:.2f}%)")
line(f"- hospitalizations affected: {nan_pbw['hospitalization_id'].nunique():,}")
# Decompose: unknown/missing sex, vs missing height, vs both
known_sex = cohort["sex_category"].isin(["Male", "Female"])
has_height = cohort["height_cm"].notna()
no_sex = nan_pbw[~nan_pbw["sex_category"].isin(["Male", "Female"])]
no_height = nan_pbw[nan_pbw["height_cm"].isna()]
both_ok_but_nan = nan_pbw[nan_pbw["sex_category"].isin(["Male", "Female"]) & nan_pbw["height_cm"].notna()]
line(f"    - cause = missing/unknown sex: {len(no_sex):,} pd")
line(f"    - cause = missing height: {len(no_height):,} pd")
line(f"    - sex+height both present but PBW still NaN (PBW<25 reject / short patient): {len(both_ok_but_nan):,} pd")
# Sex distribution overall
line(f"\n- overall sex_category (patient-days):")
for k, v in cohort["sex_category"].fillna("__missing__").value_counts().items():
    line(f"    - {k}: {v:,} ({v/n_pd*100:.2f}%)")
J["pbw_nan"] = {"n_pd": len(nan_pbw), "n_hosp": int(nan_pbw["hospitalization_id"].nunique()),
                "cause_no_sex": len(no_sex), "cause_no_height": len(no_height),
                "cause_pbw_reject": len(both_ok_but_nan)}

# ----------------------------------------------------------------------------
# 7. Fallback-height hospitalizations
# ----------------------------------------------------------------------------

section("7. Fallback-height hospitalizations")
print("  loading vitals.height_cm for cohort hosps ...")
ht = Vitals.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": list(cohort_hosps), "vital_category": ["height_cm"]},
    columns=["hospitalization_id", "vital_category", "vital_value"],
).df
ht["hospitalization_id"] = ht["hospitalization_id"].astype(str)
ht["vital_value"] = pd.to_numeric(ht["vital_value"], errors="coerce")
ht.loc[(ht["vital_value"] < HEIGHT_CM_MIN) | (ht["vital_value"] > HEIGHT_CM_MAX), "vital_value"] = np.nan
hosps_with_inhosp_height = set(ht.loc[ht["vital_value"].notna(), "hospitalization_id"].unique())

cohort_height_known = cohort[cohort["height_cm"].notna()]
hosps_height_known = set(cohort_height_known["hospitalization_id"].unique())
fallback_hosps = hosps_height_known - hosps_with_inhosp_height  # height came from patient-level fallback
line(f"- cohort hosps with a usable in-hosp height: {len(hosps_with_inhosp_height):,} "
     f"({len(hosps_with_inhosp_height)/len(cohort_hosps)*100:.1f}%)")
line(f"- cohort hosps with height ONLY from per-patient fallback: **{len(fallback_hosps):,}** "
     f"({len(fallback_hosps)/len(cohort_hosps)*100:.1f}%)")
fallback_pdays = int(cohort[cohort["hospitalization_id"].isin(fallback_hosps)].shape[0])
line(f"- patient-days relying on fallback height: {fallback_pdays:,} ({fallback_pdays/n_pd*100:.2f}%)")
no_height_anywhere = cohort_hosps - hosps_height_known
line(f"- cohort hosps with NO height anywhere (in-hosp or fallback): {len(no_height_anywhere):,}")
J["fallback_height"] = {"hosps_inhosp": len(hosps_with_inhosp_height),
                        "hosps_fallback_only": len(fallback_hosps),
                        "pdays_fallback": fallback_pdays,
                        "hosps_no_height": len(no_height_anywhere)}

# ----------------------------------------------------------------------------
# 8. Verdict
# ----------------------------------------------------------------------------

section("8. Design-doc verdict")
line("_Flags worth deciding on before `02_features.py` (auto-generated heuristics):_")
flags = []
if zero_months:
    flags.append(f"Data gap: {len(zero_months)} month(s) with zero patient-days — confirm real vs extract artifact.")
if J["pdays_per_hosp"].get("n_span_over_365d", 0) > 0:
    flags.append(f"{J['pdays_per_hosp']['n_span_over_365d']} hospitalization(s) span > 365 calendar days "
                 f"(max {J['pdays_per_hosp']['max_span_days']}d) — almost certainly reused/merged "
                 f"hospitalization_id(s); a patient-day-weighted dashboard will over-weight them. "
                 f"Consider a per-encounter calendar-span cap in 02_features.py.")
if len(both_ok_but_nan) > 0:
    flags.append(f"{len(both_ok_but_nan)} pd have sex+height but PBW rejected (<25 kg) — verify these are genuine short-stature, not bad height.")
if J["excluded"].get("pct_single_imv_row", 0) > 40:
    flags.append("Most excluded ward-only hosps have a single IMV row — likely transient intubation/transport events, not true ward vent courses.")
if not flags:
    flags.append("No automatic red flags; cohort looks ready for 02_features.py.")
for f in flags:
    line(f"- {f}")
J["flags"] = flags

# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------

(OUT_DIR / "01b_cohort_assessment.md").write_text("\n".join(report) + "\n")
(OUT_DIR / "01b_cohort_assessment.json").write_text(json.dumps(J, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '01b_cohort_assessment.md'}")
print(f"Wrote {OUT_DIR / '01b_cohort_assessment.json'}")
print(f"Wrote {FIG_DIR / 'monthly_patient_days.png'}")
print(f"Wrote {FIG_DIR / 'monthly_patient_days_by_unit.png'}")
print("\nDone.")
