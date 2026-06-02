"""
00_probe_missingness.py — Probe UChicago CLIF v2.1.0 for LPV-relevant variable
availability before any cohort/adherence code is written.

Answers the open questions in CLAUDE.md / .claude/claude-todo.md:
  1. Height availability + units sanity check (vitals.height_cm)
  2. Plateau pressure availability (respiratory_support.plateau_pressure_obs)
  3. Vt source mix (tidal_volume_obs vs tidal_volume_set)
  4. PEEP source mix (peep_obs vs peep_set)
  5. IMV universe size + mode breakdown
  6. Joint availability: % of IMV rows where Vt + plateau + PEEP are all present
     (the composite-bundle-computable rate)

DATA SAFETY: only aggregated statistics are written to stdout / output files.
No raw CLIF rows are ever printed or saved.

Run from project root:
    .venv/bin/python code/00_probe_missingness.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import Patient, Hospitalization, RespiratorySupport, Vitals


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.json") as f:
    CFG = json.load(f)

DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
SITE = CFG.get("site", "UChicago")
OUT_DIR = Path(CFG.get("output_path", ROOT / "output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


PCT_LEVELS = [0.01, 0.25, 0.50, 0.75, 0.99]


def pct_summary(s: pd.Series) -> dict:
    """Return a small distribution summary for a numeric series (no raw values)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    q = s.quantile(PCT_LEVELS)
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "p01": float(q.loc[0.01]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p99": float(q.loc[0.99]),
        "max": float(s.max()),
    }


def pct_non_null(s: pd.Series) -> dict:
    n = int(len(s))
    nn = int(s.notna().sum())
    return {"n_rows": n, "n_non_null": nn, "pct_non_null": (nn / n * 100) if n else 0.0}


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


# ----------------------------------------------------------------------------
# 1. Patient + hospitalization — define adult cohort
# ----------------------------------------------------------------------------

banner("[1] Patient + hospitalization")

pt = Patient.from_file(data_directory=DATA_DIR, filetype=FILETYPE, timezone=TZ)
pt_df = pt.df
print(f"  patient rows: {len(pt_df):,}")

hosp = Hospitalization.from_file(data_directory=DATA_DIR, filetype=FILETYPE, timezone=TZ)
hosp_df = hosp.df
print(f"  hospitalization rows: {len(hosp_df):,}")

# Identify the admission datetime column (CLIF v2.x uses admission_dttm)
admit_col = next(
    (c for c in ("admission_dttm", "admit_dttm", "hospital_admission_dttm") if c in hosp_df.columns),
    None,
)
if admit_col is None:
    raise SystemExit(f"Could not find admission datetime column in hospitalization: {list(hosp_df.columns)}")

# Compute age at admission
hosp_age = hosp_df[["hospitalization_id", "patient_id", admit_col]].merge(
    pt_df[["patient_id", "birth_date"]], on="patient_id", how="left"
)
hosp_age["age_at_admit"] = (
    (pd.to_datetime(hosp_age[admit_col]) - pd.to_datetime(hosp_age["birth_date"])).dt.days / 365.25
)
adult_mask = hosp_age["age_at_admit"] >= 18
adult_hosp_ids = hosp_age.loc[adult_mask, "hospitalization_id"].astype(str).unique().tolist()

print(f"  adult (>=18 at admit) hospitalizations: {len(adult_hosp_ids):,}")
print(f"  unique adult patients: {hosp_age.loc[adult_mask, 'patient_id'].nunique():,}")
print(f"  admission date range: {pd.to_datetime(hosp_age[admit_col]).min()} -> {pd.to_datetime(hosp_age[admit_col]).max()}")

sex_counts = pt_df["sex_category"].value_counts(dropna=False).to_dict()
print(f"  sex_category (all patients): {sex_counts}")

# ----------------------------------------------------------------------------
# 2. Respiratory support — IMV universe
# ----------------------------------------------------------------------------

banner("[2] Respiratory support — IMV rows in adult cohort")

resp_cols = [
    "hospitalization_id",
    "recorded_dttm",
    "device_category",
    "mode_category",
    "tracheostomy",
    "tidal_volume_obs",
    "tidal_volume_set",
    "plateau_pressure_obs",
    "peep_obs",
    "peep_set",
    "fio2_set",
]

resp = RespiratorySupport.from_file(
    data_directory=DATA_DIR,
    filetype=FILETYPE,
    timezone=TZ,
    filters={"hospitalization_id": adult_hosp_ids, "device_category": ["IMV"]},
    columns=resp_cols,
)
resp_df = resp.df
print(f"  IMV rows (adult cohort): {len(resp_df):,}")

imv_hosp_ids = resp_df["hospitalization_id"].astype(str).unique().tolist()
print(f"  adult hosps with >=1 IMV row: {len(imv_hosp_ids):,}")

rows_per_hosp = resp_df.groupby("hospitalization_id").size()
print(f"  IMV rows-per-hosp distribution: {pct_summary(rows_per_hosp)}")

mode_breakdown = (
    resp_df["mode_category"].fillna("__missing__").value_counts(normalize=False).head(15).to_dict()
)
print(f"  mode_category breakdown (top 15): {mode_breakdown}")

trach_breakdown = resp_df["tracheostomy"].value_counts(dropna=False).to_dict()
print(f"  tracheostomy: {trach_breakdown}")

# ----------------------------------------------------------------------------
# 3. Per-column non-null on IMV rows
# ----------------------------------------------------------------------------

banner("[3] IMV-row availability for LPV components")

col_avail = {}
for c in ["tidal_volume_obs", "tidal_volume_set", "plateau_pressure_obs", "peep_obs", "peep_set", "fio2_set"]:
    col_avail[c] = pct_non_null(resp_df[c])
    print(f"  {c:>24s}: {col_avail[c]}")

print("\n  value distributions (non-null only):")
col_dist = {}
for c in ["tidal_volume_obs", "tidal_volume_set", "plateau_pressure_obs", "peep_obs", "peep_set", "fio2_set"]:
    col_dist[c] = pct_summary(resp_df[c])
    print(f"  {c:>24s}: {col_dist[c]}")

# Vt source overlap (any-Vt = obs or set)
vt_any = resp_df["tidal_volume_obs"].notna() | resp_df["tidal_volume_set"].notna()
vt_both = resp_df["tidal_volume_obs"].notna() & resp_df["tidal_volume_set"].notna()
print(
    f"\n  any Vt (obs OR set): {vt_any.mean() * 100:.2f}%   both: {vt_both.mean() * 100:.2f}%"
)

# PEEP source overlap
peep_any = resp_df["peep_obs"].notna() | resp_df["peep_set"].notna()
peep_both = resp_df["peep_obs"].notna() & resp_df["peep_set"].notna()
print(
    f"  any PEEP (obs OR set): {peep_any.mean() * 100:.2f}%   both: {peep_both.mean() * 100:.2f}%"
)

# ----------------------------------------------------------------------------
# 4. Joint availability — composite-bundle-computable rate
# ----------------------------------------------------------------------------

banner("[4] Joint availability — LPV composite bundle computable on IMV rows")

vt_present = resp_df["tidal_volume_obs"].notna() | resp_df["tidal_volume_set"].notna()
plat_present = resp_df["plateau_pressure_obs"].notna()
peep_present = resp_df["peep_obs"].notna() | resp_df["peep_set"].notna()

all_three = vt_present & plat_present & peep_present
print(f"  IMV rows with Vt + plateau + PEEP all present: "
      f"{all_three.sum():,} / {len(resp_df):,} ({all_three.mean() * 100:.2f}%)")

# Per-hospitalization: at least one fully computable row
resp_df["_bundle_computable"] = all_three
hosp_any_bundle = resp_df.groupby("hospitalization_id")["_bundle_computable"].any()
print(f"  IMV hosps with >=1 fully computable row: "
      f"{int(hosp_any_bundle.sum()):,} / {len(imv_hosp_ids):,} "
      f"({hosp_any_bundle.mean() * 100:.2f}%)")

# Per-hospitalization: median % of their IMV rows that are computable
hosp_pct_computable = resp_df.groupby("hospitalization_id")["_bundle_computable"].mean() * 100
print(f"  per-hosp % computable IMV rows: {pct_summary(hosp_pct_computable)}")

# ----------------------------------------------------------------------------
# 5. Height availability for IMV cohort (vitals.height_cm)
# ----------------------------------------------------------------------------

banner("[5] Height availability for IMV cohort (vitals.height_cm)")

vitals = Vitals.from_file(
    data_directory=DATA_DIR,
    filetype=FILETYPE,
    timezone=TZ,
    filters={"hospitalization_id": imv_hosp_ids, "vital_category": ["height_cm"]},
    columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
)
vit_df = vitals.df
print(f"  height_cm rows (IMV cohort): {len(vit_df):,}")

hosp_with_height = vit_df["hospitalization_id"].astype(str).unique().tolist()
pct_hosp_with_height = len(hosp_with_height) / max(len(imv_hosp_ids), 1) * 100
print(f"  IMV hosps with >=1 height reading: "
      f"{len(hosp_with_height):,} / {len(imv_hosp_ids):,} ({pct_hosp_with_height:.2f}%)")

readings_per_hosp = vit_df.groupby("hospitalization_id").size()
print(f"  height readings per hosp (among hosps with any): {pct_summary(readings_per_hosp)}")

height_dist = pct_summary(vit_df["vital_value"])
print(f"  height_cm value distribution: {height_dist}")
print("  -> sanity: cm values should be ~140-210; if p99 is ~80 or p01 is ~50, units may be inches/mixed.")

# ----------------------------------------------------------------------------
# 6. Save summary
# ----------------------------------------------------------------------------

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "site": SITE,
    "clif_data_path": DATA_DIR,
    "cohort": {
        "n_patients": int(len(pt_df)),
        "n_hospitalizations": int(len(hosp_df)),
        "n_adult_hospitalizations": len(adult_hosp_ids),
        "n_adult_patients": int(hosp_age.loc[adult_mask, "patient_id"].nunique()),
        "admit_min": str(pd.to_datetime(hosp_age[admit_col]).min()),
        "admit_max": str(pd.to_datetime(hosp_age[admit_col]).max()),
        "sex_category_all": sex_counts,
    },
    "imv": {
        "n_imv_rows": int(len(resp_df)),
        "n_adult_hosps_with_imv": len(imv_hosp_ids),
        "imv_rows_per_hosp": pct_summary(rows_per_hosp),
        "mode_breakdown_top15": mode_breakdown,
        "tracheostomy": {str(k): int(v) for k, v in trach_breakdown.items()},
    },
    "imv_row_column_availability": col_avail,
    "imv_row_value_distributions": col_dist,
    "vt_overlap": {
        "any_pct": float(vt_any.mean() * 100),
        "both_pct": float(vt_both.mean() * 100),
    },
    "peep_overlap": {
        "any_pct": float(peep_any.mean() * 100),
        "both_pct": float(peep_both.mean() * 100),
    },
    "bundle_computable_on_imv_rows": {
        "n_rows_all_three_present": int(all_three.sum()),
        "pct_rows_all_three_present": float(all_three.mean() * 100),
        "n_hosps_with_any_computable_row": int(hosp_any_bundle.sum()),
        "pct_hosps_with_any_computable_row": float(hosp_any_bundle.mean() * 100),
        "per_hosp_pct_computable_rows": pct_summary(hosp_pct_computable),
    },
    "height": {
        "n_height_rows": int(len(vit_df)),
        "n_imv_hosps_with_height": len(hosp_with_height),
        "pct_imv_hosps_with_height": pct_hosp_with_height,
        "readings_per_hosp": pct_summary(readings_per_hosp),
        "value_distribution": height_dist,
    },
}

json_path = OUT_DIR / "00_probe_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nWrote {json_path}")

# Markdown summary for fast review
md_lines = [
    f"# LPV probe — {SITE} CLIF v{CFG.get('clif_version', '?')}",
    f"_Generated {summary['generated_at']}_",
    "",
    "## Cohort",
    f"- Patients: **{summary['cohort']['n_patients']:,}**",
    f"- Hospitalizations: **{summary['cohort']['n_hospitalizations']:,}**",
    f"- Adult (>=18 at admit) hospitalizations: **{summary['cohort']['n_adult_hospitalizations']:,}**",
    f"- Adult unique patients: **{summary['cohort']['n_adult_patients']:,}**",
    f"- Admit date range: {summary['cohort']['admit_min']} → {summary['cohort']['admit_max']}",
    "",
    "## IMV universe (adult cohort)",
    f"- IMV rows: **{summary['imv']['n_imv_rows']:,}**",
    f"- Adult hosps with >=1 IMV row: **{summary['imv']['n_adult_hosps_with_imv']:,}**",
    f"- IMV rows per hosp: p25={summary['imv']['imv_rows_per_hosp'].get('p25')}, "
    f"p50={summary['imv']['imv_rows_per_hosp'].get('p50')}, "
    f"p75={summary['imv']['imv_rows_per_hosp'].get('p75')}, "
    f"p99={summary['imv']['imv_rows_per_hosp'].get('p99')}",
    "",
    "## IMV-row column availability",
    "| column | % non-null |",
    "|---|---:|",
] + [
    f"| `{c}` | {col_avail[c]['pct_non_null']:.2f}% |"
    for c in ["tidal_volume_obs", "tidal_volume_set", "plateau_pressure_obs", "peep_obs", "peep_set", "fio2_set"]
] + [
    "",
    "## Joint availability — LPV bundle computable on IMV rows",
    f"- Rows with Vt + plateau + PEEP all present: **{summary['bundle_computable_on_imv_rows']['pct_rows_all_three_present']:.2f}%** "
    f"({summary['bundle_computable_on_imv_rows']['n_rows_all_three_present']:,} / {summary['imv']['n_imv_rows']:,})",
    f"- Hosps with >=1 computable row: **{summary['bundle_computable_on_imv_rows']['pct_hosps_with_any_computable_row']:.2f}%** "
    f"({summary['bundle_computable_on_imv_rows']['n_hosps_with_any_computable_row']:,} / {summary['imv']['n_adult_hosps_with_imv']:,})",
    "",
    "## Height (vitals.height_cm) for IMV cohort",
    f"- IMV hosps with >=1 height: **{summary['height']['pct_imv_hosps_with_height']:.2f}%**",
    f"- height_cm distribution: p01={summary['height']['value_distribution'].get('p01')}, "
    f"p50={summary['height']['value_distribution'].get('p50')}, "
    f"p99={summary['height']['value_distribution'].get('p99')}",
    "  (cm sanity: should be ~140–210; large deviations suggest unit issues)",
]

md_path = OUT_DIR / "00_probe_summary.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Wrote {md_path}")

print("\nDone.")
