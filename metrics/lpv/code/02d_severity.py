"""
02d_severity.py — Classify each cohort patient-day by "severe respiratory failure".

"Severe respiratory failure" (deliberately NOT full Berlin ARDS — no imaging/origin
criteria) for a patient-day = at any point that day, a low oxygenation ratio while on
PEEP:

    (P/F < 300)  OR  (S/F < 315, using SpO2/FiO2 only when SpO2 <= 97%)   AND   PEEP > 5

P/F = PaO2 (labs po2_arterial) / FiO2 ; S/F = SpO2 (vitals spo2) / FiO2. The S/F surrogate
backfills oxygenation when no arterial blood gas is available (315 is the established S/F
equivalent of P/F 300, Rice 2007). The FiO2 and PEEP paired to each PaO2/SpO2 use a
<= 4-hour backward lookback. A patient-day is classified by its WORST oxygenation that day.

Strata (per cohort patient-day):
  severe       — >= 1 usable oxygenation assessment that qualifies
  not_severe   — >= 1 usable assessment (FiO2 + PEEP present), none qualifying
  unknown      — no usable assessment that day

Output: output/02d_severity.parquet  (hospitalization_id, calendar_day, severity,
        worst_pf, worst_sf, class_source) + output/02d_severity_summary.json

Run:
    .venv/bin/python code/02d_severity.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import RespiratorySupport, Labs, Vitals

ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
DATA_DIR = CFG["clif_data_path"]
FILETYPE = CFG.get("filetype", "parquet")
TZ = CFG.get("timezone", "US/Central")
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))

# ---- Named parameters (tunable by any site) ----
PF_THRESHOLD = 300.0     # P/F < this qualifies
SF_THRESHOLD = 315.0     # S/F < this qualifies (Rice 2007 equivalent of P/F 300)
SPO2_MAX_FOR_SF = 97.0   # only use SpO2 for S/F when <= this (saturation plateau)
PEEP_MIN = 5.0           # PEEP strictly > this required
O2_FIO2_LOOKBACK = pd.Timedelta(hours=4)  # max backward lookback for FiO2/PEEP

# Plausible ranges (labs/vitals outlier ranges aren't category-exposed in clifpy config)
PAO2_RANGE = (30.0, 700.0)    # mmHg
SPO2_RANGE = (50.0, 100.0)    # %


def to_central(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)


# ----------------------------------------------------------------------------
# Load cohort + respiratory_support (FiO2 / PEEP observations)
# ----------------------------------------------------------------------------

print("[1] Cohort + respiratory_support (FiO2/PEEP) ...")
cohort = pd.read_parquet(OUT_DIR / "01_cohort_patient_days.parquet")
cohort["hospitalization_id"] = cohort["hospitalization_id"].astype(str)
cohort["calendar_day"] = pd.to_datetime(cohort["calendar_day"]).dt.date
cohort_ids = cohort["hospitalization_id"].unique().tolist()

rs_tbl = RespiratorySupport.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": cohort_ids},
    columns=["hospitalization_id", "recorded_dttm", "fio2_set", "peep_obs", "peep_set"],
)
try:
    from clifpy.utils.outlier_handler import apply_outlier_handling
    apply_outlier_handling(rs_tbl)
except Exception:
    pass
rs = rs_tbl.df
rs["hospitalization_id"] = rs["hospitalization_id"].astype(str)
rs["recorded_dttm"] = to_central(rs["recorded_dttm"])
rs = rs.dropna(subset=["recorded_dttm"])
rs["peep"] = rs["peep_obs"].fillna(rs["peep_set"])

# Raw non-null FiO2 and PEEP observations (each its own asof series)
fio2_obs = rs.loc[rs["fio2_set"].notna(), ["hospitalization_id", "recorded_dttm", "fio2_set"]] \
             .sort_values("recorded_dttm").reset_index(drop=True)
peep_obs = rs.loc[rs["peep"].notna(), ["hospitalization_id", "recorded_dttm", "peep"]] \
             .sort_values("recorded_dttm").reset_index(drop=True)
print(f"  FiO2 obs: {len(fio2_obs):,}  PEEP obs: {len(peep_obs):,}")

# ----------------------------------------------------------------------------
# Load oxygenation events: PaO2 (labs) + SpO2 (vitals, <= 97%)
# ----------------------------------------------------------------------------

print("[2] PaO2 (labs) + SpO2 (vitals) events ...")
lab = Labs.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": cohort_ids, "lab_category": ["po2_arterial"]},
    columns=["hospitalization_id", "lab_category", "lab_result_dttm", "lab_value_numeric"],
).df
lab["hospitalization_id"] = lab["hospitalization_id"].astype(str)
lab["t"] = to_central(lab["lab_result_dttm"])
lab["val"] = pd.to_numeric(lab["lab_value_numeric"], errors="coerce")
lab = lab.loc[lab["val"].between(*PAO2_RANGE) & lab["t"].notna(), ["hospitalization_id", "t", "val"]]
lab["source"] = "pao2"

vit = Vitals.from_file(
    DATA_DIR, filetype=FILETYPE, timezone=TZ,
    filters={"hospitalization_id": cohort_ids, "vital_category": ["spo2"]},
    columns=["hospitalization_id", "vital_category", "recorded_dttm", "vital_value"],
).df
vit["hospitalization_id"] = vit["hospitalization_id"].astype(str)
vit["t"] = to_central(vit["recorded_dttm"])
vit["val"] = pd.to_numeric(vit["vital_value"], errors="coerce")
vit = vit.loc[vit["val"].between(*SPO2_RANGE) & (vit["val"] <= SPO2_MAX_FOR_SF) & vit["t"].notna(),
              ["hospitalization_id", "t", "val"]]
vit["source"] = "spo2"
print(f"  PaO2 events: {len(lab):,}  SpO2 events (<= {SPO2_MAX_FOR_SF:g}%): {len(vit):,}")

ev = pd.concat([lab, vit], ignore_index=True).sort_values("t").reset_index(drop=True)

# ----------------------------------------------------------------------------
# Pair each event with concurrent FiO2 + PEEP (<= 4h backward)
# ----------------------------------------------------------------------------

print("[3] merge_asof FiO2/PEEP within 4h backward ...")
ev = pd.merge_asof(ev, fio2_obs, by="hospitalization_id", left_on="t", right_on="recorded_dttm",
                   direction="backward", tolerance=O2_FIO2_LOOKBACK).drop(columns=["recorded_dttm"])
ev = ev.sort_values("t")
ev = pd.merge_asof(ev, peep_obs, by="hospitalization_id", left_on="t", right_on="recorded_dttm",
                   direction="backward", tolerance=O2_FIO2_LOOKBACK).drop(columns=["recorded_dttm"])

# Usable = FiO2 and PEEP both present within window
ev = ev.loc[ev["fio2_set"].notna() & ev["peep"].notna()].copy()
ev["ratio"] = ev["val"] / ev["fio2_set"]
ev["qual"] = (((ev["source"] == "pao2") & (ev["ratio"] < PF_THRESHOLD)) |
              ((ev["source"] == "spo2") & (ev["ratio"] < SF_THRESHOLD))) & (ev["peep"] > PEEP_MIN)
ev["calendar_day"] = ev["t"].dt.date
print(f"  usable events (FiO2 + PEEP present): {len(ev):,}")

# ----------------------------------------------------------------------------
# Per (hosp, day) classification
# ----------------------------------------------------------------------------

print("[4] Per patient-day classification ...")
key = ["hospitalization_id", "calendar_day"]
pf = ev.loc[ev["source"] == "pao2"].groupby(key)["ratio"].min().rename("worst_pf")
sf = ev.loc[ev["source"] == "spo2"].groupby(key)["ratio"].min().rename("worst_sf")
day = ev.groupby(key).agg(severe=("qual", "any"),
                          has_pao2=("source", lambda s: (s == "pao2").any())).reset_index()
day = day.merge(pf, on=key, how="left").merge(sf, on=key, how="left")
day["class_source"] = np.where(day["has_pao2"], "pao2", "spo2")

out = cohort[key].merge(day, on=key, how="left")
out["severity"] = np.where(out["severe"] == True, "severe",  # noqa: E712
                           np.where(out["severe"].notna(), "not_severe", "unknown"))
out["class_source"] = out["class_source"].fillna("none")
out = out[["hospitalization_id", "calendar_day", "severity", "worst_pf", "worst_sf", "class_source"]]

out_path = OUT_DIR / "02d_severity.parquet"
out.to_parquet(out_path, index=False)
print(f"  wrote {out_path}  ({len(out):,} rows)")

# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

N = len(out)
sev_counts = out["severity"].value_counts()
print("\n[diag] Severity distribution (cohort patient-days):")
for k, v in sev_counts.items():
    print(f"    {k:>12s}: {v:>8,}  ({v / N * 100:.1f}%)")
classified = out[out["severity"] != "unknown"]
src = classified["class_source"].value_counts()
print(f"\n[diag] Of {len(classified):,} classified days, source:")
for k, v in src.items():
    print(f"    {k:>6s}: {v:>8,}  ({v / len(classified) * 100:.1f}%)")
if out["worst_pf"].notna().any():
    q = out["worst_pf"].quantile([0.1, 0.5, 0.9])
    print(f"\n[diag] worst P/F (ABG days) p10/p50/p90: {q.iloc[0]:.0f} / {q.iloc[1]:.0f} / {q.iloc[2]:.0f}")

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "params": {"pf_threshold": PF_THRESHOLD, "sf_threshold": SF_THRESHOLD,
               "spo2_max_for_sf": SPO2_MAX_FOR_SF, "peep_min_strict": PEEP_MIN,
               "fio2_lookback_hours": 4},
    "n_patient_days": int(N),
    "severity_counts": {str(k): int(v) for k, v in sev_counts.items()},
    "pct_unknown": float((out["severity"] == "unknown").mean() * 100),
    "class_source_counts": {str(k): int(v) for k, v in src.items()},
    "reconciles_total": bool(int(sev_counts.sum()) == N),
}
(OUT_DIR / "02d_severity_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '02d_severity_summary.json'}")
print("Done.")
