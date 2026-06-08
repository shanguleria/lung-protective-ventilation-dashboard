"""
03_aggregate.py — Roll patient-day verdicts up to (time x unit) for all 4 measures,
plus a precomputed Vt-cutoff grid for the dashboard slider.

Reads:
  output/02_patient_day_status.parquet  — per-measure status at default Vt=6 (the cohort spine)
  output/02_intervals.parquet           — mode-eligible IMV interval-pieces (Vt-grid engine)

Writes (long format; measure in {vt, plat, dp, comp}; month as 'YYYY-MM'):
  output/03_daily_unit_summary.parquet       — (calendar_day, unit[+__ALL__], measure) counts + both rates
  output/03_monthly_unit_summary.parquet     — same, monthly
  output/03_vt_grid_monthly.parquet          — (month, unit[+__ALL__], vt_cutoff, measure[vt,comp])
  output/03_vt_grid_daily_allunits.parquet   — (calendar_day, vt_cutoff, measure[vt,comp]) site-wide only
  output/03_aggregate_summary.json           — overall rates + cross-checks

Both rates: assessable = adherent/(adherent+non_adherent); crude = adherent/n_total.
Plateau<=30 & dP<=15 fixed; only the Vt cutoff varies in the grid.

Run:
    .venv/bin/python code/03_aggregate.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]            # bundle root (shared config.json)
_METRIC_ROOT = Path(__file__).resolve().parents[1]    # metrics/lpv (per-metric outputs)
CFG = json.loads((ROOT / "config.json").read_text())
OUT_DIR = Path(CFG.get("output_path", _METRIC_ROOT / "output"))

# Params (mirror 02_features.py / 02b_vt_sensitivity.py)
ADHERENCE_FRACTION = 0.80
MIN_ASSESSABLE_MIN = 60
PLATEAU_MAX, DP_MAX = 30.0, 15.0
VT_DEFAULT = 6.0
VT_GRID = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0]
MEASURES = ["vt", "plat", "dp", "comp"]
ALL = "__ALL__"
STATUSES = ["adherent", "non_adherent", "not_assessable"]


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    assessable = df["n_adherent"] + df["n_non_adherent"]
    df["assessable_rate"] = np.where(assessable > 0, df["n_adherent"] / assessable.where(assessable > 0), np.nan)
    df["crude_rate"] = np.where(df["n_total"] > 0, df["n_adherent"] / df["n_total"].where(df["n_total"] > 0), np.nan)
    return df


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

print("[0] Loading 02 outputs ...")
status = pd.read_parquet(OUT_DIR / "02_patient_day_status.parquet")
status["calendar_day"] = pd.to_datetime(status["calendar_day"])
status["month"] = status["calendar_day"].dt.strftime("%Y-%m")
_iso = status["calendar_day"].dt.isocalendar()
status["week"] = _iso["year"].astype(str) + "-W" + _iso["week"].astype(int).map("{:02d}".format)
status["calendar_day"] = status["calendar_day"].dt.date
status["hospitalization_id"] = status["hospitalization_id"].astype(str)

# Severity stratifier (severe respiratory failure) — join per (hosp, day).
SEVS = ["severe", "not_severe", "unknown"]
sev = pd.read_parquet(OUT_DIR / "02d_severity.parquet")[["hospitalization_id", "calendar_day", "severity"]]
sev["hospitalization_id"] = sev["hospitalization_id"].astype(str)
sev["calendar_day"] = pd.to_datetime(sev["calendar_day"]).dt.date
status = status.merge(sev, on=["hospitalization_id", "calendar_day"], how="left")
status["severity"] = status["severity"].fillna("unknown")
print(f"  cohort patient-days: {len(status):,}  | severity: "
      + ", ".join(f"{k} {int(v):,}" for k, v in status['severity'].value_counts().items()))


# ----------------------------------------------------------------------------
# A. Per-measure summaries (outputs 1, 2) from the status table
# ----------------------------------------------------------------------------

def summarize(bucket_col: str, by_severity: bool = False,
              unit_col: str = "assigned_unit", dim: str = "type",
              include_all: bool = True) -> pd.DataFrame:
    """Long per-measure summary for one time bucket, incl. __ALL__ pooled rows.
    When by_severity, adds a `severity` group key (strata; dashboard derives 'All' = sum).

    `unit_col` chooses the grouping grain: `assigned_unit` (location_type, dim='type')
    or `assigned_unit_name` (specific unit, dim='name'). The result always carries a
    generic `assigned_unit` key column + a `dim` tag. Name rows omit the pooled __ALL__
    (`include_all=False`) — the shared site-wide row lives only on the type dim."""
    extra = ["severity"] if by_severity else []
    src = status if dim == "type" else status.dropna(subset=[unit_col])
    frames = []
    for m in MEASURES:
        st = src[[bucket_col, unit_col, f"{m}_status"] + extra].rename(
            columns={f"{m}_status": "status", unit_col: "unit"})
        group_sets = [([bucket_col, "unit"] + extra, None)]
        if include_all:
            group_sets.append(([bucket_col] + extra, ALL))
        for unit_keys, tag in group_sets:
            counts = (st.groupby(unit_keys)["status"].value_counts().unstack(fill_value=0)
                      .reindex(columns=STATUSES, fill_value=0).reset_index())
            if tag is not None:
                counts["unit"] = tag
            counts["measure"] = m
            frames.append(counts)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"adherent": "n_adherent", "non_adherent": "n_non_adherent",
                              "not_assessable": "n_not_assessable", "unit": "assigned_unit"})
    out["n_total"] = out["n_adherent"] + out["n_non_adherent"] + out["n_not_assessable"]
    out = add_rates(out)
    out["dim"] = dim
    cols = ([bucket_col, "assigned_unit", "dim"] + extra + ["measure", "n_total",
            "n_adherent", "n_non_adherent", "n_not_assessable", "assessable_rate", "crude_rate"])
    return out[cols].sort_values([bucket_col, "assigned_unit"] + extra + ["measure"]).reset_index(drop=True)


print("[A] Per-measure summaries (day all-severity, month severity-stratified) ...")
# Type dim (location_type) — byte-identical to the prior build; name dim (specific unit)
# is additive (per-unit rows only, no __ALL__). Both concatenated, tagged via `dim`.
daily_t = summarize("calendar_day")                    # all-severity (daily drill-down)
monthly_t = summarize("month", by_severity=True)       # severity-stratified
weekly_t = summarize("week")                            # all-severity (weekly view)
daily_n = summarize("calendar_day", unit_col="assigned_unit_name", dim="name", include_all=False)
monthly_n = summarize("month", by_severity=True, unit_col="assigned_unit_name", dim="name", include_all=False)
weekly_n = summarize("week", unit_col="assigned_unit_name", dim="name", include_all=False)
daily = pd.concat([daily_t, daily_n], ignore_index=True)
monthly = pd.concat([monthly_t, monthly_n], ignore_index=True)
weekly = pd.concat([weekly_t, weekly_n], ignore_index=True)
daily.to_parquet(OUT_DIR / "03_daily_unit_summary.parquet", index=False)
monthly.to_parquet(OUT_DIR / "03_monthly_unit_summary.parquet", index=False)
weekly.to_parquet(OUT_DIR / "03_weekly_unit_summary.parquet", index=False)
print(f"  wrote 03_daily_unit_summary.parquet  ({len(daily):,} rows; "
      f"{len(daily_n):,} name-dim)")
print(f"  wrote 03_monthly_unit_summary.parquet ({len(monthly):,} rows; "
      f"{len(monthly_n):,} name-dim)")

# Name → parent type map (every specific unit rolls up to exactly one location_type).
name_parent = (status.dropna(subset=["assigned_unit_name"])
               .drop_duplicates("assigned_unit_name")
               .set_index("assigned_unit_name")["assigned_unit"].to_dict())


# ----------------------------------------------------------------------------
# B. Vt-cutoff grid (outputs 3, 4) recomputed from intervals
# ----------------------------------------------------------------------------

print("[B] Vt-cutoff grid from intervals ...")
iv = pd.read_parquet(OUT_DIR / "02_intervals.parquet")
iv["calendar_day"] = pd.to_datetime(iv["calendar_day"]).dt.date
key = ["hospitalization_id", "calendar_day"]
gk = [iv["hospitalization_id"], iv["calendar_day"]]

vt_present = iv["vt_per_pbw"].notna()
comp_present = iv["vt_per_pbw"].notna() & iv["plateau"].notna() & iv["driving_pressure"].notna()
fixed_ok = (iv["plateau"] <= PLATEAU_MAX) & (iv["driving_pressure"] <= DP_MAX)
dur = iv["duration_min"]

vt_assess = dur.where(vt_present, 0.0).groupby(gk).sum()
comp_assess = dur.where(comp_present, 0.0).groupby(gk).sum()

# Spine: every cohort patient-day with its unit + buckets (n_total denominator lives here).
spine = status[["hospitalization_id", "calendar_day", "assigned_unit", "assigned_unit_name",
                "month", "week", "severity"]].copy()

# Per-(hosp,day) assessable booleans are cutoff-independent.
day_idx = spine.set_index(key)
day_idx["calendar_day"] = day_idx.index.get_level_values("calendar_day")  # bucket col (also an index level)
day_idx["vt_assessable"] = vt_assess.reindex(day_idx.index).fillna(0.0) >= MIN_ASSESSABLE_MIN
day_idx["comp_assessable"] = comp_assess.reindex(day_idx.index).fillna(0.0) >= MIN_ASSESSABLE_MIN
vt_assess_al = vt_assess.reindex(day_idx.index)
comp_assess_al = comp_assess.reindex(day_idx.index)


def grid_counts(bucket_col: str, pooled_only: bool, by_severity: bool = False,
                unit_col: str = "assigned_unit", dim: str = "type",
                include_all: bool = True) -> pd.DataFrame:
    rows = []
    sev_key = ["severity"] if by_severity else []
    keep = day_idx if dim == "type" else day_idx[day_idx[unit_col].notna()]
    for c in VT_GRID:
        vt_in = dur.where(vt_present & (iv["vt_per_pbw"] <= c), 0.0).groupby(gk).sum().reindex(keep.index).fillna(0.0)
        comp_in = dur.where(comp_present & fixed_ok & (iv["vt_per_pbw"] <= c), 0.0).groupby(gk).sum().reindex(keep.index).fillna(0.0)
        vt_assess_k = vt_assess_al.reindex(keep.index)
        comp_assess_k = comp_assess_al.reindex(keep.index)
        frac_vt = np.where(vt_assess_k > 0, vt_in / vt_assess_k.where(vt_assess_k > 0), np.nan)
        frac_comp = np.where(comp_assess_k > 0, comp_in / comp_assess_k.where(comp_assess_k > 0), np.nan)
        cols_d = {
            "bucket": keep[bucket_col].values,
            "unit": keep[unit_col].values,
            "vt_assessable": keep["vt_assessable"].values,
            "comp_assessable": keep["comp_assessable"].values,
            "vt_adher": keep["vt_assessable"].values & (pd.Series(frac_vt).values >= ADHERENCE_FRACTION),
            "comp_adher": keep["comp_assessable"].values & (pd.Series(frac_comp).values >= ADHERENCE_FRACTION),
        }
        if by_severity:
            cols_d["severity"] = keep["severity"].values
        df = pd.DataFrame(cols_d)
        for measure, ass_col, adh_col in [("vt", "vt_assessable", "vt_adher"), ("comp", "comp_assessable", "comp_adher")]:
            if pooled_only:
                group_sets = [(["bucket"] + sev_key, ALL)]
            else:
                group_sets = [(["bucket", "unit"] + sev_key, None)]
                if include_all:
                    group_sets.append((["bucket"] + sev_key, ALL))
            for gcols, tag in group_sets:
                agg = df.groupby(gcols).agg(n_total=(ass_col, "size"),
                                            n_assessable=(ass_col, "sum"),
                                            n_adherent=(adh_col, "sum")).reset_index()
                if tag is not None:
                    agg["unit"] = tag
                agg["vt_cutoff"] = c
                agg["measure"] = measure
                rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["assessable_rate"] = np.where(out["n_assessable"] > 0, out["n_adherent"] / out["n_assessable"].where(out["n_assessable"] > 0), np.nan)
    out["crude_rate"] = np.where(out["n_total"] > 0, out["n_adherent"] / out["n_total"].where(out["n_total"] > 0), np.nan)
    out = out.rename(columns={"bucket": bucket_col, "unit": "assigned_unit"})
    out["dim"] = dim
    cols = [bucket_col, "assigned_unit", "dim"] + sev_key + ["vt_cutoff", "measure", "n_total", "n_assessable", "n_adherent", "assessable_rate", "crude_rate"]
    return out[cols].sort_values([bucket_col, "assigned_unit"] + sev_key + ["measure", "vt_cutoff"]).reset_index(drop=True)


grid_monthly_t = grid_counts("month", pooled_only=False, by_severity=True)
grid_daily = grid_counts("calendar_day", pooled_only=True)   # all-severity, site-wide only (no name split)
grid_weekly_t = grid_counts("week", pooled_only=False)       # all-severity (weekly view, per-unit)
grid_monthly_n = grid_counts("month", pooled_only=False, by_severity=True,
                             unit_col="assigned_unit_name", dim="name", include_all=False)
grid_weekly_n = grid_counts("week", pooled_only=False,
                            unit_col="assigned_unit_name", dim="name", include_all=False)
grid_monthly = pd.concat([grid_monthly_t, grid_monthly_n], ignore_index=True)
grid_weekly = pd.concat([grid_weekly_t, grid_weekly_n], ignore_index=True)
grid_monthly.to_parquet(OUT_DIR / "03_vt_grid_monthly.parquet", index=False)
grid_daily.to_parquet(OUT_DIR / "03_vt_grid_daily_allunits.parquet", index=False)
grid_weekly.to_parquet(OUT_DIR / "03_vt_grid_weekly.parquet", index=False)
print(f"  wrote 03_vt_grid_monthly.parquet       ({len(grid_monthly):,} rows; {len(grid_monthly_n):,} name-dim)")
print(f"  wrote 03_vt_grid_daily_allunits.parquet ({len(grid_daily):,} rows)")


# ----------------------------------------------------------------------------
# Verification + summary
# ----------------------------------------------------------------------------

print("\n[verify] Cross-checks (monthly is severity-stratified → sum severity out):")


def collapse(df, keys, col="n_adherent"):
    return df.groupby(keys)[col].sum()


# NOTE: checks (2)-(6) are scoped to the TYPE dim (the *_t frames) so they stay
# byte-identical to the pre-name-dim build; (7) validates the additive name dim.

# (2) Grid at c=6 == default summaries for vt & comp (sum over severity)
g6 = grid_monthly_t[grid_monthly_t["vt_cutoff"] == 6.0]
ok_grid = True
for m in ("vt", "comp"):
    a = collapse(monthly_t[monthly_t["measure"] == m], ["month", "assigned_unit"])
    b = collapse(g6[g6["measure"] == m], ["month", "assigned_unit"])
    ok_grid &= bool(a.reindex(b.index).fillna(0).astype(int).equals(b.astype(int)))
print(f"  grid(c=6) n_adherent == default summary (vt, comp): {ok_grid}")

# (3) Reconcile overall __ALL__ to 02_features (sums over months × severity)
feat = json.loads((OUT_DIR / "02_features_summary.json").read_text())["per_measure"]
recon = {}
for m in MEASURES:
    sub = monthly_t[(monthly_t["assigned_unit"] == ALL) & (monthly_t["measure"] == m)]
    n_ad = int(sub["n_adherent"].sum()); n_na = int(sub["n_non_adherent"].sum())
    ar = n_ad / (n_ad + n_na) if (n_ad + n_na) else float("nan")
    recon[m] = {"assessable_rate": ar, "feat": feat[m]["assessable_rate"],
                "match": abs(ar - feat[m]["assessable_rate"]) < 1e-9}
    print(f"  {m:>5}: assessable_rate {ar*100:.2f}%  (02_features {feat[m]['assessable_rate']*100:.2f}%)  match={recon[m]['match']}")

# (4) Pooled __ALL__ == sum over units (sum over severity)
per_unit = collapse(monthly_t[monthly_t["assigned_unit"] != ALL], ["month", "measure"])
pooled = collapse(monthly_t[monthly_t["assigned_unit"] == ALL], ["month", "measure"])
ok_pool = bool(per_unit.reindex(pooled.index).fillna(0).astype(int).equals(pooled.astype(int)))
print(f"  pooled __ALL__ == sum over units: {ok_pool}")

# (5) Internal: daily counts sum to total; daily(all-sev)->monthly(sum-sev) consistency
ok_counts = bool((daily["n_total"] == daily[["n_adherent", "n_non_adherent", "n_not_assessable"]].sum(axis=1)).all())
day2mo = (daily_t[daily_t["assigned_unit"] != ALL].assign(month=pd.to_datetime(daily_t["calendar_day"]).dt.strftime("%Y-%m"))
          .groupby(["month", "assigned_unit", "measure"])["n_adherent"].sum())
mo_chk = collapse(monthly_t[monthly_t["assigned_unit"] != ALL], ["month", "assigned_unit", "measure"])
ok_day2mo = bool(day2mo.reindex(mo_chk.index).fillna(0).astype(int).equals(mo_chk.astype(int)))
print(f"  counts sum to n_total: {ok_counts}  |  daily->monthly consistent: {ok_day2mo}")

# (6) Severity strata complete: __ALL__ vt n_total over months × severity == cohort patient-days
strata_total = int(monthly_t[(monthly_t["assigned_unit"] == ALL) & (monthly_t["measure"] == "vt")]["n_total"].sum())
ok_strata = (strata_total == len(status))
print(f"  severity strata sum to cohort ({strata_total:,} == {len(status):,}): {ok_strata}")

# (7) Name dim nests under type: for each (month, type), the specific-unit children's
#     vt n_total sum EXACTLY to that type's n_total (every cohort day gets a child unit).
nt = monthly_n[monthly_n["measure"] == "vt"].copy()
nt["parent"] = nt["assigned_unit"].map(name_parent)
name_by_parent = nt.groupby(["month", "parent"])["n_total"].sum()
type_tot = (monthly_t[(monthly_t["measure"] == "vt") & (monthly_t["assigned_unit"] != ALL)]
            .groupby(["month", "assigned_unit"])["n_total"].sum())
ok_nest = bool(name_by_parent.reindex(type_tot.index).fillna(0).astype(int).equals(type_tot.astype(int)))
ok_nest &= bool(nt["parent"].notna().all())   # every name maps to a known type
print(f"  name-dim nests under type (children sum to parent): {ok_nest}")

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "params": {"adherence_fraction": ADHERENCE_FRACTION, "min_assessable_min": MIN_ASSESSABLE_MIN,
               "plateau_max": PLATEAU_MAX, "dp_max": DP_MAX, "vt_default": VT_DEFAULT, "vt_grid": VT_GRID,
               "severity_strata": SEVS},
    "rows": {"daily": len(daily), "monthly": len(monthly),
             "vt_grid_monthly": len(grid_monthly), "vt_grid_daily_allunits": len(grid_daily),
             "monthly_name_dim": len(monthly_n)},
    "overall_assessable_rate_default": {m: recon[m]["assessable_rate"] for m in MEASURES},
    "unit_name_to_type": name_parent,
    "checks": {"grid_eq_default": ok_grid, "reconcile_02features": all(recon[m]["match"] for m in MEASURES),
               "pooled_eq_sum_units": ok_pool, "counts_sum_total": ok_counts, "daily_to_monthly": ok_day2mo,
               "severity_strata_complete": ok_strata, "name_dim_nests_under_type": ok_nest},
}
(OUT_DIR / "03_aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nWrote {OUT_DIR / '03_aggregate_summary.json'}")
print("Done.")
