"""Compute proning QI metrics, a site-aggregable summary, and the bundle
scorecard tile feed.

Denominator framing (decided 2026-06-03 — Option C, report both bounds):
    The UChicago `position` table appears to chart only proning episodes, not
    routine supine. Only ~19 % of PROSEVA-eligible patients have any position
    record, and 100 % of those were proned. We therefore report the QI rate as
    a *bounded* quantity rather than a single number:
      - lower bound  — adherent / ALL eligible        (no-data imputed not-proned)
      - upper bound  — adherent / charted (documented) subset
      - process rate — ever-proned / ALL eligible      (the tile headline)

Grain note: cohort + eligibility are one row per encounter_block (one per
patient); observation is one row per hospitalization_id. A stitched encounter
block can span several hospitalization_ids, and a prone session may be charted
under any of them — so observation is aggregated over *all* of an encounter
block's hospitalization_ids (cohort["hospitalization_ids"]), not just the
primary id.

Inputs:
    output/intermediate/cohort.parquet
    output/intermediate/proning_eligibility.parquet
    output/intermediate/proning_observation.parquet
    output/intermediate/prone_sessions.parquet

Outputs:
    output/intermediate/metrics_patient_level.parquet  (per eligible patient; keeps ids)
    output/final/metrics_site_summary.csv              (counts + rates only — federation-shareable)
    output/final/tile_feed_proning.json                (bundle scorecard tile feed, contract v1)

No raw PHI to stdout — only counts and aggregates. The tile feed is re-checked
for PHI substrings at build time and the script aborts if any appear.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

log = logging.getLogger("proning.metrics")

# Cumulative-incidence horizons (hours after T_eligible) reported in the CDF.
CDF_HORIZONS_H = [24, 48, 72, 168]

# Canonical ICU unit slugs (CLIF location_type) shared with the bundle-scorecard
# tile contract (lpv/plans/02_scorecard_tile_contract.md §3). location_type at
# this site is already exactly these values; "unknown" (T_eligible falling in a
# non-ICU gap) is kept in the dashboard but folded into __ALL__ for the feed.
CANONICAL_UNITS = [
    "medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
    "mixed_neuro_icu", "general_icu", "burn_icu",
]

# Time-bucket granularities → the patient-level column holding each period key.
GRANULARITY_COL = {"year": "period_year", "month": "period_month", "week": "period_week"}

# Tile-feed contract version (lpv/plans/02_scorecard_tile_contract.md §3).
TILE_SCHEMA_VERSION = 1
# Substrings the lpv scorecard rejects in any ingested feed (PHI guard).
PHI_FORBIDDEN = ("hospitalization_id", "patient_id")


def _load_cohort_module():
    """Import code/01_build_cohort.py via importlib (digit-prefixed name)."""
    path = CODE_DIR / "01_build_cohort.py"
    spec = importlib.util.spec_from_file_location("proning_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hids_for_block(row) -> list[str]:
    """All hospitalization_ids belonging to one cohort encounter block."""
    ids: set[str] = set()
    hids = row.get("hospitalization_ids")
    if isinstance(hids, (list, tuple, np.ndarray)):
        ids.update(str(h) for h in hids if h is not None and str(h) != "<NA>")
    elif hids is not None and str(hids) != "<NA>":
        ids.add(str(hids))
    primary = row.get("hospitalization_id")
    if primary is not None and str(primary) != "<NA>":
        ids.add(str(primary))
    return sorted(ids)


def aggregate_observation_to_eligible(
    eligible: pd.DataFrame, obs: pd.DataFrame
) -> pd.DataFrame:
    """Collapse per-hospitalization observation onto per-eligible-patient rows.

    For each eligible encounter block, OR the boolean flags and take min/max/sum
    of the timing/duration fields across all of its hospitalization_ids.
    """
    obs = obs.copy()
    obs["hospitalization_id"] = obs["hospitalization_id"].astype(str)
    obs_by_hid = obs.set_index("hospitalization_id")

    records = []
    for _, row in eligible.iterrows():
        hids = _hids_for_block(row)
        present = [h for h in hids if h in obs_by_hid.index]
        if present:
            sub = obs_by_hid.loc[present]
            rec = {
                "position_data_present": bool(sub["position_data_present"].any()),
                "any_prone": bool(sub["any_prone"].any()),
                "any_adherent": bool(sub["any_session_adherent"].any()),
                "n_sessions": int(sub["n_sessions"].sum()),
                "total_prone_hours": float(sub["total_prone_hours"].sum()),
                "longest_session_hours": float(sub["longest_session_hours"].max()),
                "first_prone_dttm": sub["first_prone_dttm"].min(),
            }
        else:
            rec = {
                "position_data_present": False,
                "any_prone": False,
                "any_adherent": False,
                "n_sessions": 0,
                "total_prone_hours": 0.0,
                "longest_session_hours": 0.0,
                "first_prone_dttm": pd.NaT,
            }
        rec["encounter_block"] = row["encounter_block"]
        records.append(rec)

    return pd.DataFrame.from_records(records)


def build_patient_level(cohort: pd.DataFrame, elig: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """One row per PROSEVA-eligible patient: demographics + T0 physiology +
    T_eligible + aggregated proning observation + time-to-prone."""
    eligible = elig[elig["eligible"]].copy()

    cohort_cols = [
        "encounter_block", "age_at_admission", "sex_category", "race_category",
        "ethnicity_category", "admission_type_category", "pao2_at_t0", "fio2_at_t0",
        "peep_at_t0", "pf_at_t0", "discharge_category", "hospitalization_ids",
        "hospitalization_id",
    ]
    merged = eligible.merge(cohort[cohort_cols], on="encounter_block", how="left",
                            suffixes=("", "_cohort"))

    agg = aggregate_observation_to_eligible(merged, obs)
    merged = merged.merge(agg, on="encounter_block", how="left")

    # Time from clinical decision-point (T_eligible) to first prone session.
    # Negative = proned during the 12h stabilization window (before formal T_eligible).
    dt = (merged["first_prone_dttm"] - merged["T_eligible"])
    merged["time_to_prone_hours"] = dt.dt.total_seconds() / 3600.0
    merged.loc[~merged["any_prone"], "time_to_prone_hours"] = np.nan

    # In-hospital mortality (Table 1 only; case-insensitive "Expired").
    merged["in_hospital_mortality"] = (
        merged["discharge_category"].astype("string").str.lower() == "expired"
    )

    return merged


# ---------------------------------------------------------------------------
# Unit (location_type) + time-period attribution
# ---------------------------------------------------------------------------
def attach_unit_and_periods(pl: pd.DataFrame, cohort_mod, tz: str) -> pd.DataFrame:
    """Attach the ICU unit (location_type at T_eligible) and week/month/year keys.

    Unit is the ICU `location_type` whose adt interval contains T_eligible
    (DuckDB range-join, mirroring restrict_to_icu in 01_build_cohort.py). Blocks
    with no containing ICU interval (T_eligible in a brief non-ICU gap) → "unknown".
    Period keys bucket by T_eligible, the PROSEVA decision-point.
    """
    adt = pd.read_parquet(cohort_mod.CACHE_DIR / "adt_stitched.parquet")
    adt["location_category"] = adt["location_category"].astype("string").str.strip().str.lower()
    icu = adt.loc[adt["location_category"] == "icu",
                  ["encounter_block", "in_dttm", "out_dttm", "location_type", "location_name"]].copy()
    icu["encounter_block"] = icu["encounter_block"].astype(str)
    icu["in_dttm"] = cohort_mod._coerce_dttm(icu["in_dttm"], tz)
    icu["out_dttm"] = cohort_mod._coerce_dttm(icu["out_dttm"], tz)
    icu["location_type"] = icu["location_type"].astype("string")
    # location_name = specific physical unit (finer than location_type); same ICU interval, so it
    # nests under the chosen type for free. Raw case (matches the other verticals' feed keys).
    icu["location_name"] = icu["location_name"].astype("string")

    base = pl[["encounter_block", "T_eligible"]].copy()
    base["encounter_block"] = base["encounter_block"].astype(str)
    base["T_eligible"] = cohort_mod._coerce_dttm(base["T_eligible"], tz)

    con = duckdb.connect()
    con.register("base", base)
    con.register("icu", icu)
    joined = con.execute(
        """
        SELECT base.encounter_block AS encounter_block,
               icu.location_type    AS unit,
               icu.location_name    AS unit_name,
               icu.in_dttm          AS in_dttm
        FROM base LEFT JOIN icu
          ON base.encounter_block = icu.encounter_block
         AND base.T_eligible BETWEEN icu.in_dttm AND icu.out_dttm
        """
    ).fetchdf()
    con.close()
    # One unit per block (earliest containing interval if several overlap); the specific unit
    # (unit_name) comes from that SAME interval, so it nests under the chosen type by construction.
    joined = (joined.sort_values(["encounter_block", "in_dttm"])
              .drop_duplicates("encounter_block", keep="first"))
    unit_map = joined.set_index("encounter_block")["unit"]
    name_map = joined.set_index("encounter_block")["unit_name"]

    pl = pl.copy()
    unit = pl["encounter_block"].astype(str).map(unit_map).astype("string")
    pl["unit"] = unit.where(unit.notna() & (unit.str.len() > 0), other="unknown")
    uname = pl["encounter_block"].astype(str).map(name_map).astype("string")
    pl["unit_name"] = uname.where(uname.notna() & (uname.str.len() > 0), other="unknown")

    te = cohort_mod._coerce_dttm(pl["T_eligible"], tz)
    iso = te.dt.isocalendar()
    pl["period_year"] = te.dt.year.astype("Int64").astype(str)
    pl["period_month"] = te.dt.strftime("%Y-%m")
    pl["period_week"] = (iso["year"].astype("Int64").astype(str) + "-W"
                         + iso["week"].astype("Int64").astype(str).str.zfill(2))
    return pl


def _slice_metrics(g: pd.DataFrame) -> dict:
    """The bounded-denominator metric bundle for one slice of eligible patients."""
    ttp = g.loc[g["any_prone"], "time_to_prone_hours"].dropna()
    return {
        "n_eligible": int(len(g)),
        "n_ever_proned": int(g["any_prone"].sum()),
        "n_adherent": int(g["any_adherent"].sum()),
        "n_documented": int(g["position_data_present"].sum()),
        "ttp_median_h": float(ttp.median()) if not ttp.empty else None,
        "ttp_q1_h": float(ttp.quantile(0.25)) if not ttp.empty else None,
        "ttp_q3_h": float(ttp.quantile(0.75)) if not ttp.empty else None,
    }


def build_slice_cells(pl: pd.DataFrame) -> pd.DataFrame:
    """Long, tidy table: one row per (unit, granularity, period). Includes the
    __ALL__ unit and the "all"/all-time granularity. Each granularity partitions
    the cohort exactly, so per-period and per-unit n_eligible both sum to the total.
    """
    rows = []

    def emit(unit, gran, period, g, dim="type", parent=None):
        rec = _slice_metrics(g)
        rec.update(unit=unit, granularity=gran, period=period, dim=dim, parent=parent)
        rows.append(rec)

    # Type dimension (location_type) — incl. __ALL__; byte-identical to the prior build.
    for unit, gu in [("__ALL__", pl)] + list(pl.groupby("unit", observed=True)):
        emit(unit, "all", "all", gu)
        for gran, col in GRANULARITY_COL.items():
            for period, g in gu.groupby(col, observed=True):
                emit(unit, gran, str(period), g)

    # Specific-unit dimension (location_name) — per-unit rows only (nested within type, no __ALL__).
    if "unit_name" in pl.columns:
        for uname, gun in pl.groupby("unit_name", observed=True):
            parent = gun["unit"].mode().iat[0] if not gun["unit"].mode().empty else "unknown"
            emit(uname, "all", "all", gun, dim="name", parent=parent)
            for gran, col in GRANULARITY_COL.items():
                for period, g in gun.groupby(col, observed=True):
                    emit(uname, gran, str(period), g, dim="name", parent=parent)

    cols = ["unit", "dim", "parent", "granularity", "period", "n_eligible", "n_ever_proned",
            "n_adherent", "n_documented", "ttp_median_h", "ttp_q1_h", "ttp_q3_h"]
    df = pd.DataFrame(rows)[cols]
    df["rate_ever_proned"] = df["n_ever_proned"] / df["n_eligible"]
    df["rate_adherent_all"] = df["n_adherent"] / df["n_eligible"]
    df["rate_adherent_charted"] = np.where(
        df["n_documented"] > 0, df["n_adherent"] / df["n_documented"], np.nan)
    return df.sort_values(["dim", "unit", "granularity", "period"]).reset_index(drop=True)


def compute_cdf(time_to_prone_hours: pd.Series, n_eligible: int, horizons: list[int]) -> dict:
    """Cumulative incidence of first prone by each horizon, over ALL eligible.

    Non-proned eligible are event-free (counted in the denominator, never an
    event). Patients proned before T_eligible (negative time) count as an event
    at every horizon.
    """
    out = {}
    times = time_to_prone_hours.dropna()
    for h in horizons:
        n_by = int((times <= h).sum())
        out[h] = {"num": n_by, "den": n_eligible,
                  "rate": (n_by / n_eligible) if n_eligible else None}
    return out


def _rate(num: int, den: int):
    return (num / den) if den else None


def build_summary_rows(site: str, cfg: dict, m: dict) -> pd.DataFrame:
    """Tidy, federation-shareable summary — counts + rates only, no row-level data."""
    rows = [
        ("ards_cohort", "ARDS cohort (one row per patient)", m["n_ards"], m["n_ards"], 1.0,
         "Berlin moderate-severe ARDS phenotype at T0"),
        ("proseva_eligible", "PROSEVA-strict eligible", m["n_eligible"], m["n_ards"],
         _rate(m["n_eligible"], m["n_ards"]), "QI denominator"),
        ("ever_proned", "Ever proned / eligible (process rate; TILE HEADLINE)",
         m["n_ever_proned"], m["n_eligible"], _rate(m["n_ever_proned"], m["n_eligible"]),
         "any documented prone session"),
        ("adherent_all_eligible", "Adherent ≥16h / ALL eligible (lower bound)",
         m["n_adherent"], m["n_eligible"], _rate(m["n_adherent"], m["n_eligible"]),
         "no-position-data imputed not-adherent"),
        ("position_data_present", "Eligible with any position record (documented subset)",
         m["n_documented"], m["n_eligible"], _rate(m["n_documented"], m["n_eligible"]),
         "position table coverage among eligible"),
        ("adherent_documented", "Adherent ≥16h / documented subset (upper bound)",
         m["n_adherent"], m["n_documented"], _rate(m["n_adherent"], m["n_documented"]),
         "charted-only; excludes 81% of eligible"),
        ("time_to_prone_median_h", "Median hours T_eligible→first prone (proned only)",
         None, None, m["ttp_median_h"], "IQR in q1/q3 rows below"),
        ("time_to_prone_q1_h", "Q1 hours T_eligible→first prone", None, None, m["ttp_q1_h"], ""),
        ("time_to_prone_q3_h", "Q3 hours T_eligible→first prone", None, None, m["ttp_q3_h"], ""),
    ]
    for h in CDF_HORIZONS_H:
        c = m["cdf"][h]
        rows.append((f"cumulative_proned_by_{h}h",
                     f"Cumulative ever-proned within {h}h of T_eligible / eligible",
                     c["num"], c["den"], c["rate"], "event-free if never proned"))

    df = pd.DataFrame(rows, columns=["metric", "label", "numerator", "denominator", "rate", "note"])
    df.insert(0, "site", site)
    df["adherent_session_hours"] = cfg["proning_observation"]["adherent_session_hours"]
    df["generated"] = m["generated"]
    return df


def build_tile_feed(cfg: dict, m: dict, slices: pd.DataFrame) -> dict:
    """tile_feed_proning.json conforming to lpv/plans/02_scorecard_tile_contract.md §3.

    Headline donut = ever-proned / all eligible (process rate).
    Segments      = adherent / all eligible (lower bound) and adherent / charted (upper bound).
    Grain         = per-unit + monthly (units = __ALL__ + canonical ICU slugs present;
                    periods = ["all","month"]). Week/year stay dashboard-only (week is too
                    sparse for the contract; "year" is not a contract period key). A cell is
                    only emitted where the slice exists (non-empty group), which naturally
                    prunes empty (unit, month) combinations.
    """
    is_type = slices["dim"] == "type" if "dim" in slices.columns else slices["unit"].notna()
    units = ["__ALL__"] + [u for u in CANONICAL_UNITS if u in set(slices.loc[is_type, "unit"])]
    # Specific-unit (location_name) dimension: name keys nest under their parent type. Drop the
    # "unknown" catch-all (T_eligible in a non-ICU gap) from the displayed name list, mirroring how
    # the type list already excludes it (only CANONICAL_UNITS).
    sl_name = slices[slices["dim"] == "name"] if "dim" in slices.columns else slices.iloc[0:0]
    name_parent = sl_name.drop_duplicates("unit").set_index("unit")["parent"].to_dict()
    name_units = sorted([n for n in name_parent if n != "unknown"],
                        key=lambda n: (CANONICAL_UNITS.index(name_parent[n])
                                       if name_parent[n] in CANONICAL_UNITS else 99, n))
    LABELS = cfg.get("unit_labels", {}) or {}
    dim_labels = {n: LABELS.get(n, n) for n in name_units}
    dim_labels.update({u: LABELS[u] for u in units if u != "__ALL__" and u in LABELS})
    all_keys = units + name_units

    months = sorted(slices.loc[slices["granularity"] == "month", "period"].unique())
    by_key = {(r.unit, "all" if r.granularity == "all" else r.period): r
              for r in slices.itertuples(index=False)
              if r.granularity in ("all", "month")}

    def cells(num_col, den_col):
        out = {}
        for u in all_keys:
            pc = {}
            for p in ["all"] + months:
                r = by_key.get((u, p))
                if r is None:
                    continue
                den = int(getattr(r, den_col))
                cell = {"num": int(getattr(r, num_col)), "den": den}
                if den_col == "n_eligible":   # headline carries the denominator count too
                    cell["n"] = den
                pc[p] = cell
            if pc:
                out[u] = pc
        return out

    return {
        "schema_version": TILE_SCHEMA_VERSION,
        "metric_id": "proning",
        "title": "ARDS Proning",
        "subtitle": "PROSEVA-eligible ARDS, proned",
        "icon": "prone",
        "detail_href": "proning_dashboard.html",
        "goal": None,
        "generated": m["generated"],
        "note": ("Position table at this site charts only proning episodes; "
                 f"only {m['n_documented']}/{m['n_eligible']} eligible have any "
                 "position record. Adherence shown as a bound (all-eligible vs charted)."),
        "grain": {"units": units, "periods": ["all", "month"]},
        # Two ICU-grouping dimensions: `type` = location_type (back-compat, also grain.units);
        # `name` = specific unit (location_name). Both key into headline/segment cells; `parent`
        # nests each name under its type; `labels` are optional friendly names (config unit_labels).
        # The scorecard's "Group ICUs by" toggle reads this block.
        "dims": {"type": [u for u in units if u != "__ALL__"], "name": name_units,
                 "parent": {n: name_parent[n] for n in name_units}, "labels": dim_labels},
        "headline": {
            "label": "ever proned",
            "den_label": "of PROSEVA-eligible",
            "n_unit": "patients",
            "cells": cells("n_ever_proned", "n_eligible"),
        },
        "segments": [
            {"key": "adherent_all", "label": "Adherent ≥16h",
             "cells": cells("n_adherent", "n_eligible")},
            {"key": "adherent_charted", "label": "Adherent (charted)",
             "cells": cells("n_adherent", "n_documented")},
        ],
    }


def _assert_phi_free(feed: dict) -> None:
    blob = json.dumps(feed)
    hits = [s for s in PHI_FORBIDDEN if s in blob]
    if hits:
        raise RuntimeError(f"tile feed contains forbidden PHI substring(s): {hits}")


def _assert_slice_integrity(slices: pd.DataFrame, m: dict) -> None:
    """The __ALL__/all cell must reproduce the headline; each granularity must
    partition the cohort (per-unit and per-period n_eligible both sum to total)."""
    # Integrity checks are scoped to the TYPE dim so they stay byte-identical to the pre-name build.
    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    a = typ[(typ["unit"] == "__ALL__") & (typ["granularity"] == "all")].iloc[0]
    for col, key in [("n_eligible", "n_eligible"), ("n_ever_proned", "n_ever_proned"),
                     ("n_adherent", "n_adherent"), ("n_documented", "n_documented")]:
        if int(a[col]) != int(m[key]):
            raise RuntimeError(f"slice __ALL__/all {col}={a[col]} != headline {m[key]}")
    units_all = typ[(typ["granularity"] == "all") & (typ["unit"] != "__ALL__")]
    if int(units_all["n_eligible"].sum()) != m["n_eligible"]:
        raise RuntimeError("per-unit n_eligible does not sum to total")
    for gran in GRANULARITY_COL:
        s = typ[(typ["unit"] == "__ALL__") & (typ["granularity"] == gran)]
        if int(s["n_eligible"].sum()) != m["n_eligible"]:
            raise RuntimeError(f"per-period ({gran}) n_eligible does not sum to total")
    # Name dim nests under type: each type's specific-unit children sum to its eligible total.
    if "dim" in slices.columns:
        nm = slices[(slices["dim"] == "name") & (slices["granularity"] == "all")]
        if not nm.empty:
            by_parent = nm.groupby("parent")["n_eligible"].sum()
            type_tot = units_all.set_index("unit")["n_eligible"]
            for t, v in by_parent.items():
                if int(v) != int(type_tot.get(t, -1)):
                    raise RuntimeError(f"name children of {t} sum to {v} != type {type_tot.get(t)}")


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cohort_mod.LOGS_DIR / "04_metrics.log", mode="w"),
        ],
    )

    cfg = cohort_mod.load_config(cohort_mod.CONFIG_PATH)
    site = cfg.get("site", "unknown")
    tz = cfg["timezone"]
    adherent_h = float(cfg["proning_observation"]["adherent_session_hours"])
    log.info("site=%s adherent threshold=%.1fh", site, adherent_h)

    inter = cohort_mod.INTERMEDIATE_DIR
    final = cohort_mod.FINAL_DIR
    cohort = pd.read_parquet(inter / "cohort.parquet")
    elig = pd.read_parquet(inter / "proning_eligibility.parquet")
    obs = pd.read_parquet(inter / "proning_observation.parquet")

    pl = build_patient_level(cohort, elig, obs)
    pl = attach_unit_and_periods(pl, cohort_mod, tz)

    # ---- headline counts -------------------------------------------------
    n_ards = int(cohort["patient_id"].nunique())
    n_eligible = len(pl)
    n_ever_proned = int(pl["any_prone"].sum())
    n_adherent = int(pl["any_adherent"].sum())
    n_documented = int(pl["position_data_present"].sum())

    # Sanity: at UChicago every documented-eligible patient was proned.
    if n_documented != n_ever_proned:
        log.warning("documented (%d) != ever-proned (%d) — position rows without a "
                    "prone session exist; 'adherent/charted' denominator uses documented.",
                    n_documented, n_ever_proned)

    ttp = pl.loc[pl["any_prone"], "time_to_prone_hours"].dropna()
    ttp_median = float(ttp.median()) if not ttp.empty else None
    ttp_q1 = float(ttp.quantile(0.25)) if not ttp.empty else None
    ttp_q3 = float(ttp.quantile(0.75)) if not ttp.empty else None
    cdf = compute_cdf(pl["time_to_prone_hours"], n_eligible, CDF_HORIZONS_H)

    import datetime as _dt
    generated = _dt.datetime.now().isoformat(timespec="minutes")

    m = {
        "n_ards": n_ards, "n_eligible": n_eligible, "n_ever_proned": n_ever_proned,
        "n_adherent": n_adherent, "n_documented": n_documented,
        "ttp_median_h": ttp_median, "ttp_q1_h": ttp_q1, "ttp_q3_h": ttp_q3,
        "cdf": cdf, "generated": generated,
    }

    # ---- sliced metrics (unit x granularity x period) --------------------
    slices = build_slice_cells(pl)
    _assert_slice_integrity(slices, m)
    n_units = pl["unit"].nunique()
    n_unknown = int((pl["unit"] == "unknown").sum())
    log.info("unit attribution: %d units (incl. %d unknown of %d eligible); "
             "%d slice cells across all/year/month/week",
             n_units, n_unknown, n_eligible, len(slices))

    # ---- write outputs ---------------------------------------------------
    pl_path = inter / "metrics_patient_level.parquet"
    pl.to_parquet(pl_path, index=False)

    summary = build_summary_rows(site, cfg, m)
    summary_path = final / "metrics_site_summary.csv"
    summary.to_csv(summary_path, index=False)

    slices_pq = inter / "metrics_slices.parquet"      # full counts — dashboard embeds this
    slices.to_parquet(slices_pq, index=False)
    slices_csv = final / "metrics_slices.csv"          # federation-shareable (counts + rates)
    slices_out = slices.copy()
    slices_out.insert(0, "site", site)
    slices_out["generated"] = generated
    slices_out.to_csv(slices_csv, index=False)

    feed = build_tile_feed(cfg, m, slices)
    _assert_phi_free(feed)
    feed_path = final / "tile_feed_proning.json"
    with open(feed_path, "w") as f:
        json.dump(feed, f, indent=2, ensure_ascii=False)

    # ---- log summary -----------------------------------------------------
    log.info("ARDS cohort:               %5d patients", n_ards)
    log.info("PROSEVA-eligible:          %5d (%.1f%% of ARDS)", n_eligible, 100 * _rate(n_eligible, n_ards))
    log.info("ever proned / eligible:    %5d (%.1f%%)  [tile headline]", n_ever_proned, 100 * _rate(n_ever_proned, n_eligible))
    log.info("adherent / all eligible:   %5d (%.1f%%)  [lower bound]", n_adherent, 100 * _rate(n_adherent, n_eligible))
    log.info("position data present:     %5d (%.1f%%)  [documented subset]", n_documented, 100 * _rate(n_documented, n_eligible))
    log.info("adherent / documented:     %5d (%.1f%%)  [upper bound]", n_adherent, 100 * _rate(n_adherent, n_documented))
    if ttp_median is not None:
        log.info("time to first prone (h):   median %.1f (IQR %.1f–%.1f), among %d proned",
                 ttp_median, ttp_q1, ttp_q3, len(ttp))
    for h in CDF_HORIZONS_H:
        c = cdf[h]
        log.info("  cumulative proned ≤%3dh:  %5d (%.1f%%)", h, c["num"], 100 * c["rate"])
    log.info("wrote: %s", pl_path.relative_to(PROJECT_ROOT))
    log.info("wrote: %s", summary_path.relative_to(PROJECT_ROOT))
    log.info("wrote: %s (%d cells)", slices_pq.relative_to(PROJECT_ROOT), len(slices))
    log.info("wrote: %s", slices_csv.relative_to(PROJECT_ROOT))
    log.info("wrote: %s  (PHI-free check passed; grain units=%d periods=all,month)",
             feed_path.relative_to(PROJECT_ROOT), len(feed["grain"]["units"]))


if __name__ == "__main__":
    main()
