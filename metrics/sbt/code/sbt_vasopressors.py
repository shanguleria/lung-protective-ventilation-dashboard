"""Norepinephrine-equivalent (NEE) vasopressor engine for the SBT QI vertical.

clifpy 0.4.9 has NO norepinephrine-equivalent helper (its `sofa.py` uses raw
per-drug dose thresholds; `ase.py` only flags vasoactive presence). So this module
builds one, reusing clifpy's `unit_converter.standardize_dose_to_base_units` for the
unit/weight plumbing and applying standard published conversion factors on top.

Pipeline:
  medication_admin_continuous (vasopressor med_category subset)
    -> standardize_dose_to_base_units(med_df, vitals_df)   # -> _base_dose in mcg/min
                                                            #    (and u/min for vasopressin),
                                                            #    weight_kg ASOF-merged from vitals
    -> mcg/kg/min = _base_dose / weight_kg                  # catecholamines only
    -> NEE contribution = factor[drug] * dose               # vasopressin uses u/min directly
    -> per (encounter_block, admin_dttm) sum across drugs   # the NEE step function

The NEE timeline is a SUM of per-drug step functions: at any instant,
  ne_equiv(t) = Σ_drug factor_drug * rate_drug(t).
We materialize it as one row per (encounter_block, change-time) carrying the total
NEE in effect from that time until the next change — a step function the stability
screen samples via merge_asof.

Standard factors (Goradia 2021 / Kotani 2023), all config-driven so a site can swap
to another reference (e.g. Jain's exact table):
  norepinephrine 1, epinephrine 1, phenylephrine/10, dopamine/100, vasopressin x2.5
  (vasopressin in u/min, NOT weight-normalized); inotropes default to 0.

Weight-missing-while-a-pressor-is-running makes that drug's mcg/kg/min UN-assessable
(NaN), which propagates so the stability screen treats the hour as not-assessable
rather than silently 0. Aggregates only; this module does no stdout I/O.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

TRAILING_CAP_H = 24          # cap the final (open-ended) record of a drug run
STOP_ACTION = "stop"         # mar_action_category value that forces dose -> 0


def coerce_dttm(series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        s = s.dt.tz_convert(tz)
    return s


def vasopressor_categories(cfg: dict) -> set[str]:
    """Lowercased vasopressor med_category set = every drug with a NEE factor key."""
    factors = cfg["sbt_vasopressors"]["ne_equivalent_factors"]
    return {k.lower() for k in factors}


def _factors(cfg: dict) -> dict[str, float]:
    return {k.lower(): float(v) for k, v in cfg["sbt_vasopressors"]["ne_equivalent_factors"].items()}


def per_drug_ne_dose(inf: pd.DataFrame, vitals_df: pd.DataFrame | None,
                     cfg: dict, tz: str) -> pd.DataFrame:
    """Standardize each vasopressor row and attach its NEE contribution.

    Returns one row per charted vasopressor record:
      [encounter_block, med_category, admin_dttm, ne_contrib, assessable]
    where `ne_contrib` is the norepinephrine-equivalent (mcg/kg/min) this single
    infusion contributes at admin_dttm (0 if dose 0 or mar_action stop), and
    `assessable` is False when a running pressor lacks the weight needed to
    weight-normalize it.
    """
    from clifpy.utils.unit_converter import standardize_dose_to_base_units

    cats = vasopressor_categories(cfg)
    vaso_units = {c.lower() for c in cfg["sbt_vasopressors"].get("vasopressin_categories", ["vasopressin"])}
    factors = _factors(cfg)

    df = inf[inf["med_category"].astype("string").str.lower().isin(cats)].copy()
    cols = ["encounter_block", "med_category", "admin_dttm", "ne_contrib", "assessable"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["encounter_block"] = df["encounter_block"].astype(str)
    df["med_category"] = df["med_category"].astype("string").str.lower()
    df["admin_dttm"] = coerce_dttm(df["admin_dttm"], tz)
    df = df.dropna(subset=["admin_dttm"])
    if "mar_action_category" in df.columns:
        df["mar_action_category"] = df["mar_action_category"].astype("string").str.strip().str.lower()

    # clifpy standardizer wants hospitalization_id + admin_dttm + med_dose +
    # med_dose_unit; it ASOF-merges weight_kg from vitals (vital_category='weight_kg')
    # on hospitalization_id when weight_kg is absent. It returns DuckDB relations, so
    # we materialize to pandas with .df().
    try:
        base_rel, _counts = standardize_dose_to_base_units(df, vitals_df)
        base = base_rel.df() if hasattr(base_rel, "df") else (
            base_rel.to_df() if hasattr(base_rel, "to_df") else base_rel)
    except Exception:
        # Fall back: no standardization available -> treat doses as already mcg/kg/min
        # for catecholamines and u/min for vasopressin (best-effort, flagged below).
        base = df.copy()
        base["_base_dose"] = pd.to_numeric(base.get("med_dose"), errors="coerce")
        base["_base_unit"] = base.get("med_dose_unit")
        base["weight_kg"] = np.nan

    base["med_category"] = base["med_category"].astype("string").str.lower()
    base["encounter_block"] = base["encounter_block"].astype(str)
    base["admin_dttm"] = coerce_dttm(base["admin_dttm"], tz)
    if "mar_action_category" in base.columns:
        base["mar_action_category"] = base["mar_action_category"].astype("string").str.strip().str.lower()
    if "_base_dose" not in base.columns:
        base["_base_dose"] = np.nan
    if "weight_kg" not in base.columns:
        base["weight_kg"] = np.nan
    base["_base_dose"] = pd.to_numeric(base["_base_dose"], errors="coerce")
    base["weight_kg"] = pd.to_numeric(base.get("weight_kg"), errors="coerce")
    base_unit = base.get("_base_unit", pd.Series([""] * len(base))).astype("string").str.lower().fillna("")

    is_vaso_units = base["med_category"].isin(vaso_units)          # dosed in u/min
    is_stop = (base.get("mar_action_category") == STOP_ACTION) if "mar_action_category" in base else False
    is_stop = is_stop if hasattr(is_stop, "__len__") else pd.Series([bool(is_stop)] * len(base))

    # mcg/kg/min for catecholamines (base is mcg/min) ; u/min stays as-is for vasopressin.
    dose_per_kg = base["_base_dose"] / base["weight_kg"]
    dose = np.where(is_vaso_units, base["_base_dose"], dose_per_kg)

    factor = base["med_category"].map(factors).fillna(0.0).to_numpy()
    ne = factor * np.asarray(dose, dtype="float64")
    ne = np.where(is_stop.to_numpy(), 0.0, ne)
    ne = np.where((base["_base_dose"].fillna(0).to_numpy() <= 0), 0.0, ne)

    # Assessability: a RUNNING catecholamine (dose>0, not stop, nonzero factor)
    # whose weight is missing cannot be weight-normalized -> un-assessable.
    running = (base["_base_dose"].fillna(0).to_numpy() > 0) & (~is_stop.to_numpy()) & (factor > 0)
    weight_missing = base["weight_kg"].isna().to_numpy()
    assessable = ~(running & (~is_vaso_units.to_numpy()) & weight_missing)

    out = base[["encounter_block", "med_category", "admin_dttm"]].copy()
    out["ne_contrib"] = ne
    out["assessable"] = assessable
    return out[cols].reset_index(drop=True)


def ne_equiv_timeline(inf: pd.DataFrame, vitals_df: pd.DataFrame | None,
                      cfg: dict, tz: str) -> pd.DataFrame:
    """The total norepinephrine-equivalent step function per encounter_block.

    Returns one row per (encounter_block, change_dttm):
      [encounter_block, t, ne_equiv, assessable]
    `ne_equiv` is the SUM of all concurrently-running infusions' NEE contributions
    in effect from `t` until the block's next change row. `assessable` is False if
    any contributing pressor at that time is un-assessable (weight-missing).

    Each drug's charted value holds until that drug's next record (consecutive-row
    step function, trailing record capped). We carry each drug forward to the union
    of all change-times within the block, then sum across drugs.
    """
    cols = ["encounter_block", "t", "ne_equiv", "assessable"]
    pd_rows = per_drug_ne_dose(inf, vitals_df, cfg, tz)
    if pd_rows.empty:
        return pd.DataFrame(columns=cols)

    pd_rows = pd_rows.sort_values(["encounter_block", "med_category", "admin_dttm"])
    # Per-drug segment end = next record of the SAME drug, trailing capped.
    grp = pd_rows.groupby(["encounter_block", "med_category"], sort=False)
    pd_rows["seg_end"] = grp["admin_dttm"].shift(-1)
    cap = pd_rows["admin_dttm"] + timedelta(hours=TRAILING_CAP_H)
    pd_rows["seg_end"] = pd_rows["seg_end"].fillna(cap)
    pd_rows = pd_rows[pd_rows["seg_end"] > pd_rows["admin_dttm"]]

    out_frames = []
    for blk, g in pd_rows.groupby("encounter_block", sort=False):
        # Union of all change-times in this block (each drug's seg starts).
        times = np.sort(g["admin_dttm"].unique())
        tdf = pd.DataFrame({"t": pd.to_datetime(times)})
        total = np.zeros(len(tdf), dtype="float64")
        assess = np.ones(len(tdf), dtype=bool)
        for _drug, dg in g.groupby("med_category", sort=False):
            dg = dg.sort_values("admin_dttm")
            # Step value of THIS drug at each union time = most recent seg covering t.
            seg = dg[["admin_dttm", "seg_end", "ne_contrib", "assessable"]].reset_index(drop=True)
            m = pd.merge_asof(tdf, seg.rename(columns={"admin_dttm": "t"}),
                              on="t", direction="backward")
            covered = m["seg_end"].notna() & (m["t"] < m["seg_end"])
            contrib = np.where(covered, m["ne_contrib"].fillna(0).to_numpy(), 0.0)
            total = total + contrib
            # un-assessable if this drug is covering t and itself un-assessable
            assess = assess & ~(covered.to_numpy() & ~m["assessable"].fillna(True).to_numpy())
        sub = pd.DataFrame({"encounter_block": str(blk), "t": tdf["t"].to_numpy(),
                            "ne_equiv": total, "assessable": assess})
        out_frames.append(sub)

    res = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=cols)
    return res[cols].sort_values(["encounter_block", "t"]).reset_index(drop=True)
