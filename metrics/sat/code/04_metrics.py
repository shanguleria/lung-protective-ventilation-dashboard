"""Stage 04 — SAT QI metrics, site summary, and the bundle-scorecard tile feed.

Unit of analysis = ventilated-ICU patient-DAYS.
  denominator (eligible) = vent-ICU days on >=1 SAT-relevant infusion, non-paralytic
  numerator   (SAT)      = eligible days with an all-infusions-held >= threshold

Denominator mode = `documented_plus_bound` (decided from the probe + user): holds
are directly observable here (explicit dose==0 rows + mar_action stop/start), so
the HEADLINE is a real documented rate. Two segments give bounds/sensitivity:
  - "of all vent-ICU days"  = SAT days / ALL vent-ICU days   (denominator-broadening)
  - "resumed-only"          = resumed-SAT days / eligible    (numerator sensitivity:
                              excludes end-of-day discontinuations that never restart)

Also emits the Kress et al. 2000 dose-resumption summary (off the tile).

Outputs:
    output/intermediate/metrics_patient_day_level.parquet  (keeps ids; not shared)
    output/intermediate/metrics_slices.parquet             (dashboard embeds this)
    output/final/metrics_site_summary.csv                  (federation-shareable)
    output/final/metrics_slices.csv                        (federation-shareable)
    output/final/kress_summary.csv                         (federation-shareable)
    output/final/tile_feed_sat.json                        (contract v1, PHI-free)

No raw PHI to stdout; the tile feed is re-checked for PHI substrings at build time.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

log = logging.getLogger("sat.metrics")

# Canonical ICU unit slugs shared with the tile contract (lpv §3). Values are the
# adt location_type at this site (lowercased); "unknown"/others fold into __ALL__.
CANONICAL_UNITS = [
    "medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
    "mixed_neuro_icu", "general_icu", "burn_icu",
]
GRANULARITY_COL = {"month": "period_month", "week": "period_week"}
TILE_SCHEMA_VERSION = 1
PHI_FORBIDDEN = ("hospitalization_id", "patient_id")


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sat_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rate(num, den):
    return (num / den) if den else None


# ---------------------------------------------------------------------------
# Period attribution (unit already attached per day in 01)
# ---------------------------------------------------------------------------
def attach_periods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["icu_day"], errors="coerce")
    iso = d.dt.isocalendar()
    df["period_month"] = d.dt.strftime("%Y-%m")
    df["period_week"] = (iso["year"].astype("Int64").astype(str) + "-W"
                         + iso["week"].astype("Int64").astype(str).str.zfill(2))
    df["unit"] = df["unit"].astype("string").fillna("unknown").replace("", "unknown")
    return df


# ---------------------------------------------------------------------------
# Sliced metrics (unit x granularity x period), day-based & cleanly partitioning
# ---------------------------------------------------------------------------
def _slice_metrics(g: pd.DataFrame) -> dict:
    elig = g["eligible"]
    sat = elig & g["sat_performed"]
    n_sat = int(sat.sum())
    n_resumed = int((sat & g["sat_resumed"]).sum())
    return {
        "n_vent_days": int(len(g)),
        "n_eligible": int(elig.sum()),
        "n_sat": n_sat,
        "n_resumed": n_resumed,
        # SAT-outcome breakdown (of SATs): resumed / not-resumed-same-day / extubated-by-EOD
        "n_notresumed": n_sat - n_resumed,
        "n_extubated": int((sat & g["extubated_eod"]).sum()) if "extubated_eod" in g.columns else 0,
    }


def build_slice_cells(pl: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def emit(unit, gran, period, g):
        rec = _slice_metrics(g)
        rec.update(unit=unit, granularity=gran, period=period)
        rows.append(rec)

    for unit, gu in [("__ALL__", pl)] + list(pl.groupby("unit", observed=True)):
        emit(unit, "all", "all", gu)
        for gran, col in GRANULARITY_COL.items():
            for period, g in gu.groupby(col, observed=True):
                emit(unit, gran, str(period), g)

    cols = ["unit", "granularity", "period", "n_vent_days", "n_eligible", "n_sat", "n_resumed",
            "n_notresumed", "n_extubated"]
    df = pd.DataFrame(rows)[cols]
    df["rate_sat"] = df["n_sat"] / df["n_eligible"].replace(0, np.nan)
    df["rate_sat_of_vent"] = df["n_sat"] / df["n_vent_days"].replace(0, np.nan)
    df["rate_resumed"] = df["n_resumed"] / df["n_eligible"].replace(0, np.nan)
    df["rate_extubated_of_sat"] = df["n_extubated"] / df["n_sat"].replace(0, np.nan)
    return df.sort_values(["unit", "granularity", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Site summary + Kress summary
# ---------------------------------------------------------------------------
def build_summary_rows(site: str, cfg: dict, m: dict) -> pd.DataFrame:
    rows = [
        ("vent_icu_days", "Ventilated-ICU patient-days", m["n_vent_days"], m["n_vent_days"], 1.0,
         "IMV ∩ ICU, day-expanded"),
        ("eligible_days", "Eligible SAT-opportunity days (denominator)", m["n_eligible"], m["n_vent_days"],
         _rate(m["n_eligible"], m["n_vent_days"]), "on >=1 SAT-relevant infusion, non-paralytic"),
        ("sat_performed", "SAT performed / eligible (HEADLINE)", m["n_sat"], m["n_eligible"],
         _rate(m["n_sat"], m["n_eligible"]), f"all SAT-relevant infusions held >= {m['hold_min']:.0f} min"),
        ("sat_of_vent", "SAT performed / ALL vent-ICU days (bound)", m["n_sat"], m["n_vent_days"],
         _rate(m["n_sat"], m["n_vent_days"]), "denominator-broadening"),
        ("sat_resumed", "Resumed-only SAT / eligible (sensitivity)", m["n_resumed"], m["n_eligible"],
         _rate(m["n_resumed"], m["n_eligible"]), "excludes end-of-day discontinuations"),
        # SAT outcomes (of SATs delivered)
        ("sat_outcome_resumed", "SAT → resumed sedation / SAT", m["n_resumed"], m["n_sat"],
         _rate(m["n_resumed"], m["n_sat"]), "sedation restarted after the hold"),
        ("sat_outcome_notresumed", "SAT → not resumed that day / SAT", m["n_sat"] - m["n_resumed"], m["n_sat"],
         _rate(m["n_sat"] - m["n_resumed"], m["n_sat"]), "stayed off continuous sedation"),
        ("sat_outcome_extubated", "SAT → off IMV (extubated) by end of day / SAT", m["n_extubated"], m["n_sat"],
         _rate(m["n_extubated"], m["n_sat"]), "off invasive vent & alive at next midnight; pure-IMV timeline"),
        ("patients_eligible", "Patients with >=1 eligible day", m["n_pts_elig"], m["n_pts_elig"], 1.0, ""),
        ("patients_ever_sat", "Patients ever SAT / eligible patients", m["n_pts_sat"], m["n_pts_elig"],
         _rate(m["n_pts_sat"], m["n_pts_elig"]), "patient-level secondary framing"),
    ]
    df = pd.DataFrame(rows, columns=["metric", "label", "numerator", "denominator", "rate", "note"])
    df.insert(0, "site", site)
    df["hold_min_minutes"] = m["hold_min"]
    df["generated"] = m["generated"]
    return df


def build_kress_summary(site: str, kress: pd.DataFrame, half: float, generated: str) -> pd.DataFrame:
    def block(label, sub):
        r = sub["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(r)
        return {"site": site, "drug": label, "n_resumptions": int(n),
                "median_ratio": float(r.median()) if n else None,
                "q1_ratio": float(r.quantile(.25)) if n else None,
                "q3_ratio": float(r.quantile(.75)) if n else None,
                "pct_at_or_below_half": float(100 * (r <= half).sum() / n) if n else None,
                "half_dose_threshold": half, "generated": generated}
    rows = [block("__ALL__", kress)]
    if not kress.empty:
        for drug, sub in kress.groupby("med_category"):
            rows.append(block(drug, sub))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tile feed (contract v1)
# ---------------------------------------------------------------------------
def build_tile_feed(cfg: dict, m: dict, slices: pd.DataFrame) -> dict:
    units = ["__ALL__"] + [u for u in CANONICAL_UNITS if u in set(slices["unit"])]
    months = sorted(slices.loc[slices["granularity"] == "month", "period"].unique())
    weeks = sorted(slices.loc[slices["granularity"] == "week", "period"].unique())
    by_key = {(r.unit, "all" if r.granularity == "all" else r.period): r
              for r in slices.itertuples(index=False)
              if r.granularity in ("all", "month", "week")}

    def cells(num_col, den_col, with_n=False):
        out = {}
        for u in units:
            pc = {}
            for p in ["all"] + months + weeks:
                r = by_key.get((u, p))
                if r is None:
                    continue
                den = int(getattr(r, den_col))
                cell = {"num": int(getattr(r, num_col)), "den": den}
                if with_n:
                    cell["n"] = den
                pc[p] = cell
            if pc:
                out[u] = pc
        return out

    cov_pct = 100 * _rate(m["n_eligible"], m["n_vent_days"])
    return {
        "schema_version": TILE_SCHEMA_VERSION,
        "metric_id": "sat",
        "title": "Spontaneous Awakening Trial",
        "subtitle": f"Sedation held ≥{m['hold_min']:.0f} min on eligible vent-sedation days",
        "icon": "sat",
        "detail_href": "sat_dashboard.html",
        "goal": None,
        "generated": m["generated"],
        "note": ("• Eligible = vent-ICU days on ≥1 sedative infusion (propofol/benzo/opioid), "
                 f"non-paralytic ({cov_pct:.0f}% of vent-ICU days); dexmedetomidine may continue"
                 "• SAT = all SAT-relevant infusions held to rate 0 (charted dose-0 / mar stop-start)"
                 "• Bars (of SATs): resumed sedation · not resumed that day · off IMV (extubated) by end "
                 "of the SAT day — alive, pure-IMV timeline so ICU-transfer-while-intubated does not count"
                 "• Crude screen — CLIF can't encode all SAT safety exclusions (seizure, withdrawal, "
                 "ischemia, ↑ICP)"),
        "grain": {"units": units, "periods": ["all", "month", "week"]},
        "headline": {
            "label": "SAT performed",
            "den_label": "of eligible vent-sedation days",
            "n_unit": "patient-days",
            "cells": cells("n_sat", "n_eligible", with_n=True),
        },
        # SAT-outcome mini-bars (added 2026-06-05): each is a % of SATs delivered. These are
        # descriptive shares, not a quality scale, so they pin maroon (the combiner's default
        # green/amber/red segColor — higher=better — would mislead here).
        "segments": [
            {"key": "resumed", "label": "Resumed sedation", "color": "#8a1f2b",
             "cells": cells("n_resumed", "n_sat")},
            {"key": "notresumed", "label": "Not resumed (day)", "color": "#8a1f2b",
             "cells": cells("n_notresumed", "n_sat")},
            {"key": "extubated", "label": "Extubated same day", "color": "#8a1f2b",
             "cells": cells("n_extubated", "n_sat")},
        ],
    }


def _assert_phi_free(feed: dict) -> None:
    blob = json.dumps(feed)
    hits = [s for s in PHI_FORBIDDEN if s in blob]
    if hits:
        raise RuntimeError(f"tile feed contains forbidden PHI substring(s): {hits}")


def _assert_slice_integrity(slices: pd.DataFrame, m: dict) -> None:
    a = slices[(slices["unit"] == "__ALL__") & (slices["granularity"] == "all")].iloc[0]
    for col, key in [("n_vent_days", "n_vent_days"), ("n_eligible", "n_eligible"),
                     ("n_sat", "n_sat"), ("n_resumed", "n_resumed")]:
        if int(a[col]) != int(m[key]):
            raise RuntimeError(f"slice __ALL__/all {col}={a[col]} != headline {m[key]}")
    units_all = slices[(slices["granularity"] == "all") & (slices["unit"] != "__ALL__")]
    if int(units_all["n_eligible"].sum()) != m["n_eligible"]:
        raise RuntimeError("per-unit n_eligible does not sum to total")
    for gran in GRANULARITY_COL:
        s = slices[(slices["unit"] == "__ALL__") & (slices["granularity"] == gran)]
        if int(s["n_eligible"].sum()) != m["n_eligible"]:
            raise RuntimeError(f"per-period ({gran}) n_eligible does not sum to total")


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "04_metrics.log", mode="w")],
    )
    cfg = cohort_mod.load_config()
    site = cfg.get("site", "unknown")
    hold_min = float(cfg["sat_observation"].get("hold_min_minutes", 30))
    half = float(cfg["sat_observation"].get("kress_half_dose_threshold", 0.5))

    inter = cohort_mod.INTERMEDIATE_DIR
    final = cohort_mod.FINAL_DIR
    obs = pd.read_parquet(inter / "sat_observation.parquet")
    obs = attach_periods(obs)
    kress = pd.read_parquet(inter / "kress_resumption.parquet")

    n_vent_days = int(len(obs))
    n_eligible = int(obs["eligible"].sum())
    n_sat = int((obs["eligible"] & obs["sat_performed"]).sum())
    n_resumed = int((obs["eligible"] & obs["sat_resumed"]).sum())
    n_extubated = int((obs["eligible"] & obs["sat_performed"] & obs["extubated_eod"]).sum()) \
        if "extubated_eod" in obs.columns else 0
    has_pid = "patient_id" in obs.columns
    elig_df = obs[obs["eligible"]]
    n_pts_elig = int(elig_df["patient_id"].nunique()) if has_pid else 0
    n_pts_sat = int(elig_df.loc[elig_df["sat_performed"], "patient_id"].nunique()) if has_pid else 0

    generated = _dt.datetime.now().isoformat(timespec="minutes")
    m = {"n_vent_days": n_vent_days, "n_eligible": n_eligible, "n_sat": n_sat,
         "n_resumed": n_resumed, "n_extubated": n_extubated, "n_pts_elig": n_pts_elig,
         "n_pts_sat": n_pts_sat, "hold_min": hold_min, "generated": generated}

    slices = build_slice_cells(obs)
    _assert_slice_integrity(slices, m)

    # ---- write outputs ----
    obs.to_parquet(inter / "metrics_patient_day_level.parquet", index=False)
    slices.to_parquet(inter / "metrics_slices.parquet", index=False)

    summary = build_summary_rows(site, cfg, m)
    summary.to_csv(final / "metrics_site_summary.csv", index=False)

    slices_out = slices.copy(); slices_out.insert(0, "site", site); slices_out["generated"] = generated
    slices_out.to_csv(final / "metrics_slices.csv", index=False)

    kress_sum = build_kress_summary(site, kress, half, generated)
    kress_sum.to_csv(final / "kress_summary.csv", index=False)

    feed = build_tile_feed(cfg, m, slices)
    _assert_phi_free(feed)
    with open(final / "tile_feed_sat.json", "w") as f:
        json.dump(feed, f, indent=2, ensure_ascii=False)

    # ---- log ----
    log.info("ventilated-ICU patient-days:   %6d", n_vent_days)
    log.info("eligible SAT-opportunity days: %6d (%.1f%% of vent-ICU days)",
             n_eligible, 100 * _rate(n_eligible, n_vent_days))
    log.info("SAT performed / eligible:      %6d (%.1f%%)  [HEADLINE]", n_sat, 100 * _rate(n_sat, n_eligible))
    log.info("SAT / all vent-ICU days:       %6d (%.1f%%)  [bound]", n_sat, 100 * _rate(n_sat, n_vent_days))
    log.info("resumed-only SAT / eligible:   %6d (%.1f%%)  [sensitivity]", n_resumed, 100 * _rate(n_resumed, n_eligible))
    log.info("patients ever SAT / eligible:  %6d / %d (%.1f%%)", n_pts_sat, n_pts_elig,
             100 * _rate(n_pts_sat, n_pts_elig) if n_pts_elig else 0)
    kr = kress_sum[kress_sum["drug"] == "__ALL__"].iloc[0]
    log.info("Kress resumption (all drugs):  n=%d median=%.2f <=half=%.0f%%",
             kr["n_resumptions"], kr["median_ratio"] or 0, kr["pct_at_or_below_half"] or 0)
    log.info("wrote: metrics_site_summary.csv, metrics_slices.csv, kress_summary.csv, tile_feed_sat.json "
             "(PHI-free; grain units=%d periods=%s)",
             len(feed["grain"]["units"]), ",".join(feed["grain"]["periods"]))


if __name__ == "__main__":
    main()
