"""Stage 04 — SBT QI metrics, site summary, and the bundle-scorecard tile feed.

Unit of analysis = ventilated-ICU patient-DAYS.
  denominator (eligible) = >=12h controlled accrued + >=2h stable window, non-trach
  numerator   (sbt)      = eligible days with a controlled->support transition >=2 min

Trach days are excluded from BOTH numerator and denominator. Days whose stability is
un-assessable are reported as a separate `not_assessable` bound (excluded from the
rate denominator).

Outputs:
    output/intermediate/metrics_patient_day_level.parquet  (keeps ids; not shared)
    output/intermediate/metrics_slices.parquet             (dashboard embeds this)
    output/final/metrics_site_summary.csv                  (federation-shareable)
    output/final/metrics_slices.csv                        (federation-shareable)
    output/final/tile_feed_sbt.json                        (contract v1, PHI-free)

No raw PHI to stdout; the tile feed is re-checked for PHI substrings at build time.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

log = logging.getLogger("sbt.metrics")

CANONICAL_UNITS = [
    "medical_icu", "mixed_cardiothoracic_icu", "surgical_icu",
    "mixed_neuro_icu", "general_icu", "burn_icu",
]
GRANULARITY_COL = {"month": "period_month", "week": "period_week"}
TILE_SCHEMA_VERSION = 1
DEFINITION_VERSION = "sbt-v1"
PHI_FORBIDDEN = ("hospitalization_id", "patient_id")


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sbt_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rate(num, den):
    return (num / den) if den else None


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


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
# Sliced metrics (unit x granularity x period)
# ---------------------------------------------------------------------------
def _slice_metrics(g: pd.DataFrame) -> dict:
    elig = g["eligible"]
    status = g["eligibility_status"]
    nontrach = status != "excluded_trach"
    rec = {
        "n_vent_days": int(len(g)),
        "n_nontrach": int(nontrach.sum()),
        "n_eligible": int(elig.sum()),
        "n_not_assessable": int((status == "not_assessable").sum()),
        "n_not_eligible": int((status == "not_eligible").sum()),
        "n_excluded_paralytic": int((status == "excluded_paralytic").sum()),
        # strict headline numerator (eligible & strict transition) — name kept for the
        # tile feed + integrity checks; do not rename.
        "n_sbt": int((elig & g["sbt_delivered"]).sum()),
        # three nested numerators × {all vent-ICU days, eligible days}
        "n_sbt_all": int(g["sbt_delivered"].sum()),
        "n_sbtany_all": int(g["sbt_delivered_any"].sum()),
        "n_sbtany_elig": int((elig & g["sbt_delivered_any"]).sum()),
        "n_spont_all": int(g["on_spontaneous"].sum()),
        "n_spont_elig": int((elig & g["on_spontaneous"]).sum()),
    }
    # patient-level (Both day + patient): nunique per slice — NOT additive across slices.
    # Two numerator flavors per numerator so num ⊆ den in each denominator mode:
    #   *_elig (ever event on an eligible day) ⊆ n_pts_elig ; *_all (ever event) ⊆ n_pts.
    if "patient_id" in g.columns:
        pid = g["patient_id"]
        rec["n_pts"] = int(pid.nunique())
        rec["n_pts_elig"] = int(pid[elig].nunique())
        rec["n_pts_strict_all"] = int(pid[g["sbt_delivered"]].nunique())
        rec["n_pts_strict_elig"] = int(pid[elig & g["sbt_delivered"]].nunique())
        rec["n_pts_any_all"] = int(pid[g["sbt_delivered_any"]].nunique())
        rec["n_pts_any_elig"] = int(pid[elig & g["sbt_delivered_any"]].nunique())
        rec["n_pts_spont_all"] = int(pid[g["on_spontaneous"]].nunique())
        rec["n_pts_spont_elig"] = int(pid[elig & g["on_spontaneous"]].nunique())
    else:
        for k in ("n_pts", "n_pts_elig", "n_pts_strict_all", "n_pts_strict_elig",
                  "n_pts_any_all", "n_pts_any_elig", "n_pts_spont_all", "n_pts_spont_elig"):
            rec[k] = 0
    return rec


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

    cols = ["unit", "granularity", "period", "n_vent_days", "n_nontrach",
            "n_eligible", "n_not_assessable", "n_not_eligible", "n_excluded_paralytic", "n_sbt",
            "n_sbt_all", "n_sbtany_all", "n_sbtany_elig", "n_spont_all", "n_spont_elig",
            "n_pts", "n_pts_elig", "n_pts_strict_all", "n_pts_strict_elig",
            "n_pts_any_all", "n_pts_any_elig", "n_pts_spont_all", "n_pts_spont_elig"]
    df = pd.DataFrame(rows)[cols]
    df["rate_sbt"] = df["n_sbt"] / df["n_eligible"].replace(0, np.nan)
    df["rate_eligible_of_nontrach"] = df["n_eligible"] / df["n_nontrach"].replace(0, np.nan)
    return df.sort_values(["unit", "granularity", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Site summary
# ---------------------------------------------------------------------------
def build_summary_rows(site: str, m: dict) -> pd.DataFrame:
    rows = [
        ("vent_icu_days", "Ventilated-ICU patient-days", m["n_vent_days"], m["n_vent_days"], 1.0,
         "IMV ∩ ICU, day-expanded"),
        ("nontrach_days", "Non-tracheostomized vent-ICU days", m["n_nontrach"], m["n_vent_days"],
         _rate(m["n_nontrach"], m["n_vent_days"]), "trach days excluded from num & den"),
        ("eligible_days", "Eligible SBT-opportunity days (denominator)", m["n_eligible"], m["n_nontrach"],
         _rate(m["n_eligible"], m["n_nontrach"]), ">=12h controlled + >=2h stable window"),
        ("not_assessable_days", "Not-assessable stability days (bound)", m["n_not_assessable"], m["n_nontrach"],
         _rate(m["n_not_assessable"], m["n_nontrach"]), "no scaffold hour with all 4 stability signals"),
        ("excluded_paralytic_days", "Continuous-paralytic days (excluded)", m["n_excluded_paralytic"],
         m["n_nontrach"], _rate(m["n_excluded_paralytic"], m["n_nontrach"]),
         "continuous NMBA infusion; no respiratory drive -> justified exclusion"),
        ("sbt_delivered", "SBT delivered / eligible (HEADLINE)", m["n_sbt"], m["n_eligible"],
         _rate(m["n_sbt"], m["n_eligible"]), "controlled->support transition >= min duration"),
        # --- liberal views (leadership ask) — all use the all-vent-ICU-days denominator ---
        ("sbt_strict_all", "Strict SBT / all vent-ICU days", m["n_sbt_all"], m["n_vent_days"],
         _rate(m["n_sbt_all"], m["n_vent_days"]), "transition >= min duration; liberal denominator"),
        ("sbt_any_all", "SBT any-duration / all vent-ICU days", m["n_sbtany_all"], m["n_vent_days"],
         _rate(m["n_sbtany_all"], m["n_vent_days"]), "controlled->support transition, any duration"),
        ("sbt_any_elig", "SBT any-duration / eligible", m["n_sbtany_elig"], m["n_eligible"],
         _rate(m["n_sbtany_elig"], m["n_eligible"]), "any-duration transition on eligible days"),
        ("on_spont_all", "On a spontaneous mode / all vent-ICU days", m["n_spont_all"], m["n_vent_days"],
         _rate(m["n_spont_all"], m["n_vent_days"]), "any support-mode time that day; no transition, no PEEP gate"),
        ("on_spont_elig", "On a spontaneous mode / eligible", m["n_spont_elig"], m["n_eligible"],
         _rate(m["n_spont_elig"], m["n_eligible"]), "any support-mode time on eligible days"),
        # --- patient-level secondary framing ---
        ("patients_cohort", "Ventilated-ICU patients", m["n_pts"], m["n_pts"], 1.0, ""),
        ("patients_eligible", "Patients with >=1 eligible day", m["n_pts_elig"], m["n_pts"],
         _rate(m["n_pts_elig"], m["n_pts"]), ""),
        ("patients_ever_sbt", "Patients ever strict-SBT / eligible patients", m["n_pts_sbt"], m["n_pts_elig"],
         _rate(m["n_pts_sbt"], m["n_pts_elig"]), "patient-level secondary framing"),
        ("patients_ever_sbt_any", "Patients ever SBT any-duration / vent-ICU patients",
         m["n_pts_sbt_any"], m["n_pts"], _rate(m["n_pts_sbt_any"], m["n_pts"]), "patient-level, liberal"),
        ("patients_ever_spont", "Patients ever on a spontaneous mode / vent-ICU patients",
         m["n_pts_spont"], m["n_pts"], _rate(m["n_pts_spont"], m["n_pts"]), "patient-level, liberal"),
    ]
    df = pd.DataFrame(rows, columns=["metric", "label", "numerator", "denominator", "rate", "note"])
    df.insert(0, "site", site)
    df["generated"] = m["generated"]
    return df


# ---------------------------------------------------------------------------
# Tile feed (contract v1)
# ---------------------------------------------------------------------------
def build_tile_feed(cfg: dict, m: dict, slices: pd.DataFrame, diag: dict) -> dict:
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

    cov_pct = 100 * _rate(m["n_eligible"], m["n_nontrach"]) if m["n_nontrach"] else 0
    pct_native = diag.get("pct_native_support_rows")
    native_clause = (f"~{pct_native:.0f}% of support readings are native-resolution; "
                     if pct_native is not None else "")
    return {
        "schema_version": TILE_SCHEMA_VERSION,
        "metric_id": "sbt",
        "title": "Spontaneous Breathing Trial",
        "subtitle": "Controlled→support transition on eligible vent-days (Jain et al.)",
        "icon": "sbt",
        "detail_href": "sbt_dashboard.html",
        "goal": None,
        "generated": m["generated"],
        "note": ("• Eligible = ≥12 h controlled + ≥2 h stable window, non-trach, non-paralytic "
                 f"({cov_pct:.0f}% of non-trach vent-ICU days)"
                 "• Donut = strict SBT (controlled→support transition ≥2 min); bars = any-length SBT "
                 "and on a spontaneous mode (of eligible days / of all vent-days)"
                 "• Transition rates are a lower bound where charting is hourly; CPAP read from PEEP"),
        "grain": {"units": units, "periods": ["all", "month", "week"]},
        "headline": {
            "label": "SBT delivered",
            "den_label": "of eligible vent-days",
            "n_unit": "patient-days",
            "cells": cells("n_sbt", "n_eligible", with_n=True),
        },
        # Mini-bars: the other SBT views vs the strict-delivery donut (react to the selector).
        "segments": [
            {"key": "sbt_any", "label": "SBT, any length",
             "cells": cells("n_sbtany_elig", "n_eligible")},
            {"key": "spont_elig", "label": "On spont · elig",
             "cells": cells("n_spont_elig", "n_eligible")},
            {"key": "spont_all", "label": "On spont · all-days",
             "cells": cells("n_spont_all", "n_vent_days")},
        ],
        "provenance": {
            "site_id": cfg.get("site", "unknown"),
            "code_version": _git_sha(),
            "clif_version": cfg.get("primary_dataset", {}).get("clif_version"),
            "definition_version": DEFINITION_VERSION,
            "generated": m["generated"],
        },
    }


def _assert_phi_free(feed: dict) -> None:
    blob = json.dumps(feed)
    hits = [s for s in PHI_FORBIDDEN if s in blob]
    if hits:
        raise RuntimeError(f"tile feed contains forbidden PHI substring(s): {hits}")


def _assert_slice_integrity(slices: pd.DataFrame, m: dict) -> None:
    a = slices[(slices["unit"] == "__ALL__") & (slices["granularity"] == "all")].iloc[0]
    for col, key in [("n_vent_days", "n_vent_days"), ("n_eligible", "n_eligible"),
                     ("n_sbt", "n_sbt"), ("n_not_assessable", "n_not_assessable"),
                     ("n_sbt_all", "n_sbt_all"), ("n_sbtany_all", "n_sbtany_all"),
                     ("n_spont_all", "n_spont_all")]:
        if int(a[col]) != int(m[key]):
            raise RuntimeError(f"slice __ALL__/all {col}={a[col]} != headline {m[key]}")
    # additive day-count columns must sum to the total across units and across periods
    additive = ["n_vent_days", "n_nontrach", "n_eligible", "n_sbt", "n_sbt_all",
                "n_sbtany_all", "n_spont_all"]
    units_all = slices[(slices["granularity"] == "all") & (slices["unit"] != "__ALL__")]
    for col in additive:
        if int(units_all[col].sum()) != int(m[col]):
            raise RuntimeError(f"per-unit {col} does not sum to total")
    for gran in GRANULARITY_COL:
        s = slices[(slices["unit"] == "__ALL__") & (slices["granularity"] == gran)]
        for col in additive:
            if int(s[col].sum()) != int(m[col]):
                raise RuntimeError(f"per-period ({gran}) {col} does not sum to total")
    # CONSORT funnel monotonicity (totals)
    if not (m["n_vent_days"] >= m["n_nontrach"] >= m["n_eligible"] >= m["n_sbt"]):
        raise RuntimeError("CONSORT funnel not monotone: "
                           f"vent={m['n_vent_days']} nontrach={m['n_nontrach']} "
                           f"elig={m['n_eligible']} sbt={m['n_sbt']}")
    # numerator nesting (totals): strict ⊆ any-duration ⊆ on-spontaneous; eligible⊆all
    if not (m["n_spont_all"] >= m["n_sbtany_all"] >= m["n_sbt_all"] >= m["n_sbt"]):
        raise RuntimeError("numerator nesting violated: "
                           f"spont={m['n_spont_all']} any={m['n_sbtany_all']} "
                           f"strict_all={m['n_sbt_all']} strict_elig={m['n_sbt']}")


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

    inter = cohort_mod.INTERMEDIATE_DIR
    final = cohort_mod.FINAL_DIR
    obs = pd.read_parquet(inter / "sbt_observation.parquet")
    obs = attach_periods(obs)
    diag = {}
    diag_path = inter / "sbt_diag.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())

    n_vent_days = int(len(obs))
    n_nontrach = int((obs["eligibility_status"] != "excluded_trach").sum())
    n_eligible = int(obs["eligible"].sum())
    n_not_assessable = int((obs["eligibility_status"] == "not_assessable").sum())
    n_not_eligible = int((obs["eligibility_status"] == "not_eligible").sum())
    n_excluded_paralytic = int((obs["eligibility_status"] == "excluded_paralytic").sum())
    n_sbt = int((obs["eligible"] & obs["sbt_delivered"]).sum())
    # liberal numerators × {all vent-ICU days, eligible days}
    n_sbt_all = int(obs["sbt_delivered"].sum())
    n_sbtany_all = int(obs["sbt_delivered_any"].sum())
    n_sbtany_elig = int((obs["eligible"] & obs["sbt_delivered_any"]).sum())
    n_spont_all = int(obs["on_spontaneous"].sum())
    n_spont_elig = int((obs["eligible"] & obs["on_spontaneous"]).sum())
    has_pid = "patient_id" in obs.columns
    elig_df = obs[obs["eligible"]]
    n_pts = int(obs["patient_id"].nunique()) if has_pid else 0
    n_pts_elig = int(elig_df["patient_id"].nunique()) if has_pid else 0
    n_pts_sbt = int(elig_df.loc[elig_df["sbt_delivered"], "patient_id"].nunique()) if has_pid else 0
    n_pts_sbt_any = int(obs.loc[obs["sbt_delivered_any"], "patient_id"].nunique()) if has_pid else 0
    n_pts_spont = int(obs.loc[obs["on_spontaneous"], "patient_id"].nunique()) if has_pid else 0

    generated = _dt.datetime.now().isoformat(timespec="minutes")
    m = {"n_vent_days": n_vent_days, "n_nontrach": n_nontrach, "n_eligible": n_eligible,
         "n_not_assessable": n_not_assessable, "n_not_eligible": n_not_eligible,
         "n_excluded_paralytic": n_excluded_paralytic, "n_sbt": n_sbt,
         "n_sbt_all": n_sbt_all, "n_sbtany_all": n_sbtany_all, "n_sbtany_elig": n_sbtany_elig,
         "n_spont_all": n_spont_all, "n_spont_elig": n_spont_elig,
         "n_pts": n_pts, "n_pts_elig": n_pts_elig, "n_pts_sbt": n_pts_sbt,
         "n_pts_sbt_any": n_pts_sbt_any, "n_pts_spont": n_pts_spont, "generated": generated}

    slices = build_slice_cells(obs)
    _assert_slice_integrity(slices, m)

    obs.to_parquet(inter / "metrics_patient_day_level.parquet", index=False)
    slices.to_parquet(inter / "metrics_slices.parquet", index=False)

    summary = build_summary_rows(site, m)
    summary.to_csv(final / "metrics_site_summary.csv", index=False)

    slices_out = slices.copy(); slices_out.insert(0, "site", site); slices_out["generated"] = generated
    slices_out.to_csv(final / "metrics_slices.csv", index=False)

    feed = build_tile_feed(cfg, m, slices, diag)
    _assert_phi_free(feed)
    with open(final / "tile_feed_sbt.json", "w") as f:
        json.dump(feed, f, indent=2, ensure_ascii=False)

    log.info("ventilated-ICU patient-days:   %6d", n_vent_days)
    log.info("non-trach vent-ICU days:       %6d (%.1f%%)", n_nontrach, 100 * _rate(n_nontrach, n_vent_days))
    log.info("continuous-paralytic days:     %6d (excluded from eligible denominator)", n_excluded_paralytic)
    log.info("eligible SBT-opportunity days: %6d (%.1f%% of non-trach)",
             n_eligible, 100 * _rate(n_eligible, n_nontrach) if n_nontrach else 0)
    log.info("not-assessable stability days: %6d", n_not_assessable)
    log.info("SBT delivered / eligible:      %6d (%.1f%%)  [HEADLINE]",
             n_sbt, 100 * _rate(n_sbt, n_eligible) if n_eligible else 0)
    log.info("patients ever SBT / eligible:  %6d / %d (%.1f%%)", n_pts_sbt, n_pts_elig,
             100 * _rate(n_pts_sbt, n_pts_elig) if n_pts_elig else 0)
    log.info("--- liberal views (denominator = all %d vent-ICU days) ---", n_vent_days)
    log.info("  strict SBT:        %6d (%.1f%%)", n_sbt_all, 100 * _rate(n_sbt_all, n_vent_days))
    log.info("  SBT any duration:  %6d (%.1f%%)", n_sbtany_all, 100 * _rate(n_sbtany_all, n_vent_days))
    log.info("  on spontaneous:    %6d (%.1f%%)", n_spont_all, 100 * _rate(n_spont_all, n_vent_days))
    log.info("  patients ever on spontaneous: %d / %d (%.1f%%)", n_pts_spont, n_pts,
             100 * _rate(n_pts_spont, n_pts) if n_pts else 0)
    log.info("wrote: metrics_site_summary.csv, metrics_slices.csv, tile_feed_sbt.json "
             "(PHI-free; grain units=%d periods=%s)",
             len(feed["grain"]["units"]), ",".join(feed["grain"]["periods"]))


if __name__ == "__main__":
    main()
