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
# Exclusion-toggle model (plan 04): stable bit order for the per-day criterion mask.
# The dashboard JS engine references bits by this exact name->index order. 6 denominator
# bits + 8 numerator-subset bits (on_spontaneous = the empty subset).
MASK_BITS = ["db_trach", "db_paralytic", "db_accrued12",
             "db_stable_oxy", "db_stable_vaso", "db_stable_both",
             "on_spontaneous", "nb_t", "nb_d", "nb_p", "nb_td", "nb_tp", "nb_dp", "nb_tdp"]
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
    if "unit_name" in df.columns:
        df["unit_name"] = df["unit_name"].astype("string").fillna("unknown").replace("", "unknown")
    else:
        df["unit_name"] = "unknown"
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
        # eligible AND a transition candidate (drop eligible days parked on a spontaneous mode with
        # no transition) — the scorecard-tile denominator under the "require transition" den+num rule.
        "n_eligible_txcand": int((elig & ~(g["on_spontaneous"] & ~g["nb_t"])).sum()),
        "n_not_assessable": int((status == "not_assessable").sum()),
        "n_not_eligible": int((status == "not_eligible").sum()),
        # not_eligible subdivided by driver (partitions n_not_eligible; vasopressor split out)
        "n_notelig_lt12h": int((g["notelig_reason"] == "lt12h_controlled").sum()),
        "n_notelig_vaso": int((g["notelig_reason"] == "failed_vasopressor").sum()),
        "n_notelig_oxypeep": int((g["notelig_reason"] == "failed_oxy_peep").sum()),
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

    cols = ["unit", "dim", "parent", "granularity", "period", "n_vent_days", "n_nontrach",
            "n_eligible", "n_eligible_txcand", "n_not_assessable", "n_not_eligible",
            "n_notelig_lt12h", "n_notelig_vaso", "n_notelig_oxypeep", "n_excluded_paralytic", "n_sbt",
            "n_sbt_all", "n_sbtany_all", "n_sbtany_elig", "n_spont_all", "n_spont_elig",
            "n_pts", "n_pts_elig", "n_pts_strict_all", "n_pts_strict_elig",
            "n_pts_any_all", "n_pts_any_elig", "n_pts_spont_all", "n_pts_spont_elig"]
    df = pd.DataFrame(rows)[cols]
    # not_eligible sub-reasons partition n_not_eligible in every slice
    _part = df["n_notelig_lt12h"] + df["n_notelig_vaso"] + df["n_notelig_oxypeep"]
    if not (_part == df["n_not_eligible"]).all():
        raise RuntimeError("not_eligible sub-reason partition mismatch in a slice")
    df["rate_sbt"] = df["n_sbt"] / df["n_eligible"].replace(0, np.nan)
    df["rate_eligible_of_nontrach"] = df["n_eligible"] / df["n_nontrach"].replace(0, np.nan)
    return df.sort_values(["dim", "unit", "granularity", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exclusion-toggle criterion-mask histogram (plan 04) — the dashboard computes
# num/den LIVE in JS from this; PHI-free (counts of days per 14-bit mask per slice).
# ---------------------------------------------------------------------------
def build_mask_histogram(obs: pd.DataFrame) -> pd.DataFrame:
    df = obs.copy()
    m = np.zeros(len(df), dtype=np.int64)
    for i, col in enumerate(MASK_BITS):
        m |= (df[col].astype(bool).to_numpy().astype(np.int64) << i)
    df = df.assign(mask=m)
    rows = []

    def emit(unit, gran, period, g):
        for mask_val, cnt in g["mask"].value_counts().items():
            rows.append({"unit": unit, "granularity": gran, "period": period,
                         "mask": int(mask_val), "count": int(cnt)})

    for unit, gu in [("__ALL__", df)] + list(df.groupby("unit", observed=True)):
        emit(unit, "all", "all", gu)
        for gran, col in GRANULARITY_COL.items():
            for period, g in gu.groupby(col, observed=True):
                emit(unit, gran, str(period), g)
    # Specific-unit (location_name) keys too, so the dashboard's live toggle engine can split by
    # specific unit. Name keys are distinct from type slugs; the reconciliation merges per (unit,…).
    if "unit_name" in df.columns:
        for uname, gu in df.groupby("unit_name", observed=True):
            emit(uname, "all", "all", gu)
            for gran, col in GRANULARITY_COL.items():
                for period, g in gu.groupby(col, observed=True):
                    emit(uname, gran, str(period), g)
    return pd.DataFrame(rows, columns=["unit", "granularity", "period", "mask", "count"])


# ---------------------------------------------------------------------------
# Site summary
# ---------------------------------------------------------------------------
def build_summary_rows(site: str, m: dict) -> pd.DataFrame:
    rows = [
        ("vent_icu_days", "Ventilated-ICU patient-days", m["n_vent_days"], m["n_vent_days"], 1.0,
         "IMV ∩ ICU, day-expanded"),
        ("nontrach_days", "Non-tracheostomized vent-ICU days", m["n_nontrach"], m["n_vent_days"],
         _rate(m["n_nontrach"], m["n_vent_days"]), "trach days excluded from num & den"),
        ("eligible_days", "Eligible SBT-opportunity days", m["n_eligible"], m["n_nontrach"],
         _rate(m["n_eligible"], m["n_nontrach"]), ">=12h controlled + >=2h stable window"),
        ("eligible_txcand_days", "Eligible transition-candidate days (tile denominator)",
         m["n_eligible_txcand"], m["n_nontrach"], _rate(m["n_eligible_txcand"], m["n_nontrach"]),
         "eligible minus days parked on a spontaneous mode with no transition"),
        ("not_assessable_days", "Not-assessable stability days (bound)", m["n_not_assessable"], m["n_nontrach"],
         _rate(m["n_not_assessable"], m["n_nontrach"]), "no scaffold hour with all 4 stability signals"),
        # not_eligible subdivided by driver (federation: lets sites harmonize the denominator)
        ("notelig_lt12h_days", "Not eligible — <12h controlled accrued", m["n_notelig_lt12h"], m["n_nontrach"],
         _rate(m["n_notelig_lt12h"], m["n_nontrach"]), "controlled-vent accrual gate not met"),
        ("notelig_vaso_days", "Not eligible — vasopressor (NEE>0.2)", m["n_notelig_vaso"], m["n_nontrach"],
         _rate(m["n_notelig_vaso"], m["n_nontrach"]),
         "stability failed on pressors only; a site not screening on pressors would call eligible"),
        ("notelig_oxypeep_days", "Not eligible — oxygenation/PEEP", m["n_notelig_oxypeep"], m["n_nontrach"],
         _rate(m["n_notelig_oxypeep"], m["n_nontrach"]), "stability failed on FiO2/PEEP/SpO2"),
        ("excluded_paralytic_days", "Continuous-paralytic days (excluded)", m["n_excluded_paralytic"],
         m["n_nontrach"], _rate(m["n_excluded_paralytic"], m["n_nontrach"]),
         "continuous NMBA infusion; no respiratory drive -> justified exclusion"),
        ("sbt_delivered", "SBT delivered / transition-candidate days (HEADLINE)", m["n_sbt"],
         m["n_eligible_txcand"], _rate(m["n_sbt"], m["n_eligible_txcand"]),
         "controlled->support transition >= min duration; tile headline"),
        ("sbt_delivered_legacy", "SBT delivered / all eligible (legacy)", m["n_sbt"], m["n_eligible"],
         _rate(m["n_sbt"], m["n_eligible"]), "parked-on-spontaneous days kept in denominator"),
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
    is_type = slices["dim"] == "type" if "dim" in slices.columns else slices["unit"].notna()
    units = ["__ALL__"] + [u for u in CANONICAL_UNITS if u in set(slices.loc[is_type, "unit"])]
    # Specific-unit (location_name) dimension: name keys nest under their parent type.
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
    weeks = sorted(slices.loc[slices["granularity"] == "week", "period"].unique())
    by_key = {(r.unit, "all" if r.granularity == "all" else r.period): r
              for r in slices.itertuples(index=False)
              if r.granularity in ("all", "month", "week")}

    def cells(num_col, den_col, with_n=False):
        out = {}
        for u in all_keys:
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
        "subtitle": "Controlled→support transition among transition-candidate vent-days",
        "icon": "sbt",
        "detail_href": "sbt_dashboard.html",
        "goal": None,
        "generated": m["generated"],
        "note": ("• Denominator = eligible transition candidates: ≥12 h controlled + ≥2 h stable window "
                 "(FiO₂/PEEP/SpO₂ + NEE≤0.2), non-trach, non-paralytic, excluding days already parked on a "
                 "spontaneous mode with no transition"
                 "• Donut = strict SBT (controlled→support transition ≥2 min); bars = any-length SBT "
                 "and on a spontaneous mode (of eligible days / of all vent-days)"
                 "• Transition rates are a lower bound where charting is hourly; CPAP read from PEEP"
                 "• The detail dashboard makes every exclusion togglable"),
        "grain": {"units": units, "periods": ["all", "month", "week"]},
        # Two ICU-grouping dimensions: `type` = location_type (back-compat, also grain.units);
        # `name` = specific unit (location_name). Both key into headline/segment cells; `parent`
        # nests each name under its type; `labels` are optional friendly names (config unit_labels).
        # The scorecard's "Group ICUs by" toggle reads this block.
        "dims": {"type": [u for u in units if u != "__ALL__"], "name": name_units,
                 "parent": name_parent, "labels": dim_labels},
        "headline": {
            "label": "SBT delivered",
            "den_label": "of transition-candidate vent-days",
            "n_unit": "patient-days",
            "cells": cells("n_sbt", "n_eligible_txcand", with_n=True),
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
    # Integrity checks are scoped to the TYPE dim so they stay byte-identical to the pre-name build.
    typ = slices[slices["dim"] == "type"] if "dim" in slices.columns else slices
    a = typ[(typ["unit"] == "__ALL__") & (typ["granularity"] == "all")].iloc[0]
    for col, key in [("n_vent_days", "n_vent_days"), ("n_eligible", "n_eligible"),
                     ("n_sbt", "n_sbt"), ("n_not_assessable", "n_not_assessable"),
                     ("n_sbt_all", "n_sbt_all"), ("n_sbtany_all", "n_sbtany_all"),
                     ("n_spont_all", "n_spont_all")]:
        if int(a[col]) != int(m[key]):
            raise RuntimeError(f"slice __ALL__/all {col}={a[col]} != headline {m[key]}")
    # additive day-count columns must sum to the total across units and across periods
    additive = ["n_vent_days", "n_nontrach", "n_eligible", "n_sbt", "n_sbt_all",
                "n_sbtany_all", "n_spont_all"]
    units_all = typ[(typ["granularity"] == "all") & (typ["unit"] != "__ALL__")]
    for col in additive:
        if int(units_all[col].sum()) != int(m[col]):
            raise RuntimeError(f"per-unit {col} does not sum to total")
    for gran in GRANULARITY_COL:
        s = typ[(typ["unit"] == "__ALL__") & (typ["granularity"] == gran)]
        for col in additive:
            if int(s[col].sum()) != int(m[col]):
                raise RuntimeError(f"per-period ({gran}) {col} does not sum to total")
    # Name dim nests under type: each type's specific-unit children sum EXACTLY to its vent-day total.
    if "dim" in slices.columns:
        nm = slices[(slices["dim"] == "name") & (slices["granularity"] == "all")]
        if not nm.empty:
            by_parent = nm.groupby("parent")["n_vent_days"].sum()
            type_tot = units_all.set_index("unit")["n_vent_days"]
            for t, v in by_parent.items():
                if int(v) != int(type_tot.get(t, -1)):
                    raise RuntimeError(f"name children of {t} sum to {v} != type {type_tot.get(t)}")
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
    n_eligible_txcand = int((obs["eligible"] & ~(obs["on_spontaneous"] & ~obs["nb_t"])).sum())
    n_not_assessable = int((obs["eligibility_status"] == "not_assessable").sum())
    n_not_eligible = int((obs["eligibility_status"] == "not_eligible").sum())
    n_notelig_lt12h = int((obs["notelig_reason"] == "lt12h_controlled").sum())
    n_notelig_vaso = int((obs["notelig_reason"] == "failed_vasopressor").sum())
    n_notelig_oxypeep = int((obs["notelig_reason"] == "failed_oxy_peep").sum())
    assert n_notelig_lt12h + n_notelig_vaso + n_notelig_oxypeep == n_not_eligible, \
        "not_eligible sub-reason partition does not sum to n_not_eligible"
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
         "n_eligible_txcand": n_eligible_txcand,
         "n_not_assessable": n_not_assessable, "n_not_eligible": n_not_eligible,
         "n_notelig_lt12h": n_notelig_lt12h, "n_notelig_vaso": n_notelig_vaso,
         "n_notelig_oxypeep": n_notelig_oxypeep,
         "n_excluded_paralytic": n_excluded_paralytic, "n_sbt": n_sbt,
         "n_sbt_all": n_sbt_all, "n_sbtany_all": n_sbtany_all, "n_sbtany_elig": n_sbtany_elig,
         "n_spont_all": n_spont_all, "n_spont_elig": n_spont_elig,
         "n_pts": n_pts, "n_pts_elig": n_pts_elig, "n_pts_sbt": n_pts_sbt,
         "n_pts_sbt_any": n_pts_sbt_any, "n_pts_spont": n_pts_spont, "generated": generated}

    slices = build_slice_cells(obs)
    _assert_slice_integrity(slices, m)

    obs.to_parquet(inter / "metrics_patient_day_level.parquet", index=False)
    slices.to_parquet(inter / "metrics_slices.parquet", index=False)

    # Exclusion-toggle criterion-mask histogram (the dashboard slices live in JS off this).
    masks = build_mask_histogram(obs)
    # Integrity: every slice's mask counts must sum to that slice's n_vent_days.
    chk = (masks.groupby(["unit", "granularity", "period"])["count"].sum()
                .reset_index(name="m_sum")
                .merge(slices[["unit", "granularity", "period", "n_vent_days"]],
                       on=["unit", "granularity", "period"], how="outer"))
    bad = chk[chk["m_sum"].fillna(0) != chk["n_vent_days"].fillna(0)]
    if len(bad):
        raise RuntimeError(f"mask-histogram slice sums != n_vent_days on {len(bad)} slices")
    masks.to_parquet(inter / "metrics_masks.parquet", index=False)
    (inter / "metrics_masks_bits.json").write_text(json.dumps(MASK_BITS))   # bit-order contract for 05
    log.info("mask histogram: %d (slice,mask) rows · %d distinct masks · %d slices",
             len(masks), masks["mask"].nunique(),
             masks[["unit", "granularity", "period"]].drop_duplicates().shape[0])

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
    log.info("SBT delivered / transition-candidate days: %6d / %d (%.1f%%)  [TILE HEADLINE]",
             n_sbt, n_eligible_txcand, 100 * _rate(n_sbt, n_eligible_txcand) if n_eligible_txcand else 0)
    log.info("  (legacy SBT / all eligible:               %6d / %d (%.1f%%))",
             n_sbt, n_eligible, 100 * _rate(n_sbt, n_eligible) if n_eligible else 0)
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
