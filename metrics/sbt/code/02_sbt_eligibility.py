"""Stage 02 — eligible SBT-opportunity days (the SBT denominator), per Jain et al.

Starting from the ventilated-ICU patient-days (01/cohort.parquet), a day is an
ELIGIBLE SBT opportunity iff:
  - >= controlled_min_hours (12h) of CONTROLLED ventilation has accrued before the
    day's opportunity (cumulative-since-intubation), AND
  - there is a >= stability_min_hours (2h) contiguous window that day of stable
    physiology: FiO2 <= 0.50, PEEP <= 8, SpO2 >= 88, norepinephrine-equiv <= 0.2
    mcg/kg/min, AND
  - the patient is NOT tracheostomized that day (excluded from numerator AND
    denominator).

Eligibility status per non-trach day:
  eligible        — accrued 12h controlled AND a >=2h stable window
  not_assessable  — accrued 12h but stability un-assessable (no scaffold hour with
                    all four signals) -> reported as a bound, excluded from the rate
  not_eligible    — accrued 12h but assessed not-stable, OR < 12h controlled
  excluded_trach  — tracheostomized that day (dropped from num & den)

Inputs: cohort.parquet, the warm resp_waterfall cache, vitals (spo2 + weight_kg),
medication_admin_continuous (vasopressors). Aggregates only to stdout.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clifpy.tables import Vitals, MedicationAdminContinuous

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
import sbt_detect as sd            # noqa: E402
import sbt_vasopressors as sv      # noqa: E402

log = logging.getLogger("sbt.eligibility")

SPO2_RANGE = (50.0, 100.0)        # plausible SpO2 %


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sbt_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "02_sbt_eligibility.log", mode="w")],
    )
    cfg = cohort_mod.load_config()
    tz = cfg["timezone"]
    ds = cfg["primary_dataset"]
    elig_cfg = cfg["sbt_eligibility"]
    ctrl_min_h = int(elig_cfg.get("controlled_min_hours", 12))
    exclude_trach = bool(elig_cfg.get("exclude_trach_days", True))

    inter = cohort_mod.INTERMEDIATE_DIR

    # ---- cohort ----
    cohort = pd.read_parquet(inter / "cohort.parquet")
    cohort["encounter_block"] = cohort["encounter_block"].astype(str)
    cohort["day_in"] = cohort_mod._coerce_dttm(cohort["day_in"], tz)
    cohort["day_out"] = cohort_mod._coerce_dttm(cohort["day_out"], tz)
    cohort_blocks = set(cohort["encounter_block"].unique())
    log.info("ventilated-ICU patient-days in: %d (%d blocks)", len(cohort), len(cohort_blocks))

    # ---- waterfall (cache) restricted to cohort blocks ----
    wf = pd.read_parquet(cohort_mod.cpath("resp_waterfall"))
    wf = cohort_mod._normalize_waterfall(wf, tz)
    wf["encounter_block"] = wf["encounter_block"].astype(str)
    wf = wf[wf["encounter_block"].isin(cohort_blocks)]
    log.info("waterfall rows (cohort blocks): %d  (scaffold=%d)",
             len(wf), int(wf["is_scaffold"].fillna(False).sum()))

    # ---- hosp ids + mapping for vitals/meds loads ----
    mapping = pd.read_parquet(cohort_mod.cpath("encounter_mapping"))
    mapping["hospitalization_id"] = mapping["hospitalization_id"].astype(str)
    mapping["encounter_block"] = mapping["encounter_block"].astype(str)
    map_cohort = mapping[mapping["encounter_block"].isin(cohort_blocks)]
    hosp_ids = sorted(map_cohort["hospitalization_id"].unique())
    h2b = map_cohort.set_index("hospitalization_id")["encounter_block"].to_dict()

    # ---- vitals: SpO2 (stability) + weight_kg (vasopressor normalization) ----
    vit = Vitals.from_file(
        ds["data_path"], filetype=ds["file_format"], timezone=tz,
        filters={"hospitalization_id": hosp_ids, "vital_category": ["spo2", "weight_kg"]},
        columns=["hospitalization_id", "vital_category", "recorded_dttm", "vital_value"],
    ).df
    vit["hospitalization_id"] = vit["hospitalization_id"].astype(str)
    vit["recorded_dttm"] = cohort_mod._coerce_dttm(vit["recorded_dttm"], tz)
    vit["vital_value"] = pd.to_numeric(vit["vital_value"], errors="coerce")

    spo2 = vit[vit["vital_category"] == "spo2"].copy()
    spo2 = spo2[spo2["vital_value"].between(*SPO2_RANGE) & spo2["recorded_dttm"].notna()]
    spo2["encounter_block"] = spo2["hospitalization_id"].map(h2b).astype("string")
    spo2 = spo2.dropna(subset=["encounter_block"]).rename(
        columns={"recorded_dttm": "t", "vital_value": "spo2"})[["encounter_block", "t", "spo2"]]
    weight_vitals = vit[vit["vital_category"] == "weight_kg"][
        ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]].copy()
    log.info("vitals: spo2=%d obs, weight_kg=%d obs", len(spo2), len(weight_vitals))

    # ---- vasopressors -> norepinephrine-equivalent timeline ----
    vaso_cats = sorted(sv.vasopressor_categories(cfg))
    mac = MedicationAdminContinuous.from_file(
        ds["data_path"], filetype=ds["file_format"], timezone=tz,
        filters={"hospitalization_id": hosp_ids, "med_category": vaso_cats},
        columns=["hospitalization_id", "admin_dttm", "med_category", "med_dose",
                 "med_dose_unit", "mar_action_category"],
    ).df
    mac["hospitalization_id"] = mac["hospitalization_id"].astype(str)
    mac["encounter_block"] = mac["hospitalization_id"].map(h2b).astype("string")
    mac = mac.dropna(subset=["encounter_block"])
    log.info("vasopressor rows: %d (med_categories present: %s)",
             len(mac), sorted(mac["med_category"].astype("string").str.lower().dropna().unique().tolist()))
    ne_tl = sv.ne_equiv_timeline(mac, weight_vitals, cfg, tz)
    log.info("NE-equiv timeline change-points: %d (blocks=%d)",
             len(ne_tl), ne_tl["encounter_block"].nunique() if not ne_tl.empty else 0)

    # ---- per-day computations ----
    ctrl = sd.controlled_hours_before(wf, cohort, cfg)
    trach = sd.trach_day_flag(wf, cohort)
    stab = sd.hourly_stability_window(wf, cohort, spo2, ne_tl, cfg)

    out = (cohort
           .merge(ctrl, on=["encounter_block", "icu_day"], how="left")
           .merge(trach, on=["encounter_block", "icu_day"], how="left")
           .merge(stab, on=["encounter_block", "icu_day"], how="left"))
    out["prior_controlled_h"] = out["prior_controlled_h"].fillna(0).astype(int)
    out["trach_day"] = out["trach_day"].fillna(False).astype(bool)
    out["stable_window"] = out["stable_window"].fillna(False).astype(bool)
    for c in ("n_stable_hours", "n_scaffold_hours", "n_assessable_hours"):
        out[c] = out[c].fillna(0).astype(int)

    out["accrued_12h"] = out["prior_controlled_h"] >= ctrl_min_h
    trach_flag = out["trach_day"] & exclude_trach

    # Status (per the docstring).
    cond_trach = trach_flag
    cond_elig = (~trach_flag) & out["accrued_12h"] & out["stable_window"]
    cond_notassess = (~trach_flag) & out["accrued_12h"] & (~out["stable_window"]) & (out["n_assessable_hours"] == 0)
    out["eligibility_status"] = np.select(
        [cond_trach, cond_elig, cond_notassess],
        ["excluded_trach", "eligible", "not_assessable"],
        default="not_eligible",
    )
    out["eligible"] = out["eligibility_status"] == "eligible"

    out.to_parquet(inter / "sbt_eligibility.parquet", index=False)

    # ---- log ----
    n = len(out)
    n_trach = int(cond_trach.sum())
    n_nontrach = n - n_trach
    n_accrued = int(((~trach_flag) & out["accrued_12h"]).sum())
    vc = out["eligibility_status"].value_counts()
    n_elig = int(vc.get("eligible", 0))
    n_notassess = int(vc.get("not_assessable", 0))
    log.info("vent-ICU days:                 %6d", n)
    log.info("  tracheostomized (excluded):  %6d", n_trach)
    log.info("  non-trach vent-ICU days:     %6d", n_nontrach)
    log.info("  >=%dh controlled accrued:     %6d (%.1f%% of non-trach)",
             ctrl_min_h, n_accrued, 100 * n_accrued / max(n_nontrach, 1))
    log.info("  not_assessable stability:    %6d", n_notassess)
    log.info("ELIGIBLE SBT-opportunity days: %6d (%.1f%% of non-trach vent-ICU days)",
             n_elig, 100 * n_elig / max(n_nontrach, 1))
    log.info("wrote: sbt_eligibility.parquet")


if __name__ == "__main__":
    main()
