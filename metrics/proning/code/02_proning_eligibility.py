"""Compute PROSEVA-strict proning eligibility per cohort encounter.

PROSEVA eligibility (defaults from config; overridable for sensitivity):
    - device_category == "imv"
    - peep_set    ≥ 5  cmH2O
    - fio2_set    ≥ 0.6
    - pf_ratio    ≤ 150
    - sustained ≥ 12 h: a second qualifying ABG exists at or after T_first + 12h,
      AND the waterfall shows no extubation event in (T_first, T_first + 12h].

This matches PROSEVA's post-stabilization re-evaluation pattern: enrollment
required severity to persist after a 12-24 h stabilization period, not that
every intermediate ABG remained at the severity threshold (clinically, ABGs
during weaning attempts are common and shouldn't disqualify a patient who
ultimately remained severe). T_eligible = T_first_qualifying_ABG + 12 h.

Inputs (cached by code/01_build_cohort.py):
    - output/intermediate/cohort.parquet                     (one row per patient)
    - output/intermediate/_cache/resp_waterfall.parquet      (vent timeline)
    - output/intermediate/_cache/abgs.parquet                (PaO2 events)
    - output/intermediate/_cache/adt_stitched.parquet        (ICU localization)
    - output/intermediate/_cache/encounter_mapping.parquet   (encounter blocks)

Output:
    - output/intermediate/proning_eligibility.parquet
        keyed on hospitalization_id (one row per cohort patient), columns:
        encounter_block, hospitalization_id, patient_id, T0, eligible (bool),
        T_first_qualifying_abg, T_eligible, n_qualifying_abgs_in_window,
        ineligibility_reason

No raw PHI is printed; only counts and aggregates.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

log = logging.getLogger("proning.eligibility")


def _load_cohort_module():
    """Import code/01_build_cohort.py via importlib (digit-prefixed name).

    Pattern documented in .claude/lessons.md ("importlib for digit-prefixed
    pipeline modules"). 01_build_cohort.py is shape-correct: heavy work runs
    only inside its main(), top-level only sets constants and defines helpers.
    """
    path = CODE_DIR / "01_build_cohort.py"
    spec = importlib.util.spec_from_file_location("proning_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_eligibility(
    cohort: pd.DataFrame,
    pf_icu: pd.DataFrame,
    wf: pd.DataFrame,
    cfg: dict,
    imv_category: str,
) -> pd.DataFrame:
    """Per cohort encounter, decide PROSEVA-strict eligibility.

    Persistent re-evaluation interpretation (chosen 2026-04-28):
    1. Find the first post-T₀ ABG meeting all four criteria — call it T_first.
    2. Require a *second* ABG at or after T_first + 12 h that also meets all
       four criteria. (Severity persisted past the stabilization window.)
    3. Require no extubation event in (T_first, T_first + 12 h] — the patient
       was on IMV continuously through the stabilization period.

    If all three hold, eligible. T_eligible = T_first + 12 h.

    Intermediate ABGs that don't meet criteria (e.g., during a brief weaning
    attempt) do NOT disqualify, matching how PROSEVA enrolled — they screened
    pre-stabilization and re-checked at the post-stabilization timepoint.

    Returns a DataFrame keyed on encounter_block (one row per cohort entry),
    including non-eligible patients with reason codes.
    """
    pf_max = cfg["proning_eligibility"]["pf_max"]
    fio2_min = cfg["proning_eligibility"]["fio2_min"]
    peep_min = cfg["proning_eligibility"]["peep_min"]
    sustained = pd.Timedelta(hours=cfg["proning_eligibility"]["sustained_hours"])

    # Restrict pf_icu to cohort encounters and to ABGs at or after T0
    keys = cohort[["encounter_block", "T0"]].drop_duplicates()
    pf = pf_icu.merge(keys, on="encounter_block", how="inner")
    pf = pf[pf["abg_time"] >= pf["T0"]].copy()

    # Per-ABG meets-criteria flag
    pf["meets"] = (
        (pf["device_category"] == imv_category)
        & (pf["peep_set"].ge(peep_min))
        & (pf["fio2_set"].ge(fio2_min))
        & (pf["pf_ratio"].le(pf_max))
    )
    pf = pf.sort_values(["encounter_block", "abg_time"])

    # Build a per-encounter waterfall index for the extubation check
    wf_slim = wf[["encounter_block", "recorded_dttm", "device_category"]].dropna(
        subset=["encounter_block", "recorded_dttm"]
    )

    # Group ABGs by encounter
    pf_by_eb = {eb: g for eb, g in pf.groupby("encounter_block", sort=False)}
    wf_by_eb = {eb: g for eb, g in wf_slim.groupby("encounter_block", sort=False)}

    rows = []
    for _, c in cohort.iterrows():
        eb = c["encounter_block"]
        result = {
            "encounter_block": eb,
            "hospitalization_id": c["hospitalization_id"],
            "patient_id": c["patient_id"],
            "T0": c["T0"],
            "eligible": False,
            "T_first_qualifying_abg": pd.NaT,
            "T_eligible": pd.NaT,
            "n_qualifying_abgs_in_window": 0,
            "ineligibility_reason": None,
        }

        g = pf_by_eb.get(eb)
        if g is None or g.empty:
            result["ineligibility_reason"] = "no post-T0 ABGs"
            rows.append(result)
            continue

        meets = g["meets"].to_numpy(dtype=bool)
        meets_idx = np.flatnonzero(meets)
        if meets_idx.size == 0:
            result["ineligibility_reason"] = "no ABG meets PROSEVA-strict thresholds"
            rows.append(result)
            continue

        # tz-aware pandas types throughout — .to_numpy() drops tz info.
        times = g["abg_time"].reset_index(drop=True)

        wfg = wf_by_eb.get(eb)
        if wfg is not None and len(wfg) > 0:
            wf_dt = wfg["recorded_dttm"].reset_index(drop=True)
            wf_dev = wfg["device_category"].reset_index(drop=True)
        else:
            wf_dt = pd.Series([], dtype=times.dtype)
            wf_dev = pd.Series([], dtype="string")

        # T_first = earliest qualifying ABG.
        t_first = times.iloc[meets_idx[0]]
        t_eligible_target = t_first + sustained

        # Need ≥1 qualifying ABG at or after the stabilization window end.
        post_window_qualifying = (times >= t_eligible_target) & pd.Series(meets)
        if not bool(post_window_qualifying.any()):
            result["ineligibility_reason"] = (
                "qualifying ABG at T_first but no qualifying ABG ≥12h later "
                "(weaned, extubated, or no ABG coverage)"
            )
            rows.append(result)
            continue

        # No extubation in (T_first, T_first + 12h].
        if len(wf_dt) > 0:
            in_win = (wf_dt > t_first) & (wf_dt <= t_eligible_target)
            non_imv = wf_dev.ne(imv_category) & wf_dev.notna()
            if bool((in_win & non_imv).any()):
                result["ineligibility_reason"] = (
                    "extubation event during 12h stabilization window"
                )
                rows.append(result)
                continue

        n_qualifying_in_window = int(((times >= t_first) & (times <= t_eligible_target) & pd.Series(meets)).sum())

        result["eligible"] = True
        result["T_first_qualifying_abg"] = t_first
        result["T_eligible"] = t_eligible_target
        result["n_qualifying_abgs_in_window"] = n_qualifying_in_window
        result["ineligibility_reason"] = None
        rows.append(result)

    out = pd.DataFrame(rows)
    return out


def main() -> None:
    cohort_mod = _load_cohort_module()

    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cohort_mod.LOGS_DIR / "02_proning_eligibility.log", mode="w"),
        ],
    )

    cfg = cohort_mod.load_config(cohort_mod.CONFIG_PATH)
    tz = cfg["timezone"]
    log.info("site=%s timezone=%s", cfg.get("site"), tz)
    log.info(
        "PROSEVA-strict thresholds: P/F ≤ %s, FiO2 ≥ %s, PEEP ≥ %s, sustained %s h",
        cfg["proning_eligibility"]["pf_max"],
        cfg["proning_eligibility"]["fio2_min"],
        cfg["proning_eligibility"]["peep_min"],
        cfg["proning_eligibility"]["sustained_hours"],
    )

    cohort_path = cohort_mod.INTERMEDIATE_DIR / "cohort.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"{cohort_path} not found. Run code/01_build_cohort.py first."
        )
    cohort = pd.read_parquet(cohort_path)
    log.info("loaded cohort: %d encounters / %d patients",
             cohort["encounter_block"].nunique(), cohort["patient_id"].nunique())

    # Reuse cohort module's cached IO — fast since 01 already populated _cache/
    co = cohort_mod.build_orchestrator(cfg)
    cohort_mod.load_small_tables(co)
    hosp_s, adt_s, mapping = cohort_mod.stitch_cached(co)
    abg_df = cohort_mod.load_abgs_cached(co)
    wf, _ = cohort_mod.waterfall_cached(co, abg_df, mapping, tz)
    abg = cohort_mod.extract_abgs(abg_df, mapping)
    pf = cohort_mod.attach_vent_and_compute_pf(abg, wf, tz)
    pf_icu = cohort_mod.restrict_to_icu(pf, adt_s)

    elig = compute_eligibility(cohort, pf_icu, wf, cfg, cohort_mod.IMV_CATEGORY)

    out_path = cohort_mod.INTERMEDIATE_DIR / "proning_eligibility.parquet"
    elig.to_parquet(out_path, index=False)

    n_total = len(elig)
    n_eligible = int(elig["eligible"].sum())
    pct = 100.0 * n_eligible / n_total if n_total else 0.0
    log.info("eligibility result: %d / %d (%.1f%%) of cohort meet PROSEVA-strict",
             n_eligible, n_total, pct)
    reason_counts = elig.loc[~elig["eligible"], "ineligibility_reason"].value_counts(dropna=False)
    log.info("ineligibility breakdown:")
    for reason, n in reason_counts.items():
        log.info("  %4d  %s", n, reason)
    log.info("wrote: %s", out_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
