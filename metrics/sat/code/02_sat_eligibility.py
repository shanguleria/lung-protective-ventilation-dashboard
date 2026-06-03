"""Stage 02 — eligible SAT-opportunity days (the SAT denominator).

Starting from the ventilated-ICU patient-days (01/cohort.parquet), a day is an
ELIGIBLE SAT opportunity iff, during that day's ventilated-ICU window:
  - >= 1 SAT-relevant continuous infusion is active (propofol / midazolam /
    fentanyl / other benzo or opioid), AND
  - the patient is NOT on a continuous paralytic (neuromuscular blockade) — the
    one SAT safety-screen exclusion CLIF can observe (config-gated).

Dex-only days are excluded automatically: eligibility REQUIRES a SAT-relevant
infusion, and dexmedetomidine is not in that set (it may continue during a SAT).

Other classic SAT safety-screen exclusions (active seizures, alcohol withdrawal,
myocardial ischemia, raised ICP) are NOT reliably codable in CLIF -> this is
crude eligibility, not full safety-screen-passed eligibility (surfaced as a
dashboard/tile caveat).

Aggregates only to stdout.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
import sat_infusions as si  # noqa: E402

log = logging.getLogger("sat.eligibility")


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sat_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def overlap_flag(days: pd.DataFrame, intervals: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Per (encounter_block, icu_day) boolean: does any [start,end] interval
    overlap the day's [day_in, day_out] window?"""
    base = days[["encounter_block", "icu_day", "day_in", "day_out"]].copy()
    base["encounter_block"] = base["encounter_block"].astype(str)
    if intervals is None or intervals.empty:
        base[colname] = False
        return base[["encounter_block", "icu_day", colname]]
    iv = intervals.copy()
    iv["encounter_block"] = iv["encounter_block"].astype(str)
    con = duckdb.connect()
    con.register("d", base)
    con.register("i", iv)
    out = con.execute(
        f"""
        SELECT d.encounter_block AS encounter_block, d.icu_day AS icu_day,
               COUNT(i.start) > 0 AS {colname}
        FROM d LEFT JOIN i
          ON d.encounter_block = i.encounter_block
         AND i.start < d.day_out AND i.end > d.day_in
        GROUP BY d.encounter_block, d.icu_day
        """
    ).fetchdf()
    con.close()
    return out


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "02_sat_eligibility.log", mode="w")],
    )
    cfg = cohort_mod.load_config()
    tz = cfg["timezone"]
    med_sets = cohort_mod.sat_med_sets(cfg)
    exclude_paralytic = bool(cfg["sat_eligibility"].get("exclude_paralytic_days", True))

    inter = cohort_mod.INTERMEDIATE_DIR
    cohort = pd.read_parquet(inter / "cohort.parquet")
    cohort["encounter_block"] = cohort["encounter_block"].astype(str)
    cohort["day_in"] = si.coerce_dttm(cohort["day_in"], tz)
    cohort["day_out"] = si.coerce_dttm(cohort["day_out"], tz)
    log.info("ventilated-ICU patient-days in: %d", len(cohort))

    inf = pd.read_parquet(cohort_mod.cpath("infusions"))
    inf["encounter_block"] = inf["encounter_block"].astype(str)

    au_sat = si.active_union(si.build_drug_segments(inf, med_sets["sat_relevant"], tz))
    au_par = si.active_union(si.build_drug_segments(inf, med_sets["paralytic"], tz))
    au_dex = si.active_union(si.build_drug_segments(inf, med_sets["dex"], tz))
    log.info("active-interval unions: sat=%d par=%d dex=%d (blocks with any)",
             au_sat["encounter_block"].nunique() if not au_sat.empty else 0,
             au_par["encounter_block"].nunique() if not au_par.empty else 0,
             au_dex["encounter_block"].nunique() if not au_dex.empty else 0)

    f_sat = overlap_flag(cohort, au_sat, "on_sat_sedation")
    f_par = overlap_flag(cohort, au_par, "on_paralytic")
    f_dex = overlap_flag(cohort, au_dex, "on_dex")

    out = (cohort
           .merge(f_sat, on=["encounter_block", "icu_day"], how="left")
           .merge(f_par, on=["encounter_block", "icu_day"], how="left")
           .merge(f_dex, on=["encounter_block", "icu_day"], how="left"))
    for c in ("on_sat_sedation", "on_paralytic", "on_dex"):
        out[c] = out[c].fillna(False).astype(bool)

    out["eligible"] = out["on_sat_sedation"] & ~(out["on_paralytic"] if exclude_paralytic else False)
    out.to_parquet(inter / "sat_eligibility.parquet", index=False)

    n = len(out)
    n_sat = int(out["on_sat_sedation"].sum())
    n_par = int(out["on_paralytic"].sum())
    n_par_excl = int((out["on_sat_sedation"] & out["on_paralytic"]).sum()) if exclude_paralytic else 0
    n_dexonly = int((~out["on_sat_sedation"] & out["on_dex"]).sum())
    n_elig = int(out["eligible"].sum())
    log.info("vent-ICU days:                 %6d", n)
    log.info("  on SAT-relevant sedation:    %6d (%.1f%%)", n_sat, 100 * n_sat / max(n, 1))
    log.info("  on continuous paralytic:     %6d (excluded from eligible: %d)", n_par, n_par_excl)
    log.info("  dex-only (no SAT-relevant):  %6d  [excluded — nothing to interrupt]", n_dexonly)
    log.info("ELIGIBLE SAT-opportunity days: %6d (%.1f%% of vent-ICU days)", n_elig, 100 * n_elig / max(n, 1))
    log.info("wrote: sat_eligibility.parquet")


if __name__ == "__main__":
    main()
