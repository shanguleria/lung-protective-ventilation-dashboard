"""Stage 03 — SBT delivery detection (the numerator), transition-only per Jain et al.

For each ventilated-ICU day we look on the NATIVE-resolution waterfall rows for a
CONTROLLED -> SUPPORT mode transition sustained >= support_min_minutes, where the
support episode is `pressure support/cpap` with PEEP <= ps_peep_max (pressure-support
arm) or PEEP <= cpap_peep_max (CPAP arm). A transition whose start falls inside the
day's ventilated-ICU window marks that day as "SBT delivered". Transition-only: a
patient parked on support all day with no controlled->support edge does NOT count.

Each transition episode is attributed to its US/Central calendar day and LEFT-joined
onto the cohort/eligibility skeleton (cohort-restriction discipline — never group raw
events without restricting to the cohort (block, day) set). The metric numerator
(stage 04) is `eligible & sbt_delivered`.

Also emits a coverage diagnostic: pct_native = share of support-mode readings that
come from native vs hourly-scaffold rows (sites charting only hourly cannot resolve
sub-hourly trials -> delivery is a lower bound).

Outputs:
    output/intermediate/sbt_observation.parquet  (one row per cohort day + SBT flags)
    output/intermediate/sbt_diag.json            (coverage diagnostics for the tile note)

Aggregates only to stdout.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
import sbt_detect as sd            # noqa: E402

log = logging.getLogger("sbt.observation")


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sbt_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attribute_transitions(days: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """Per (encounter_block, icu_day): SBT-delivery flags from transition episodes
    whose ep_start falls inside the day window."""
    base = days[["encounter_block", "icu_day", "day_in", "day_out"]].copy()
    base["encounter_block"] = base["encounter_block"].astype(str)
    cols = ["encounter_block", "icu_day", "sbt_delivered", "n_transitions",
            "longest_support_min", "sbt_arm"]
    if trans is None or trans.empty:
        base["sbt_delivered"] = False
        base["n_transitions"] = 0
        base["longest_support_min"] = 0.0
        base["sbt_arm"] = ""
        return base[cols]
    t = trans.copy()
    t["encounter_block"] = t["encounter_block"].astype(str)
    con = duckdb.connect()
    con.register("d", base)
    con.register("t", t)
    out = con.execute(
        """
        SELECT d.encounter_block AS encounter_block, d.icu_day AS icu_day,
               COUNT(t.ep_start) > 0           AS sbt_delivered,
               COUNT(t.ep_start)               AS n_transitions,
               COALESCE(MAX(t.dur_min), 0.0)   AS longest_support_min,
               COALESCE(string_agg(DISTINCT t.arm, ','), '') AS sbt_arm
        FROM d LEFT JOIN t
          ON d.encounter_block = t.encounter_block
         AND t.ep_start >= d.day_in AND t.ep_start < d.day_out
        GROUP BY d.encounter_block, d.icu_day
        """
    ).fetchdf()
    con.close()
    out["sbt_delivered"] = out["sbt_delivered"].fillna(False).astype(bool)
    out["n_transitions"] = out["n_transitions"].fillna(0).astype(int)
    out["longest_support_min"] = out["longest_support_min"].fillna(0.0)
    # normalize arm label: ps / cpap / both
    out["sbt_arm"] = out["sbt_arm"].fillna("").apply(
        lambda s: "both" if ("ps" in s and "cpap" in s) else s)
    return out[cols]


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "03_sbt_observation.log", mode="w")],
    )
    cfg = cohort_mod.load_config()
    tz = cfg["timezone"]
    obs_cfg = cfg["sbt_observation"]
    log.info("support_min_minutes=%.1f ps_peep_max=%.0f cpap_peep_max=%.0f | support_modes=%s",
             float(obs_cfg.get("support_min_minutes", 2)), float(obs_cfg["ps_peep_max"]),
             float(obs_cfg["cpap_peep_max"]), sorted(sd.support_modes(cfg)))

    inter = cohort_mod.INTERMEDIATE_DIR
    elig = pd.read_parquet(inter / "sbt_eligibility.parquet")
    elig["encounter_block"] = elig["encounter_block"].astype(str)
    elig["day_in"] = cohort_mod._coerce_dttm(elig["day_in"], tz)
    elig["day_out"] = cohort_mod._coerce_dttm(elig["day_out"], tz)
    cohort_blocks = set(elig["encounter_block"].unique())

    wf = pd.read_parquet(cohort_mod.cpath("resp_waterfall"))
    wf = cohort_mod._normalize_waterfall(wf, tz)
    wf["encounter_block"] = wf["encounter_block"].astype(str)
    wf = wf[wf["encounter_block"].isin(cohort_blocks)]

    trans = sd.support_transitions(wf, cfg)
    log.info("controlled->support transition episodes (>=min, arm-qualified): %d", len(trans))

    obs_flags = attribute_transitions(elig, trans)
    out = elig.merge(obs_flags, on=["encounter_block", "icu_day"], how="left")
    out["sbt_delivered"] = out["sbt_delivered"].fillna(False).astype(bool)
    out["n_transitions"] = out["n_transitions"].fillna(0).astype(int)
    out["longest_support_min"] = out["longest_support_min"].fillna(0.0)
    out["sbt_arm"] = out["sbt_arm"].fillna("")
    out.to_parquet(inter / "sbt_observation.parquet", index=False)

    # ---- coverage diagnostic: native vs scaffold share of support-mode readings ----
    supp = wf[wf["mode_category"].astype("string").str.lower().isin(sd.support_modes(cfg))]
    n_supp = int(len(supp))
    n_supp_native = int((~supp["is_scaffold"].fillna(False)).sum())
    pct_native = (100.0 * n_supp_native / n_supp) if n_supp else None
    diag = {
        "pct_native_support_rows": pct_native,
        "n_support_rows": n_supp,
        "n_support_rows_native": n_supp_native,
        "n_transition_episodes": int(len(trans)),
    }
    (inter / "sbt_diag.json").write_text(json.dumps(diag, indent=2))

    # ---- log ----
    n_elig = int(out["eligible"].sum())
    n_sbt = int((out["eligible"] & out["sbt_delivered"]).sum())
    n_sbt_anyday = int(out["sbt_delivered"].sum())
    log.info("eligible SBT-opportunity days: %6d", n_elig)
    log.info("  SBT delivered / eligible:    %6d (%.1f%%)  [headline numerator]",
             n_sbt, 100 * n_sbt / max(n_elig, 1))
    log.info("  (transitions on any day:     %6d)", n_sbt_anyday)
    if pct_native is not None:
        log.info("coverage: %.1f%% of support-mode readings are native-resolution (%d/%d)",
                 pct_native, n_supp_native, n_supp)
    log.info("wrote: sbt_observation.parquet, sbt_diag.json")


if __name__ == "__main__":
    main()
