"""Stage 03 — SAT detection + the Kress dose-resumption benchmark.

For each ELIGIBLE SAT-opportunity day (02/sat_eligibility.parquet) we look inside
that day's ventilated-ICU window for an OFF gap during which ALL SAT-relevant
infusions are simultaneously held to rate 0 for >= `hold_min_minutes`, occurring
AFTER sedation had been running that day (a genuine interruption, not a
pre-sedation lead-in). Such a day is "SAT performed". Dexmedetomidine running is
ignored (it may continue).

Kress et al. 2000 add-on: among holds that RESUME sedation (a SAT-relevant
infusion restarts after the gap), capture the per-drug pre-hold and post-resume
steady-state rate and the unitless ratio resumed/pre-hold, to compare against
Kress's "restart at half the prior dose" recommendation. Like-to-like per drug;
denominator for this metric is resumed SATs only.

Outputs:
    output/intermediate/sat_observation.parquet  (one row per eligible day + SAT flags)
    output/intermediate/kress_resumption.parquet (one row per resumed-SAT x drug)

Aggregates only to stdout.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
import sat_infusions as si  # noqa: E402

log = logging.getLogger("sat.observation")

# How far after a hold a resumption may occur (within the vent-ICU day window).
# Kept generous; resumption is also bounded by the day window.


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("sat_cohort", CODE_DIR / "01_build_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _group_intervals(au: pd.DataFrame) -> dict:
    """block -> sorted list of (start, end) active intervals."""
    g = defaultdict(list)
    if au is None or au.empty:
        return g
    for blk, s, e in zip(au["encounter_block"].astype(str), au["start"], au["end"]):
        g[blk].append((s, e))
    for blk in g:
        g[blk].sort()
    return g


def _group_active_segs(segs: pd.DataFrame) -> dict:
    """block -> sorted list of (med_category, seg_start, seg_end, dose) for ACTIVE segs."""
    g = defaultdict(list)
    act = segs[segs["active"]]
    for blk, drug, s, e, dose in zip(act["encounter_block"].astype(str), act["med_category"],
                                     act["seg_start"], act["seg_end"], act["dose"]):
        g[blk].append((drug, s, e, float(dose)))
    for blk in g:
        g[blk].sort(key=lambda r: r[1])
    return g


def _kress_for_gap(active_segs: list, gs, ge, day_out) -> list[dict]:
    """Per-drug pre-hold vs post-resume rate for one resumed hold."""
    rows = []
    drugs = {r[0] for r in active_segs}
    for drug in drugs:
        ds = [r for r in active_segs if r[0] == drug]
        pre = [r for r in ds if r[1] < gs]                       # active started before hold
        post = [r for r in ds if r[1] >= ge and r[1] <= day_out]  # resumed within the day window
        if not pre or not post:
            continue
        pre_dose = max(pre, key=lambda r: r[1])[3]               # most recent pre-hold dose
        post_dose = min(post, key=lambda r: r[1])[3]             # first post-resume dose
        if pre_dose and pre_dose > 0:
            rows.append({"med_category": drug, "pre_rate": pre_dose,
                         "post_rate": post_dose, "ratio": post_dose / pre_dose})
    return rows


def main() -> None:
    cohort_mod = _load_cohort_module()
    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cohort_mod.LOGS_DIR / "03_sat_observation.log", mode="w")],
    )
    cfg = cohort_mod.load_config()
    tz = cfg["timezone"]
    med_sets = cohort_mod.sat_med_sets(cfg)
    hold_min = float(cfg["sat_observation"].get("hold_min_minutes", 30))
    log.info("hold_min_minutes=%.0f | SAT-relevant=%s", hold_min, sorted(med_sets["sat_relevant"]))

    inter = cohort_mod.INTERMEDIATE_DIR
    elig = pd.read_parquet(inter / "sat_eligibility.parquet")
    elig["encounter_block"] = elig["encounter_block"].astype(str)
    elig["day_in"] = si.coerce_dttm(elig["day_in"], tz)
    elig["day_out"] = si.coerce_dttm(elig["day_out"], tz)

    inf = pd.read_parquet(cohort_mod.cpath("infusions"))
    inf["encounter_block"] = inf["encounter_block"].astype(str)
    unit_map = (inf[inf["med_category"].isin(med_sets["sat_relevant"])]
                .dropna(subset=["med_dose_unit"])
                .groupby("med_category")["med_dose_unit"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
                .to_dict())

    segs = si.build_drug_segments(inf, med_sets["sat_relevant"], tz)
    au = si.active_union(segs)
    au_by_block = _group_intervals(au)
    segs_by_block = _group_active_segs(segs)

    hold_td = pd.Timedelta(minutes=hold_min)
    obs_rows = []
    kress_rows = []
    dur_rows = []   # per qualifying SAT hold (PHI-free): the off-sedation duration panel

    elig_only = elig[elig["eligible"]]
    for r in elig_only.itertuples(index=False):
        blk = str(r.encounter_block)
        win_lo, win_hi = r.day_in, r.day_out
        clipped = []
        for s, e in au_by_block.get(blk, ()):
            cs, ce = max(s, win_lo), min(e, win_hi)
            if ce > cs:
                clipped.append((cs, ce))
        clipped.sort()
        _, gaps = si.off_gaps_in_window(clipped, win_lo, win_hi)
        holds = [(gs, ge, res) for (gs, ge, res) in gaps if (ge - gs) >= hold_td]

        sat_performed = len(holds) > 0
        n_holds = len(holds)
        longest_off_min = max(((ge - gs).total_seconds() / 60.0 for gs, ge, _ in holds), default=0.0)
        any_resumed = any(res for _, _, res in holds)
        first_hold_dttm = holds[0][0] if holds else pd.NaT

        # per-hold off-sedation durations (PHI-free: unit/icu_day/off_min/resumed) for the panel
        for (gs, ge, res) in holds:
            dur_rows.append({"unit": r.unit, "icu_day": r.icu_day,
                             "off_min": (ge - gs).total_seconds() / 60.0, "resumed": bool(res)})

        obs_rows.append({
            "encounter_block": blk, "icu_day": r.icu_day,
            "sat_performed": sat_performed, "n_holds": n_holds,
            "longest_off_min": longest_off_min, "sat_resumed": any_resumed,
            "first_hold_dttm": first_hold_dttm,
        })

        # Kress: first RESUMED hold of the day (the primary SAT) -> per-drug ratio.
        resumed_holds = [(gs, ge) for gs, ge, res in holds if res]
        if resumed_holds:
            gs, ge = resumed_holds[0]
            for k in _kress_for_gap(segs_by_block.get(blk, []), gs, ge, win_hi):
                k.update(encounter_block=blk, icu_day=r.icu_day,
                         med_dose_unit=unit_map.get(k["med_category"]))
                kress_rows.append(k)

    obs = pd.DataFrame(obs_rows)
    out = elig.merge(obs, on=["encounter_block", "icu_day"], how="left")
    for c in ("sat_performed", "sat_resumed"):
        out[c] = out[c].fillna(False).astype(bool)
    out["n_holds"] = out["n_holds"].fillna(0).astype(int)
    out["longest_off_min"] = out["longest_off_min"].fillna(0.0)

    # --- same-day-extubation outcome: off IMV at the end of the SAT calendar day -------------
    # "extubated_eod" = the patient is NOT on an invasive-vent device at the next midnight (local)
    # after the day, AND alive then. Uses the PURE IMV device timeline (build_imv_intervals), so a
    # patient transferred out of ICU while still intubated is NOT counted, reintubation before
    # midnight keeps them "on", and death-on-vent is excluded via death_dttm. The trailing-IMV cap
    # makes end-of-day status near a charting gap a bound (same caveat the rate carries).
    wf = pd.read_parquet(cohort_mod.cpath("resp_waterfall"))
    wf = cohort_mod._normalize_waterfall(wf, tz)
    imv = cohort_mod.build_imv_intervals(wf)
    imv["encounter_block"] = imv["encounter_block"].astype(str)
    imv["seg_start"] = si.coerce_dttm(imv["seg_start"], tz)
    imv["seg_end"] = si.coerce_dttm(imv["seg_end"], tz)
    ivl = {}
    for r in imv.itertuples(index=False):
        ivl.setdefault(str(r.encounter_block), []).append((r.seg_start, r.seg_end))
    eod = (pd.to_datetime(out["icu_day"].astype(str)) + pd.Timedelta(days=1)) \
        .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")

    def _off_imv(blk, t):
        if pd.isna(t):
            return False
        for s, e in ivl.get(str(blk), ()):
            if s <= t < e:
                return False
        return True
    off_imv = [_off_imv(b, t) for b, t in zip(out["encounter_block"], eod)]
    if "death_dttm" in out.columns:
        dd = si.coerce_dttm(out["death_dttm"], tz)
        alive = dd.isna() | (dd > eod)
    else:
        alive = pd.Series(True, index=out.index)
    out["extubated_eod"] = pd.Series(off_imv, index=out.index) & alive.reset_index(drop=True).values
    out.to_parquet(inter / "sat_observation.parquet", index=False)

    _sat = out["eligible"] & out["sat_performed"]
    _nsat = int(_sat.sum())
    log.info("same-day extubation among SATs: %d / %d (%.1f%%)  [off IMV & alive at end of SAT day]",
             int((_sat & out["extubated_eod"]).sum()), _nsat,
             100 * int((_sat & out["extubated_eod"]).sum()) / max(_nsat, 1))

    kress = pd.DataFrame(kress_rows, columns=["encounter_block", "icu_day", "med_category",
                                              "med_dose_unit", "pre_rate", "post_rate", "ratio"]) \
        if kress_rows else pd.DataFrame(
            columns=["encounter_block", "icu_day", "med_category", "med_dose_unit",
                     "pre_rate", "post_rate", "ratio"])
    kress.to_parquet(inter / "kress_resumption.parquet", index=False)

    durs = pd.DataFrame(dur_rows, columns=["unit", "icu_day", "off_min", "resumed"])
    durs.to_parquet(inter / "sat_durations.parquet", index=False)
    if not durs.empty:
        log.info("SAT holds (per-hold off-sedation durations): %d  (median %.0f min, p90 %.0f min)",
                 len(durs), durs["off_min"].median(), durs["off_min"].quantile(0.9))

    n_elig = int(out["eligible"].sum())
    n_sat = int(out.loc[out["eligible"], "sat_performed"].sum())
    n_resumed = int(out.loc[out["eligible"], "sat_resumed"].sum())
    log.info("eligible SAT-opportunity days: %6d", n_elig)
    log.info("  SAT performed (>=%.0f min):   %6d (%.1f%%)  [headline numerator]",
             hold_min, n_sat, 100 * n_sat / max(n_elig, 1))
    log.info("  of which resumed sedation:   %6d", n_resumed)
    if not kress.empty:
        ratios = kress["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        n_half = int((ratios <= float(cfg["sat_observation"].get("kress_half_dose_threshold", 0.5))).sum())
        log.info("Kress resumption ratios: n=%d (drug-level) | median=%.2f (IQR %.2f-%.2f) | <=half-dose: %d (%.1f%%)",
                 len(ratios), ratios.median(), ratios.quantile(.25), ratios.quantile(.75),
                 n_half, 100 * n_half / max(len(ratios), 1))
    log.info("wrote: sat_observation.parquet, kress_resumption.parquet")


if __name__ == "__main__":
    main()
