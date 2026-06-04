"""Build the ARDS cohort for the proning QI project.

Screening question: "did this ICU stay ever look like ARDS?". The cohort is a
Berlin moderate-severe ARDS phenotype on invasive ventilation, defined purely
on physiology so any CLIF site can reproduce it. Trial-specific machinery
(enrollment-enrichment windows, ECMO/pregnancy/influenza/DNR exclusions,
fuzzy-window enrollment ABG) is deliberately omitted — that machinery exists to
clean up a causal effect estimate, and proning QI is descriptive.

ARDS screening at T₀ (Berlin moderate-severe gate):
    - age ≥ 18
    - device_category == "imv"
    - peep_set   ≥ 5  cmH2O
    - fio2_set   ≥ 0.4
    - pf_ratio  ≤ 300
    - in an ICU location at the ABG time

T₀ = earliest ABG meeting all criteria within an encounter_block.
One row per patient (earliest T₀ across encounter_blocks).

No raw PHI is printed to stdout; only aggregate counts and summary stats.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd

import clifpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
CACHE_DIR = INTERMEDIATE_DIR / "_cache"
FINAL_DIR = OUTPUT_DIR / "final"
LOGS_DIR = OUTPUT_DIR / "logs"

IMV_CATEGORY = "imv"  # UChicago stores device_category lowercase

log = logging.getLogger("proning.cohort")


def _ensure_dirs() -> None:
    for d in (INTERMEDIATE_DIR, CACHE_DIR, FINAL_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def cpath(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_orchestrator(cfg: dict) -> clifpy.ClifOrchestrator:
    ds = cfg["primary_dataset"]
    _ensure_dirs()
    return clifpy.ClifOrchestrator(
        data_directory=ds["data_path"],
        filetype=ds["file_format"],
        timezone=cfg["timezone"],
        output_directory=str(OUTPUT_DIR),
    )


def _coerce_dttm(series: pd.Series, tz: str) -> pd.Series:
    """Normalize a datetime column to tz-aware ``datetime64[us, tz]``.
    Cached parquets and waterfall scaffold rows can demote to object dtype.
    """
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        s = s.dt.tz_convert(tz)
    return s


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_abgs_cached(co: clifpy.ClifOrchestrator) -> pd.DataFrame:
    if cpath("abgs").exists():
        log.info("cache hit: abgs")
        return pd.read_parquet(cpath("abgs"))
    co.load_table("labs", filters={"lab_category": ["po2_arterial"]})
    df = co.labs.df
    df.to_parquet(cpath("abgs"), index=False)
    log.info("wrote cache: abgs (%d rows)", len(df))
    return df


def load_small_tables(co: clifpy.ClifOrchestrator) -> None:
    for t in ("patient", "hospitalization", "adt"):
        co.load_table(t)
        df = getattr(co, t).df
        log.info("loaded %s: %d rows", t, 0 if df is None else len(df))


# ---------------------------------------------------------------------------
# Encounter stitching (cached)
# ---------------------------------------------------------------------------
def stitch_cached(co: clifpy.ClifOrchestrator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cpath("encounter_mapping").exists():
        log.info("cache hit: stitched hosp/adt/mapping")
        hosp_s = pd.read_parquet(cpath("hosp_stitched"))
        adt_s = pd.read_parquet(cpath("adt_stitched"))
        mapping = pd.read_parquet(cpath("encounter_mapping"))
        if co.hospitalization is not None:
            co.hospitalization.df = hosp_s
        if co.adt is not None:
            co.adt.df = adt_s
        co.encounter_mapping = mapping
        return hosp_s, adt_s, mapping
    co.stitch_time_interval = 6
    co.run_stitch_encounters()
    mapping = co.encounter_mapping
    if mapping is None:
        raise RuntimeError("encounter stitching did not produce a mapping")
    hosp_s = co.hospitalization.df
    adt_s = co.adt.df
    hosp_s.to_parquet(cpath("hosp_stitched"), index=False)
    adt_s.to_parquet(cpath("adt_stitched"), index=False)
    mapping.to_parquet(cpath("encounter_mapping"), index=False)
    log.info("wrote cache: stitched %d hospitalizations → %d encounter_blocks",
             mapping["hospitalization_id"].nunique(), mapping["encounter_block"].nunique())
    return hosp_s, adt_s, mapping


# ---------------------------------------------------------------------------
# Respiratory waterfall (cached — expensive ~35 min step)
# ---------------------------------------------------------------------------
def waterfall_cached(
    co: clifpy.ClifOrchestrator,
    abg_df: pd.DataFrame,
    mapping: pd.DataFrame,
    tz: str,
) -> tuple[pd.DataFrame, str]:
    if cpath("resp_waterfall").exists():
        log.info("cache hit: resp_waterfall  ← the 35-min step")
        wf = pd.read_parquet(cpath("resp_waterfall"))
        wf = _normalize_waterfall(wf, tz)
        return wf, "from cache"

    abg_hosp_ids = abg_df["hospitalization_id"].dropna().astype(str).unique().tolist()
    log.info("waterfall input: %d hospitalizations (filtered to ABG-having)", len(abg_hosp_ids))
    co.load_table("respiratory_support", filters={"hospitalization_id": abg_hosp_ids})
    rs = co.respiratory_support.df
    log.info("loaded respiratory_support: %d rows", len(rs))

    wf = clifpy.process_resp_support_waterfall(
        rs, id_col="hospitalization_id", bfill=False, verbose=True
    )
    wf = wf.merge(mapping[["hospitalization_id", "encounter_block"]],
                  on="hospitalization_id", how="left")

    # Cache the raw-ish waterfall BEFORE normalization so cache stays valid
    # when we tweak normalization rules.
    wf.to_parquet(cpath("resp_waterfall"), index=False)
    log.info("wrote cache: resp_waterfall (%d rows)", len(wf))

    wf = _normalize_waterfall(wf, tz)
    return wf, "fresh + normalized"


def _normalize_waterfall(wf: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Post-waterfall cleanup applied every time (whether fresh or from cache).

    - Coerce ``recorded_dttm`` to tz-aware.
    - Lowercase device_category, mode_category (UChicago site convention).
    - FiO2 unit detection via p95; clip implausible FiO2 ∈ [0.15, 1.0] and
      PEEP ∈ [0, 40] to NaN.
    """
    wf = wf.copy()
    wf["recorded_dttm"] = _coerce_dttm(wf["recorded_dttm"], tz)
    for col in ("device_category", "mode_category"):
        if col in wf.columns:
            wf[col] = wf[col].astype("string").str.strip().str.lower()
    fio2 = wf["fio2_set"]
    p95 = fio2.dropna().quantile(0.95) if fio2.notna().any() else None
    if p95 is not None and p95 > 1.5:
        wf["fio2_set"] = fio2 / 100.0
        note = f"percent-encoded (p95={p95:.2f}, max={fio2.max():.1f}) → /100"
    else:
        note = f"fraction (p95={p95:.3f}, max={fio2.max():.3f})" if p95 is not None else "empty"
    bad_mask = wf["fio2_set"].notna() & ~wf["fio2_set"].between(0.15, 1.0)
    n_bad = int(bad_mask.sum())
    wf.loc[bad_mask, "fio2_set"] = pd.NA
    peep = wf["peep_set"]
    peep_bad_mask = peep.notna() & ~peep.between(0, 40)
    n_peep_bad = int(peep_bad_mask.sum())
    wf.loc[peep_bad_mask, "peep_set"] = pd.NA
    log.info("normalize: device_category/mode_category lowercased; "
             "fio2 %s; clipped %d implausible fio2, %d implausible peep",
             note, n_bad, n_peep_bad)
    return wf


# ---------------------------------------------------------------------------
# ABG extraction + as-of merge + P/F
# ---------------------------------------------------------------------------
def extract_abgs(abg_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mask = abg_df["lab_value_numeric"].notna() & (abg_df["lab_value_numeric"] > 0)
    abg = abg_df.loc[mask, ["hospitalization_id", "lab_collect_dttm", "lab_value_numeric"]].copy()
    abg = abg.rename(columns={"lab_collect_dttm": "abg_time", "lab_value_numeric": "pao2"})
    abg = abg.merge(
        mapping[["hospitalization_id", "encounter_block"]], on="hospitalization_id", how="left"
    )
    abg = abg.dropna(subset=["abg_time", "encounter_block"])
    log.info("arterial PaO2 events: %d (across %d encounter_blocks)",
             len(abg), abg["encounter_block"].nunique())
    return abg


def attach_vent_and_compute_pf(abg: pd.DataFrame, wf: pd.DataFrame, tz: str) -> pd.DataFrame:
    cols = ["encounter_block", "recorded_dttm", "device_category", "peep_set",
            "fio2_set", "mode_category"]
    wf_s = wf[cols].dropna(subset=["encounter_block", "recorded_dttm"]).copy()
    wf_s["recorded_dttm"] = _coerce_dttm(wf_s["recorded_dttm"], tz)
    abg = abg.copy()
    abg["abg_time"] = _coerce_dttm(abg["abg_time"], tz)
    wf_s = wf_s.sort_values("recorded_dttm")
    abg_s = abg.sort_values("abg_time")
    pf = pd.merge_asof(
        abg_s, wf_s,
        left_on="abg_time", right_on="recorded_dttm",
        by="encounter_block",
        direction="backward",
        tolerance=pd.Timedelta("6h"),
    )
    n_stale = pf["fio2_set"].isna().sum()
    log.info("ABGs dropped for stale/absent vent state (>6h) or bad fio2: %d", int(n_stale))
    pf = pf.dropna(subset=["fio2_set"])
    pf["pf_ratio"] = pf["pao2"] / pf["fio2_set"]
    in_band = pf["pf_ratio"].between(10, 1000)
    log.info("ABGs dropped for implausible P/F (<10 or >1000): %d", int((~in_band).sum()))
    return pf.loc[in_band].copy()


def restrict_to_icu(pf: pd.DataFrame, adt_s: pd.DataFrame) -> pd.DataFrame:
    if "encounter_block" not in adt_s.columns:
        raise RuntimeError("adt was not stitched — missing encounter_block")
    icu = adt_s.loc[
        adt_s["location_category"] == "icu",
        ["encounter_block", "in_dttm", "out_dttm"],
    ].copy()
    con = duckdb.connect()
    con.register("pf", pf)
    con.register("icu", icu)
    joined = con.execute(
        """
        SELECT pf.*,
               icu.in_dttm  AS icu_in_dttm,
               icu.out_dttm AS icu_out_dttm
        FROM pf
        JOIN icu
          ON pf.encounter_block = icu.encounter_block
         AND pf.abg_time BETWEEN icu.in_dttm AND icu.out_dttm
        """
    ).fetchdf()
    con.close()
    joined = (
        joined.sort_values(["encounter_block", "abg_time", "icu_in_dttm"])
        .drop_duplicates(subset=["encounter_block", "abg_time"], keep="first")
    )
    log.info("ABGs in ICU: %d (from %d pre-ICU-filter)", len(joined), len(pf))
    return joined


# ---------------------------------------------------------------------------
# T₀: ARDS screening
# ---------------------------------------------------------------------------
def compute_t0(pf_icu: pd.DataFrame, hosp_s: pd.DataFrame) -> pd.DataFrame:
    hp = hosp_s[["hospitalization_id", "patient_id", "age_at_admission"]].drop_duplicates()
    pf_icu = pf_icu.merge(hp, on="hospitalization_id", how="left")
    candidates = pf_icu[
        (pf_icu["device_category"] == IMV_CATEGORY)
        & (pf_icu["peep_set"] >= 5)
        & (pf_icu["fio2_set"] >= 0.4)
        & (pf_icu["pf_ratio"] <= 300)
        & (pf_icu["age_at_admission"] >= 18)
    ].copy()
    t0 = (
        candidates.sort_values("abg_time")
        .drop_duplicates(subset=["encounter_block"], keep="first")
        .rename(columns={
            "abg_time": "T0",
            "pao2": "pao2_at_t0",
            "fio2_set": "fio2_at_t0",
            "peep_set": "peep_at_t0",
            "pf_ratio": "pf_at_t0",
            "icu_in_dttm": "icu_in_dttm_at_t0",
        })
    )
    return t0[[
        "encounter_block", "hospitalization_id", "patient_id", "age_at_admission",
        "T0", "pao2_at_t0", "fio2_at_t0", "peep_at_t0", "pf_at_t0", "icu_in_dttm_at_t0",
    ]]


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------
def assemble_cohort_row(
    cohort: pd.DataFrame,
    co: clifpy.ClifOrchestrator,
    hosp_s: pd.DataFrame,
) -> pd.DataFrame:
    pat_cols = ["patient_id", "sex_category", "race_category", "ethnicity_category", "death_dttm"]
    pat = co.patient.df[pat_cols].drop_duplicates(subset=["patient_id"])
    hosp_cols = [
        "hospitalization_id", "admission_dttm", "discharge_dttm",
        "admission_type_category", "discharge_category",
    ]
    hosp_slim = hosp_s[hosp_cols]
    mapping = co.encounter_mapping[["hospitalization_id", "encounter_block"]]
    hosp_ids_per_block = (
        mapping.groupby("encounter_block")["hospitalization_id"].apply(list).rename("hospitalization_ids")
    )
    out = cohort.merge(pat, on="patient_id", how="left", suffixes=("", "_pat"))
    out = out.merge(hosp_slim, on="hospitalization_id", how="left")
    out = out.merge(hosp_ids_per_block, on="encounter_block", how="left")
    keep = [
        "patient_id", "encounter_block", "hospitalization_id", "hospitalization_ids",
        "icu_in_dttm_at_t0",
        "T0", "pao2_at_t0", "fio2_at_t0", "peep_at_t0", "pf_at_t0",
        "age_at_admission", "sex_category", "race_category", "ethnicity_category",
        "admission_type_category", "admission_dttm", "discharge_dttm", "discharge_category",
        "death_dttm",
    ]
    return out[[c for c in keep if c in out.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Delete output/intermediate/_cache/ and rebuild everything.")
    ap.add_argument("--refresh-waterfall", action="store_true",
                    help="Keep other caches; force waterfall rebuild.")
    args = ap.parse_args()

    _ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "01_build_cohort.log", mode="w"),
        ],
    )

    if args.refresh and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR); CACHE_DIR.mkdir()
        log.info("cleared full cache")
    elif args.refresh_waterfall and cpath("resp_waterfall").exists():
        cpath("resp_waterfall").unlink()
        log.info("cleared waterfall cache")

    cfg = load_config(CONFIG_PATH)
    tz = cfg["timezone"]
    log.info("site=%s timezone=%s", cfg.get("site"), tz)

    co = build_orchestrator(cfg)
    load_small_tables(co)
    hosp_s, adt_s, mapping = stitch_cached(co)
    abg_df = load_abgs_cached(co)
    wf, fio2_note = waterfall_cached(co, abg_df, mapping, tz)

    abg = extract_abgs(abg_df, mapping)
    pf = attach_vent_and_compute_pf(abg, wf, tz)
    pf_icu = restrict_to_icu(pf, adt_s)

    t0 = compute_t0(pf_icu, hosp_s)
    log.info("encounters with a T₀: %d (patients: %d)",
             t0["encounter_block"].nunique(), t0["patient_id"].nunique())

    # One row per patient — earliest T₀
    t0_one = t0.sort_values("T0").drop_duplicates(subset=["patient_id"], keep="first")
    log.info("after one-per-patient (earliest T₀): %d patients", t0_one["patient_id"].nunique())

    cohort_final = assemble_cohort_row(t0_one, co, hosp_s)
    cohort_final.to_parquet(INTERMEDIATE_DIR / "cohort.parquet", index=False)

    # Concise CONSORT-like flow for downstream reporting
    flow = pd.DataFrame([
        {"step": 1, "label": "encounter_blocks meeting ARDS screen at T₀",
         "n_encounter_blocks": int(t0["encounter_block"].nunique()),
         "n_patients": int(t0["patient_id"].nunique())},
        {"step": 2, "label": "one row per patient (earliest T₀)",
         "n_encounter_blocks": int(t0_one["encounter_block"].nunique()),
         "n_patients": int(t0_one["patient_id"].nunique())},
    ])
    flow.to_csv(FINAL_DIR / "cohort_flow.csv", index=False)

    log.info("CONSORT flow:")
    for _, row in flow.iterrows():
        log.info("  [%d] %-50s n_patients=%d  n_blocks=%d",
                 row["step"], row["label"], row["n_patients"], row["n_encounter_blocks"])
    log.info("fio2 convention: %s", fio2_note)
    log.info("wrote: cohort.parquet, cohort_flow.csv, 01_build_cohort.log")


if __name__ == "__main__":
    main()
