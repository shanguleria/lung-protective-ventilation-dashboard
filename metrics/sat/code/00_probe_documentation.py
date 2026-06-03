"""Quantify-first documentation probe for the SAT adherence QI vertical.

Before locking the SAT numerator/denominator we must answer the proning
"denominator trap" for sedation infusions:

    Does `medication_admin_continuous` chart explicit rate-0 rows when an
    infusion is PAUSED, or does charting simply STOP (leaving a gap)?

A gap is ambiguous — a SAT hold-then-resume vs a permanent discontinuation
(de-escalation / extubation / death) vs a charting gap. This script profiles
that convention plus the denominator size, the drug inventory (so we can fill
`config.json -> sat_medications` with the site's real `med_category` values),
and RASS coverage (a secondary lens).

DATA SAFETY: this prints AGGREGATES ONLY (counts, fractions, quantiles,
value_counts on low-cardinality columns). It queries the parquet files with
DuckDB so raw rows are never materialised into the conversation. No
hospitalization_id / patient_id / dates are printed.

Run on demand (not part of run_pipeline.sh):
    python code/00_probe_documentation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
FINAL_DIR = PROJECT_ROOT / "output" / "final"
LOGS_DIR = PROJECT_ROOT / "output" / "logs"

log = logging.getLogger("sat.probe")

# Coarse keyword buckets used ONLY for probe-time classification of whatever
# med_category strings the site actually uses. The authoritative lists live in
# config.json -> sat_medications (which this probe helps populate).
SEDATIVE_ANALGESIC_KW = [
    "propofol", "midazolam", "lorazepam", "diazepam",
    "fentanyl", "hydromorphone", "morphine", "remifentanil", "sufentanil",
    "ketamine",
]
DEX_KW = ["dexmedetomidine", "precedex"]
PARALYTIC_KW = ["cisatracurium", "vecuronium", "rocuronium", "atracurium",
                "pancuronium", "succinylcholine"]


def _ensure_dirs() -> None:
    for d in (FINAL_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_table(data_path: Path, table: str, fmt: str) -> str | None:
    """Locate a CLIF table file under data_path (handles clif_/CLIF_ prefixes)."""
    ext = "parquet" if fmt == "parquet" else fmt
    pats = [f"{table}.{ext}", f"clif_{table}.{ext}", f"*{table}.{ext}", f"*{table}*.{ext}"]
    for pat in pats:
        hits = sorted(data_path.glob(pat))
        if hits:
            return str(hits[0])
    return None


def _scan(path: str) -> str:
    """DuckDB scan expression for a parquet (or csv) file path."""
    if path.lower().endswith(".parquet"):
        return f"read_parquet('{path}')"
    return f"read_csv_auto('{path}')"


def _columns(con, scan: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()]


def _classify(cat: str) -> str:
    c = (cat or "").strip().lower()
    if any(k in c for k in PARALYTIC_KW):
        return "paralytic"
    if any(k in c for k in DEX_KW):
        return "dexmedetomidine"
    if any(k in c for k in SEDATIVE_ANALGESIC_KW):
        return "sat_relevant"
    return "other"


def _print_vc(con, scan: str, col: str, limit: int = 40) -> pd.DataFrame:
    df = con.execute(
        f"SELECT {col} AS value, COUNT(*) AS n FROM {scan} "
        f"GROUP BY {col} ORDER BY n DESC LIMIT {limit}"
    ).fetchdf()
    with pd.option_context("display.max_rows", limit, "display.width", 120):
        log.info("\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Section 1+2+3 — medication_admin_continuous: convention, inventory, cadence
# ---------------------------------------------------------------------------
def probe_med_continuous(con, scan: str, summary: dict) -> dict:
    cols = _columns(con, scan)
    log.info("medication_admin_continuous columns: %s", cols)

    cat_col = next((c for c in ("med_category", "med_group", "med_name") if c in cols), None)
    dose_col = next((c for c in ("med_dose", "med_dose_continuous", "dose") if c in cols), None)
    unit_col = next((c for c in ("med_dose_unit", "dose_unit") if c in cols), None)
    dttm_col = next((c for c in ("admin_dttm", "recorded_dttm", "med_admin_dttm") if c in cols), None)
    action_col = next((c for c in ("mar_action_category", "mar_action_name", "mar_action") if c in cols), None)
    hid = "hospitalization_id" if "hospitalization_id" in cols else None

    n_rows = con.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()[0]
    log.info("total rows: %s", f"{n_rows:,}")
    summary["mac_total_rows"] = int(n_rows)
    summary["mac_cat_col"] = cat_col
    summary["mac_dose_col"] = dose_col
    summary["mac_dttm_col"] = dttm_col
    summary["mac_action_col"] = action_col

    if cat_col is None:
        log.warning("no category column found; cannot classify drugs")
        return summary

    log.info("\n--- %s value_counts (top 40) ---", cat_col)
    vc = _print_vc(con, scan, cat_col)
    vc["bucket"] = vc["value"].map(_classify)
    log.info("\n--- inferred SAT buckets (keyword-classified; CONFIRM before writing config) ---")
    log.info("\n%s", vc.groupby("bucket")["n"].agg(["count", "sum"]).to_string())
    # Persist the inventory for config population.
    inv_path = FINAL_DIR / "probe_med_category_inventory.csv"
    vc.to_csv(inv_path, index=False)
    log.info("wrote %s", inv_path.relative_to(PROJECT_ROOT))

    if unit_col:
        log.info("\n--- %s value_counts (top 40) ---", unit_col)
        _print_vc(con, scan, unit_col)

    if action_col:
        log.info("\n--- %s value_counts (top 40) [explicit stop/pause markers?] ---", action_col)
        _print_vc(con, scan, action_col)

    # Build a SAT-relevant category list for the convention probe.
    sat_cats = [v for v, b in zip(vc["value"], vc["bucket"]) if b == "sat_relevant"]
    sat_cats_sql = ", ".join("'" + str(c).replace("'", "''") + "'" for c in sat_cats) or "''"

    if dose_col:
        log.info("\n=== INFUSION CHARTING CONVENTION (the pivotal question) ===")
        conv = con.execute(
            f"""
            SELECT
              COUNT(*)                                              AS n,
              SUM(CASE WHEN {dose_col} IS NULL THEN 1 ELSE 0 END)   AS n_null_dose,
              SUM(CASE WHEN {dose_col} = 0 THEN 1 ELSE 0 END)       AS n_zero_dose,
              SUM(CASE WHEN {dose_col} > 0 THEN 1 ELSE 0 END)       AS n_pos_dose
            FROM {scan}
            WHERE lower(CAST({cat_col} AS VARCHAR)) IN ({sat_cats_sql})
            """
        ).fetchdf().iloc[0]
        n = max(int(conv["n"]), 1)
        log.info("SAT-relevant infusion rows: %s", f"{int(conv['n']):,}")
        log.info("  dose == 0  : %s (%.2f%%)  <- explicit rate-0 rows charted?",
                 f"{int(conv['n_zero_dose']):,}", 100 * conv["n_zero_dose"] / n)
        log.info("  dose NULL  : %s (%.2f%%)", f"{int(conv['n_null_dose']):,}", 100 * conv["n_null_dose"] / n)
        log.info("  dose > 0   : %s (%.2f%%)", f"{int(conv['n_pos_dose']):,}", 100 * conv["n_pos_dose"] / n)
        summary["mac_sat_rows"] = int(conv["n"])
        summary["mac_sat_zero_dose_pct"] = round(100 * conv["n_zero_dose"] / n, 3)
        summary["mac_sat_null_dose_pct"] = round(100 * conv["n_null_dose"] / n, 3)
        verdict = ("CHARTS ZERO ROWS (holds directly observable)"
                   if conv["n_zero_dose"] / n > 0.005
                   else "DOES NOT chart zeros (holds must be GAP-inferred — ambiguous)")
        log.info("  -> convention verdict: %s", verdict)

        # Per-drug dose distribution (helps set plausibility + see units).
        log.info("\n--- per-drug dose distribution (SAT-relevant, dose>0) ---")
        dist = con.execute(
            f"""
            SELECT lower(CAST({cat_col} AS VARCHAR)) AS drug,
                   COUNT(*) AS n_pos,
                   median({dose_col}) AS med,
                   quantile_cont({dose_col}, 0.05) AS p05,
                   quantile_cont({dose_col}, 0.95) AS p95
            FROM {scan}
            WHERE lower(CAST({cat_col} AS VARCHAR)) IN ({sat_cats_sql}) AND {dose_col} > 0
            GROUP BY 1 ORDER BY n_pos DESC
            """
        ).fetchdf()
        log.info("\n%s", dist.to_string(index=False))

    # Section 2 — charting cadence (median minutes between consecutive records
    # within one hospitalization's infusion of one drug).
    if dttm_col and hid:
        log.info("\n=== CHARTING CADENCE (median minutes between consecutive infusion records) ===")
        cad = con.execute(
            f"""
            WITH ordered AS (
              SELECT {hid} AS hid, lower(CAST({cat_col} AS VARCHAR)) AS drug,
                     CAST({dttm_col} AS TIMESTAMP) AS t,
                     LAG(CAST({dttm_col} AS TIMESTAMP)) OVER (
                       PARTITION BY {hid}, {cat_col} ORDER BY {dttm_col}) AS prev_t
              FROM {scan}
              WHERE lower(CAST({cat_col} AS VARCHAR)) IN ({sat_cats_sql})
            )
            SELECT median(date_diff('minute', prev_t, t)) AS median_gap_min,
                   quantile_cont(date_diff('minute', prev_t, t), 0.90) AS p90_gap_min,
                   COUNT(*) AS n_intervals
            FROM ordered WHERE prev_t IS NOT NULL AND t > prev_t
            """
        ).fetchdf().iloc[0]
        log.info("median gap = %.1f min | p90 gap = %.1f min | intervals = %s",
                 cad["median_gap_min"], cad["p90_gap_min"], f"{int(cad['n_intervals']):,}")
        summary["mac_median_gap_min"] = round(float(cad["median_gap_min"]), 2)
        summary["mac_p90_gap_min"] = round(float(cad["p90_gap_min"]), 2)
    return summary


# ---------------------------------------------------------------------------
# Section 4 — denominator sizing (approximate; no waterfall)
# ---------------------------------------------------------------------------
def probe_denominator(con, paths: dict, cfg: dict, summary: dict) -> dict:
    tz = cfg["timezone"]
    log.info("\n=== DENOMINATOR SIZING (approximate; raw device_category, no waterfall) ===")

    # ICU patient-days from adt.
    adt = paths.get("adt")
    if adt:
        cols = _columns(con, _scan(adt))
        loc = next((c for c in ("location_category", "location_type") if c in cols), None)
        din = next((c for c in ("in_dttm", "intime", "location_in_dttm") if c in cols), None)
        if loc and din and "hospitalization_id" in cols:
            icu_days = con.execute(
                f"""
                SELECT COUNT(*) AS icu_patient_days FROM (
                  SELECT DISTINCT hospitalization_id,
                         CAST(timezone('{tz}', CAST({din} AS TIMESTAMP)) AS DATE) AS d
                  FROM {_scan(adt)}
                  WHERE lower(CAST({loc} AS VARCHAR)) = 'icu'
                )
                """
            ).fetchone()[0]
            log.info("ICU patient-days (by adt in_dttm date): %s", f"{icu_days:,}")
            summary["icu_patient_days_approx"] = int(icu_days)

    # IMV hospitalization-days from raw respiratory_support.
    rs = paths.get("respiratory_support")
    if rs:
        cols = _columns(con, _scan(rs))
        dev = next((c for c in ("device_category", "device_name") if c in cols), None)
        rdt = next((c for c in ("recorded_dttm", "record_dttm") if c in cols), None)
        if dev and rdt and "hospitalization_id" in cols:
            imv_days = con.execute(
                f"""
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT hospitalization_id,
                         CAST(timezone('{tz}', CAST({rdt} AS TIMESTAMP)) AS DATE) AS d
                  FROM {_scan(rs)}
                  WHERE lower(CAST({dev} AS VARCHAR)) = 'imv'
                )
                """
            ).fetchone()[0]
            log.info("IMV hospitalization-days (raw device_category=='imv'): %s", f"{imv_days:,}")
            summary["imv_hosp_days_approx"] = int(imv_days)

    return summary


# ---------------------------------------------------------------------------
# Section 5 — RASS coverage (secondary lens)
# ---------------------------------------------------------------------------
def probe_rass(con, paths: dict, summary: dict) -> dict:
    log.info("\n=== RASS COVERAGE (secondary validation lens) ===")
    for tbl in ("vitals", "patient_assessments"):
        p = paths.get(tbl)
        if not p:
            log.info("%s: file not found", tbl)
            continue
        cols = _columns(con, _scan(p))
        catc = next((c for c in ("vital_category", "assessment_category",
                                 "assessment_name", "category") if c in cols), None)
        log.info("\n--- %s: %s value_counts (top 30) ---", tbl, catc)
        if catc:
            vc = con.execute(
                f"SELECT lower(CAST({catc} AS VARCHAR)) AS value, COUNT(*) AS n "
                f"FROM {_scan(p)} GROUP BY 1 ORDER BY n DESC LIMIT 30"
            ).fetchdf()
            log.info("\n%s", vc.to_string(index=False))
            rass = vc[vc["value"].str.contains("rass", na=False)]
            if not rass.empty:
                summary[f"{tbl}_rass_rows"] = int(rass["n"].sum())
                log.info("  RASS-like rows in %s: %s", tbl, f"{int(rass['n'].sum()):,}")
    return summary


def main() -> None:
    _ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOGS_DIR / "00_probe_documentation.log", mode="w")],
    )
    cfg = load_config(CONFIG_PATH)
    data_path = Path(cfg["primary_dataset"]["data_path"])
    fmt = cfg["primary_dataset"]["file_format"]
    log.info("site=%s  data_path=%s  format=%s", cfg.get("site"), data_path, fmt)

    tables = ["medication_admin_continuous", "adt", "respiratory_support",
              "vitals", "patient_assessments", "hospitalization", "patient"]
    paths = {t: find_table(data_path, t, fmt) for t in tables}
    for t, p in paths.items():
        log.info("  %-32s -> %s", t, (Path(p).name if p else "NOT FOUND"))

    con = duckdb.connect()
    summary: dict = {"site": cfg.get("site")}

    if paths.get("medication_admin_continuous"):
        log.info("\n" + "=" * 70 + "\nSECTION 1-3: medication_admin_continuous\n" + "=" * 70)
        summary = probe_med_continuous(con, _scan(paths["medication_admin_continuous"]), summary)
    else:
        log.error("medication_admin_continuous not found — cannot probe the SAT signal")

    log.info("\n" + "=" * 70 + "\nSECTION 4: denominator sizing\n" + "=" * 70)
    summary = probe_denominator(con, paths, cfg, summary)

    log.info("\n" + "=" * 70 + "\nSECTION 5: RASS coverage\n" + "=" * 70)
    summary = probe_rass(con, paths, summary)

    con.close()
    out = pd.DataFrame([summary])
    out_path = FINAL_DIR / "probe_summary.csv"
    out.to_csv(out_path, index=False)
    log.info("\nwrote %s", out_path.relative_to(PROJECT_ROOT))
    log.info("PROBE COMPLETE — review the convention verdict + inventory before locking config.")


if __name__ == "__main__":
    main()
