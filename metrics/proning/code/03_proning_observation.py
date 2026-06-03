"""Detect prone sessions from the CLIF position table.

Approach: bookended sessions. The CLIF `position` table emits both `prone`
and `not_prone` rows; a session opens at a `prone` row and closes at the
next `not_prone` row for the same hospitalization (or end-of-record). This
is the most clinically faithful reconstruction — it uses both states rather
than inferring duration from gap heuristics on prone-only rows.

Outputs:
    - output/intermediate/prone_sessions.parquet
        one row per prone session: hospitalization_id, session_start_dttm,
        session_end_dttm, duration_hours, ended_by ("not_prone" or "end_of_record").
    - output/intermediate/proning_observation.parquet
        one row per cohort hospitalization, columns:
            hospitalization_id, position_data_present (bool),
            any_prone (bool), n_sessions (int), total_prone_hours (float),
            any_session_>=_adherent (bool), longest_session_hours (float),
            first_prone_dttm (datetime), last_prone_end_dttm (datetime).

UChicago coverage caveat (2026-04-28 probe): only ~19 % of PROSEVA-eligible
hospitalizations have any position records at all. Patients with zero
position rows get `position_data_present = False`; how to treat them in
metrics (impute as not-proned vs exclude) is a 04_metrics.py decision.

No raw PHI to stdout — only counts and aggregates.
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

log = logging.getLogger("proning.observation")

PRONE = "prone"
NOT_PRONE = "not_prone"


def _load_cohort_module():
    path = CODE_DIR / "01_build_cohort.py"
    spec = importlib.util.spec_from_file_location("proning_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cohort_hosp_ids(cohort: pd.DataFrame) -> list[str]:
    """All hospitalization_ids in the cohort, including stitched encounter blocks."""
    ids: set[str] = set()
    for hids in cohort["hospitalization_ids"]:
        if isinstance(hids, (list, tuple, np.ndarray)):
            ids.update(str(h) for h in hids if h is not None and str(h) != "<NA>")
        elif hids is not None:
            ids.add(str(hids))
    # Plus the primary hospitalization_id column for safety
    ids.update(cohort["hospitalization_id"].astype(str).tolist())
    return sorted(ids)


def build_sessions(pos: pd.DataFrame) -> pd.DataFrame:
    """Bookend prone sessions per hospitalization.

    A `prone` row opens a session; the next chronological row in the same
    hospitalization that is `not_prone` closes it. If no closing `not_prone`
    row exists before end-of-record, the session is closed at the last
    `prone` row in that hospitalization (best estimate; flagged via
    `ended_by == "end_of_record"`).
    """
    if pos.empty:
        return pd.DataFrame(columns=[
            "hospitalization_id", "session_start_dttm", "session_end_dttm",
            "duration_hours", "ended_by",
        ])

    pos = pos.sort_values(["hospitalization_id", "recorded_dttm"]).reset_index(drop=True)
    pos["is_prone"] = (pos["position_category"] == PRONE).astype("int64")  # numpy-backed for cumsum
    pos["prev_is_prone"] = pos.groupby("hospitalization_id")["is_prone"].shift(1, fill_value=0)

    # Mark "session-start" rows: prone with previous row being non-prone (or first row of hosp)
    pos["is_session_start"] = (pos["is_prone"] == 1) & (pos["prev_is_prone"] == 0)
    # Mark "session-end" rows: not_prone with previous row being prone
    pos["is_session_end"] = (pos["position_category"] == NOT_PRONE) & (pos["prev_is_prone"] == 1)

    sessions = []
    for hid, g in pos.groupby("hospitalization_id", sort=False):
        starts = g.loc[g["is_session_start"], "recorded_dttm"].tolist()
        ends_idx = g.index[g["is_session_end"]].tolist()
        ends = [g.at[i, "recorded_dttm"] for i in ends_idx]

        for k, t_start in enumerate(starts):
            # Find the first end >= this start
            t_end = None
            for e in ends:
                if e >= t_start:
                    t_end = e
                    ended_by = "not_prone"
                    break
            if t_end is None:
                # No closing not_prone row — close at last prone row in this hosp
                last_prone = g.loc[g["position_category"] == PRONE, "recorded_dttm"]
                t_end = last_prone.max() if not last_prone.empty else t_start
                ended_by = "end_of_record"
            duration_h = (pd.Timestamp(t_end) - pd.Timestamp(t_start)).total_seconds() / 3600.0
            sessions.append({
                "hospitalization_id": hid,
                "session_start_dttm": t_start,
                "session_end_dttm": t_end,
                "duration_hours": duration_h,
                "ended_by": ended_by,
            })

    return pd.DataFrame(sessions)


def aggregate_per_hospitalization(
    cohort_hids: list[str],
    sessions: pd.DataFrame,
    pos: pd.DataFrame,
    adherent_hours: float,
) -> pd.DataFrame:
    """One row per cohort hospitalization_id."""
    pos_present = set(pos["hospitalization_id"].astype(str).unique()) if not pos.empty else set()
    by_hid = sessions.groupby("hospitalization_id") if not sessions.empty else None

    rows = []
    for hid in cohort_hids:
        present = hid in pos_present
        if by_hid is not None and hid in by_hid.groups:
            g = by_hid.get_group(hid)
            n_sessions = len(g)
            total_h = float(g["duration_hours"].sum())
            longest_h = float(g["duration_hours"].max())
            any_adherent = bool((g["duration_hours"] >= adherent_hours).any())
            first_start = g["session_start_dttm"].min()
            last_end = g["session_end_dttm"].max()
            any_prone = True
        else:
            n_sessions = 0
            total_h = 0.0
            longest_h = 0.0
            any_adherent = False
            first_start = pd.NaT
            last_end = pd.NaT
            any_prone = False
        rows.append({
            "hospitalization_id": hid,
            "position_data_present": present,
            "any_prone": any_prone,
            "n_sessions": n_sessions,
            "total_prone_hours": total_h,
            "longest_session_hours": longest_h,
            "any_session_adherent": any_adherent,
            "first_prone_dttm": first_start,
            "last_prone_end_dttm": last_end,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cohort_mod = _load_cohort_module()

    cohort_mod._ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cohort_mod.LOGS_DIR / "03_proning_observation.log", mode="w"),
        ],
    )

    cfg = cohort_mod.load_config(cohort_mod.CONFIG_PATH)
    adherent_hours = float(cfg["proning_observation"]["adherent_session_hours"])
    log.info("site=%s adherent threshold=%.1fh", cfg.get("site"), adherent_hours)

    cohort_path = cohort_mod.INTERMEDIATE_DIR / "cohort.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"{cohort_path} not found. Run code/01_build_cohort.py first.")
    cohort = pd.read_parquet(cohort_path)
    cohort_hids = _cohort_hosp_ids(cohort)
    log.info("cohort hospitalization_ids: %d", len(cohort_hids))

    co = cohort_mod.build_orchestrator(cfg)
    co.load_table("position", filters={"hospitalization_id": cohort_hids})
    pos = co.position.df.copy()
    log.info("loaded position rows for cohort: %d (across %d hospitalizations)",
             len(pos), pos["hospitalization_id"].nunique() if not pos.empty else 0)

    if pos.empty:
        log.warning("no position rows for any cohort hospitalization — emitting empty observation")
    else:
        # Cast hospitalization_id to concrete str (lessons.md trigger) and normalize category
        pos["hospitalization_id"] = pos["hospitalization_id"].astype(str)
        pos["position_category"] = pos["position_category"].astype("string").str.strip().str.lower()

        # Drop rows with values outside the schema enum (unexpected at UChicago, but defensive)
        unknown = ~pos["position_category"].isin([PRONE, NOT_PRONE])
        if int(unknown.sum()) > 0:
            log.warning("dropping %d rows with unexpected position_category values: %s",
                        int(unknown.sum()),
                        pos.loc[unknown, "position_category"].value_counts(dropna=False).to_dict())
            pos = pos.loc[~unknown].copy()

    sessions = build_sessions(pos)
    sessions_path = cohort_mod.INTERMEDIATE_DIR / "prone_sessions.parquet"
    sessions.to_parquet(sessions_path, index=False)
    log.info("built %d prone sessions (across %d hospitalizations); wrote %s",
             len(sessions),
             sessions["hospitalization_id"].nunique() if not sessions.empty else 0,
             sessions_path.relative_to(PROJECT_ROOT))

    obs = aggregate_per_hospitalization(cohort_hids, sessions, pos, adherent_hours)
    obs_path = cohort_mod.INTERMEDIATE_DIR / "proning_observation.parquet"
    obs.to_parquet(obs_path, index=False)

    n_total = len(obs)
    n_with_data = int(obs["position_data_present"].sum())
    n_proned = int(obs["any_prone"].sum())
    n_adherent = int(obs["any_session_adherent"].sum())
    log.info("aggregate (over %d cohort hospitalization_ids):", n_total)
    log.info("  position data present:        %5d (%.1f%%)", n_with_data, 100*n_with_data/n_total)
    log.info("  any prone session:            %5d (%.1f%%)", n_proned, 100*n_proned/n_total)
    log.info("  any session ≥ %.0fh (adherent):%5d (%.1f%%)", adherent_hours, n_adherent, 100*n_adherent/n_total)
    if not sessions.empty:
        log.info("  session duration distribution (hours):")
        for q in [0.25, 0.5, 0.75, 0.95]:
            log.info("    p%d: %.1f", int(q*100), sessions["duration_hours"].quantile(q))
        log.info("    max: %.1f", sessions["duration_hours"].max())
    log.info("wrote: %s", obs_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
