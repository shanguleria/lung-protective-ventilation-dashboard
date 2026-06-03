"""Shared infusion-timeline engine for the SAT QI vertical.

Reconstructs, from `medication_admin_continuous`, when each continuous infusion
is ACTIVE (running at dose > 0) vs OFF, so that both stages can agree on:
  - eligibility (02): is any SAT-relevant infusion active during a vent-ICU day?
  - SAT detection (03): is there an OFF gap (all SAT-relevant infusions at rate 0)
    of >= threshold within a vent-ICU day, after sedation had been running?

Charting convention at UChicago (confirmed by 00_probe_documentation.py): dose==0
rows ARE charted (4.88% of SAT-relevant rows) and `mar_action_category` carries
explicit start/stop markers, so holds are directly observable — we do not have to
gap-infer. A record's value holds until the next record of the same drug
(consecutive-row step function), the trailing record capped.

Aggregates only; no PHI printed here (this module does no I/O to stdout).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

TRAILING_CAP_H = 24          # cap the final (open-ended) record of a drug run
STOP_ACTION = "stop"         # mar_action_category value that forces "off"


def coerce_dttm(series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        s = s.dt.tz_convert(tz)
    return s


def build_drug_segments(inf: pd.DataFrame, cats: set[str], tz: str,
                        trailing_cap_h: int = TRAILING_CAP_H) -> pd.DataFrame:
    """Per (encounter_block, med_category) consecutive-row step function.

    Returns one row per segment: [encounter_block, med_category, seg_start,
    seg_end, dose, active]. `active` = dose > 0 AND mar_action != stop.
    """
    df = inf[inf["med_category"].isin(cats)].copy()
    if df.empty:
        return pd.DataFrame(columns=["encounter_block", "med_category", "seg_start",
                                     "seg_end", "dose", "active"])
    df["encounter_block"] = df["encounter_block"].astype(str)
    df["admin_dttm"] = coerce_dttm(df["admin_dttm"], tz)
    df = df.dropna(subset=["admin_dttm"]).sort_values(
        ["encounter_block", "med_category", "admin_dttm"])

    df["seg_end"] = df.groupby(["encounter_block", "med_category"])["admin_dttm"].shift(-1)
    cap = df["admin_dttm"] + timedelta(hours=trailing_cap_h)
    df["seg_end"] = df["seg_end"].fillna(cap)

    dose = pd.to_numeric(df["med_dose"], errors="coerce") if "med_dose" in df.columns else 0.0
    is_stop = (df["mar_action_category"] == STOP_ACTION) if "mar_action_category" in df.columns else False
    df["dose"] = dose
    df["active"] = (dose > 0) & (~is_stop if hasattr(is_stop, "__len__") else ~bool(is_stop))

    seg = df.rename(columns={"admin_dttm": "seg_start"})[
        ["encounter_block", "med_category", "seg_start", "seg_end", "dose", "active"]]
    seg = seg[seg["seg_end"] > seg["seg_start"]].reset_index(drop=True)
    return seg


def active_union(segments: pd.DataFrame) -> pd.DataFrame:
    """Union of ACTIVE segments per encounter_block (merge overlapping/adjacent).

    Returns [encounter_block, start, end] — the intervals during which AT LEAST
    ONE drug in the segment set is running.
    """
    act = segments[segments["active"]][["encounter_block", "seg_start", "seg_end"]].copy()
    if act.empty:
        return pd.DataFrame(columns=["encounter_block", "start", "end"])
    act = act.sort_values(["encounter_block", "seg_start"])
    out = []
    for blk, g in act.groupby("encounter_block", sort=False):
        cur_s = cur_e = None
        for s, e in zip(g["seg_start"], g["seg_end"]):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:                # overlap or touch -> extend
                if e > cur_e:
                    cur_e = e
            else:
                out.append((blk, cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((blk, cur_s, cur_e))
    return pd.DataFrame(out, columns=["encounter_block", "start", "end"])


def clip_intervals_to_window(intervals: pd.DataFrame, win_start, win_end) -> list[tuple]:
    """Clip [start,end] intervals (already for one block) to a [win_start,win_end]
    window; return sorted, non-empty (start, end) tuples."""
    out = []
    for s, e in zip(intervals["start"], intervals["end"]):
        cs, ce = max(s, win_start), min(e, win_end)
        if ce > cs:
            out.append((cs, ce))
    out.sort()
    return out


def off_gaps_in_window(active_clipped: list[tuple], win_start, win_end):
    """Given active intervals clipped to a window, return:
      - first_active_start (or None if never active in window)
      - list of OFF gaps (g_start, g_end, resumed) that occur AT/AFTER the first
        active start (i.e. true interruptions, not pre-sedation lead-in).
    A gap BETWEEN two active blocks has resumed=True (sedation restarted). The
    trailing region from the last active end to win_end (sedation stopped while
    still ventilated, never restarted) is returned with resumed=False."""
    if not active_clipped:
        return None, []
    first_active_start = active_clipped[0][0]
    gaps = []
    cursor = active_clipped[0][1]           # end of first active block
    for s, e in active_clipped[1:]:
        if s > cursor:
            gaps.append((cursor, s, True))  # off between two active blocks -> resumed
        cursor = max(cursor, e)
    if win_end > cursor:                    # trailing off until window end -> not resumed
        gaps.append((cursor, win_end, False))
    return first_active_start, gaps
