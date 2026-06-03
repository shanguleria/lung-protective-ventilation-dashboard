# sat_dashboard — Descriptive QI for Spontaneous Awakening Trial (SAT) Adherence

## Research Question

**Across eligible ventilated ICU patient-days, how often is a Spontaneous Awakening Trial (SAT) —
daily interruption of sedation — actually performed?** This is descriptive process / quality-
improvement epidemiology, **not** a causal-effect study. There is **no** outcome modeling (mortality,
IMV duration, ventilator-free days). The output is one tile on the lpv-owned ICU ventilator/liberation
**bundle scorecard**, alongside LPV, ARDS proning, SBT, and mobilization.

> **Not the same project as `/CLIF/early_sat`.** That is a *causal* study (does SAT-within-48h shorten
> IMV duration?), a different unit of analysis. Its `CLAUDE.md` is a reference for SAT clinical logic and
> the CLIF table list only — this project does not build on it.

### Specific aims
1. Build the denominator: **ventilated ICU patient-days on continuous sedation** (the eligible SAT
   opportunities).
2. Detect the numerator: **patient-days on which a SAT was performed** — an interval where all
   SAT-relevant continuous infusions are held to rate 0.
3. Describe the **SAT adherence rate** (% of eligible patient-days with a SAT), sliced by ICU unit and
   time period, with an explicit documentation-coverage caveat (bounds, not a single number).
4. Describe the **Kress et al. 2000 "half-dose restart" benchmark**: among SATs that resume sedation,
   the distribution of resumed-vs-prior dose and the share restarted at ≤ 50% of the prior dose.
5. Emit a PHI-free, site-aggregable tile feed so other consortium sites can contribute without sharing
   row-level data.

---

## Definitions

### Unit of analysis — ventilated ICU patient-DAY
A calendar day (in `US/Central`) on which an encounter_block is on IMV **and** in an ICU location.
Patient-level ("ever-SAT") is reported as a secondary segment only.

### SAT-relevant medications (held to rate 0 for a SAT)
The set of continuous infusions that must be at **rate 0** for a SAT to count: **propofol, midazolam,
other benzodiazepine infusions, fentanyl, other opioid infusions** (and ketamine). **Dexmedetomidine
may continue** — its presence does *not* block a SAT and it is **not** in this set. Exact `med_category`
values are confirmed at this site by `code/00_probe_documentation.py` and live in
`config.json → sat_medications` (CLIF permissible-values are aspirational; site casing varies).

### Eligible SAT-opportunity day (denominator)
A ventilated ICU patient-day on which the patient is receiving ≥ 1 **SAT-relevant** continuous
infusion. Continuous **neuromuscular-blockade (paralytic) days are excluded** (the one SAT safety-screen
exclusion CLIF can observe). Other classic safety-screen exclusions (active seizures, alcohol
withdrawal, myocardial ischemia, raised ICP) are **not reliably codable in CLIF** → this is **crude
eligibility, not full safety-screen-passed eligibility** (surfaced as a dashboard + tile caveat).
**Dex-only days** (only dexmedetomidine, nothing to interrupt): handling set in
`config.json → sat_eligibility.dex_only_days`.

### SAT performed (numerator)
On an eligible day, an interval where **all** SAT-relevant infusions are simultaneously at **rate 0**
(held) for ≥ `sat_observation.hold_min_minutes`, while the patient remains ventilated and SAT-relevant
sedation was present earlier that day. Dexmedetomidine running is ignored. Whether a zero is read from
*charted* rate-0 rows or *gap-inferred*, and whether resumption is required, are configured in
`config.json → sat_observation` (locked from the probe — see the **denominator trap** below).

### Documentation handling (the proning denominator trap)
The pivotal question is the **continuous-infusion charting convention**: does
`medication_admin_continuous` chart explicit rate-0 rows when an infusion is paused, or does charting
simply stop (a gap)? A gap is ambiguous — SAT-hold-then-resume vs permanent discontinuation
(de-escalation / extubation / death) vs charting gap. `00_probe_documentation.py` quantifies this
**before** the metric is locked, and the rate is reported as a **bound** (à la proning's Option C), not
a single number. Mode is set in `config.json → reporting.denominator_mode`
(`documented` | `impute` | `bounds`). The caveat is carried in the tile feed's `note`.

### Kress dose-resumption ratio (descriptive add-on)
For each SAT that **resumes** sedation (a SAT-relevant infusion restarts after the hold), capture the
per-`med_category` steady-state **pre-hold** and **post-resume** rate and the unitless ratio
`resumed / pre_hold` (like-to-like on the same drug; cross-drug switches flagged, not ratioed). Report
the distribution (median, IQR) and the **% restarted at ≤ 50% of prior dose** (Kress et al. 2000). This
is a dashboard figure + site-summary rows only — it stays **off** the headline tile.

### Unit & time-period slicing (dashboard filters)
- **Unit** = ICU `location_type` of the patient-day's ICU interval (DuckDB range-join on stitched
  `adt`); a day with no ICU `location_type` → `"unknown"` (folded into `__ALL__` for the tile).
- **Time period** keys by the patient-day's calendar date: month `"YYYY-MM"`, ISO week `"YYYY-Www"`.
  Each granularity partitions the patient-days exactly.
- Tile-feed grain target: `units:[__ALL__ + canonical ICU slugs present]`, `periods:["all","month"]`
  (fall back to `["__ALL__"]`/`["all"]` if N is sparse). Slices below
  `reporting.small_cell_min_den` are grayed, not hidden.

---

## Project Structure

```
config.json                 # site-specific paths + SAT knobs (gitignored)
config.example.json         # committed template; copy to config.json
code/
  00_probe_documentation.py # quantify-first coverage probe (aggregates only; run on demand)
  01_build_cohort.py        # ventilated-ICU patient-DAYS + SAT-relevant infusion load
  02_sat_eligibility.py     # eligible SAT-opportunity days (denominator rules)
  03_sat_observation.py     # SAT hold detection + Kress dose-resumption ratio
  04_metrics.py             # rates/bounds + unit/period slices + site summary + tile feed
  05_dashboard.py           # interactive maroon/cream HTML dashboard
output/
  intermediate/
    _cache/                 # checkpoints (stitched adt/hosp/mapping, resp waterfall, infusions)
    cohort.parquet          # ventilated-ICU patient-days
    sat_eligibility.parquet
    sat_observation.parquet
    metrics_patient_day_level.parquet   # per-day detail + unit/period keys (keeps ids; not shared)
    metrics_slices.parquet
  final/
    cohort_flow.csv         # CONSORT-like counts
    metrics_site_summary.csv# consortium-aggregable (counts + rates only)
    metrics_slices.csv      # consortium-aggregable slices
    tile_feed_sat.json      # bundle scorecard tile feed (contract v1, PHI-free)
    sat_dashboard.html      # self-contained dashboard
    graphs/
  logs/
run_pipeline.sh             # entry point — timestamped log → output/logs/pipeline_*.log
```

`output/` and `config.json` are gitignored.

---

## Data — CLIF Tables In Use

| Table | Purpose |
|---|---|
| `patient` | demographics, death_dttm |
| `hospitalization` | admission/discharge times, age_at_admission |
| `adt` | ICU localization (`location_category == "icu"`), unit (`location_type`) |
| `respiratory_support` | IMV window (waterfall `device_category == "imv"`), extubation |
| `medication_admin_continuous` | sedative/analgesic infusions (the SAT signal) + dexmedetomidine + paralytics |
| `medication_admin_intermittent` | PRN/bolus sedatives (context only; not the primary signal) |
| `vitals` | RASS (sedation depth) — secondary validation lens |
| `patient_assessments` | RASS if charted here rather than in vitals — secondary lens |

Primary dataset: **UChicago CLIF v2.1.0**. Secondary (validation): **MIMIC-IV CLIF v1.1.0**.
The pipeline is **config-driven** — no hard-coded paths. Other sites copy `config.example.json` to
`config.json` and edit the data path + (after running the probe) the `sat_medications` categories.

---

## Reuse from `proning`

The cohort/loader machinery is **adapted from** (not imported from)
`/Users/shanguleria/Desktop/Research/CLIF/proning/code/01_build_cohort.py` so this project is
self-contained for federation: `build_orchestrator`, `_coerce_dttm`, `stitch_cached`,
`waterfall_cached` + `_normalize_waterfall`, `restrict_to_icu` (DuckDB ICU range-join), the cache
architecture. The metrics/slice/tile machinery is adapted from `proning/code/04_metrics.py`
(`build_slice_cells`, `_assert_slice_integrity`, `attach_unit_and_periods`, `build_tile_feed`,
`_assert_phi_free`). **Dropped** from proning (irrelevant here): ARDS phenotype, P/F waterfall to
ABG-having hospitalizations, PROSEVA eligibility, the position table. The waterfall here is scoped to
hospitalizations with a SAT-relevant infusion + an ICU stay.

---

## Key Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt

python code/00_probe_documentation.py     # coverage probe (run first, on demand)
./run_pipeline.sh                          # full pipeline (01–05)

python code/01_build_cohort.py             # --refresh / --refresh-waterfall available
```

The waterfall step is checkpointed in `output/intermediate/_cache/`; re-running reuses the cache.

---

## Implementation Notes

- **Timezone:** UChicago → `US/Central`. clifpy orchestrator is constructed with this timezone; all
  datetime columns pass through `_coerce_dttm` (handles tz-naive cached parquets). (early_sat's
  "CLIF stores UTC" note does not apply — we use clifpy's localized loads, as proning does.)
- **Site case normalization:** UChicago stores `device_category` / `med_category` lowercase. All
  comparisons lowercase both sides; `config.json` category lists are matched case-insensitively.
- **`hospitalization_id` dtype:** cast to `str` immediately after every clifpy load before merging —
  pyarrow-backed extension dtypes silently fail merges.
- **Day bucketing:** a vent-ICU stay spanning midnight yields multiple patient-day rows; eligibility
  and SAT detection are per calendar day in `US/Central`.
- **No raw PHI to stdout:** only counts and aggregates. Raw files are blocked by the global
  `~/.claude/hooks/protect-clif-data.sh` hook. The tile feed is re-checked for PHI substrings at build
  time and the script aborts if any appear.

---

## References

- **Kress JP, et al.** Daily interruption of sedative infusions in critically ill patients undergoing
  mechanical ventilation. *N Engl J Med* 2000;342:1471-77. (The "restart at half the prior dose"
  benchmark and the canonical daily-interruption protocol.)
- **Girard TD, et al.** (Wake Up and Breathe / ABC trial) *Lancet* 2008;371:126-34. (SAT safety screen
  and pass/fail criteria — context for the un-codable safety-screen caveat.)
- **Bundle scorecard tile contract:** `/Users/shanguleria/Desktop/Research/CLIF/lpv/plans/02_scorecard_tile_contract.md`
  — `04_metrics.py` emits `output/final/tile_feed_sat.json` (schema_version 1, PHI-free) per this
  contract. The lpv `05_scorecard.py` ingests it via its `scorecard_tiles` config list.
- **Sibling QI vertical (structure cloned):** `/Users/shanguleria/Desktop/Research/CLIF/proning`.
- **Dashboard design language:** `~/.claude/templates/dashboard_design_guide.md` (CLIF maroon-cream).
- CLIF 2.1.0 schema: see the `clifpy` package.
