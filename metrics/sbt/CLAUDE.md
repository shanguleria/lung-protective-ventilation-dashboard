# sbt — Descriptive QI for Spontaneous Breathing Trial (SBT) Delivery

## Research Question

**Across eligible ventilated ICU patient-days, how often is a Spontaneous Breathing Trial (SBT) —
a transition from controlled ventilation to a spontaneous support mode — actually delivered?** This is
descriptive process / quality-improvement epidemiology, **not** a causal study. There is **no** outcome
modeling (extubation success, IMV duration, ventilator-free days) in v1. The output is one tile on the
ICU ventilator/liberation **bundle scorecard**, alongside LPV, ARDS proning, SAT, and mobilization. SBT
is the breathing-trial half of the ABCDE liberation bundle and the natural pair to SAT.

The clinical definitions follow **Jain et al.** (Critical Care Medicine; DOI
10.1097/01.ccm.0001184980.06827.9a), who identified SBTs from CLIF data.

### Specific aims
1. Build the denominator: **eligible ventilated-ICU patient-days** — ≥12h controlled ventilation
   accrued and a ≥2h window of stable physiology, excluding tracheostomized days.
2. Detect the numerator: **patient-days on which an SBT was delivered** — a controlled→support mode
   transition sustained ≥2 min meeting the PEEP/CPAP criteria.
3. Describe the **SBT delivery rate** (% of eligible patient-days with an SBT), sliced by ICU unit and
   time period, with explicit data-quality caveats (charting cadence, CPAP-from-PEEP, sedation scope).
4. Emit a PHI-free, site-aggregable tile feed so other consortium sites can contribute without sharing
   row-level data.

---

## Definitions

### Unit of analysis — ventilated ICU patient-DAY
A calendar day (in `US/Central`) on which an encounter_block is on IMV **and** in an ICU location.
Identical universe to the SAT vertical. Patient-level framing is reported as a secondary segment only.

### Eligible SBT-opportunity day (denominator) — Jain et al.
A ventilated-ICU patient-day is eligible iff **all** hold:
- **≥12h of controlled ventilation accrued** before the day's opportunity (cumulative-since-intubation,
  counted from hourly waterfall scaffold rows in a controlled mode; not reset by a transient support
  episode). Controlled modes (config `sbt_modes.controlled_modes`): *assist control-volume control,
  pressure control, pressure-regulated volume control, SIMV*.
- **A ≥2h contiguous window of stable physiology** that day: **FiO2 ≤ 0.50** (fraction; the waterfall
  scales FiO2 to ≤1.0), **PEEP ≤ 8**, **SpO2 ≥ 88%**, and **norepinephrine-equivalent ≤ 0.2 mcg/kg/min**.
  FiO2/PEEP come from the waterfall (hourly, forward-filled); SpO2 from `vitals` (`spo2`, merge_asof
  backward ≤1h); NE-equivalent from `medication_admin_continuous` vasopressors (see below). The four
  signals are resampled onto the hourly scaffold grid; a run of ≥2 consecutive stable hours qualifies.
- **Not tracheostomized that day.** Trach patients are excluded from numerator **and** denominator (the
  waterfall's forward-filled `tracheostomy` flag). Continuous-spontaneous / never-controlled patients
  are excluded automatically by the ≥12h-controlled gate.

A day whose stability is **un-assessable** (no scaffold hour has all four signals present) is reported
as a separate `not_assessable` bound, excluded from the rate denominator (mirrors LPV's `not_assessable`).

### Norepinephrine-equivalent (NEE)
clifpy has **no** NEE helper. We standardize each vasopressor dose to mcg/min via
`clifpy.utils.unit_converter.standardize_dose_to_base_units` (which ASOF-merges `weight_kg` from
`vitals`), divide by weight → mcg/kg/min, then apply standard published factors (Goradia 2021 /
Kotani 2023, all in `config.json → sbt_vasopressors.ne_equivalent_factors`): **norepinephrine 1,
epinephrine 1, phenylephrine /10, dopamine /100, vasopressin ×2.5** (vasopressin in U/min, NOT
weight-normalized). Inotropes (dobutamine/milrinone/isoproterenol) and angiotensin default to 0. The
NEE timeline is the sum of concurrent per-drug step functions, sampled onto scaffold hours. A running
pressor with missing weight makes that hour **un-assessable** (not silently 0). Swap to Jain's exact
factors in config if obtained.

### SBT delivered (numerator) — transition-only
On an eligible day, a **controlled→support mode transition** sustained ≥ `support_min_minutes` (default
2), where the support episode is `pressure support/cpap` with **PEEP ≤ 8** (pressure-support arm) or
**PEEP ≤ 5** (CPAP arm). Detection runs on the **native-resolution** waterfall rows (not the hourly
scaffold) so sub-hourly trials are visible where charted. **Transition-only** (user decision): a patient
parked on support all day with no controlled→support edge does **not** count.

### Documentation / data-quality caveats (carried in the tile `note`)
- **Charting cadence:** a ≥2-min support episode can be invisible at sites charting ventilator settings
  only hourly → delivery is a **lower bound**. `pct_native` (share of support readings from native vs
  scaffold rows) is surfaced as a coverage diagnostic.
- **CPAP pressure** is read from `peep_set` — CLIF `respiratory_support` has no dedicated CPAP column.
- **Sedation scope:** the cohort reuses the SAT vertical's warm waterfall cache (ICU ∩ SAT-sedation
  hospitalizations). Essentially all controlled-vent patients are sedated, so coverage is near-complete;
  a few never-sedated ventilated-ICU patients are not represented (`cohort.seed_cache_from` → null +
  `--refresh-waterfall` builds the full ICU∩IMV cohort).

### Unit & time-period slicing (dashboard filters)
- **Unit** = ICU `location_type` of the patient-day (attached per day in 01).
- **Time period** keys by the patient-day's calendar date: month `"YYYY-MM"` and ISO week
  `"YYYY-Www"` — both in the tile grain `["all","month","week"]` (weekly added 2026-06-04; SBT weekly
  denominators are robust, `__ALL__` median ~114 patient-days/week, so the scorecard answers week picks
  exactly). Slices below `reporting.small_cell_min_den` are grayed, not hidden.

---

## Project Structure

```
config.json                 # site paths + SBT knobs (gitignored)
config.example.json         # committed template; copy to config.json
code/
  sbt_vasopressors.py       # UTIL: norepinephrine-equivalent step-function engine
  sbt_detect.py             # UTIL: controlled-hour accrual, stability screen, trach flag, transitions
  00_probe_documentation.py # quantify-first coverage probe (aggregates only; run on demand)
  01_build_cohort.py        # ventilated-ICU patient-DAYS (reuses SAT's warm waterfall cache)
  02_sbt_eligibility.py     # >=12h controlled + >=2h stable + non-trach -> eligible/not/not_assessable
  03_sbt_observation.py     # controlled->support transition detection (numerator)
  04_metrics.py             # rates + unit/period slices + site summary + tile feed
  05_dashboard.py           # interactive maroon/cream HTML dashboard
output/
  intermediate/
    _cache/                 # checkpoints (seeded from SAT: stitched adt/hosp/mapping, resp waterfall)
    cohort.parquet          # ventilated-ICU patient-days
    sbt_eligibility.parquet
    sbt_observation.parquet
    metrics_patient_day_level.parquet   # per-day detail + unit/period keys (keeps ids; not shared)
    metrics_slices.parquet
  final/
    cohort_flow.csv         # CONSORT-like counts (vent-days -> non-trach -> eligible -> SBT)
    metrics_site_summary.csv# consortium-aggregable (counts + rates only)
    metrics_slices.csv      # consortium-aggregable slices
    tile_feed_sbt.json      # bundle scorecard tile feed (contract v1, PHI-free)
    sbt_dashboard.html      # self-contained dashboard
    graphs/
  logs/
run_pipeline.sh             # entry point — uses the bundle-root shared .venv
```

`output/` and `config.json` are gitignored.

---

## Data — CLIF Tables In Use

| Table | Purpose |
|---|---|
| `patient` | demographics, death_dttm |
| `hospitalization` | admission/discharge times, age_at_admission |
| `adt` | ICU localization (`location_category == "icu"`), unit (`location_type`) |
| `respiratory_support` | waterfall: device (`imv`), `mode_category` (controlled vs support), `fio2_set`, `peep_set`, `pressure_support_set`, `tracheostomy` |
| `medication_admin_continuous` | vasopressors (norepinephrine-equivalent for the stability screen) |
| `vitals` | `spo2` (stability screen) + `weight_kg` (vasopressor mcg/kg/min normalization) |

Primary dataset: **UChicago CLIF v2.1.0**. Secondary (validation): **MIMIC-IV CLIF v1.1.0**.
The pipeline is **config-driven** — no hard-coded paths. Other sites copy `config.example.json` to
`config.json`, edit the data path, and (after the probe) confirm the `sbt_modes` strings.

---

## Reuse

- **Cohort / loader / waterfall machinery** is adapted from the sibling **SAT** vertical
  (`../sat/code/01_build_cohort.py`): `build_orchestrator`, `_coerce_dttm`, `stitch_cached`,
  `waterfall_cached` + `_normalize_waterfall`, `build_imv_intervals`/`build_icu_intervals`/
  `intersect_imv_icu`/`expand_to_days`/`attach_demographics`. The cohort is the **same**
  ventilated-ICU patient-day universe, so SBT **seeds its `_cache/` from SAT's** (config
  `cohort.seed_cache_from`) to skip the ~35-min waterfall.
- **SpO2 load + merge_asof backward** pattern from `../lpv/code/02d_severity.py`.
- **Vasopressor unit/weight standardization** from clifpy `unit_converter.standardize_dose_to_base_units`.
- **Metrics / slice / tile-feed machinery** adapted from `../sat/code/04_metrics.py`
  (`build_slice_cells`, `_assert_slice_integrity`, `build_tile_feed`, `_assert_phi_free`).

---

## Key Commands

```bash
source ../../.venv/bin/activate          # shared monorepo venv at the bundle root
python code/00_probe_documentation.py    # coverage probe (run first, on demand)
./run_pipeline.sh                        # full pipeline (01–05)
python code/01_build_cohort.py           # --refresh / --refresh-waterfall available
```

The waterfall step is checkpointed in `output/intermediate/_cache/` (seeded from SAT on first run).

---

## Implementation Notes

- **Timezone:** UChicago → `US/Central`. All datetimes pass through `_coerce_dttm` (handles tz-naive
  cached parquets). The waterfall cache already stores `recorded_dttm` tz-aware.
- **Site case normalization:** UChicago stores `device_category` / `mode_category` / `med_category`
  lowercase; the waterfall lowercases device/mode. All comparisons lowercase both sides; config category
  lists are matched case-insensitively.
- **`hospitalization_id` dtype:** cast to `str` immediately after every clifpy load before merging.
- **Day bucketing:** a vent-ICU stay spanning midnight yields multiple patient-day rows; eligibility,
  stability windows, and transitions are attributed per `US/Central` calendar day. DST fall-back days are
  25h — never hardcode 24h.
- **Cohort-restriction discipline:** every per-day flag is LEFT-joined onto the cohort skeleton and
  `fillna(False)`; raw events are never grouped without restricting to the cohort `(block, day)` set.
- **No raw PHI to stdout:** only counts and aggregates. The tile feed is re-checked for PHI substrings
  at build time and the script aborts if any appear.

---

## References

- **Jain S, et al.** Identifying spontaneous breathing trials in the Common Longitudinal ICU data Format
  (CLIF). *Crit Care Med* (CCM); DOI 10.1097/01.ccm.0001184980.06827.9a. (SBT eligibility + delivery
  definitions used here.)
- **Goradia S, et al.** Vasopressor dose equivalence: a systematic review. *J Crit Care* 2021. /
  **Kotani Y, et al.** norepinephrine-equivalent conversions. (NEE conversion factors.)
- **Bundle scorecard tile contract:** `../../contract/tile_feed_contract.md` — `04_metrics.py` emits
  `output/final/tile_feed_sbt.json` (schema_version 1, PHI-free) per this contract; the combiner
  (`scorecard/build_scorecard.py`) collects it for every metric in `config.json → metrics`.
- **Sibling QI verticals (structure cloned):** `../sat`, `../lpv`, `../proning`.
- **Dashboard design language:** `~/.claude/templates/dashboard_design_guide.md` (CLIF maroon-cream).
- CLIF 2.1.0 schema: see the `clifpy` package.
