# proning — Multi-Site Federated QI for Prone Positioning in ARDS

## Research Question

Across CLIF consortium sites, **how often do ARDS patients who meet PROSEVA-strict eligibility for prone positioning actually receive it, and how quickly?** This is descriptive process / quality-improvement work, not a causal-effect study. Output is intended to feed a larger multi-metric ICU quality-of-care dashboard alongside other adherence indicators (LTV ventilation, sedation interruption, etc.).

### Specific aims
1. Build an ARDS cohort (Berlin moderate-severe phenotype).
2. Among ARDS patients, identify the subset who reach PROSEVA-strict proning eligibility (the QI denominator).
3. Among the eligible, describe (a) the proportion ever proned, (b) the proportion with an adherent (≥16 h) prone session, (c) time from eligibility to first prone session.
4. Emit a site-aggregable summary so other consortium sites can run this code on their CLIF data and contribute results without sharing row-level data.

---

## Definitions

### ARDS cohort (T₀ — first ARDS-qualifying ABG)
A patient enters at the first time point during a hospitalization where **all** of:
- age ≥ 18
- on invasive mechanical ventilation (`device_category == "imv"`)
- PEEP ≥ 5 cmH₂O
- FiO₂ ≥ 0.4
- PaO₂/FiO₂ ≤ 300 mmHg
- in an ICU location at the ABG time

One row per patient (earliest T₀ across encounter blocks). Berlin imaging criteria are not used (CLIF 2.1.0 lacks structured imaging); P/F ≤ 300 is the physiologic proxy for the Berlin oxygenation criterion.

### Proning eligibility (PROSEVA-strict)
**Persistent re-evaluation interpretation** (chosen 2026-04-28 — see `plans/experimental_approach.md` Change Log).

A cohort encounter is eligible iff:
1. There is a **first qualifying ABG** post-T₀ — meeting all of: `device_category == "imv"`, PEEP ≥ 5, FiO₂ ≥ 0.6, P/F ≤ 150. Call its time `T_first`.
2. There is a **second qualifying ABG** at or after `T_first + 12 h`. (Severity persisted past the stabilization window.)
3. The waterfall shows **no extubation event** in `(T_first, T_first + 12 h]`. (Patient was on IMV continuously through stabilization.)

`T_eligible = T_first + 12 h` — the clinical decision-point at which proning should be considered.

This matches PROSEVA's enrollment protocol: screen, allow 12-24 h stabilization, re-check severity. Intermediate ABGs that don't meet criteria (during weaning attempts, recovery) do not disqualify — only an ABG that confirms the patient is no longer severe at the 12 h mark would. At UChicago this yields **17.9 %** of the ARDS cohort eligible (1,854 / 10,369), in line with the consortium prior of 25-50 % "ever severe" (37.9 % here).

Thresholds live in `config/config.json` under `proning_eligibility` so sites can run sensitivity analyses without code changes.

Thresholds live in `config/config.json` under `proning_eligibility` so sites can run sensitivity analyses without code changes.

### Proning observation
Reconstructed from the CLIF `position` table. Sessions are contiguous runs of `position_category == "prone"`; gaps > `proning_observation.session_gap_minutes` (default 60 min) end a session. A session ≥ `proning_observation.adherent_session_hours` (default 16 h) counts as PROSEVA-adherent.

### QI metrics (Option C — bounded denominator)
The UChicago `position` table charts only proning episodes (only ~19 % of eligible have any position record; all of those were proned), so the adherence rate is reported as a **bound**, not a single number:
- **Process rate (tile headline):** ever-proned / all eligible = 350/1,854 = **18.9 %**.
- **Lower bound:** adherent ≥16 h / all eligible = 213/1,854 = **11.5 %** (no-data imputed not-adherent).
- **Upper bound:** adherent ≥16 h / charted subset = 213/350 = **60.9 %**.
- **Time-to-prone:** median (IQR) hours `T_eligible`→first prone among proned, plus a 7-day cumulative-incidence CDF over all eligible (non-proned treated as event-free, not censored).

Observation is joined to eligibility at the **encounter_block grain** — each eligible block aggregates over *all* of its `hospitalization_ids` (a prone session may be charted under any stitched id). `04_metrics.py` emits `metrics_site_summary.csv` (federation-shareable) and `tile_feed_proning.json` for the lpv bundle scorecard (see References).

### Unit & time-period slicing (dashboard filters)
The metrics are sliceable by **ICU unit** and **time granularity** so the dashboard can answer "in MICU during 2023, what fraction of eligible patients were proned / proned adherently?".
- **Time anchor = `T_eligible`** (the PROSEVA decision-point). Period keys: year `"YYYY"`, month `"YYYY-MM"`, ISO week `"YYYY-Www"`. Each granularity partitions the cohort exactly (per-unit and per-period eligible counts both sum to the total).
- **Unit = ICU `location_type` at `T_eligible`**, attached via a DuckDB range-join on the stitched `adt` intervals (mirrors `restrict_to_icu` in stage 01). T_eligible falling in a non-ICU gap → `"unknown"` (kept as a dashboard filter; folded into `__ALL__` for the tile feed).
- `04_metrics.py` writes `metrics_slices.parquet` (full counts) + `metrics_slices.csv` (shareable) and enriches the tile feed to grain `units:[__ALL__ + canonical ICU slugs]`, `periods:["all","month"]`.
- `05_dashboard.py` embeds the slices as a PHI-free `SLICES[unit][granularity][period]` JS object; **the 4 metric cards + a "Proning rate over time" trend chart react** to Unit × granularity × period (granularities All-time/Yearly/Monthly/Weekly). CONSORT, the time-to-prone CDF, and Table 1 stay **site-wide / all-time** (fine-slice stats are unreliable at this N). Slices with eligible < `reporting.small_cell_min_den` (default 10) are grayed, not hidden.
- Config: `config.json → reporting.{unit_attribution_anchor, small_cell_min_den}`.

---

## Project Structure

```
config/
  config.json                 # site-specific paths (gitignored)
  config_template.json        # committed; copy to config.json
  README.md
code/
  01_build_cohort.py          # ARDS screening → T₀ → cohort.parquet
  02_proning_eligibility.py   # PROSEVA-strict 12h sustained window
  03_proning_observation.py   # prone sessions from position table
  04_metrics.py               # QI rates (Option C bounds) + unit/period slices + site summary + scorecard tile feed
  05_dashboard.py             # interactive HTML dashboard (filterable cards + trend, CONSORT, Table 1, CDF)
output/
  intermediate/
    _cache/                   # checkpoints (abgs, waterfall, stitched, mapping)
    cohort.parquet            # ARDS cohort, one row per patient
    proning_eligibility.parquet
    prone_sessions.parquet
    proning_observation.parquet
    metrics_patient_level.parquet             # per-eligible detail + unit/period keys (keeps ids; not shared)
    metrics_slices.parquet                    # full counts per unit×granularity×period (dashboard embeds this)
  final/
    cohort_flow.csv           # CONSORT counts
    metrics_site_summary.csv  # consortium-aggregable (counts + rates only)
    metrics_slices.csv        # consortium-aggregable slices (unit×{all,year,month,week}, counts + rates)
    tile_feed_proning.json    # bundle scorecard tile feed (contract v1, PHI-free; grain units+month)
    proning_dashboard.html    # self-contained dashboard
    graphs/cohort_consort.png # standalone CONSORT funnel
  logs/                       # timestamped pipeline logs + per-step logs
plans/
  experimental_approach.md    # living design doc — append a Change Log bullet on every design change
run_pipeline.sh               # entry point — timestamped log → output/logs/pipeline_*.log
```

`output/` and `config/config.json` are gitignored.

---

## Data — CLIF Tables In Use

| Table | Purpose |
|---|---|
| `patient` | demographics, death_dttm |
| `hospitalization` | admission/discharge times, age_at_admission |
| `adt` | ICU localization (`location_category == "icu"`) |
| `respiratory_support` | vent mode, PEEP, FiO2, extubation events |
| `labs` | arterial PaO2 (`lab_category == "po2_arterial"`) |
| `position` | prone session detection (used in stage 03) |
| `patient_assessments` | reserved for awake-prone / RASS sub-analyses |

Primary dataset: **UChicago CLIF v2.1.0**.
Secondary (validation): **MIMIC-IV CLIF v1.1.0**.

The pipeline is **config-driven** — no hard-coded paths. Other CLIF consortium sites can run the pipeline against their data by copying `config/config_template.json` to `config/config.json` and editing the data path.

---

## ARDS Cohort — Design Notes

The cohort builder (`code/01_build_cohort.py`) is fully self-contained — no
cross-project imports — so any CLIF site can run it against its own data for
federated use.

Trial-specific machinery is deliberately **omitted** (irrelevant for descriptive QI):
- T_enroll = T₀ + 24 h enrichment window
- Pregnancy / influenza / DNR / chronic-steroid / ECMO exclusions
- Fuzzy-window enrollment ABG within ±6 h of T_enroll

What the builder **does** (the physiologic ARDS phenotype + its plumbing):
- Cache architecture (`output/intermediate/_cache/`, raw waterfall before normalization)
- `_normalize_waterfall` (FiO2 unit detection via p95, clip implausible FiO2 ∈ [0.15, 1.0] and PEEP ∈ [0, 40])
- `waterfall_cached` flow (filter resp_support to ABG-having hospitalizations, clifpy `process_resp_support_waterfall`, attach encounter_block, cache)
- `attach_vent_and_compute_pf` (merge_asof with 6 h tolerance, P/F = pao2 / fio2_set, plausibility band [10, 1000])
- `restrict_to_icu` (DuckDB range join on adt ICU intervals)

---

## Key Commands

```bash
# Activate env
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run full pipeline (stages 01 + 02 active; 03/04/05 are stubs)
./run_pipeline.sh

# Or run individual stages
python code/01_build_cohort.py            # --refresh / --refresh-waterfall available
python code/02_proning_eligibility.py
```

The waterfall step (~35 min on a fresh cache at UChicago) is checkpointed; re-running 01 reuses the cache for free.

---

## Implementation Notes

- **Timezone:** UChicago site → `US/Central`. All datetime columns are localized via `_coerce_dttm` (handles tz-naive cached parquets).
- **Site case normalization:** UChicago stores `device_category` lowercase (`"imv"`). All comparisons use lowercase constants. CLIF schema permissible-values are aspirational — see `.claude/lessons.md`.
- **`hospitalization_id` dtype:** Cast to `str` immediately after any clifpy load before merging — pyarrow-backed extension dtypes silently fail merges. See `.claude/lessons.md`.
- **Eligibility windowing:** The 12 h sustained-criteria window is checked against ABGs *and* the waterfall (no extubation event allowed in the window). See `code/02_proning_eligibility.py:compute_eligibility`.
- **`death_dttm` quirks at UChicago:** When `discharge_category == "Expired"` and `death_dttm` is missing or > `discharge_dttm`, fall back to `discharge_dttm`. (Relevant once `04_metrics.py` adds death-censoring.)
- **No raw PHI to stdout:** Only counts and aggregates. Raw data files are blocked by the global `~/.claude/hooks/protect-clif-data.sh` hook.

---

## References

- Guérin C, et al. Prone positioning in severe acute respiratory distress syndrome. *N Engl J Med* 2013;368:2159-68. (PROSEVA — the eligibility-and-duration definition this project targets.)
- CLIF 2.1.0 schema: see `clifpy` package.
- **Bundle scorecard tile contract:** `../../contract/tile_feed_contract.md` — `04_metrics.py` emits `output/final/tile_feed_proning.json` (schema_version 1, PHI-free, grain `units:[__ALL__ + canonical ICU slugs]`/`periods:["all","month"]`) per this contract. The bundle combiner (`scorecard/build_scorecard.py`) collects it for every metric listed in `config.json → metrics`, with grain-fallback for any (unit, period) the feed doesn't carry. Tile mapping: donut = ever-proned, segments = adherent bounds. (This project now lives at `metrics/proning/` in the `clif-ventilator-qi-dashboard` monorepo.)
