# LPV Adherence Dashboard — Design Spec

_Written 2026-06-01. Locks the analytic choices before the pipeline is built. Revisit and update as choices change — do NOT silently diverge._

---

## 1. Question

For a given **time window** (day / week / month) and a given **ICU unit**, what fraction of mechanically-ventilated patient-days at UChicago meet the composite lung-protective ventilation (LPV) bundle?

Goal: enable queries like "in March 2024, what % of IMV patient-days in the MICU were LPV-adherent each day?"

No outcome modeling. Descriptive only.

---

## 2. Cohort

- All adult (≥18 at hospitalization admit) hospitalizations at UChicago, CLIF v2.1.0.
- Restriction: must have ≥1 `respiratory_support` row with `device_category == 'IMV'` during the hospitalization.
- No diagnosis-based exclusions. ARDS is not required.
- Tracheostomy patients are included; trach status will be a stratifier but not an exclusion.

Per the `00_probe_summary.md`: 18,503 hospitalizations qualify (1.06M IMV rows over 2018–2024).

---

## 3. Unit of analysis

**Patient-day** = a (`hospitalization_id`, calendar day in `US/Central`) pair where the patient was both:
- on IMV at any point during the calendar day, **and**
- in an ICU (`adt.location_category == 'icu'`) at any point during the calendar day.

A hospitalization can contribute multiple patient-days. A patient-day with both ICU and non-ICU time still counts (assignment-by-most-time below).

---

## 4. LPV composite bundle (adherence definition)

A row meets the LPV bundle if **all three** simultaneously:

1. `tidal_volume / PBW_kg ≤ 6` mL/kg, where:
   - `tidal_volume`: `tidal_volume_obs` if non-null, else `tidal_volume_set` (obs-first fallback).
   - `PBW_kg`: Devine / ARDSnet formula, using patient `sex_category` and a representative height.
     - Male: PBW = 50 + 2.3 × (height_in − 60)
     - Female: PBW = 45.5 + 2.3 × (height_in − 60)
     - Unknown sex: patient-days excluded from adherence (not-assessable).
   - Height = **median of all `vitals.height_cm` readings for that hospitalization**, converted to inches. Held constant within a hospitalization.
2. `plateau_pressure_obs ≤ 30` cm H₂O.
3. `∆P = plateau_pressure_obs − PEEP ≤ 15` cm H₂O, where `PEEP = peep_obs` if non-null, else `peep_set` (obs-first fallback).

---

## 4a. Component-separated measures (decided 2026-06-01)

The single composite above is reported, but it is **not the primary framing**. Because the three components have very different missingness — Vt is densely charted, plateau is sparse (Q4), and ∆P inherits plateau's sparsity — a single composite forces the densely-measured Vt to share the scarce plateau's denominator, discarding ~17 pts of Vt-assessable patient-days and conflating "how protective" with "how completely documented."

So `02_features.py` computes **four measures, each on its OWN denominator**, all using the same time-weighted ≥80%/≥60-min rule (§6) but over that measure's own assessable minutes:

| Measure (`m`) | Assessable when (mode-eligible IMV interval has…) | Adherent when… |
|---|---|---|
| `vt` — Vt/kg | Vt(obs→set) **and** PBW present | `vt_per_pbw ≤ VT_MAX` |
| `plat` — Pplat | plateau present | `plateau ≤ 30` |
| `dp` — Pdriving | plateau **and** PEEP present | `∆P ≤ 15` |
| `comp` — Composite | all three present | all three pass (joint, **not** the product of marginals) |

- **`VT_MAX` is a dashboard slider** (default 6). Plateau ≤ 30 and ∆P ≤ 15 are **fixed** ("less negotiable"). The slider recomputes from `02_intervals.parquet`; plateau/∆P could become advanced toggles later.
- **Dashboard headline = the Vt measure** (largest denominator, the tunable one); Pplat, Pdriving, and the composite are supporting panels.
- `dp`'s assessable set ⊆ `plat`'s (adds the PEEP requirement); `comp`'s ⊆ both `vt` and `dp`. Verified as an invariant.
- **Empirical magnitudes** (default Vt≤6, UChicago; src `output/02_features_summary.json`): Vt 74.2% assessable / 24.6% adherent · Pplat 58.5% / **86.0%** (mostly documentation-limited, rarely clinically high) · Pdriving 58.4% / **48.2%** (the real pressure limiter) · Composite 57.1% / 11.3%.

---

## 5. Mode eligibility

LPV thresholds are not meaningful when the patient is breathing spontaneously without volume targeting. We restrict the **assessable denominator** to mode-eligible rows:

- **Eligible:** `Assist Control-Volume Control`, `Pressure-Regulated Volume Control`, `SIMV`, `Pressure Control`.
- **Not eligible (excluded from assessable time):** `Pressure Support/CPAP`, `Volume Support`, `Blow by`, `Other`, missing.

Per probe: ~67.7% of IMV rows have an eligible mode; 21.1% have missing mode (excluded — likely T-piece / wean / transition states).

---

## 6. Time-weighting (the headline rule)

Adherence is computed on a **time-weighted** basis within each patient-day:

### Step A — Snapshot construction
Each row of `respiratory_support` is treated as a settings snapshot at `recorded_dttm`. Within a snapshot, missing values are forward-filled from prior rows in the same hospitalization, subject to **carry-forward windows**:

| Variable | Carry-forward window | Rationale |
|---|---|---|
| `tidal_volume_obs`/`set`, `peep_obs`/`set`, `fio2_set` | 2 hours | Densely recorded (~hourly); short window prevents stale settings being attributed. |
| `plateau_pressure_obs` | **6 hours** | UChicago practice: per-hosp median gap = 4.25h, p75 = 5.2h, p90 = 12.9h (per `00b` probe). 6h covers the Q4-shift practice without extrapolating past clinical action points. |
| `mode_category`, `tracheostomy` | 6 hours | Slow-changing. |

If a variable's most recent value is older than its window, the snapshot has that variable as null.

### Step B — Interval assignment
Each snapshot covers the interval `[recorded_dttm, next_recorded_dttm in same hospitalization)`, **clipped to the calendar day boundary** in `US/Central`. Intervals beyond 24h with no following row are clipped at +1h (treats long silence as off-IMV / transport / extubation).

### Step C — Per-interval classification
Each interval is classified as one of:

- **assessable & in-bundle**: mode-eligible AND Vt + plateau + PEEP all present in snapshot AND all three thresholds met.
- **assessable & out-of-bundle**: mode-eligible AND Vt + plateau + PEEP all present, but ≥1 threshold violated.
- **not assessable**: mode-ineligible OR ≥1 of (Vt, plateau, PEEP) missing in snapshot OR `device_category != IMV`.

### Step D — Per-patient-day rollup
For each patient-day:

- `assessable_minutes` = total interval minutes classified assessable.
- `in_bundle_minutes` = total interval minutes classified assessable AND in-bundle.
- `bundle_fraction` = `in_bundle_minutes / assessable_minutes` (undefined if denominator 0).
- **Patient-day status:**
  - `not_assessable` if `assessable_minutes < 60`.
  - else `adherent` if `bundle_fraction ≥ 0.80`.
  - else `non_adherent`.

The 60-minute and 80% thresholds are recorded as **named parameters** in the pipeline so they can be flexed in sensitivity panels.

---

## 7. Unit assignment per patient-day

Multiple ICU stays from `adt` can overlap a patient-day. We assign the patient-day to the ICU `location_type` where the patient spent the **most ICU time during IMV** on that calendar day. Ties broken by alphabetical `location_type`.

UChicago ICU types (per `00b` probe, by ADT row count):
- `mixed_cardiothoracic_icu` (16,834)
- `medical_icu` (15,754)
- `surgical_icu` (10,012)
- `mixed_neuro_icu` (6,862)
- `general_icu` (6,041)
- `burn_icu` (2,594)

Patient-days with **no overlapping ICU stay** are excluded from the cohort (these may be ward-IMV anomalies or pre-ICU intubations; they will be quantified and reported separately).

---

## 8. Outlier handling

Before any feature computation:
- Use clifpy's built-in outlier ranges on `vitals` (`height_cm`) and `respiratory_support` (`tidal_volume_obs/set`, `plateau_pressure_obs`, `peep_obs/set`, `fio2_set`).
- For `vitals.height_cm`: exclude values < 100 cm or > 230 cm (per the WHO adult range; the probe showed min=3, max=226).
- Negative values for any pressure/volume/FiO2 → null.
- Per-row outliers become null (and therefore "not-assessable" for that row).

**Decided 2026-06-01 (cohort assessment, `01b`):**
- **Merged/long-span hospitalizations:** keep ALL hospitalizations in the cohort and rollups — do **not** drop or cap. `02_features.py` adds an `encounter_span_days` (and derived `long_span_flag`, threshold 200d) column at the hospitalization level so the dashboard can optionally filter. The assessment found exactly 1 such artifact (1,004-day span, 627 pd across 5 units; 0.6% of all patient-days); it is flagged, not removed.
- **PBW < 25 kg rejects:** keep current behavior — these (168 pd, heights 100–129 cm) stay PBW=NaN → **not-assessable** for the Vt component. No height-floor change; the 100 cm floor stands.

---

## 9. Output schema

Persisted artifacts (the pipeline splits cohort-building from classification, so the patient-day status lives in a `02_` file, not the `01_` skeleton):

1. **`output/01_cohort_patient_days.parquet`** _(by `01_cohort.py`)_ — the cohort skeleton, one row per (hospitalization_id, calendar_day) for the IMV-on-ICU cohort. Columns: hospitalization_id, patient_id, calendar_day, assigned_unit, sex_category, age_at_admit, height_cm, pbw_kg, n_imv_rows. **No adherence columns** — this file just defines the cohort.
2. **`output/02_patient_day_status.parquet`** _(by `02_features.py`)_ — one row per cohort patient-day, **wide with four component-separated measures** (`m` ∈ {`vt`, `plat`, `dp`, `comp`}; see §4a). Columns: hospitalization_id, patient_id, calendar_day, assigned_unit, sex_category, age_at_admit, height_cm, pbw_kg, encounter_span_days, long_span_flag, total_imv_minutes, mode_eligible_minutes, and for each measure `{m}_assessable_min, {m}_in_min, {m}_bundle_fraction, {m}_status` (status ∈ adherent/non_adherent/not_assessable). Vt + composite status use the **default cutoff `VT_MAX_DEFAULT = 6`**; the slider recomputes downstream from artifact #3. _(`encounter_span_days`/`long_span_flag` per the 2026-06-01 keep-all-flag-only decision; flag = encounter span > 200 d.)_
3. **`output/02_intervals.parquet`** _(by `02_features.py`)_ — one row per **mode-eligible IMV** interval-piece (NOT just all-three-present), the engine for the Vt-cutoff slider and the settings-distribution histograms. Columns: hospitalization_id, calendar_day, assigned_unit, duration_min, vt_per_pbw, plateau, driving_pressure, peep, fio2. **Component values are nullable and presence-encoding**: `vt_per_pbw` null ⇒ Vt not assessable for that piece; `plateau` null ⇒ plateau absent; `driving_pressure` null ⇒ plateau or PEEP absent. Histograms/recomputes over this table must be **time-weighted by `duration_min`**, with each measure filtered to its own non-null subset. _(Replaces the earlier composite-only `02_assessable_intervals.parquet`, deleted; rebuilt 2026-06-01 for component separation.)_
4. **`output/03_daily_unit_summary.parquet`** _(by `03_aggregate.py`)_ — **long** format: one row per (calendar_day, assigned_unit [incl. pooled `__ALL__`], `measure` ∈ {vt, plat, dp, comp}). Columns: n_total, n_adherent, n_non_adherent, n_not_assessable, assessable_rate, crude_rate. At default Vt = 6.
5. **`output/03_monthly_unit_summary.parquet`** _(by `03_aggregate.py`)_ — same long shape, `month` = `'YYYY-MM'`.
6. **`output/03_vt_grid_monthly.parquet`** _(by `03_aggregate.py`)_ — the Vt-slider engine: one row per (month, assigned_unit [incl. `__ALL__`], `vt_cutoff` ∈ VT_GRID, `measure` ∈ {vt, comp}). Columns: n_total, n_assessable, n_adherent, assessable_rate, crude_rate. Plateau/∆P fixed; only Vt varies.
7. **`output/03_vt_grid_daily_allunits.parquet`** _(by `03_aggregate.py`)_ — same as #6 but per (calendar_day, vt_cutoff, measure), **site-wide (`__ALL__`) only** (granular daily slider series).
8. **`output/04_lpv_dashboard.html`** _(by `04_dashboard.py`)_ — the single self-contained interactive dashboard (Plotly.js inlined; ~5.3 MB). Four tabs (Vt headline + slider · component breakdown · by-unit/temporal · distributions + Table 1). Two global controls: the **Vt-cutoff slider** (indexes the precomputed grid — no in-browser recompute) and a **time-period selector** (Year + Month dropdowns; All-time default). Period-filtering sums per-month counts in JS for headline/components/per-unit bars/histograms; Table 1 + cohort header swap among 92 precomputed per-period aggregated tables. **Trend granularity follows the period:** all-time = monthly full range; a year zooms the x-axis to that year (per-unit monthly lines); a single month switches to a daily site-wide line (from the `days`/`vtd`/`std` daily payload built off `03_vt_grid_daily_allunits` + `03_daily_unit_summary`). Panels lazy-draw on tab-show (Plotly renders 0×0 in `display:none`); deep-linkable via `#p-vt|p-comp|p-trend|p-dist`. Vendors `output/_vendor/plotly.min.js` from the `plotly` Python package (offline, no network). Embeds aggregated data + binned histograms only — no raw patient rows.

Both rates exposed:
- **Assessable adherence rate** = `n_adherent / (n_adherent + n_non_adherent)`
- **Crude adherence rate** = `n_adherent / n_patient_days_total` (treats not-assessable as not-adherent — more pessimistic but harder to game)

The dashboard reports both with clear labels; the editorially-preferred default is **assessable rate**, with the crude rate and the % assessable shown as supporting context.

---

## 10. Pipeline structure

```
code/
  01_cohort.py        # build (hosp_id × calendar_day) patient-day table; attach unit, sex, height, PBW
  02_features.py      # build snapshots + intervals; classify; persist patient-day status
  03_aggregate.py     # roll up to (day × unit) and (month × unit)
  04_dashboard.py     # HTML dashboard per global template
```

Each script is idempotent, reads from `output/` for prior step outputs, and writes its own artifact.

---

## 11. Open / deferred questions

- ETT vs trach stratification: include as a dashboard stratifier but not as an exclusion. Surface in `01_cohort.py` as a column.
- BMI / obesity stratification: deferred — would need weight from vitals.
- Severity (P/F ratio) stratification: deferred — would need joined `labs.pao2` + `vitals.fio2_set`.
- Time-of-day adherence (day shift vs night shift): deferred to v2 dashboard.
- Patients with no available sex: excluded from PBW → not assessable. Quantify in cohort report.
