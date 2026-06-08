# lpv — Lung Protective Ventilation Adherence Dashboard

A CLIF project that builds a **descriptive dashboard** characterizing adherence to a composite lung-protective-ventilation (LPV) bundle in adult ICU patients receiving invasive mechanical ventilation at UChicago.

This is descriptive epidemiology only. Outcome modeling (mortality, VFD-28, etc.) is explicitly **out of scope** for this phase. Do not build outcome regressions, survival models, or causal estimators unless the user expands the scope.

---

## Research framing

- **Cohort:** all adult ICU hospitalizations with at least one episode of **invasive mechanical ventilation** during the ICU stay.
  - Definition of invasive MV: `respiratory_support.device_category == "imv"` for any duration during an ICU window.
  - Adult: age ≥ 18 at hospitalization start.
  - No restriction to ARDS / Berlin criteria.
- **Exposure / metric:** LPV bundle adherence, three components:
  1. Tidal volume **≤ 6 mL/kg predicted body weight (PBW)**
  2. Plateau pressure **≤ 30 cm H₂O**
  3. Driving pressure (∆P = Plateau − PEEP) **≤ 15 cm H₂O**
  - **As of 2026-06-01 these are reported COMPONENT-SEPARATED — each on its own denominator — not only as a single all-three composite** (the components have very different missingness; a lone composite forces densely-charted Vt to share sparse plateau's denominator). The composite is still computed alongside. The **Vt/kg cutoff is a dashboard slider** (default 6); plateau ≤ 30 and ∆P ≤ 15 are fixed ("less negotiable"). See `plans/01_design.md` §4a for the locked design and `code/02_features.py` for the four-measure implementation.
- **Outcome:** none. The dashboard summarizes adherence patterns; it does not model patient outcomes.
- **Design:** cross-sectional / longitudinal description of adherence.

### PBW formula (Devine, ARDSnet)
- Male: PBW_kg = 50 + 2.3 × (height_in − 60)
- Female: PBW_kg = 45.5 + 2.3 × (height_in − 60)
- Inputs are `patient.sex_category` and patient height.
- Height is stored in **`vitals`** under `vital_category == 'height_cm'` (not on the `patient` table). Convert to inches: `height_in = height_cm / 2.54`. Per-patient height is taken as a representative value (median across recorded height readings, since height should not change meaningfully during an admission).

---

## Dashboard scope (working draft — confirm with user before building)

Possible panels:
- **Overall adherence:** % of patient-time-on-IMV meeting the composite bundle.
- **Component breakdown:** marginal adherence to each of the three components (Vt, plateau, ∆P) and pairwise overlap.
- **Per-encounter:** distribution of % time-in-bundle across encounters; CDF/violin.
- **By severity stratum:** P/F ratio bands (mild/moderate/severe per Berlin), if computable from `vitals` / `labs`.
- **Temporal:** first 24h vs sustained vent course; possibly time-of-day.
- **Settings distributions:** marginal histograms of Vt/kg PBW, plateau, ∆P, PEEP, FiO₂.

Final panel set TBD. See `.claude/claude-todo.md`.

---

## CLIF tables expected

- `patient` — demographics (`birth_date`, `sex_category`).
- `hospitalization` — encounter timing, age at admission (computed from `birth_date` + `admission_dttm`).
- `adt` — ICU location windows (`location_category == 'icu'`).
- `respiratory_support` — filter `device_category == 'IMV'`; key columns: `tidal_volume_obs`, `tidal_volume_set`, `plateau_pressure_obs`, `peep_obs`, `peep_set`, `fio2_set`, `mode_category`, `tracheostomy`.
- `vitals` — `vital_category == 'height_cm'` (for PBW); `spo2` if a P/F surrogate is needed.
- `labs` — `pao2` (for P/F ratio if used for severity strata).

Other tables (medications, microbiology, scores) are **not** needed in this phase.

---

## Conventions

- **Site:** UChicago. Data path in `config.json` → `clif_data_path`. Timezone: `US/Central`.
- **CLIF version:** 2.1.0.
- **Python env:** `.venv/` at project root. Activate before running anything.
- **Outputs:** write to `output/` (gitignored). Anonymize site names in any audience-facing artifact (per global CLAUDE.md).
- **Data safety:** never print raw rows of CLIF tables into the conversation. Always summarize via code; only aggregated outputs may appear in chat.
- **Source traceability:** any specific numerical claim in `.md` files must carry an inline source pointer (`<!-- src: ... -->` or parenthetical), per global CLAUDE.md.

---

## ICU grouping dimensions (location_type vs location_name)

By-unit breakdowns support **two ICU-grouping grains**, switchable via a "Group ICUs by" toggle in
the dashboard's "By unit & over time" tab and on the bundle scorecard:

- **ICU type** (default) — `adt.location_type` (e.g. `medical_icu`). Back-compatible; all prior
  numbers are unchanged.
- **Specific unit** — `adt.location_name` (e.g. `N09S`). A single `location_type` can cover several
  physical units; this grain is the actionable one for site-level QI. At UChicago only `medical_icu`
  splits (→ `N09S` + `N09N`); other sites may fan out further.

**How it threads through the pipeline:**
- `01_cohort.py` assigns `assigned_unit` (type, by most-IMV-rows/day — *unchanged*) and, nested
  *within* that chosen type, `assigned_unit_name` (the specific unit). Deriving the name inside the
  already-chosen type guarantees every name rolls up to exactly one type and keeps type-level numbers
  byte-identical (a hard cross-check asserts the nesting).
- `02_features.py` carries `assigned_unit_name` as a passthrough id column.
- `03_aggregate.py` emits the daily/monthly/weekly summaries + Vt grid for **both** dims, tagged with
  a `dim` column (`type` | `name`); the name dim has per-unit rows only (no `__ALL__` — the shared
  site-wide row lives on the type dim). Check (7) asserts the name children sum to their parent type.
- `05_tile_feed.py` publishes both sets of unit keys in `headline`/`segment` cells plus a `dims`
  block: `{type:[...], name:[...], parent:{name→type}, labels:{...}}`.
- `04_dashboard.py` / `scorecard/build_scorecard.py` read `dims` to drive the toggle; un-migrated
  feeds (no name cells) fall back to site-wide with the existing grain-fallback badge.

**Friendly unit labels:** specific units are raw codes (`N09S`) by default. Set an optional map in
`config.json` → `"unit_labels": {"N09S": "MICU North", ...}` to display friendly names; unmapped
codes fall back to the raw value.

---

## Open questions

These need answers before the dashboard can be built. Track in `.claude/claude-todo.md`.

1. **Height availability at UChicago:** how complete is `vitals.height_cm` for the IMV cohort? Units sanity check (values should be ~140–210 cm; if some look like inches the data needs unit reconciliation).
2. **Plateau pressure availability:** how complete is `respiratory_support.plateau_pressure_obs` at UChicago during IMV rows? Plateau is manually obtained and expected to be sparse — quantify before deciding adherence definition.
3. **Vt source:** `tidal_volume_obs` vs `tidal_volume_set` — which feeds adherence? Default: observed if present, else set; quantify both.
4. **Time-on-IMV granularity:** treat adherence per row/event, per hour, or per fixed window?
5. **Pediatric exclusion:** confirm age ≥ 18 cutoff at hospitalization (vs ICU admit).
6. **ICU stay window:** use `adt` ICU intervals only, or any IMV row regardless of location?

---

## Files & directories

- `config.json` — site + data path (gitignored; copy from `config.example.json`)
- `config.example.json` — config template for new sites
- `requirements.txt` — Python deps
- `run_pipeline.sh` — one-command build (01 → 04)
- `README.md` — setup/run/customize guide for other CLIF sites
- `LICENSE` — MIT
- `code/` — pipeline (`01_cohort` → `02_features` → `03_aggregate` → `04_dashboard`) + data-quality probes (`00*`, `01b`, `02b`, `02c`)
- `output/` — generated artifacts incl. the dashboard HTML (gitignored)
- `plans/01_design.md` — authoritative methodology / locked analytic choices
- `references/` — papers, prior work
- `.claude/` — running session logs (`claude-progress.md`, `claude-todo.md`, `lessons.md`); gitignored, not shipped
