# CLIF ICU Ventilator-QI Bundle Dashboard

A reproducible [CLIF](https://clif-consortium.github.io/website/) **monorepo** that builds a
glanceable **ICU ventilator / liberation QI bundle scorecard** — one tile per metric — plus a
detailed drill-down per metric. Each metric is its own self-contained pipeline that emits a small,
PHI-free **tile feed**; the scorecard is a combiner that collects them.

Any CLIF 2.x site can clone this repo, point it at their own CLIF data, and run one command to
produce their own scorecard. It is **descriptive epidemiology only** — no outcome modeling.

![pipeline](https://img.shields.io/badge/CLIF-2.x-blue) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

The metrics shipped today: **LPV** (lung-protective ventilation — the reference implementation),
**ARDS proning**, and **SAT** (spontaneous awakening trials). SBT and mobilization are placeholders.

---

## What it produces

The build writes a self-contained, shippable bundle to **`output/dashboard/`**:

- **`scorecard.html`** — the glanceable ICU ventilator-QI bundle scorecard (open this).
- **`lpv_dashboard.html`**, **`proning_dashboard.html`**, **`sat_dashboard.html`** — each metric's
  detailed drill-down (the scorecard tiles link here).

The whole `output/dashboard/` folder travels together (the HTML files cross-link by relative name).

`lpv_dashboard.html` (~8 MB, Plotly inlined, works offline) has four tabs: **Tidal Volume** (adherence
to a tunable Vt/kg cutoff, slider 4–10 mL/kg), **Component breakdown** (the three components reported
separately, each on its own denominator), **By unit & over time**, and **Distributions & cohort**
(time-weighted settings histograms + a cohort Table 1). Two global controls: a **Vt-cutoff slider**
and a **time-period selector**.

### The LPV bundle

Three components, evaluated on **mode-eligible IMV time**, time-weighted within each patient-day
(a day is "adherent" for a measure if ≥80% of its assessable time meets the threshold, with a
≥60-minute assessable floor):

1. **Tidal volume** ≤ 6 mL/kg predicted body weight (PBW) — *cutoff is adjustable in the dashboard*
2. **Plateau pressure** ≤ 30 cm H₂O — fixed
3. **Driving pressure** (∆P = Plateau − PEEP) ≤ 15 cm H₂O — fixed

Each component is reported on **its own denominator** (plus a strict joint composite), because the
components have very different missingness (Vt densely charted, plateau sparse) — a single composite
would force well-measured Vt to share plateau's small denominator. PBW uses the Devine/ARDSnet
formula from `patient.sex_category` and height (`vitals.height_cm`).

---

## CLIF tables required (LPV metric)

CLIF 2.x, as the `filetype` in `config.json` (default `parquet`):

| Table | Used for |
|---|---|
| `patient` | `birth_date`, `sex_category` |
| `hospitalization` | admission timing, age at admit |
| `adt` | ICU location windows (`location_category == 'icu'`, `location_type`) |
| `respiratory_support` | `device_category == 'IMV'`, `mode_category`, `tidal_volume_obs/set`, `plateau_pressure_obs`, `peep_obs/set`, `fio2_set` |
| `vitals` | `vital_category == 'height_cm'` (PBW); `vital_category == 'spo2'` (severity S/F surrogate) |
| `labs` | `lab_category == 'po2_arterial'` (severity P/F ratio) |

(The proning and SAT metrics use additional tables — see each metric's `CLAUDE.md`.) Standard CLIF
mCIDE category values are assumed; outlier handling uses clifpy's built-in ranges.

### Severity stratifier (LPV)

The LPV dashboard includes a **Severity filter** ("severe respiratory failure" = P/F < 300, or the
S/F surrogate < 315 at SpO₂ ≤ 97%, **and** PEEP > 5; FiO₂/PEEP paired within a 4-hour lookback;
worst oxygenation of the day) — **not** full Berlin ARDS. Thresholds are named constants in
`metrics/lpv/code/02d_severity.py`.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/shanguleria/clif-ventilator-qi-dashboard.git && cd clif-ventilator-qi-dashboard

# 2. One shared Python environment for the whole bundle
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configure for your site
cp config.example.json config.json     # then edit (see below)

# 4. Build the bundle (LPV pipeline + scorecard combiner)
./run_bundle.sh

# 5. Open the scorecard (tiles link to each metric's drill-down)
open output/dashboard/scorecard.html   # macOS  (Linux: xdg-open)
```

### `config.json`

| Field | Meaning |
|---|---|
| `clif_data_path` | Absolute path to your CLIF tables directory |
| `filetype` | `parquet` (default), `csv`, etc. — passed to clifpy |
| `timezone` | Your site's local tz (e.g. `US/Central`) — used for calendar-day binning |
| `site` | Your site label — appears in the dashboard title + each feed's provenance |
| `clif_version` | Your CLIF version string |
| `metrics` | Which metric tiles to build, in slot order, e.g. `["lpv", "proning", "sat"]`. A listed metric with no feed yet shows a "Coming soon…" placeholder; omit a metric you don't run |
| `unit_labels` | *(optional)* Friendly display names for specific ICU units, e.g. `{"N09S": "MICU North", "N09N": "MICU South"}`. See **Grouping ICUs by specific unit** below. Defaults to `{}` (raw codes shown) |

`config.json` is **gitignored** so your data path stays local; commit only `config.example.json`.

### Grouping ICUs by specific unit (`location_type` vs `location_name`)

Every by-unit breakdown — the bundle scorecard and each metric's drill-down — has a **"Group ICUs
by"** toggle: **ICU type** (the CLIF `location_type`, e.g. `medical_icu`) or **Specific unit** (the
CLIF `location_name`, the actual physical unit, e.g. `N09S`). The specific-unit grain is the
actionable one for site-level QI when a single `location_type` covers several physical units; it is
computed nested within the type, so the type-level numbers never change.

The toggle appears only when at least one `location_type` at your site splits into multiple
`location_name`s. Specific units display as their **raw `location_name` code** by default. To show
friendly names instead, add a `unit_labels` map to `config.json` — no code change, no pipeline re-run
beyond the scorecard rebuild:

```json
"unit_labels": { "N09S": "MICU North", "N09N": "MICU South", "N04E": "CT-ICU" }
```

Unmapped codes fall back to the raw value. Keys are the exact `location_name` strings as they appear
in your `adt` table.

---

## Repository layout

```
config.example.json     # copy to config.json and edit
requirements.txt        # one shared venv for the whole bundle
run_bundle.sh           # one-command build: LPV pipeline -> scorecard
refresh_scorecard.sh    # fast scorecard-only rebuild (no CLIF re-read)
CODEOWNERS              # per-metric ownership (stub; solo for now)
contract/               # the tile-feed spec + JSON Schema (the only thing the scorecard depends on)
metrics/                # one folder per QI vertical (see metrics/README.md)
  lpv/      code/ ...    #   the reference metric: 01_cohort -> 04_dashboard + 05_tile_feed
  proning/  code/ ...
  sat/      code/ ...
scorecard/              # build_scorecard.py — the combiner (collects feeds, renders scorecard.html)
feeds/                  # collected PHI-free tile feeds (the per-site submission set; build artifact)
output/dashboard/       # the shippable bundle: scorecard.html + each metric's drill-down (gitignored)
```

Each metric emits `metrics/<id>/output/final/tile_feed_<id>.json` (+ its `<id>_dashboard.html`). The
combiner collects every metric in `config.json → metrics`, stages feeds into `feeds/`, ships the
drill-downs into `output/dashboard/`, and renders the scorecard.

## Pipeline (LPV metric + combiner)

`run_bundle.sh` runs these in order:

| Step | Script | Output |
|---|---|---|
| 1 | `metrics/lpv/code/01_cohort.py` | adult IMV-on-ICU patient-day cohort + PBW |
| 2 | `metrics/lpv/code/02_features.py` | per-component adherence (`02_patient_day_status`, `02_intervals`) |
| 2d | `metrics/lpv/code/02d_severity.py` | severe-respiratory-failure flag per patient-day |
| 3 | `metrics/lpv/code/03_aggregate.py` | (time × unit, severity) rollups + Vt-cutoff grid |
| 4 | `metrics/lpv/code/04_dashboard.py` | `metrics/lpv/output/final/lpv_dashboard.html` |
| 5 | `metrics/lpv/code/05_tile_feed.py` | `metrics/lpv/output/final/tile_feed_lpv.json` |
| → | `scorecard/build_scorecard.py` | **`output/dashboard/scorecard.html`** (collects every enabled metric) |

Other metrics (proning, sat) are their own pipelines under `metrics/<id>/`; run those in their own
dir when their data updates. The combiner just collects whatever feeds exist.

### The scorecard is a combiner, not a place to add metric logic

It is **registry-driven**: each metric is its own vertical that emits a PHI-free
`tile_feed_<metric>.json` (spec: [`contract/tile_feed_contract.md`](contract/tile_feed_contract.md),
schema: `contract/tile_feed.schema.json`). The combiner renders each through the **same tile
component** (donut + up to 3 segments + optional goal bar + sparkline) and copies its detail dashboard
into `output/dashboard/`. A coarse feed (e.g. proning is site-wide / all-time only) shows a
**`· site-wide` / `· all-time` badge** when the global Unit/Week filters are finer than it provides,
so a number is never silently mislabeled. A slot with no feed shows a **"Coming soon…"** placeholder.

So adding a metric is: *build the vertical → emit a tile feed → add its id to `config.json → metrics`*
— no scorecard code change. See [`metrics/README.md`](metrics/README.md).

Tile artwork is read from `assets/<LPV|SAT|SBT|Proning|Mobilization>.png` (downscaled + embedded at
build time); that folder is gitignored, and the scorecard **falls back to inline SVG icons** when the
images are absent — so a fresh clone still builds.

### Recommended first: check your data

Before trusting the dashboard, run the **data-quality probes** (aggregated summaries only — no
patient rows):

```bash
.venv/bin/python metrics/lpv/code/00_probe_missingness.py     # variable completeness for the IMV cohort
.venv/bin/python metrics/lpv/code/01b_cohort_assessment.py    # cohort sanity checks (after step 1)
.venv/bin/python metrics/lpv/code/02c_component_probe.py      # per-component assessability (after step 2)
.venv/bin/python metrics/lpv/code/02b_vt_sensitivity.py       # adherence vs Vt cutoff (after step 2)
```

Pay attention to **plateau-pressure completeness** and **height availability** — these drive how much
of the cohort is assessable at your site.

---

## Customizing the LPV analytic choices

The key parameters are named constants near the top of `metrics/lpv/code/02_features.py` (mirrored in
`03`/`04`):

| Parameter | Default | Where |
|---|---|---|
| Vt/kg cutoff (default; slider overrides) | 6 mL/kg | `VT_MAX_DEFAULT` |
| Plateau / driving-pressure thresholds | 30 / 15 cm H₂O | `PLATEAU_MAX`, `DP_MAX` |
| Adherence fraction / assessable floor | 80% / 60 min | `ADHERENCE_FRACTION`, `MIN_ASSESSABLE_MIN` |
| Carry-forward windows | Vt/PEEP 2 h, plateau/mode 6 h | `CF_FAST`, `CF_SLOW` |
| Eligible ventilator modes | AC-VC, PRVC, SIMV, PC | `ELIGIBLE_MODES` |
| Scorecard headline Vt cutoff / goal | 8 mL/kg / 90% | `SCORECARD_VT_CUTOFF`, `LPV_GOAL` (in `05_tile_feed.py`) |

The **6-hour plateau carry-forward** reflects a Q4-shift plateau-charting cadence; widen it if your
site charts plateau less often. Re-run `run_bundle.sh` after any change.

---

## Data safety

- The pipelines read CLIF tables but **embed only aggregated values** (rates, counts, binned
  histograms, per-period aggregated Table 1s) — **no patient-level rows**. Tile feeds carry only
  `num`/`den` counts and are PHI-checked at build time.
- `output/`, `feeds/*.json`, and `assets/` are gitignored; nothing patient-adjacent is committed.
- Dashboards use real ICU **unit** labels (within-site). For audience-facing / consortium use, review
  your consortium's anonymization expectations (per-site displays should use anonymized "Site N").

## Acknowledgements

Built on the [Common Longitudinal ICU Format (CLIF)](https://clif-consortium.github.io/website/) and
the [`clifpy`](https://pypi.org/project/clifpy/) library. Licensed under MIT (see `LICENSE`).
