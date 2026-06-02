# LPV — Lung-Protective Ventilation Adherence Dashboard

A reproducible [CLIF](https://clif-consortium.github.io/website/) pipeline that builds an
**interactive, self-contained HTML dashboard** characterizing adherence to a lung-protective
ventilation (LPV) bundle among adult ICU patients receiving invasive mechanical ventilation (IMV).

Any CLIF 2.x site can clone this repo, point it at their own CLIF data, and run one command to
produce their own dashboard. It is **descriptive epidemiology only** — no outcome modeling.

![pipeline](https://img.shields.io/badge/CLIF-2.x-blue) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## What it produces

A single file — `output/04_lpv_dashboard.html` (~6 MB, Plotly inlined, works offline) — with four tabs:

- **Tidal Volume** — adherence to a tunable Vt/kg cutoff, with a slider (4–10 mL/kg) and time trends.
- **Component breakdown** — the three components reported **separately, each on its own denominator**.
- **By unit & over time** — adherence by ICU unit and month/day.
- **Distributions & cohort** — time-weighted ventilator-settings histograms + a cohort Table 1.

Two global controls: a **Vt-cutoff slider** and a **time-period selector** (all-time / year / month).

### The LPV bundle

Three components, evaluated on **mode-eligible IMV time**, time-weighted within each patient-day
(a day is "adherent" for a measure if ≥80% of its assessable time meets the threshold, with a
≥60-minute assessable floor):

1. **Tidal volume** ≤ 6 mL/kg predicted body weight (PBW) — *cutoff is adjustable in the dashboard*
2. **Plateau pressure** ≤ 30 cm H₂O — fixed
3. **Driving pressure** (∆P = Plateau − PEEP) ≤ 15 cm H₂O — fixed

Each component is reported on **its own denominator** (the patient-days where that component is
measurable), plus a strict joint **composite** (all three). This separation matters because the
components have very different missingness — tidal volume is densely charted, plateau is sparse —
so a single composite would force the well-measured Vt to share plateau's small denominator.
PBW uses the Devine/ARDSnet formula from `patient.sex_category` and height (`vitals.height_cm`).

---

## CLIF tables required

CLIF 2.x, all as the `filetype` in `config.json` (default `parquet`):

| Table | Used for |
|---|---|
| `patient` | `birth_date`, `sex_category` |
| `hospitalization` | admission timing, age at admit |
| `adt` | ICU location windows (`location_category == 'icu'`, `location_type`) |
| `respiratory_support` | `device_category == 'IMV'`, `mode_category`, `tidal_volume_obs/set`, `plateau_pressure_obs`, `peep_obs/set`, `fio2_set` |
| `vitals` | `vital_category == 'height_cm'` (for PBW) |

Standard CLIF mCIDE category values are assumed (e.g. `device_category` `IMV`, the volume/pressure
modes in `mode_category`). Outlier handling uses clifpy's built-in ranges.

---

## Quick start

```bash
# 1. Clone
git clone <your-fork-url> lpv && cd lpv

# 2. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configure for your site
cp config.example.json config.json
#   then edit config.json (see below)

# 4. Run the whole pipeline
./run_pipeline.sh

# 5. Open the dashboard
open output/04_lpv_dashboard.html      # macOS  (Linux: xdg-open)
```

### `config.json`

| Field | Meaning |
|---|---|
| `clif_data_path` | Absolute path to your CLIF tables directory |
| `filetype` | `parquet` (default), `csv`, etc. — passed to clifpy |
| `timezone` | Your site's local tz (e.g. `US/Central`) — used for calendar-day binning |
| `site` | Your site label — appears in the dashboard title (display only) |
| `clif_version` | Your CLIF version string (display only) |
| `output_path` | Where artifacts are written (default `output/`) |

`config.json` is **gitignored** so your data path stays local; commit only `config.example.json`.

---

## Pipeline

`run_pipeline.sh` runs four steps in order (each is idempotent and reads the previous step's output):

| Step | Script | Output |
|---|---|---|
| 1 | `code/01_cohort.py` | `01_cohort_patient_days.parquet` — adult IMV-on-ICU patient-day cohort + PBW |
| 2 | `code/02_features.py` | `02_patient_day_status.parquet`, `02_intervals.parquet` — per-component adherence |
| 3 | `code/03_aggregate.py` | `03_*_unit_summary.parquet`, `03_vt_grid_*.parquet` — (time × unit) rollups + Vt grid |
| 4 | `code/04_dashboard.py` | **`04_lpv_dashboard.html`** — the dashboard |

### Recommended first: check your data

Before trusting the dashboard, run the **data-quality probes** to confirm your site's coverage
(these print aggregated summaries only — no patient rows):

```bash
.venv/bin/python code/00_probe_missingness.py        # variable completeness for the IMV cohort
.venv/bin/python code/01b_cohort_assessment.py        # cohort sanity checks (after step 1)
.venv/bin/python code/02c_component_probe.py          # per-component assessability (after step 2)
.venv/bin/python code/02b_vt_sensitivity.py           # adherence vs Vt cutoff (after step 2)
```

Pay attention to **plateau-pressure completeness** and **height availability** — these drive how
much of the cohort is assessable at your site. See `output/00_probe_summary.md` and
`output/01b_cohort_assessment.md`.

---

## Customizing the analytic choices

The locked design is documented in [`plans/01_design.md`](plans/01_design.md). The key parameters
are named constants near the top of `code/02_features.py` (and mirrored in `03`/`04`):

| Parameter | Default | Where |
|---|---|---|
| Vt/kg cutoff (default; slider overrides) | 6 mL/kg | `VT_MAX_DEFAULT` |
| Plateau / driving-pressure thresholds | 30 / 15 cm H₂O | `PLATEAU_MAX`, `DP_MAX` |
| Adherence fraction / assessable floor | 80% / 60 min | `ADHERENCE_FRACTION`, `MIN_ASSESSABLE_MIN` |
| Carry-forward windows | Vt/PEEP 2 h, plateau/mode 6 h | `CF_FAST`, `CF_SLOW` |
| Eligible ventilator modes | AC-VC, PRVC, SIMV, PC | `ELIGIBLE_MODES` |
| Vt-cutoff grid (dashboard slider) | 4.0–10.0 by 0.5 | `VT_GRID` (in `03`/`04`) |

The **6-hour plateau carry-forward** reflects a Q4-shift plateau-charting cadence; if your site
charts plateau less often, widen it. Re-run `run_pipeline.sh` after any change.

---

## Data safety

- The pipeline reads CLIF tables but **embeds only aggregated values** (rates, counts, binned
  histograms, and per-period aggregated Table 1s) into the dashboard — **no patient-level rows**.
- `output/` is gitignored; the dashboard HTML is not committed. Treat generated outputs per your
  site's data-governance rules before sharing.
- The dashboard uses real ICU **unit** labels (within-site unit types). If you intend to share it
  outside your institution, review your consortium's anonymization expectations first.

---

## Repository layout

```
config.example.json   # copy to config.json and edit
requirements.txt
run_pipeline.sh        # one-command build (01 -> 04)
code/                  # pipeline + data-quality probes
plans/01_design.md     # authoritative methodology / locked analytic choices
output/                # generated artifacts (gitignored)
```

## Acknowledgements

Built on the [Common Longitudinal ICU Format (CLIF)](https://clif-consortium.github.io/website/)
and the [`clifpy`](https://pypi.org/project/clifpy/) library. Licensed under MIT (see `LICENSE`).
