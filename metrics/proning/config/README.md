# Config

Site-specific configuration lives in `config.json`. To set up a new site:

```bash
cp config/config_template.json config/config.json
# edit config/config.json — fill in site name and data_path for the primary dataset
```

`config.json` is gitignored. `config_template.json` is committed and safe to share.

Fields:
- `site`: short identifier (e.g., `UChicago`, `MIMIC`, `RUSH`).
- `timezone`: IANA zone used to localize all CLIF timestamps (e.g., `US/Central`, `US/Eastern`).
- `primary_dataset.data_path`: absolute path to the CLIF data directory.
- `secondary_dataset`: optional external-validation dataset; ignored in primary analysis.
- `tables_in_use`: CLIF tables the pipeline will load.
- `proning_eligibility`: thresholds for the PROSEVA-strict eligibility window. Defaults match the trial (P/F ≤150, FiO2 ≥0.6, PEEP ≥5, sustained ≥12h on IMV). Sites should not edit these unless running a sensitivity analysis.
- `proning_observation`: parameters for prone-session reconstruction from the `position` table.
  - `session_gap_minutes`: gap between consecutive `prone` records that ends a session (default 60 min).
  - `adherent_session_hours`: PROSEVA recommends ≥16h per session. Sessions ≥ this threshold count as adherent.
- `output_path`: relative to project root; deliverables land in `output/final/`, intermediates in `output/intermediate/`, logs in `output/logs/`.
