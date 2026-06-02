#!/usr/bin/env bash
#
# run_pipeline.sh — build the LPV adherence dashboard end-to-end.
#
# Runs the pipeline steps in order using the project virtualenv:
#   01_cohort.py     -> output/01_cohort_patient_days.parquet
#   02_features.py   -> output/02_patient_day_status.parquet, 02_intervals.parquet
#   02d_severity.py  -> output/02d_severity.parquet  (severe respiratory failure flag)
#   03_aggregate.py  -> output/03_*_unit_summary.parquet, 03_vt_grid_*.parquet
#   04_dashboard.py  -> output/04_lpv_dashboard.html   (LPV drill-down)
#   05_scorecard.py  -> output/index.html              (QI bundle scorecard — open this)
#
# Prereqs (see README.md): a .venv with requirements installed, and a config.json
# (copy config.example.json -> config.json and edit it for your site).
#
# Usage:
#   ./run_pipeline.sh
#
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found. Create the venv first:"
  echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ ! -f config.json ]]; then
  echo "ERROR: config.json not found. Copy the template and edit it:"
  echo "    cp config.example.json config.json   # then set clif_data_path, site, timezone"
  exit 1
fi

steps=(
  "code/01_cohort.py"
  "code/02_features.py"
  "code/02d_severity.py"
  "code/03_aggregate.py"
  "code/04_dashboard.py"
  "code/05_scorecard.py"
)

for step in "${steps[@]}"; do
  echo ""
  echo "=================================================================="
  echo ">>> $step"
  echo "=================================================================="
  "$PY" "$step"
done

echo ""
echo "Done. Open the QI scorecard:  output/index.html   (LPV tile → output/04_lpv_dashboard.html)"
