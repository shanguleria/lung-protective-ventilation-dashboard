#!/usr/bin/env bash
#
# refresh_scorecard.sh — re-render the QI bundle scorecard ONLY (no CLIF re-read).
#
# Runs just the scorecard combiner, which is registry-driven: it reads every feed listed in
# config.json -> scorecard_tiles (e.g. ../proning/..., ../Sedation/sat_dashboard/...),
# validates each (schema_version == 1 + PHI-free), copies each feed's detail dashboard
# into output/dashboard/ so its "View details ->" link resolves, and rebuilds
# output/dashboard/scorecard.html. Reuses the existing metrics/lpv/output/*.parquet artifacts,
# so it is fast and safe to run anytime a sibling repo re-emits its tile feed.
#
# To rebuild the underlying LPV data first, run ./run_pipeline.sh (01 -> scorecard) instead.
#
# Usage:
#   ./refresh_scorecard.sh
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
if [[ ! -f metrics/lpv/output/02_patient_day_status.parquet ]]; then
  echo "ERROR: metrics/lpv/output/02_patient_day_status.parquet not found — the LPV pipeline hasn't been built."
  echo "    Run the full pipeline first:  ./run_pipeline.sh"
  exit 1
fi

"$PY" scorecard/build_scorecard.py

echo ""
echo "Done. Open the QI scorecard:  output/dashboard/scorecard.html"
