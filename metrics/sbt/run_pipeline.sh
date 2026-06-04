#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
BUNDLE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

if [ ! -f "config.json" ]; then
    echo "ERROR: config.json not found."
    echo "       Copy config.example.json to config.json and fill in the data path."
    exit 1
fi

# Monorepo uses ONE shared venv at the bundle root.
if [ -f "$BUNDLE_ROOT/.venv/bin/activate" ]; then
    source "$BUNDLE_ROOT/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: no .venv found at $BUNDLE_ROOT/.venv or ./.venv"
    exit 1
fi

mkdir -p output/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="output/logs/pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1
echo "===== SBT (spontaneous breathing trial) QI pipeline — $(date) ====="
echo "Log: $LOGFILE"

PIPELINE_START=$SECONDS

run_step() {
    local step_name="$1"; shift
    echo ""
    echo "-----> $step_name"
    local start=$SECONDS
    "$@"
    local elapsed=$((SECONDS - start))
    printf "       done in %d:%02d\n" $((elapsed/60)) $((elapsed%60))
}

# Stage 00 (documentation probe) is run on demand, not in the standard pipeline:
#   python code/00_probe_documentation.py
run_step "01 build cohort"     python code/01_build_cohort.py
run_step "02 sbt eligibility"  python code/02_sbt_eligibility.py
run_step "03 sbt observation"  python code/03_sbt_observation.py
run_step "04 metrics"          python code/04_metrics.py
run_step "05 dashboard"        python code/05_dashboard.py

TOTAL=$((SECONDS - PIPELINE_START))
echo ""
printf "===== pipeline complete in %d:%02d =====\n" $((TOTAL/60)) $((TOTAL%60))
