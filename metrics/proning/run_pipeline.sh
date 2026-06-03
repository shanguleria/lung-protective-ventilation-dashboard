#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f "config/config.json" ]; then
    echo "ERROR: config/config.json not found."
    echo "       Copy config/config_template.json to config/config.json and fill in paths."
    exit 1
fi

mkdir -p output/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="output/logs/pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1
echo "===== proning QI pipeline — $(date) ====="
echo "Log: $LOGFILE"

source .venv/bin/activate

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

run_step "01 build cohort"        python code/01_build_cohort.py
run_step "02 proning eligibility" python code/02_proning_eligibility.py
run_step "03 proning observation" python code/03_proning_observation.py
run_step "04 metrics"             python code/04_metrics.py
run_step "05 dashboard"           python code/05_dashboard.py

TOTAL=$((SECONDS - PIPELINE_START))
echo ""
printf "===== pipeline complete in %d:%02d =====\n" $((TOTAL/60)) $((TOTAL%60))
