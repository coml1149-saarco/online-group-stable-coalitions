#!/bin/sh
# Regenerate the summaries, figures and tables of the paper from the saved trial records
# (results/saved). The benchmark data must have been downloaded (scripts/fetch_benchmark_data.sh): the
# dataset-derived games are rebuilt from it first, and the four example trials of the game figures are
# rerun locally. No other learning run is repeated. Set SNS_RUNS to analyse rerun records instead.
set -e
mkdir -p "$(dirname "$0")/../output"
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
for s in rebuild_benchmark_games analyze_synthetic analyze_robustness analyze_bandit analyze_benchmark \
         make_figures make_bandit_figures make_feedback_figure make_robustness_figures make_benchmark_figures; do
    echo "== $s"
    "$PY" "scripts/$s.py" > "output/log_$s.txt" 2>&1 || { cat "output/log_$s.txt"; exit 1; }
done
"$PY" scripts/check_outputs.py "$@"
