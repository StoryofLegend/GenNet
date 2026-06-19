#!/bin/bash
# Compute weight-based (2A) importance for every experiment in a results folder.
#
# Lightweight (weights only, no forward pass / GPU), so it runs fine on the login
# node — no SLURM needed. Each run reloads the model (~30 s).
#
# Usage:
#   bash pipeline/05_importance/run_importance.sh results/tanh
#   bash pipeline/05_importance/run_importance.sh results/relu --edge-agg mean
#
# Output: reports/importance/<exp-folder>/{gene,pathway,pair}_importance.csv

set -euo pipefail

RESULTS_DIR="${1:?Usage: run_importance.sh <results_folder> [extra args]}"
shift || true

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

for exp in "$RESULTS_DIR"/GenNet_experiment_*_; do
    [ -f "$exp/bestweights_job.h5" ] || { echo "skip $exp (no weights)"; continue; }
    echo "=== $exp ==="
    python pipeline/05_importance/compute_weight_importance.py "$exp" "$@"
done
