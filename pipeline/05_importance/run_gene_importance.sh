#!/bin/bash
# Weight-based gene (node) importance for every experiment in a results folder.
#
# Runs `GenNet.py interpret -type get_gene_importance`, which calls
# interpretation/weight_importance.py::make_gene_importance and writes one
# gene_importance.csv (one row per gene) into each experiment folder.
#
# Interpret.py skips when gene_importance.csv already exists, so any stale file is
# removed first -- the point of running this is to regenerate it.
#
# Submit:
#   sbatch pipeline/05_importance/run_gene_importance.sh              # defaults to results/tanh
#   sbatch pipeline/05_importance/run_gene_importance.sh results/relu
#
#SBATCH --job-name=GenNet_gene_importance
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/gene_importance_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/gene_importance_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=all_5hrs
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

shopt -s nullglob
EXPS=()
for d in "$RESULTS_DIR"/GenNet_experiment_*_; do
    [ -f "$d/bestweights_job.h5" ] && EXPS+=("$d")
done
[ ${#EXPS[@]} -gt 0 ] || { echo "No trained experiments in $RESULTS_DIR"; exit 1; }

echo "=== gene importance for ${#EXPS[@]} experiments in $RESULTS_DIR ==="

for EXP in "${EXPS[@]}"; do
    rm -f "$EXP/gene_importance.csv"
    echo "--- $EXP"
    python GenNet.py interpret -type get_gene_importance -resultpath "$EXP"
done

echo "=== done -> $RESULTS_DIR/GenNet_experiment_*_/gene_importance.csv ==="
