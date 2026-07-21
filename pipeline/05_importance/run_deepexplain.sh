#!/bin/bash
# Node-level SHAP (method 2B) via GenNet's DeepExplainer — one SLURM task per tanh seed.
#
# Runs `GenNet.py interpret -type DeepExplain`, which builds shap.DeepExplainer on the
# trained model (background = validation CONTROLS, explained set = test CASES) and writes
# `DeepExplain_test.npy` (one |SHAP| value per SNP, length = inputsize) into each
# experiment folder. Lift SNP -> gene afterwards via topology.csv for gene importance.
#
# NOTE: this uses the model as-is (`model.output`), so it still has the double-sigmoid
# — i.e. this is the "before/buggy" node run for the co-author's comparison. Only the
# missing-background bug is fixed (Interpret.py:103).
#
# --- Why num_sample_pat = 9942 ---
# get_data() samples this many subjects (seeded, random_state=1) from BOTH the validation
# and test sets, then Interpret.py filters: controls -> background, cases -> explained set.
#   val  = 13257 (6732 controls / 6525 cases)
#   test =  9942 (5049 controls / 4893 cases)
# 9942 is the MAX safe value (= min(val, test)); anything larger overflows .sample(n=...)
# and crashes. At 9942 the run uses ~5050 controls as background and ALL 4893 test cases
# as the explained set — the whole case cohort, zero subsampling variance = the most
# stable node ranking this method can give. (Same split for every seed 42-46, so one
# value is safe across the whole array.)
#
# Submit:
#   sbatch pipeline/05_importance/run_deepexplain.sh              # defaults to results/tanh
#   sbatch pipeline/05_importance/run_deepexplain.sh results/tanh
#
#SBATCH --job-name=GenNet_deepexplain
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/deepexplain_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/deepexplain_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --array=1-5

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"
NUM_SAMPLE_PAT=9942

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

# Map the SLURM array index -> Nth trained experiment folder (sorted, stable order).
mapfile -t EXPS < <(for d in "$RESULTS_DIR"/GenNet_experiment_*_; do
    [ -f "$d/bestweights_job.h5" ] && echo "$d"
done | sort)

EXP="${EXPS[$((SLURM_ARRAY_TASK_ID - 1))]:-}"
[ -n "$EXP" ] || { echo "No trained experiment at array index $SLURM_ARRAY_TASK_ID in $RESULTS_DIR"; exit 1; }

echo "=== DeepExplain: $EXP  (num_sample_pat=$NUM_SAMPLE_PAT) ==="
python GenNet.py interpret -type DeepExplain \
    -resultpath "$EXP" \
    -num_sample_pat "$NUM_SAMPLE_PAT"
echo "=== done: $EXP -> $EXP/DeepExplain_test.npy ==="
