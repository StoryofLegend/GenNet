#!/bin/bash
# Node-level SHAP via GradientExplainer (expected gradients) — one SLURM task per seed.
#
# Runs `GenNet.py interpret -type GradientExplain`, which builds shap.GradientExplainer
# on the GENETIC LOGIT sub-model (`genetic_logit_model`) with BatchNorm KEPT INTACT.
# GradientExplainer uses ordinary autodiff, so it differentiates through BN natively —
# no BN removal, hence NO logit collapse (the DeepExplainer path had to delete BN, which
# flattened the logit to std~1e-4 and produced ~1e-5 "dust" SHAP values). It writes three
# per-SNP arrays (length = inputsize) into each experiment folder:
#   GradientExplain_test_meanabs.npy  = mean_patients |SHAP|  (node importance, for gene ranking)
#
# Defaults to results/tanh_gradexplain (the lightweight copy of the tanh experiments,
# without connection_weights.csv). Originals in results/tanh keep their DeepExplain_test.npy
# so the two methods can be compared.
#
# --- Knobs ---
# -num_sample_pat 9942 : MAX safe value (= min(val,test)); background = val CONTROLS (~5050),
#                        explained = ALL test CASES (~4893). Same split for every seed 42-46.
# -gx_nsamples 200     : background references drawn per explained case (expected-gradients
#                        Monte-Carlo samples). 200 is SHAP's default; raise for smoother
#                        magnitudes, lower to save time.
#
# Submit:
#   sbatch pipeline/05_importance/run_gradexplain.sh                     # results/tanh_gradexplain
#   sbatch pipeline/05_importance/run_gradexplain.sh results/tanh_gradexplain
#
#SBATCH --job-name=GenNet_gradexplain
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/gradexplain_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/gradexplain_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
# partition=gpu is ptfgpu001: 64 CPU / 931 GB / 2x H100, Hidden=YES (so plain
# `sinfo` omits it -- use `sinfo -a`), AllowGroups=ALL, MaxTime=UNLIMITED, and in
# practice never queued. We take it for the CPUs and RAM only.
# NO --gres: probed 2026-09-04, env_GenNet TF 2.11 wants libcudart.so.11.0 /
# libcudnn.so.8 and the node has CUDA 12.8, so tf.config.list_physical_devices('GPU')
# is empty and this runs on CPU regardless. Requesting a GPU would only block a
# real GPU user. See CLAUDE.md.
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --array=1-5

set -euo pipefail

RESULTS_DIR="${1:-results/tanh_gradexplain}"
NUM_SAMPLE_PAT=9942
GX_NSAMPLES=200

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

# Map the SLURM array index -> Nth trained experiment folder (sorted, stable order).
mapfile -t EXPS < <(for d in "$RESULTS_DIR"/GenNet_experiment_*_; do
    [ -f "$d/bestweights_job.h5" ] && echo "$d"
done | sort)

EXP="${EXPS[$((SLURM_ARRAY_TASK_ID - 1))]:-}"
[ -n "$EXP" ] || { echo "No trained experiment at array index $SLURM_ARRAY_TASK_ID in $RESULTS_DIR"; exit 1; }

echo "=== GradientExplain: $EXP  (num_sample_pat=$NUM_SAMPLE_PAT gx_nsamples=$GX_NSAMPLES) ==="
python GenNet.py interpret -type GradientExplain \
    -resultpath "$EXP" \
    -num_sample_pat "$NUM_SAMPLE_PAT" \
    -gx_nsamples "$GX_NSAMPLES"
echo "=== done: $EXP -> $EXP/GradientExplain_test_meanabs.npy ==="
