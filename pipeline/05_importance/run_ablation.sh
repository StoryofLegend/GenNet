#!/bin/bash
# Node-ablation gene importance (method 2C) — one SLURM task per seed.
#
# Runs `GenNet.py interpret -type Ablation`, which calls
# interpretation/ablation.py::make_ablation_importance and writes one
# ablation_importance.csv (one row per gene) into each experiment folder.
#
#   Delta_y(g) = y_full - y_{g ablated}
#
# --- What is ablated, and where ---
# The gene is zeroed at the GENE-LAYER ACTIVATION (the post-BatchNorm tensor feeding
# LocallyDirected_1), NOT at the genotype input. Zeroing genotype would set everyone to
# homozygous-reference — a real genotype, not a neutral baseline. The gene layer is
# BatchNorm'd (center=False, scale=False), so it is already mean-~0 across the cohort:
# zero there means "replace this gene by the population-average gene", which is the
# baseline the ablation formula actually asks for. All layer1_nodes carrying the same
# gene name are zeroed together, so the score is per GENE, not per node.
#
# --- What is measured ---
# The delta on the GENETIC LOGIT (pre-sigmoid output_layer), like the GradientExplain run
# and unlike the old DeepExplain run — the sigmoid saturates and squashes the differences.
# BatchNorm is kept intact throughout (no remove_batchnorm_model, no logit collapse).
#
# --- Exactness ---
# Zeroing gene g only perturbs the pathway nodes g connects to, so the delta is computed
# in closed form over those pathways instead of re-running the whole network ~6000 times.
# This is exact, not an approximation, and the code proves it: before the sweep it ablates
# `-ablation_verify` random genes with a real forward pass through the reused Keras layers
# and raises if the two disagree by more than 1e-3 on the logit scale. Watch for the
# "Verified." line in the log.
#
# --- Knobs ---
# -num_sample_pat 9942   : MAX safe value (= min(val,test) subject counts). get_data()
#                          samples this many test subjects (seeded, random_state=1), i.e.
#                          the whole test set, then -ablation_set cases keeps the ~4893
#                          CASES — the same cohort DeepExplain/GradientExplain explain, so
#                          2B and 2C rank over identical patients. Zero subsampling
#                          variance. Use `-ablation_set all` for the full test set.
# -ablation_verify 3     : genes cross-checked against a full forward pass (0 disables).
# -ablation_per_patient  : also write ablation_per_patient.csv (genes x subjects) — the
#                          per-subject deltas the ISN step (06_isn) will need. ~6000 x 4893
#                          floats, so only pass it when you want that matrix.
#
# Consensus across seeds afterwards. Run BOTH and report them side by side: the raw
# score, and the connectivity-normalised one the guidelines (Sec 1A) ask for. The
# second needs an explicit --out or it overwrites the first (the default output name
# is derived from --input-name, which is the same file for both).
#
#   python pipeline/04_report/seed_consensus.py results/tanh \
#       --input-name ablation_importance.csv --score-column ablation_meanabs
#
#   python pipeline/04_report/seed_consensus.py results/tanh \
#       --input-name ablation_importance.csv --score-column ablation_meanabs_per_degree \
#       --out results/tanh/ablation_importance_per_degree_consensus.csv
#
# Ablating the whole gene at once already captures its joint effect through every
# pathway it touches, so ablation_meanabs is only weakly hub-driven (Spearman vs
# degree 0.23, against 0.32 for the 2A weight sum). Treat the per-degree ranking as a
# robustness check on the raw one, not as a replacement for it.
#
# Submit:
#   sbatch pipeline/05_importance/run_ablation.sh              # defaults to results/tanh
#   sbatch pipeline/05_importance/run_ablation.sh results/relu
#
#SBATCH --job-name=GenNet_ablation
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/ablation_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/05_importance/ablation_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=all_5hrs
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --array=1-5

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"
NUM_SAMPLE_PAT=9942
ABLATION_SET=cases
ABLATION_VERIFY=3

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

# Map the SLURM array index -> Nth trained experiment folder (sorted, stable order).
mapfile -t EXPS < <(for d in "$RESULTS_DIR"/GenNet_experiment_*_; do
    [ -f "$d/bestweights_job.h5" ] && echo "$d"
done | sort)

EXP="${EXPS[$((SLURM_ARRAY_TASK_ID - 1))]:-}"
[ -n "$EXP" ] || { echo "No trained experiment at array index $SLURM_ARRAY_TASK_ID in $RESULTS_DIR"; exit 1; }

# Interpret.py overwrites both outputs, but drop any stale file so a crashed run never
# leaves a half-written CSV behind that looks current.
rm -f "$EXP/ablation_importance.csv" "$EXP/ablation_per_patient.csv"

echo "=== Ablation (2C): $EXP  (num_sample_pat=$NUM_SAMPLE_PAT set=$ABLATION_SET) ==="
python GenNet.py interpret -type Ablation \
    -resultpath "$EXP" \
    -num_sample_pat "$NUM_SAMPLE_PAT" \
    -ablation_set "$ABLATION_SET" \
    -ablation_verify "$ABLATION_VERIFY" \
    -ablation_per_patient
echo "=== done: $EXP -> $EXP/ablation_importance.csv + ablation_per_patient.csv ==="
