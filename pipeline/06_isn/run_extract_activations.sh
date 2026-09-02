#!/bin/bash
# Per-patient node activations for ISN construction (step 2 of the ISN/HotZone flow).
#
# Runs pipeline/06_isn/extract_activations.py, which does ONE forward pass per seed
# over the whole test set (cases + controls, 9942 subjects) and writes into each
# experiment folder:
#
#   isn_gene_act.npy       (n_patients, n_slots)     gene-layer activations
#   isn_gene_slots.csv     slot -> layer1_node, gene
#   isn_pathway_act.npy    (n_patients, 4510)        pathway-layer activations
#   isn_pathway_nodes.csv  slot -> layer2_node, pathway
#   isn_subjects.csv       patient_id, labels, cov_* in row order
#
# --- Which subjects ---
# The WHOLE test set, not just cases. get_data(sample_pat=0) skips the
# .sample(random_state=1) permutation, so rows follow subjects.csv order; the script
# then ASSERTS that subjects.csv labels equal the loader's y before writing anything.
# Sec 3.4 compares ISNs between high- and low-risk individuals, so controls are
# needed - ablation_per_patient.csv (2C) is cases-only and positionally indexed, and
# cannot be used here.
#
# --- Which tensor ---
# Post-BatchNorm by default: the value that actually propagates into the next
# LocallyDirected layer, and the same point method 2C ablates at. `--layer activation`
# gives the pre-BN tanh instead. These BN layers are center=False, scale=False, so at
# inference they apply a POSITIVE per-node affine map - Pearson correlation across
# individuals is invariant to it, so the choice changes nothing for the LIONESS route
# and matters only for the multiplicative GenNet-native ISN.
#
# --- Size ---
# Only the slots belonging to the ISN gene sets are kept (union over every
# gene_set_*.csv written by pipeline/06_isn/gene_sets.py), so a seed costs tens of MB
# instead of 1.75 GB. Re-running is cheap (~1 min/seed), so widen the gene sets and
# re-run rather than hoarding the full layer; --all-genes is there if you need it.
#
# Prerequisite:
#   python pipeline/06_isn/gene_sets.py results/tanh
#
# Submit:
#   sbatch pipeline/06_isn/run_extract_activations.sh              # defaults to results/tanh
#   sbatch pipeline/06_isn/run_extract_activations.sh results/relu
#
#SBATCH --job-name=GenNet_isn_act
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/extract_activations_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/extract_activations_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=all_5hrs
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

echo "=== ISN activations: $RESULTS_DIR ==="
python pipeline/06_isn/extract_activations.py "$RESULTS_DIR" "${@:2}"
echo "=== done ==="
