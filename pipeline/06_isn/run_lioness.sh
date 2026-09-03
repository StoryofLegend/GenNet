#!/bin/bash
# LIONESS individual-specific networks over the method x cutoff x seed grid (step 4).
#
# One SLURM array task per configuration. Each runs pipeline/06_isn/lioness.py on the
# matching results/tanh/isn_input/isn_input_<method>_top<N>_seed<S>.csv and writes
# results/tanh/isn/<method>_top<N>_seed<S>/ containing:
#
#   isn_edges.csv       source, target (upper triangle, symmetrised, no self-loops)
#   isn_weights.npy     (n_edges, n_patients) float32
#   isn_patients.csv    real patient_ids, in column order
#   isn_edge_stats.csv  per-edge mean/sd/var across individuals -> HotZone input
#   isn_run_info.json   shapes, timings, asymmetry diagnostics
#
# --- Grid ---
# METHODS x CUTOFFS x SEEDS x REFS, in that nesting order.
#
# Gene sets are already a 5-seed consensus (pipeline/06_isn/gene_sets.py ranks by mean
# rank across seeds), so the seed here selects only which model's ACTIVATIONS are used,
# not which genes. Default is one seed; add more to SEEDS to test whether HotZones are
# stable across training runs. Note the 5 seeds have near-disjoint test cohorts (~15%
# overlap), so cross-seed comparison must be done at the edge/gene level, never patient
# by patient.
#
# --- Reference population (REFS) ---
# The supervisor's two strategies:
#   REFS=01  all test subjects, cases and controls -> ISN_01,i   (default)
#   REFS=1   cases only                            -> ISN_1,i
# The strategy is baked into the input matrix by make_isn_input.py --reference, not
# chosen here: LIONESS takes its reference from every sample in the file it is given.
# Run BOTH arms with `REFS="01 1"`, having built both sets of matrices first.
# 01 keeps the historical unsuffixed paths; 1 writes isn/<...>_ref1/.
#
# --- INPUT_0 ---
# The supervisor's INPUT_0 = union of the top-N sets of 2A, 2B and 2C, so it is much
# larger than its cutoff label: top50 -> 78 genes, top100 -> 181, top150 -> 283,
# top200 -> 387, top250 -> 486 (results/tanh; see gene_set_sizes.csv). INPUT_B is
# method 2B at the same cutoff. Memory scales with the SQUARE of the gene count, so
# INPUT_0 top100 costs roughly what a real top181 would - budget from the gene count
# in gene_set_sizes.csv, never from the cutoff. Rough guide at 9,942 patients:
#   INPUT_0 top50  (78 genes)  ~0.5 GB float -> fits the 60G default
#   INPUT_0 top100 (181 genes) ~2.6 GB float -> raise to --mem=120G
#   INPUT_0 top150+ (283+)     >6 GB float   -> use the 180G/%2 density arm below
# The cases-only arm halves n (9,942 -> 4,893), so it costs about half the above.
#
# --- Cutoffs ---
# 50 and 100 by default. At top250 more than half of every gene set is inactive
# (sd < 1e-4 across patients: 2A 70%, 2C 59%), and a HotZone is by definition a
# subnetwork that VARIES across individuals, so those genes contribute only noise.
# Run 150/200/250 as the edge-density sensitivity arm, not as primary results.
#
# --- Cost ---
# Output is symmetrised upper-triangle float32: ~49 MB at top50, ~197 MB at top100 per
# configuration. The old full-N2 CSV format was 1.9 GB at top100 and 12 GB at top250,
# which is what made a 4-method grid impossible.
#
# Submit (array size must equal METHODS x CUTOFFS x SEEDS x REFS):
#   sbatch --array=1-10 pipeline/06_isn/run_lioness.sh                 # 5 methods x 2 cutoffs
#   sbatch --array=1-10 pipeline/06_isn/run_lioness.sh results/tanh
#
# Both reference strategies, one seed, two cutoffs (5 x 2 x 1 x 2 = 20):
#   REFS="01 1" sbatch --export=ALL --array=1-20%4 pipeline/06_isn/run_lioness.sh
#
# Just the supervisor's two arms, INPUT_0 vs INPUT_B (= 2B), both references (2x2x1x2=8):
#   METHODS="INPUT_0 2B" CUTOFFS="50 100" REFS="01 1" sbatch --export=ALL \
#       --mem=120G --array=1-8%2 pipeline/06_isn/run_lioness.sh
#
# Density arm (12 configs). Memory is set by netZooPy, not by our output: it holds the
# FULL directed N x N block per patient (not the N(N-1)/2 upper triangle we keep), as
# float64, and then builds an object-dtype copy of the whole thing to label it.
#   top100: 9942 x 100^2 x 8 B  = 0.8 GB float  -> ~9 GB peak observed
#   top250: 9942 x 250^2 x 8 B  = 5.0 GB float  -> tens of GB peak
# lioness.py drops that labelled copy as soon as netZooPy returns, but the peak happens
# inside netZooPy itself. urtgen nodes have 183 GB, so --mem=180G takes a whole node and
# only one task can run at a time; throttle with %N rather than raising memory further.
#   CUTOFFS="150 200 250" sbatch --export=ALL --mem=180G --array=1-12%2 \
#       pipeline/06_isn/run_lioness.sh
#
#SBATCH --job-name=GenNet_lioness
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/lioness_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/lioness_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=all_5hrs
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"
# Grid is overridable from the environment so the density arm can be submitted without
# editing the file:  CUTOFFS="150 200 250" sbatch --array=1-12 ... --export=ALL,CUTOFFS
read -r -a METHODS <<< "${METHODS:-2A 2B 2C combined INPUT_0}"
read -r -a CUTOFFS <<< "${CUTOFFS:-50 100}"
read -r -a SEEDS   <<< "${SEEDS:-42}"
read -r -a REFS    <<< "${REFS:-01}"
NCORES="${NCORES:-4}"

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

# Build the grid in a fixed order so an array index always maps to the same config.
CONFIGS=()
for m in "${METHODS[@]}"; do
  for n in "${CUTOFFS[@]}"; do
    for s in "${SEEDS[@]}"; do
      for r in "${REFS[@]}"; do
        CONFIGS+=("$m $n $s $r")
      done
    done
  done
done

TOTAL=${#CONFIGS[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "$TOTAL" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID out of range; the grid has $TOTAL configs"
    echo "submit with --array=1-$TOTAL"
    exit 1
fi

read -r METHOD CUTOFF SEED REF <<< "${CONFIGS[$IDX]}"
# reference 01 (all subjects) keeps the historical, unsuffixed names so nothing
# already computed is invalidated; cases-only gets _ref1 on both input and outdir.
SUFFIX=""
[ "$REF" = "01" ] || SUFFIX="_ref${REF}"
INPUT="$RESULTS_DIR/isn_input/isn_input_${METHOD}_top${CUTOFF}_seed${SEED}${SUFFIX}.csv"
OUTDIR="$RESULTS_DIR/isn/${METHOD}_top${CUTOFF}_seed${SEED}${SUFFIX}"

[ -f "$INPUT" ] || { echo "missing input: $INPUT (run make_isn_input.py --reference $REF first)"; exit 1; }

echo "=== LIONESS [$SLURM_ARRAY_TASK_ID/$TOTAL]: method=$METHOD cutoff=$CUTOFF seed=$SEED ref=$REF (ISN_${REF},i) ==="
echo "    input : $INPUT"
echo "    outdir: $OUTDIR"

python pipeline/06_isn/lioness.py --input "$INPUT" --outdir "$OUTDIR" \
    --ncores "$NCORES" --reference "$REF"

echo "=== done: $OUTDIR ==="
