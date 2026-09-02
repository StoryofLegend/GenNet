#!/bin/bash
# Run the SUPERVISOR'S LIONESS script over the method x cutoff x seed grid.
#
# This deliberately runs the supervisor's code, not the rewrite. The file executed is
# pipeline/06_isn/lioness_original.py. Only two things were changed in it (23 lines):
# --input/--outdir replacing the hardcoded methylation CSV path, and the pip-install
# block turned into an error (it must not rewrite the conda env mid-job). The untouched
# copy is kept as lioness_original.py.supervisor_untouched. Prior, orientation, PANDA
# and LIONESS calls are the supervisor's, untouched.
#
# Each task writes results/tanh/isn_original/<method>_top<N>_seed<S>/ containing:
#   labeled_lioness_data.csv        tf, gene, Sample_1..Sample_N  (both directions + self-loops)
#   labeled_lioness_data_clean.csv  after clean_lioness.py (self-loops and one direction dropped)
#   lioness_top_10.png, lioness_top_100.png
#   lioness_output/                 netZooPy's raw dump
#
# --- Disk ---
# The output is a full N x N CSV per config, NOT an upper triangle:
#   top50  (2,500 rows x 9,942 patients) ~  480 MB, ~960 MB with the cleaned copy
#   top100 (10,000 rows x 9,942 patients) ~ 1.9 GB, ~3.8 GB with the cleaned copy
# The full 4-method x {50,100} x 5-seed grid is therefore ~95 GB. Check free space
# before submitting the whole thing; this account has hit its quota once already.
#
# Submit:
#   sbatch --array=1-40%4 pipeline/06_isn/run_lioness_original.sh              # everything
#   CUTOFFS=50 SEEDS=42 sbatch --export=ALL --array=1-4 pipeline/06_isn/run_lioness_original.sh
#
#SBATCH --job-name=GenNet_lioness_orig
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/lioness_orig_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/06_isn/lioness_orig_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=k.kovacev@campus.unimib.it
#SBATCH --partition=all_5hrs
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

set -euo pipefail

RESULTS_DIR="${1:-results/tanh}"
read -r -a METHODS <<< "${METHODS:-2A 2B 2C combined}"
read -r -a CUTOFFS <<< "${CUTOFFS:-50 100}"
read -r -a SEEDS   <<< "${SEEDS:-42 43 44 45 46}"

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

cd "$(git rev-parse --show-toplevel)"

CONFIGS=()
for m in "${METHODS[@]}"; do
  for n in "${CUTOFFS[@]}"; do
    for s in "${SEEDS[@]}"; do
      CONFIGS+=("$m $n $s")
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

read -r METHOD CUTOFF SEED <<< "${CONFIGS[$IDX]}"
INPUT="$(pwd)/$RESULTS_DIR/isn_input/isn_input_${METHOD}_top${CUTOFF}_seed${SEED}.csv"
OUTDIR="$(pwd)/$RESULTS_DIR/isn_original/${METHOD}_top${CUTOFF}_seed${SEED}"

[ -f "$INPUT" ] || { echo "missing input: $INPUT (run make_isn_input.py first)"; exit 1; }
mkdir -p "$OUTDIR"

echo "=== LIONESS (supervisor's script) [$SLURM_ARRAY_TASK_ID/$TOTAL]: $METHOD top$CUTOFF seed$SEED ==="
echo "    input : $INPUT"
echo "    outdir: $OUTDIR"

# The supervisor's script prints a PANDA progress line per solver step per patient
# (17 MB per run). Keep that out of the SLURM log: 40 tasks writing that concurrently
# is what exhausted the disk quota before. It is still on disk as netzoo_progress.log.
python pipeline/06_isn/lioness_original.py \
    --input "$INPUT" --outdir "$OUTDIR" > "$OUTDIR/netzoo_progress.log" 2>&1

echo "--- cleaning: self-loops and bidirectional duplicates ---"
python pipeline/06_isn/clean_lioness.py "$OUTDIR"

# netZooPy's raw dump: 192 MB per top50 config, 768 MB per top100, ~18 GB over the full
# grid. The supervisor's script has already extracted everything from it into
# labeled_lioness_data.csv and nothing downstream reads it. Set KEEP_RAW=1 to retain.
if [ "${KEEP_RAW:-0}" != "1" ] && [ -d "$OUTDIR/lioness_output" ]; then
    echo "--- removing lioness_output/ ($(du -sh "$OUTDIR/lioness_output" | cut -f1) netZooPy raw dump; KEEP_RAW=1 to retain) ---"
    rm -rf "$OUTDIR/lioness_output"
fi

ls -la "$OUTDIR"
echo "=== done: $OUTDIR ==="
