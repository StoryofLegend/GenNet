#!/bin/bash
# Multi-seed stability run — BEST softplus config from the grid search.
#
# TODO: fill in the winning hyperparameters once the softplus grid (exp 300-307)
#       is complete. Select by validation AUC from:
#         python pipeline/04_report/summarize_experiments.py results/softplus
#       then set LR and L1 below (and update this comment with the winner exp ID).
#
# Each seed_N directory holds a DIFFERENT train/val/test split (the seed is the
# random_state of the StratifiedShuffleSplit in create_subjects.py). Re-training
# the same config across seeds therefore measures stability across data splits.
#
# seed_42 = the grid winner (already trained) is the 5th member of the stability set.
#
#   task 1 -> exp 343: seed_43
#   task 2 -> exp 344: seed_44
#   task 3 -> exp 345: seed_45
#   task 4 -> exp 346: seed_46
#
#SBATCH --job-name=GenNet_ms_softplus
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_softplus_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_softplus_%a_%j.err
#SBATCH --partition=all_5days
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

# TODO: set these to the softplus grid winner before submitting.
LR=PLACEHOLDER_LR     # e.g. 0.001
L1=PLACEHOLDER_L1     # e.g. 0.001

SEEDS=(43 44 45 46)
EXP_IDS=(343 344 345 346)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=${SEEDS[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}

echo "exp ${EXP_ID}: seed_${SEED}, lr=${LR}, L1=${L1}, activation=softplus"

python GenNet.py train \
    -path processed_data/seed_${SEED}/ \
    -out results/ \
    -ID "$EXP_ID" \
    -epochs 200 \
    -bs 64 \
    -lr "$LR" \
    -L1 "$L1" \
    -hidden_activation softplus \
    -patience 15 \
    -workers 4 \
    -problem_type classification
