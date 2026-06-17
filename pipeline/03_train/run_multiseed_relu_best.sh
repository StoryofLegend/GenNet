#!/bin/bash
# Multi-seed stability run — BEST relu config from the grid search.
# Best config selected by validation AUC: lr=0.001, L1=0.001, activation=relu
# (grid winner = exp 205 on seed_42, valAUC 0.7255 / testAUC 0.7282).
#
# Each seed_N directory holds a DIFFERENT train/val/test split (the seed is the
# random_state of the StratifiedShuffleSplit in create_subjects.py). Re-training
# the same config across seeds therefore measures stability across data splits.
#
# seed_42 = exp 205 (already trained) is the 5th member of the stability set.
#
#   task 1 -> exp 243: seed_43
#   task 2 -> exp 244: seed_44
#   task 3 -> exp 245: seed_45
#   task 4 -> exp 246: seed_46
#
#SBATCH --job-name=GenNet_ms_relu
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_relu_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_relu_%a_%j.err
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

SEEDS=(43 44 45 46)
EXP_IDS=(243 244 245 246)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=${SEEDS[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}

echo "exp ${EXP_ID}: seed_${SEED}, lr=0.001, L1=0.001, activation=relu"

python GenNet.py train \
    -path processed_data/seed_${SEED}/ \
    -out results/ \
    -ID "$EXP_ID" \
    -epochs 200 \
    -bs 64 \
    -lr 0.001 \
    -L1 0.001 \
    -hidden_activation relu \
    -patience 15 \
    -workers 4 \
    -problem_type classification
