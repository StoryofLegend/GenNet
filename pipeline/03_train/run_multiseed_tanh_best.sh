#!/bin/bash
# Multi-seed stability run — BEST tanh config from the grid search.
# Best config selected by validation AUC: lr=0.001, L1=0.001, activation=tanh
# (grid winner = exp 105 on seed_42, valAUC 0.7036 / testAUC 0.7073).
#
# Each seed_N directory holds a DIFFERENT train/val/test split (the seed is the
# random_state of the StratifiedShuffleSplit in create_subjects.py). Re-training
# the same config across seeds therefore measures stability across data splits.
#
# seed_42 = exp 105 (already trained) is the 5th member of the stability set.
#
#   task 1 -> exp 143: seed_43
#   task 2 -> exp 144: seed_44
#   task 3 -> exp 145: seed_45
#   task 4 -> exp 146: seed_46
#
#SBATCH --job-name=GenNet_ms_tanh
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_tanh_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/ms_tanh_%a_%j.err
#SBATCH --partition=all_5days
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

SEEDS=(43 44 45 46)
EXP_IDS=(143 144 145 146)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=${SEEDS[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}

echo "exp ${EXP_ID}: seed_${SEED}, lr=0.001, L1=0.001, activation=tanh"

python GenNet.py train \
    -path processed_data/seed_${SEED}/ \
    -out results/ \
    -ID "$EXP_ID" \
    -epochs 200 \
    -bs 64 \
    -lr 0.001 \
    -L1 0.001 \
    -patience 15 \
    -workers 4 \
    -problem_type classification
