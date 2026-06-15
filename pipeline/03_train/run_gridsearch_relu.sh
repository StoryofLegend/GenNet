#!/bin/bash
# Grid search — relu hidden activation, seed 42.
# 8 combinations: lr ∈ {0.0001, 0.001, 0.01, 0.1} × L1 ∈ {0.01, 0.001}
#
#   task 1 -> exp 200: lr=0.0001, L1=0.01
#   task 2 -> exp 201: lr=0.001,  L1=0.01
#   task 3 -> exp 202: lr=0.01,   L1=0.01
#   task 4 -> exp 203: lr=0.1,    L1=0.01
#   task 5 -> exp 204: lr=0.0001, L1=0.001
#   task 6 -> exp 205: lr=0.001,  L1=0.001
#   task 7 -> exp 206: lr=0.01,   L1=0.001
#   task 8 -> exp 207: lr=0.1,    L1=0.001
#
#SBATCH --job-name=GenNet_gs_relu
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/gs_relu_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/gs_relu_%a_%j.err
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=90000
#SBATCH --array=1-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

LR_VALUES=(0.0001 0.001 0.01 0.1 0.0001 0.001 0.01 0.1)
L1_VALUES=(0.01   0.01  0.01 0.01 0.001  0.001 0.001 0.001)
EXP_IDS=(200 201 202 203 204 205 206 207)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
LR=${LR_VALUES[$IDX]}
L1=${L1_VALUES[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}

echo "exp ${EXP_ID}: lr=${LR}, L1=${L1}, activation=relu"

python GenNet.py train \
    -path processed_data/seed_42/ \
    -out results/ \
    -ID "$EXP_ID" \
    -epochs 200 \
    -bs 64 \
    -lr "$LR" \
    -L1 "$L1" \
    -patience 15 \
    -hidden_activation relu \
    -problem_type classification
