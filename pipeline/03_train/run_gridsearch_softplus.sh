#!/bin/bash
# Grid search — softplus hidden activation, seed 42.
# 8 combinations: lr ∈ {0.0001, 0.001, 0.01, 0.1} × L1 ∈ {0.01, 0.001}
#
#   task 1 -> exp 300: lr=0.0001, L1=0.01
#   task 2 -> exp 301: lr=0.001,  L1=0.01
#   task 3 -> exp 302: lr=0.01,   L1=0.01
#   task 4 -> exp 303: lr=0.1,    L1=0.01
#   task 5 -> exp 304: lr=0.0001, L1=0.001
#   task 6 -> exp 305: lr=0.001,  L1=0.001
#   task 7 -> exp 306: lr=0.01,   L1=0.001
#   task 8 -> exp 307: lr=0.1,    L1=0.001
#
#SBATCH --job-name=GenNet_gs_softplus
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/gs_softplus_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/03_train/gs_softplus_%a_%j.err
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
EXP_IDS=(300 301 302 303 304 305 306 307)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
LR=${LR_VALUES[$IDX]}
L1=${L1_VALUES[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}

echo "exp ${EXP_ID}: lr=${LR}, L1=${L1}, activation=softplus"

python GenNet.py train \
    -path processed_data/seed_42/ \
    -out results/ \
    -ID "$EXP_ID" \
    -epochs 200 \
    -bs 64 \
    -lr "$LR" \
    -L1 "$L1" \
    -patience 15 \
    -hidden_activation softplus \
    -problem_type classification
