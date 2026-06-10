#!/bin/bash
# Step 3a — Create subjects.csv for each seed (array job 42-46).
# Output: processed_data/seed_N/subjects.csv
#
#SBATCH --job-name=GenNet_subjects
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/02_subjects/subjects_%a_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/02_subjects/subjects_%a_%j.err
#SBATCH --partition=all_5hrs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=42-46
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

python pipeline/02_subjects/create_subjects.py --seed "$SLURM_ARRAY_TASK_ID"
