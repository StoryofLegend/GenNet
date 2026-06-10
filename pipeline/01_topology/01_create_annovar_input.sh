#!/bin/bash
# Step 2a — Generate ANNOVAR input from processed_data/probes/.
# Output: processed_data/annovar_input_gennnet_ibd.csv
#
#SBATCH --job-name=GenNet_annovar_input
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/annovar_input_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/annovar_input_%j.err
#SBATCH --partition=all_5hrs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

python GenNet.py topology -type create_annovar_input \
    -path processed_data/ \
    -study_name gennnet_ibd \
    -out processed_data/
