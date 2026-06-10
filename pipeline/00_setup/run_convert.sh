#!/bin/bash
#
# Converts PLINK files to HDF5 format required by GenNet.
# Input:  raw_data_input/ (symlinks to PLINK .bed/.bim/.fam)
# Output: processed_data/genotype.h5
#
#SBATCH --job-name=GenNet_convert
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/00_setup/conv_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/00_setup/conv_%j.err
#SBATCH --partition=all_5hrs
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"
cd "$BASE_DIR"

python GenNet.py convert \
    -g raw_data_input \
    -study_name gennnet_ibd \
    -o processed_data/

