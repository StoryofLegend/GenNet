#!/bin/bash
#
# Step 2c — Build SNP→gene topology from ANNOVAR annotations.
# Reads the .variant_function file and writes topology.csv with columns:
#   chr, layer0_node, layer0_name (SNP), layer1_node, layer1_name (gene)
#
# NOTE: for intergenic SNPs ANNOVAR reports the two nearest genes, e.g.
#   GENEX(dist=z),GENEY(dist=k)
# GenNet keeps only the FIRST nearest gene. The next step (04_clean_topology.py)
# fixes this by adding back the second gene so both are represented.
#
# Input:  processed_data/gennnet_ibd_RefGene_RefGene.variant_function
# Output: processed_data/topology.csv  (SNP→gene, no pathway layer yet)
#
#SBATCH --job-name=GenNet_gene_network
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/gene_network_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/gene_network_%j.err
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

python GenNet.py topology -type create_gene_network \
    -path processed_data/ \
    -study_name gennnet_ibd \
    -out processed_data/
