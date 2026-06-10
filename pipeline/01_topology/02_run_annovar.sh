#!/bin/bash
#
# Step 2b — Run ANNOVAR to annotate SNPs with gene information (refGene, hg19).
#
# PREREQUISITES:
#   1. Download ANNOVAR from https://annovar.openbioinformatics.org/en/latest/user-guide/download/
#      (requires free registration)
#   2. Set ANNOVAR_DIR below to the folder containing annotate_variation.pl
#   3. Download the refGene database:
#        cd $ANNOVAR_DIR
#        perl annotate_variation.pl -buildver hg19 -downdb -webfrom annovar refGene humandb/
#
# Input:  processed_data/annovar_input_gennnet_ibd.csv
# Output: processed_data/gennnet_ibd_RefGene_RefGene.variant_function
#         processed_data/gennnet_ibd_RefGene.exonic_variant_function
#
#SBATCH --job-name=GenNet_annovar
#SBATCH --output=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/annovar_%j.out
#SBATCH --error=/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet/logs/01_topology/annovar_%j.err
#SBATCH --partition=all_5hrs
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.kovacev@campus.unimib.it

ANNOVAR_DIR="/home/u/f099193/annovar"

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"

perl "${ANNOVAR_DIR}/annotate_variation.pl" \
    -geneanno \
    -dbtype refGene \
    -buildver hg19 \
    "${BASE_DIR}/processed_data/annovar_input_gennnet_ibd.csv" \
    "${ANNOVAR_DIR}/humandb/" \
    --outfile "${BASE_DIR}/processed_data/gennnet_ibd_RefGene"
