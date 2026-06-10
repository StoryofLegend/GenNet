#!/bin/bash
# Step 3b — Add topology.csv symlink and genotype.h5 symlink to each seed directory.
# Run after create_subjects.py has populated processed_data/seed_N/subjects.csv.

BASE_DIR="/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet"

for SEED in 42 43 44 45 46; do
    SEED_DIR="${BASE_DIR}/processed_data/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    cp  "${BASE_DIR}/processed_data/topology_final.csv" "${SEED_DIR}/topology.csv"
    ln -sf "${BASE_DIR}/processed_data/genotype.h5"    "${SEED_DIR}/genotype.h5"

    echo "seed_${SEED}: OK"
done
