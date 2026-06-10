"""
Step 3a — Create subjects.csv for a given seed.

Split: 65% train (set=1), 20% val (set=2), 15% test (set=3), stratified by label.
Each seed produces a different shuffle; all other columns are identical across seeds.

Usage: python create_subjects.py --seed 42
Output: processed_data/seed_<N>/subjects.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from pandas_plink import read_plink1_bin
from sklearn.model_selection import StratifiedShuffleSplit

PLINK_DIR  = "/massstorage/URT/GEN/BIO3/Arslan_Ahmed/IBD_students"
PLINK_BASE = "CD_UC_CON_QCed_rel1_without_relatives_maf0.05_hwe0.001_Liu2015_232SNPs_LD0.75_noFilter_binary"
PCA_FILE   = os.path.join(PLINK_DIR, "CD_UC_CON_QCed_rel1_without_relatives.pca.evec")
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

OUT_FILE = os.path.join(BASE_DIR, "processed_data", f"seed_{args.seed}", "subjects.csv")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

G = read_plink1_bin(
    os.path.join(PLINK_DIR, f"{PLINK_BASE}.bed"),
    os.path.join(PLINK_DIR, f"{PLINK_BASE}.bim"),
    os.path.join(PLINK_DIR, f"{PLINK_BASE}.fam"),
    verbose=False,
)
print(f"Subjects: {G.shape[0]}, SNPs: {G.shape[1]}")

subjects = pd.DataFrame({"patient_id": G.iid.values, "labels": G.trait.values})
subjects["labels"] = subjects["labels"].astype(str).replace({"1": 0, "2": 1}).astype(int)

# 15% test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
train_val_idx, test_idx = next(sss.split(np.zeros(len(subjects)), subjects["labels"]))
subjects.loc[test_idx, "set"] = 3

# 20% of total = 23.53% of remaining
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2353, random_state=args.seed)
train_idx, val_idx = next(sss.split(np.zeros(len(train_val_idx)), subjects.loc[train_val_idx, "labels"]))
subjects.loc[train_val_idx[train_idx], "set"] = 1
subjects.loc[train_val_idx[val_idx],   "set"] = 2
subjects["set"] = subjects["set"].astype(int)

print(f"Seed {args.seed} split:")
for s, name in [(1,"Train"),(2,"Val"),(3,"Test")]:
    sub = subjects[subjects["set"]==s]
    print(f"  {name:5}: {len(sub):6}  ({100*sub['labels'].mean():.1f}% cases)")

subjects["chunk_id"] = 0

pca = pd.read_csv(PCA_FILE, sep=r"\s+")
pca = pca.rename(columns={f"PC{i}": f"cov_{i}" for i in range(1, 33)})
subjects = subjects.merge(pca[["IID"] + [f"cov_{i}" for i in range(1, 8)]],
                          left_on="patient_id", right_on="IID", how="left")
subjects = subjects.drop(columns=["IID"])
subjects["genotype_row"] = np.arange(len(subjects))
subjects = subjects[["patient_id","labels","genotype_row","set","chunk_id"] +
                    [f"cov_{i}" for i in range(1, 8)]]

subjects.to_csv(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE}")
