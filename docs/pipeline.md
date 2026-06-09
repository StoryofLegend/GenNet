# GenNet IBD Pipeline

**Author:** Kristian Kovacev  
**Dataset:** IBD GWAS — Crohn's Disease + Ulcerative Colitis + Controls  
**Goal:** Train GenNet on IBD GWAS data, extract gene/pathway importance, build Individual-Specific Networks (ISNs)

---

## Data

Raw PLINK files (read-only, do not modify):
```
/massstorage/URT/GEN/BIO3/Arslan_Ahmed/IBD_students/
  CD_UC_CON_QCed_rel1_without_relatives_maf0.05_hwe0.001_Liu2015_232SNPs_LD0.75_noFilter_binary.bed
  CD_UC_CON_QCed_rel1_without_relatives_maf0.05_hwe0.001_Liu2015_232SNPs_LD0.75_noFilter_binary.bim
  CD_UC_CON_QCed_rel1_without_relatives_maf0.05_hwe0.001_Liu2015_232SNPs_LD0.75_noFilter_binary.fam
  CD_UC_CON_QCed_rel1_without_relatives.pca.evec
```

| Stat | Value |
|---|---|
| Total subjects | 66,280 |
| Cases (CD + UC) | 32,622 |
| Controls | 33,658 |
| SNPs | 38,225 |

---

## Step 0 — Setup: symlinks to PLINK files

Instead of copying the large PLINK files (~3 GB), symlinks are created in `raw_data_input/`:

```bash
PLINK_DIR="/massstorage/URT/GEN/BIO3/Arslan_Ahmed/IBD_students"
PLINK_BASE="CD_UC_CON_QCed_rel1_without_relatives_maf0.05_hwe0.001_Liu2015_232SNPs_LD0.75_noFilter_binary"

mkdir -p raw_data_input
ln -sf ${PLINK_DIR}/${PLINK_BASE}.bed raw_data_input/
ln -sf ${PLINK_DIR}/${PLINK_BASE}.bim raw_data_input/
ln -sf ${PLINK_DIR}/${PLINK_BASE}.fam raw_data_input/
```

---

## Step 1 — Convert PLINK to HDF5

GenNet requires genotype data in HDF5 format. The Dataloader reads it in mini-batches during training so the full matrix (66,280 × 38,225) never needs to fit in RAM.

**Script:** `pipeline/00_setup/run_convert.sh`

```bash
python GenNet.py convert \
    -g raw_data_input \
    -study_name gennnet_ibd \
    -o processed_data/
```

**Output:**

| File | Description |
|---|---|
| `processed_data/genotype.h5` | Main genotype matrix (~700 MB) |
| `processed_data/gennnet_ibd_std.npy` | Per-SNP mean and std for normalisation |
| `processed_data/gennnet_ibd_step2_merged_genotype.h5` | Intermediate merge |
| `processed_data/gennnet_ibd_step3_genotype_no_missing.h5` | Missing values removed |

**Run:**
```bash
sbatch pipeline/00_setup/run_convert.sh
```
