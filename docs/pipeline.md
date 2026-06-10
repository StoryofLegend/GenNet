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

---

## Step 2 — Build topology (SNP → gene → pathway)

The topology defines the network connectivity: which SNPs connect to which genes, and which genes connect to which pathways. It is shared across all seeds.

### Step 2a — Generate ANNOVAR input

Reads the converted probe file (`processed_data/probes/`) and writes the SNP coordinates in the 6-column format required by ANNOVAR.

**Script:** `pipeline/01_topology/01_create_annovar_input.sh`

```bash
python GenNet.py topology -type create_annovar_input \
    -path processed_data/ \
    -study_name gennnet_ibd \
    -out processed_data/
```

**Output:** `processed_data/annovar_input_gennnet_ibd.csv`

**Run:**
```bash
sbatch pipeline/01_topology/01_create_annovar_input.sh
```

---

### Step 2b — Run ANNOVAR

ANNOVAR annotates each SNP with the nearest gene(s) using the hg19 RefGene database.

**Prerequisites:**
1. Register and download ANNOVAR from https://annovar.openbioinformatics.org/en/latest/user-guide/download/
2. Set `ANNOVAR_DIR` in the script to the ANNOVAR installation folder
3. Download the RefGene database (one-time):
   ```bash
   cd $ANNOVAR_DIR
   perl annotate_variation.pl -buildver hg19 -downdb -webfrom annovar refGene humandb/
   ```

**Script:** `pipeline/01_topology/02_run_annovar.sh`

**Output:**
| File | Description |
|---|---|
| `processed_data/gennnet_ibd_RefGene_RefGene.variant_function` | Gene annotation per SNP |
| `processed_data/gennnet_ibd_RefGene.exonic_variant_function` | Exonic variant details |

**Run:**
```bash
sbatch pipeline/01_topology/02_run_annovar.sh
```

---

### Step 2c — Build SNP→gene topology

Parses the ANNOVAR output and creates `topology.csv` with the SNP→gene layer.

**Script:** `pipeline/01_topology/03_create_gene_network.sh`

```bash
python GenNet.py topology -type create_gene_network \
    -path processed_data/ \
    -study_name gennnet_ibd \
    -out processed_data/
```

**Output:** `processed_data/topology.csv`

**Known limitation:** for intergenic SNPs ANNOVAR reports two nearest genes (e.g. `GENEX(dist=z),GENEY(dist=k)`). GenNet keeps only the first. Step 2d adds back the second gene.

**Run:**
```bash
sbatch pipeline/01_topology/03_create_gene_network.sh
```

---

### Step 2d — Fix intergenic multi-gene annotations

Reads the raw ANNOVAR variant_function and adds the second nearest gene for every intergenic SNP. Both genes appear as separate nodes — biologically correct since the SNP is in the regulatory neighbourhood of both.

**Script:** `pipeline/01_topology/04_fix_intergenic.py`

**Output:** `processed_data/topology_with_intergenic.csv`

**Run:**
```bash
conda activate env_GenNet
python pipeline/01_topology/04_fix_intergenic.py
```

---

### Step 2e — Fix duplicate layer1_node IDs

After step 2d, both genes of an intergenic SNP share the same `layer1_node` ID (copied from the first gene). This makes two distinct genes map to the same network node. This script assigns unique IDs to the duplicates.

**Script:** `pipeline/01_topology/05_fix_topo_id.py`

**Output:**
- `processed_data/topology_fixed.csv`
- `processed_data/topology_fix_report.csv` (audit log of all reassignments)

**Run:**
```bash
python pipeline/01_topology/05_fix_topo_id.py
```

---

### Step 2f — Add pathway layer (ConsensusPathDB)

Joins genes to CPDB pathways. Genes with no pathway annotation are removed (they would be network dead-ends). CPDB was chosen over KEGG/Enrichr because it covers more genes.

**Prerequisites:** Download `CPDB_pathways_genes.tab` from http://cpdb.molgen.mpg.de/ → Downloads → Gene sets (HGNC) → tab-separated. Save to `processed_data/CPDB_pathways_genes.tab`.

**Script:** `pipeline/01_topology/06_add_cpdb_pathways.py`

**Output:** `processed_data/topology_final.csv`

| Stat | Value (IBD run) |
|---|---|
| Rows | 567,575 |
| Unique genes | 6,072 |
| Unique pathways | 4,513 |

**Run:**
```bash
python pipeline/01_topology/06_add_cpdb_pathways.py
```
