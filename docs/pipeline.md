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

---

## Step 3 — subjects.csv and seed directories

### Step 3a — Create subjects.csv

Reads the PLINK .fam file and the PCA eigenvector file, creates a stratified 65/20/15 train/val/test split, and saves `subjects.csv` into the seed directory. The `--seed` value is the `random_state` of the `StratifiedShuffleSplit`, so **each seed produces a different split** — this is the only thing that varies across seeds. (GenNet training itself has no seed flag, so model-weight initialisation is a separate, uncontrolled source of variation.) All other columns are identical across seeds.

**Script:** `pipeline/02_subjects/create_subjects.py` (`--seed N`), driven across seeds 42–46 by the array job `pipeline/02_subjects/run_create_subjects.sh`.

| Column | Description |
|---|---|
| `patient_id` | Subject identifier from .fam |
| `labels` | 0 = control, 1 = case (converted from PLINK 1/2 encoding) |
| `genotype_row` | Row index in genotype.h5 |
| `set` | 1 = train, 2 = val, 3 = test |
| `cov_1…cov_7` | First 7 principal components |

**Output:** `processed_data/seed_<N>/subjects.csv` (one per seed)

**Run:**
```bash
sbatch pipeline/02_subjects/run_create_subjects.sh
```

---

### Step 3b — Set up per-seed directories

Run *after* Step 3a (which already wrote each `seed_N/subjects.csv`). For seeds 42–46 it adds a copy of `topology_final.csv` (as `topology.csv`) and a symlink to the shared `genotype.h5` into each seed directory.

**Script:** `pipeline/02_subjects/setup_seed_dirs.sh`

**Run:**
```bash
bash pipeline/02_subjects/setup_seed_dirs.sh
```

---

## Step 4 — Train GenNet (hyperparameter grid search)

Trains the GenNet model on `processed_data/seed_42/`. Three grid searches are
run, one per hidden-layer activation function (tanh, relu, softplus), to compare
activation choices (Month 2 methodological comparison — see
[`project_context.md`](project_context.md)).

Each grid search is a SLURM **job array of 8 tasks** sweeping:

- learning rate ∈ `{0.0001, 0.001, 0.01, 0.1}`
- L1 regularisation ∈ `{0.01, 0.001}`

Common settings: `-epochs 200`, `-bs 64`, `-patience 15`,
`-problem_type classification`. Each task writes to its own experiment ID under
`results/GenNet_experiment_<ID>_/`.

### Step 4a — tanh activation (default)

`-hidden_activation` is omitted, so GenNet uses its default for classification
(**tanh**; relu is the default for regression — see `GenNet.py:261`).

**Script:** `pipeline/03_train/run_gridsearch_tanh.sh`

| Array task | Exp ID | lr | L1 |
|---|---|---|---|
| 1 | 100 | 0.0001 | 0.01 |
| 2 | 101 | 0.001 | 0.01 |
| 3 | 102 | 0.01 | 0.01 |
| 4 | 103 | 0.1 | 0.01 |
| 5 | 104 | 0.0001 | 0.001 |
| 6 | 105 | 0.001 | 0.001 |
| 7 | 106 | 0.01 | 0.001 |
| 8 | 107 | 0.1 | 0.001 |

**Run:**
```bash
sbatch pipeline/03_train/run_gridsearch_tanh.sh
```

### Step 4b — relu activation

Same grid, with `-hidden_activation relu`, experiment IDs offset to the 200 range.

**Script:** `pipeline/03_train/run_gridsearch_relu.sh`

| Array task | Exp ID | lr | L1 |
|---|---|---|---|
| 1 | 200 | 0.0001 | 0.01 |
| 2 | 201 | 0.001 | 0.01 |
| 3 | 202 | 0.01 | 0.01 |
| 4 | 203 | 0.1 | 0.01 |
| 5 | 204 | 0.0001 | 0.001 |
| 6 | 205 | 0.001 | 0.001 |
| 7 | 206 | 0.01 | 0.001 |
| 8 | 207 | 0.1 | 0.001 |

**Run:**
```bash
sbatch pipeline/03_train/run_gridsearch_relu.sh
```

### Step 4c — softplus activation

Same grid, with `-hidden_activation softplus`, experiment IDs in the 300 range.
Softplus (`log(1 + eˣ)`) is a smooth, strictly-positive relu analog, chosen as a
third activation to test (Month 2 comparison). Its smoothness gives more stable
gradient/SHAP-based importance than relu's kink at 0, and its non-negativity
avoids sign-cancellation when ISN edges are formed as products of node
activations — both relevant to the project's importance-stability focus (see
[`project_context.md`](project_context.md)).

**Script:** `pipeline/03_train/run_gridsearch_softplus.sh`

| Array task | Exp ID | lr | L1 |
|---|---|---|---|
| 1 | 300 | 0.0001 | 0.01 |
| 2 | 301 | 0.001 | 0.01 |
| 3 | 302 | 0.01 | 0.01 |
| 4 | 303 | 0.1 | 0.01 |
| 5 | 304 | 0.0001 | 0.001 |
| 6 | 305 | 0.001 | 0.001 |
| 7 | 306 | 0.01 | 0.001 |
| 8 | 307 | 0.1 | 0.001 |

**Run:**
```bash
sbatch pipeline/03_train/run_gridsearch_softplus.sh
```

### Output (per experiment)

Each `results/GenNet_experiment_<ID>_/` contains:

| File | Description |
|---|---|
| `bestweights_job.h5` | Best model weights (lowest val loss) |
| `connection_weights.csv` | Learned layer connection weights (input for importance analysis) |
| `model_architecture.txt` | Layer-by-layer architecture |
| `train_args.json` | Exact arguments used for the run |
| `train_log.csv` | Per-epoch loss/metrics |
| `train_val_loss.png` | Train vs validation loss curve |
| `pval.npy`, `ptest.npy` | Predicted probabilities (val / test) |
| `results_summary.txt`, `pd_summary_results.csv` | Final performance summary |

---

## Step 5 — Report grids, then multi-seed stability (best config per activation)

After each grid search finishes, the runs are organised by activation, summarised
into a CSV, and the best config **per activation** is re-trained on every seed
directory. Because each `seed_N` holds a different train/val/test split (Step 3a),
this measures stability of performance and of gene/pathway importance across
splits (a core goal — see [`project_context.md`](project_context.md)).

> **Both activations are carried forward, not just the single best one.** Per the
> supervisor's methodological clarification (see [`project_context.md`](project_context.md)
> §2), the *whole* pipeline is evaluated per configuration, so tanh and relu each
> get a multi-seed run and continue into ISN/HotZone — we do not pick one
> activation at the GenNet level and drop the other.

### Step 5a — Organise results by activation and summarise the grids

The experiment folders are grouped into per-activation subfolders for browsing.
The experiment IDs already namespace the activation (100s = tanh, 200s = relu,
300s = softplus), so the move is a same-filesystem rename (instant, no copy of the
multi-GB `connection_weights.csv`):

```bash
mkdir -p results/tanh results/relu results/softplus
mv results/GenNet_experiment_10?_ results/tanh/
mv results/GenNet_experiment_20?_ results/relu/
mv results/GenNet_experiment_30?_ results/softplus/   # when softplus is done
```

> GenNet always writes to `results/` (top level); it has no notion of these
> subfolders. So every later run (including the multi-seed jobs below) lands in
> `results/GenNet_experiment_<ID>_/` and must be moved into its activation folder
> afterwards.

**Script:** `pipeline/04_report/summarize_gridsearch.py` — point it at one
activation folder; it reads each run's `train_args.json` (hyperparameters) and
`pd_summary_results.csv` (AUCs), and writes one CSV sorted by validation AUC into
`reports/gridsearch/`. The activation is read from `hidden_activation` in the
JSON (not the folder name), so a misfiled run still lands in the right CSV.

```bash
python pipeline/04_report/summarize_gridsearch.py results/tanh   # -> reports/gridsearch/tanh_gridsearch.csv
python pipeline/04_report/summarize_gridsearch.py results/relu   # -> reports/gridsearch/relu_gridsearch.csv
```

`results/` is gitignored (large weight files), but `reports/` is **not** — the
small CSVs there are the publishable artefacts. Each row carries the
hyperparameters, `epochs_trained`, `best_val_loss`, and `auc_val`/`auc_test`; the
`seed` column makes the multi-seed AUC spread readable directly once Step 5c/5d
runs are added.

### Step 5b — Select the best config (per activation)

Selection is by **validation** AUC (not test).

**tanh grid** — winner exp 105:

| Rank | Exp | lr | L1 | val AUC | test AUC |
|---|---|---|---|---|---|
| 1 | **105** | 0.001 | 0.001 | 0.7036 | 0.7073 |
| 2 | 104 | 0.0001 | 0.001 | 0.7035 | 0.7010 |
| 3 | 101 | 0.001 | 0.01 | 0.6218 | 0.6205 |
| … | … | … | … | (rest ≈ 0.61) | |

**relu grid** — winner exp 205:

| Rank | Exp | lr | L1 | val AUC | test AUC |
|---|---|---|---|---|---|
| 1 | **205** | 0.001 | 0.001 | 0.7255 | 0.7282 |
| 2 | 200 | 0.0001 | 0.01 | 0.7209 | 0.7149 |
| 3 | 202 | 0.01 | 0.01 | 0.6148 | 0.6129 |
| … | … | … | … | (rest ≈ 0.61) | |

Both activations select the **same coordinate** (`lr=0.001, L1=0.001`,
`L1_act=0.01`), and relu outperforms tanh at the optimum (val 0.7255 vs 0.7036,
test 0.7282 vs 0.7073).

### Step 5c — Re-train the tanh winner across seeds

Trains the tanh winner on seeds 43–46 (seed_42 = exp 105 is already the 5th
member of the stability set). GenNet has no seed flag, so model-weight init is an
additional uncontrolled source of variation on top of the split differences.

**Script:** `pipeline/03_train/run_multiseed_tanh_best.sh`

| Array task | Exp ID | Seed dir |
|---|---|---|
| (anchor) | 105 | seed_42 |
| 1 | 143 | seed_43 |
| 2 | 144 | seed_44 |
| 3 | 145 | seed_45 |
| 4 | 146 | seed_46 |

### Step 5d — Re-train the relu winner across seeds

Same scheme for relu (seed_42 = exp 205 is the anchor), with
`-hidden_activation relu` and exp IDs in the 24x range.

**Script:** `pipeline/03_train/run_multiseed_relu_best.sh`

| Array task | Exp ID | Seed dir |
|---|---|---|
| (anchor) | 205 | seed_42 |
| 1 | 243 | seed_43 |
| 2 | 244 | seed_44 |
| 3 | 245 | seed_45 |
| 4 | 246 | seed_46 |

**Run (both):**
```bash
# validate the per-epoch time on one seed first, then launch the rest
sbatch --array=1 pipeline/03_train/run_multiseed_relu_best.sh
sbatch --array=2-4 pipeline/03_train/run_multiseed_relu_best.sh
sbatch pipeline/03_train/run_multiseed_tanh_best.sh
```

Afterwards, move the new runs into their activation folders and re-run the
summary (it picks up all 5 seeds per activation automatically):
```bash
mv results/GenNet_experiment_14?_ results/tanh/
mv results/GenNet_experiment_24?_ results/relu/
python pipeline/04_report/summarize_gridsearch.py results/tanh
python pipeline/04_report/summarize_gridsearch.py results/relu
```

### Resource settings (CPU-only cluster)

Training runs **on CPU** — the `env_GenNet` TensorFlow cannot load the cluster's
CUDA libraries (it wants CUDA 11 / cuDNN 8; the module is `cuda12.2`), so it falls
back to CPU at ~10 min/epoch (~13 h per model at the winner config). The
multi-seed scripts are tuned for this:

| `#SBATCH` | Value | Reason |
|---|---|---|
| `--partition` | `all_5days` | a real CPU partition (the old `gpu` was not in the partition list and gave no GPU) |
| `--mem` | `32G` | GenNet streams genotypes from HDF5 in minibatches; the old 360 GB request (`mem-per-cpu=90000 × 4`) blocked parallel packing so only 2 of 4 array tasks ran at once |
| `--time` | `24:00:00` | the ~13 h winner config fits with margin; aids backfill |
| `-workers` | `4` | the data generator opens `genotype.h5` once **per batch** (`Dataloader.py:145`), single-threaded → I/O-bound; `>1` workers auto-enable multiprocessing (`Train_network.py:47`) to prefetch batches in parallel and overlap I/O with compute. Throughput only — does not change results. |

`-bs` and `-lr` are **not** changed for the multi-seed runs: they must reproduce
the grid winner exactly, or the cross-seed stability comparison is no longer
apples-to-apples.

### Assessing stability

Stability is assessed **within each activation** across its 5 seeds — tanh exp
{105, 143, 144, 145, 146} and relu exp {205, 243, 244, 245, 246}: AUC spread, and
overlap / rank-correlation of gene/pathway importances from each run's
`connection_weights.csv`.
