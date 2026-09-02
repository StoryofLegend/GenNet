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

**Script:** `pipeline/04_report/summarize_experiments.py` — point it at one
activation folder; it reads each run's `train_args.json` (hyperparameters) and
`pd_summary_results.csv` (AUCs), and writes one CSV into `reports/`. The
activation is read from `hidden_activation` in the JSON (not the folder name), so
a misfiled run still lands in the right CSV. It has two `--mode`s:

* `gridsearch` (default) — rows sorted by validation AUC (winner first), written
  to `reports/gridsearch/<folder>_gridsearch.csv`.
* `multiseed` — for a stability run (same config across seeds): rows sorted by
  seed plus `mean`/`std`/`min`/`max` aggregate rows, written to
  `reports/multiseed/<folder>_multiseed.csv`. It auto-restricts the aggregate to
  the config shared across seeds (so the grid points kept in the same folder
  don't pollute it); override with `--ids 142-146` if needed.

```bash
python pipeline/04_report/summarize_experiments.py results/tanh   # -> reports/gridsearch/tanh_gridsearch.csv
python pipeline/04_report/summarize_experiments.py results/relu   # -> reports/gridsearch/relu_gridsearch.csv
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
python pipeline/04_report/summarize_experiments.py results/tanh --mode multiseed
python pipeline/04_report/summarize_experiments.py results/relu --mode multiseed
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
{105, 143, 144, 145, 146} and relu exp {205, 243, 244, 245, 246}: AUC spread (Step
5a `--mode multiseed`), and rank-correlation of gene importances (Step 6,
`summarize_importance.py`).

## Step 6 — Gene importance (weight-based, 2A)

First step of the gene-importance / ISN work (see `docs/project_context.md` §3).
Weight-based importance is the **GenNet-native baseline** — deterministic, needs
no background data — that SHAP (2B) and perturbation (2C) later validate. Model
hierarchy: **SNP → gene → pathway → phenotype**.

It sources directly from the trained **weights**, *not* from
`connection_weights.csv` (~31 GB/seed): it rebuilds the model with GenNet's own
`load_trained_network` and reuses GenNet's weight↔edge alignment (the same one
`create_importance_csv` uses, validated to match `connection_weights.csv`
exactly), keeping only the small per-layer edge tables instead of the
combinatorial SNP→gene→pathway join that makes the CSV huge. Weights-only, so no
forward pass / GPU — it runs on the login node (~30 s/model).

**Scripts:** `pipeline/05_importance/`
- `compute_weight_importance.py` — one experiment → gene / pathway / pair CSVs
- `run_importance.sh` — every experiment in a results folder
- `summarize_importance.py` — cross-seed Spearman stability + merged CSV

```bash
bash pipeline/05_importance/run_importance.sh results/tanh
python pipeline/05_importance/summarize_importance.py \
    reports/importance/GenNet_experiment_{105,143,144,145,146} --name tanh
```

Outputs per run: `reports/importance/<exp>/{gene,pathway,pair}_importance.csv` plus
`top_genes.png`, `top_pathways.png` and `hub_bias_diagnostic.png`; across seeds,
`reports/importance/tanh_gene_stability.csv`.

**Importance definitions** (all node importance = magnitude of the node's
**outgoing** weights, for consistency):

| Object | Measure | Notes |
|---|---|---|
| Gene (node) | `importance_sum = Σ|w_gene→pathway|`; `importance_mean = sum ÷ degree` | `mean` is the connectivity-normalised view (guideline 1A) so hub genes don't dominate |
| Pathway (node) | `importance = |w_pathway→output|`, plus `degree` (# genes) | direct contribution to the logit; a quick biological check (IBD/immune pathways should rank top) |
| Gene pair (i,j) | `Σ_p |w_i→p| · |w_j→p|` over **shared** pathways `p` | non-zero only on existing co-memberships (guideline 1B — GenNet has no gene–gene edge); hub pathways with >200 genes skipped (`--max-genes-per-pathway`) |

SNP-level importance is intentionally **not** produced: SNPs are inputs, not ISN
nodes, so it is interpretation/QC only — compute it on demand, not per seed.

**Duplicate handling (important).** The topology repeats each gene→pathway
connection once per SNP under the gene, and a gene name can span several node
indices (e.g. SMAD3 = 3990 rows for 105 real pathways; ACOT7 = 2 nodes) — on
seed_45, ~66% of `(gene, pathway)` connections carry more than one weight. Both
sources are collapsed to one effective weight per connection by the **mean** of
the signed weights (then magnitude). **Rule: aggregate duplicates, never
`drop_duplicates`.** Mean is chosen over median/sum because it is the
representative central value and uses all duplicates, whereas `sum` would scale
importance with annotation density (a topology artefact, not signal).

**The hub-bias diagnostic** plots gene degree vs raw importance: for tanh seed_45,
raw `r = 0.74` (hubs dominate) collapses to normalised `r = −0.01` — i.e. the
degree-normalisation fully decouples connectivity from importance (guideline 1A).

**Result (tanh, 5 seeds):** ranking is stable across data splits — mean pairwise
Spearman **ρ = 0.94** over ~6,000 genes. Top pathway is *Inflammatory bowel
disease*; top genes are signaling/immune hubs (MAPK3, SRC, RAC1, RHOA, …).

---

## Step 7 — Gene importance (SHAP, 2B)

Second of the three importance methods. Where 2A reads the **weights**, 2B measures
each node's contribution to the **prediction**, per patient, then averages.

**Explainer choice.** `GradientExplainer` (expected gradients), not `DeepExplainer`.
DeepExplainer requires BatchNorm to be removed from the graph, and removing it here
collapses the genetic logit to std ≈ 1e-4, producing ~1e-5 "dust" values.
GradientExplainer differentiates through BN natively, so the logit is untouched. It
runs on `genetic_logit_model` — the genetic tail alone, before the covariates are
concatenated — so the scores are attributable to genotype rather than to age/sex/PCs.

**Scripts:** `pipeline/05_importance/`
- `run_gradexplain.sh` — one SLURM task per seed; writes per-SNP arrays into each experiment folder
- `shap_gene_importance.py` — lifts the per-SNP array through `topology.csv` to genes

```bash
sbatch pipeline/05_importance/run_gradexplain.sh results/tanh
python pipeline/05_importance/shap_gene_importance.py results/tanh
```

**Output:**

| File | Description |
|---|---|
| `<exp>/GradientExplain_test_meanabs.npy` | per-SNP `mean_patients |SHAP|`, length 38,225 |
| `<exp>/shap_gene_importance.csv` | per-gene: `shap_sum`, `shap_per_snp`, `shap_per_degree`, `shap_max`, `n_snps`, `degree` |

**Two caveats, both deliberate.**

`shap_sum` is an **upper bound**, not the exact gene attribution. The saved array is
already `mean |φ|` per SNP, so summing over a gene's SNPs adds magnitudes that could
have cancelled. The exact value needs GradientExplain re-run saving the signed
`(n_cases, n_snps)` matrix (~1.4 GB/seed) — worth doing only if a specific gene's
number is being quoted, not for ranking.

4,497 of the 25,188 mapped SNPs sit in **more than one gene** (up to 9), so column
totals exceed the array total. This is the topology, not a bug. The 13,037 unmapped
inputs carry **0.0%** of the score — they have no outgoing edge, so their SHAP is
identically zero.

**Result:** 6,014 genes per seed.

---

## Step 8 — Gene importance (node ablation, 2C)

Third method, and the most direct: set a gene's node to zero, re-run the forward pass,
and measure how much the prediction moves.

```
Delta_y(g) = y_full - y_{g ablated}
```

**Where the ablation happens matters.** The gene is zeroed at the **gene-layer
activation** (the post-BatchNorm tensor feeding `LocallyDirected_1`), not at the
genotype input. Zeroing genotype would set every patient to homozygous-reference,
which is a different question — and one the network was never trained to answer.

**Scripts:** `pipeline/05_importance/run_ablation.sh` → `GenNet.py interpret -type Ablation`
→ `interpretation/ablation.py::make_ablation_importance`

```bash
sbatch pipeline/05_importance/run_ablation.sh results/tanh
```

**Output:**

| File | Description |
|---|---|
| `<exp>/ablation_importance.csv` | one row per gene: `ablation_meanabs`, `ablation_meanabs_per_degree` |
| `<exp>/ablation_per_patient.csv` | genes × subjects (~400 MB), with `-ablation_per_patient` |

**Crash safety.** Both CSVs are written via `.tmp` + `os.replace` (`GenNet_utils/Interpret.py::_atomic_to_csv`), and `run_ablation.sh` no longer deletes the previous CSVs before starting. A pre-empted run used to destroy a good previous result; now it leaves it in place and only refuses to start if a stale `.tmp` is lying around.

---

## Step 9 — Consensus across seeds, then across methods (2D)

### Step 9a — Cross-seed consensus, per method

Single-seed importance is **not** reproducible at the head of the ranking. Each method
is therefore collapsed to one consensus ranking over the 5 tanh seeds, sorted by **mean
rank** — not mean score, because the three methods' scores are on incomparable scales.

**Script:** `pipeline/04_report/seed_consensus.py`

```bash
# raw, then the connectivity-normalised variant (guideline 1A).
# The second needs an explicit --out or it overwrites the first.
python pipeline/04_report/seed_consensus.py results/tanh \
    --input-name ablation_importance.csv --score-column ablation_meanabs
python pipeline/04_report/seed_consensus.py results/tanh \
    --input-name ablation_importance.csv --score-column ablation_meanabs_per_degree \
    --out results/tanh/ablation_importance_per_degree_consensus.csv
```

Repeat with `--input-name gene_importance.csv` (2A) and `shap_gene_importance.csv` (2B).

**Output:** `results/tanh/{gene,shap_gene,ablation}_importance[_per_degree]_consensus.csv`,
each carrying `rank`, `mean_rank`, `best`, `worst`, `n_top20`, `cv`, `spread`.

### Step 9b — Cross-method comparison

**Script:** `pipeline/04_report/compare_methods.py`

```bash
python pipeline/04_report/compare_methods.py results/tanh
```

**Output:** `results/tanh/method_comparison.csv` — one row per gene with
`rank_combined` and a `rank_<method>` column for each of the six variants, plus
`mean_rank_primary`, `n_methods_top50`, `degree`.

**Results.**

*Consensus core:* **31 genes** are in the top-50 of all three methods — NOD2, SMAD3,
IL23R, RORC, STAT4, IL10, IL18RAP, HLA-DQB1, IL21, HLA-DQA1, IL12RB2, STAT3, IFNGR2,
IL12B, HLA-DOB, NFATC1, IL5, HLA-DQA2, IL18R1, IL4, IL2, HLA-DRB1, HLA-DOA, HLA-DRA,
IL13, IL4R, TBX21, HLA-DMB, HLA-DPA1, IL6, TNF. These are textbook IBD genes, which is
the sanity check the three-method design exists to provide.

*Hub bias, and why per-degree is not a free fix (guideline 1A):*

| Method | raw vs degree | per-degree vs degree |
|---|---|---|
| 2A weight | +0.357 | **−0.780** |
| 2B SHAP | +0.449 | +0.183 |
| 2C ablation | +0.292 | **−0.028** |

2A's per-degree variant **over**corrects — it inverts the bias rather than removing it,
and its top-15 becomes IFNL4/TFF2/MIR187, sharing only 2–4 of its top-50 with any other
method. 2C is the cleanest on both axes.

*Seed stability* (genes in the top-20 of all 5 seeds): **2C = 6, 2B = 3, 2A = 1.**

---

## Step 10 — Individual-Specific Networks (ISN)

Goal (see `docs/project_context.md` §3): one network **per patient**, so that a HotZone
— a subnetwork whose wiring varies most strongly across individuals — can be scored.

**Where the seed enters.** The gene sets are the 5-seed consensus from Step 9 and carry
no seed. But a network needs a *value per gene per patient*, and that value is a
gene-node activation, which only exists inside a trained model. So each ISN is labelled
with the seed whose model supplied the numbers — **not** the genes.

Averaging activations across seeds does **not** work: correlating each gene between two
seeds on the patients they share gives 22% agreeing (r > 0.5), 22% **anti**-correlated
(r < −0.5), median r = −0.011. A hidden node has no canonical sign, so the average
cancels. Combine across seeds at the **edge** level instead — an edge's sd across
patients is invariant to a node's sign flipping.

### Step 10a — Gene sets from the consensus

**Script:** `pipeline/06_isn/gene_sets.py`

```bash
python pipeline/06_isn/gene_sets.py results/tanh --cutoffs 50 100 150 200 250
```

Selection uses `nsmallest(n, <that method's rank column>)`. Using `head(n)` would return
the *same* genes for all four methods, because `method_comparison.csv` is sorted by
combined rank.

**Output:** `results/tanh/isn_gene_sets/` — `gene_set_{2A,2B,2C,combined}_top{N}.csv`,
`gene_set_membership_top{N}.csv`, `gene_set_overlap.csv`.
Mean pairwise Jaccard falls from 0.585 (top50) to 0.383 (top250); 2A is the outlier.

### Step 10b — Extract activations

**Scripts:** `pipeline/06_isn/extract_activations.py`, `run_extract_activations.sh`

```bash
sbatch pipeline/06_isn/run_extract_activations.sh results/tanh
```

Layers are resolved **structurally**, not by name: Keras uniquifies layer names per
process, so the second model built in one run gets `batch_normalization_2` and a
name lookup fails. The block order (`LocallyDirected1D → Activation →
BatchNormalization`) is stable, so the code indexes off the `LocallyDirected1D`
positions and asserts the class of what it lands on.

The patient axis is asserted, not assumed: `subjects.csv` labels must equal the
loader's `y`, or the run aborts.

**Output** (per experiment): `isn_gene_act.npy` (9942 × 5263), `isn_pathway_act.npy`
(9942 × 4510), `isn_gene_slots.csv`, `isn_pathway_nodes.csv`, `isn_subjects.csv`.

**Most gene nodes are dead.** Activation sd spans 8 orders of magnitude; the median
per-slot sd is 5.5e-6 and 4,607 of 5,263 slots are below 1e-4. The most variable are
SMAD3 (0.76), STAT4 (0.36), RORC (0.29), DOCK3, C5, NOD2, NFATC1, STAT3. Dead-gene
fraction per set (%, mean over seeds):

| cutoff | 2A | 2B | 2C | combined |
|---|---|---|---|---|
| top50 | 30.8 | 24.8 | **12.8** | 17.2 |
| top250 | 69.7 | 54.5 | 58.7 | 57.6 |

A HotZone is by definition a subnetwork that *varies* across individuals, so dead genes
contribute only noise — treat top150/200/250 as a density-sensitivity arm, not as
primary results.

### Step 10c — Build the LIONESS input matrices

**Script:** `pipeline/06_isn/make_isn_input.py`

```bash
python pipeline/06_isn/make_isn_input.py results/tanh
```

**Output:** `results/tanh/isn_input/isn_input_{method}_top{N}_seed{S}.csv` (patients ×
genes, index = `patient_id`) plus `isn_input_manifest.csv`.

`--min-sd` defaults to **0** (report-only). Filtering would leave each method a
different gene count (2A ~53 vs 2C ~82 at top-100), making network sizes and HotZone
scores non-comparable across methods.

**The 5 seeds have near-disjoint test cohorts** — seed42 ∩ seed43 = 1,508 of 9,942
patients (15%), and only **6** patients appear in all five. Cross-seed comparison must
be at the gene/edge level, never patient by patient.

### Step 10d — LIONESS

**Scripts:** `pipeline/06_isn/lioness.py`, `run_lioness.sh`

```bash
sbatch --array=1-8 pipeline/06_isn/run_lioness.sh                       # seed 42, top50/100
SEEDS="42 43 44 45 46" sbatch --export=ALL --array=1-40%6 pipeline/06_isn/run_lioness.sh
CUTOFFS="150 200 250" sbatch --export=ALL --mem=180G --array=1-12%2 pipeline/06_isn/run_lioness.sh
```

**Output** (per config, `results/tanh/isn/<method>_top<N>_seed<S>/`):

| File | Description |
|---|---|
| `isn_edges.csv` | `source`, `target` — upper triangle, symmetrised, no self-loops |
| `isn_weights.npy` | `(n_edges, n_patients)` float32 |
| `isn_patients.csv` | real `patient_id`s, in column order |
| `isn_edge_stats.csv` | per-edge mean/sd/var across individuals → HotZone input |
| `isn_run_info.json` | shapes, timings, asymmetry diagnostics |

**Six changes from the supervisor's script** (`lioness_original.py`, kept for reference):

1. **TF-axis permutation.** netZooPy sorts the TF axis (`unique_tfs = sorted(...)`) but
   leaves the gene axis in expression order, so the two axes of the reshaped array are
   the same genes in **different orders**. Without permuting, the transpose pairs up
   unrelated genes: forward/reverse r = 0.23 and the "self-loop" diagonal contains
   negative values. With it, the diagonal is all-positive.
2. **Symmetrise by averaging** instead of dropping one direction (see below).
3. **Patient IDs kept.** The original renames columns `Sample_1 … Sample_N`, after which
   no network can be traced to a patient — fatal for HotZones.
4. **`.npy` upper triangle** instead of a full N² CSV: 197 MB vs 1.9 GB at top100.
5. **No `pip install` at import** — it must not rewrite the conda env on a compute node.
6. **netZooPy's stdout silenced** and its raw dump deleted. netZooPy prints 417,608
   lines / 17 MB per run; a 40-task array writing that concurrently exhausted the disk
   quota and killed the jobs mid-write, with empty logs because stderr was on the same
   mount. Silenced output is 658 bytes/run. `--verbose-netzoo` / `--keep-raw` restore
   the old behaviour.

**Why averaging, not dropping.** LIONESS output is genuinely asymmetric: on
`2C_top50_seed42`, forward vs reverse r = 0.578, **16.5% of pairs have opposite signs**,
89% differ by more than 10%. The old `clean_lioness.py` keeps whichever direction sorts
first lexicographically, discarding half the values non-randomly (discarded half: mean
−0.064 / sd 2.32; kept half: −0.016 / sd 2.05). Averaging keeps both, and the upper
triangle with `k=1` removes self-loops and duplicate pairs by construction — **no
cleaning step is needed**.

The two approaches are not interchangeable downstream:

| | value |
|---|---|
| edge weights, cleaned vs averaged | r = 0.887 |
| **per-edge sd** (the HotZone signal) | **r = 0.624** |
| top-100 most variable edges shared | **64 / 100** |

**The PANDA prior is noise, by construction.** The prior is `(gene-gene correlation >
0)`, inherited from the supervisor's script — but GenNet is **bipartite**, gene→pathway,
with no gene-gene edges. Measured gene-gene r has median |r| = 0.008 against a sampling
SE of 0.010 at n = 9,942; the only pair above 0.1 is HLA-DQA1/HLA-DQB1 (r = 0.132, LD).
So `prior_median_abs_corr ≈ 0.008` in every run and the binarised motif is coin flips.
This is fine for HotZones, which measure variance across individuals — but the prior
must not be described as biological.

### Step 10e — Running the supervisor's script instead

For comparison, the supervisor's original chain can be run over the same grid.

**Scripts:** `pipeline/06_isn/lioness_original.py`, `run_lioness_original.sh`,
`clean_lioness.py`

```bash
CUTOFFS=100 SEEDS=42 sbatch --export=ALL --array=1-4 pipeline/06_isn/run_lioness_original.sh
sbatch --array=1-40%4 pipeline/06_isn/run_lioness_original.sh          # full grid, ~85 GB
```

`lioness_original.py` was changed in **two** places only (23 lines); the untouched copy
is `lioness_original.py.supervisor_untouched`:

1. `file_path = "DNAm_imputed_corr_1PC_filt_pcgene_python.csv"` → `--input` / `--outdir`.
   The hardcoded path is a methylation file; the script cannot read a GenNet matrix
   without this.
2. The `pip install` block raises instead of installing.

Prior, orientation, PANDA and LIONESS calls are unchanged, and the edited file produces
**byte-identical** output to the original. Verified against the current script on the
same input: raw output correlates at **r = 0.9999999999999998** once the two directions
are averaged, i.e. the methods are identical and only the output handling differs.

**Output** (per config, `results/tanh/isn_original/<method>_top<N>_seed<S>/`):
`labeled_lioness_data.csv` (2,500 rows at top50 = both directions + self-loops),
`labeled_lioness_data_clean.csv` (1,225 rows), `lioness_top_{10,100}.png`,
`netzoo_progress.log`.

**Disk.** The full N² CSV is 480 MB per top50 config and 1.9 GB per top100 — the whole
40-config grid is ~85 GB, against ~5 GB for the same grid in `.npy`. This is the reason
the rewrite exists, not a stylistic preference.
