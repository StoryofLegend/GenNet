# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [GenNet](https://github.com/ArnovanHilten/GenNet) (`upstream/master`) carrying an
IBD-specific research pipeline on top of it. Two layers live side by side:

- **Upstream framework** — `GenNet.py` (CLI), `GenNet_utils/`, `interpretation/`, `tests/`,
  `examples/`. Modified only where the project needs it (see *Fork-specific changes*).
- **Project layer** — `pipeline/` (numbered stages, SLURM + Python), `docs/`, `reports/`.

Read `docs/pipeline.md` first: it is the authoritative, step-by-step record of the actual runs
(data, hyperparameters, experiment IDs, results, and the reasoning behind each design choice).
`docs/project_context.md` holds the research plan and the supervisor's methodological
constraints. Keep both updated when adding a pipeline stage — the docs are the deliverable, not
an afterthought.

## Environment and paths

```bash
source /home/u/f099193/miniconda3/etc/profile.d/conda.sh
conda activate env_GenNet          # python 3.10.12, TF 2.x, CPU-only on this cluster
```

TensorFlow here **cannot use the cluster GPUs** (it wants CUDA 11/cuDNN 8, the module is
cuda12.2), so training runs on CPU at ~10 min/epoch. Do not add GPU `#SBATCH` directives.

`/massstorage/URT/GEN/BIO3/Kristian/exp/GenNet` and
`/home/u/f099193/_SHARE_/Research/GEN/BIO3/Kristian/exp/GenNet` are **the same directory** via
two mounts. SLURM scripts hardcode the `_SHARE_` path in `#SBATCH --output/--error`; new job
scripts should follow that convention and `cd "$(git rev-parse --show-toplevel)"`.

## Commands

```bash
# upstream CLI (modes: convert | topology | train | plot | interpret)
python GenNet.py train -path processed_data/seed_42/ -out results/ -ID 105 \
    -epochs 200 -bs 64 -lr 0.001 -L1 0.001 -patience 15 -workers 4 \
    -problem_type classification [-hidden_activation relu|tanh|softplus]
python GenNet.py interpret -type <get_gene_importance|GradientExplain|Ablation|NID|DFIM|...> \
    -resultpath results/tanh/GenNet_experiment_105_
python GenNet.py <mode> --help

# tests (CI runs bare `pytest` on ubuntu/py3.10.12; they train tiny example models)
pytest
pytest tests/test_GenNet.py::TestTrain::test_train_classification -x
```

Pipeline stages are submitted with `sbatch`; each script's header comment documents its array
mapping and knobs. Typical sequence — see `docs/pipeline.md` for the full narrative:

```bash
sbatch pipeline/00_setup/run_convert.sh                       # PLINK -> genotype.h5
bash   pipeline/01_topology/*.sh + python 04/05/06_*.py       # -> topology_final.csv
sbatch pipeline/02_subjects/run_create_subjects.sh            # seeds 42-46 splits
sbatch pipeline/03_train/run_gridsearch_tanh.sh               # 8-task grid per activation
python pipeline/04_report/summarize_experiments.py results/tanh [--mode multiseed]
sbatch pipeline/05_importance/run_gene_importance.sh results/tanh   # 2A weights
sbatch pipeline/05_importance/run_gradexplain.sh    results/tanh    # 2B SHAP
sbatch pipeline/05_importance/run_ablation.sh       results/tanh    # 2C ablation
python pipeline/04_report/seed_consensus.py results/tanh --input-name ... --score-column ...
python pipeline/04_report/compare_methods.py results/tanh           # 2D
python pipeline/06_isn/gene_sets.py results/tanh --cutoffs 50 100 150 200 250
sbatch pipeline/06_isn/run_extract_activations.sh results/tanh
python pipeline/06_isn/make_isn_input.py results/tanh                  # ISN_01,i (all subjects)
python pipeline/06_isn/make_isn_input.py results/tanh --reference 1    # ISN_1,i  (cases only)
REFS="01 1" sbatch --export=ALL --array=1-50%6 pipeline/06_isn/run_lioness.sh
```

## Architecture

**The model.** GenNet is a *biologically constrained, bipartite* net:
`SNP → gene → pathway → phenotype`. Connectivity comes from `topology.csv` (one row per
input→…→output path), compiled into sparse COO masks and applied by the custom
`GenNet_utils/LocallyDirected1D.py` layer. Each hidden block is
`LocallyDirected1D → Activation → BatchNormalization`; covariates (7 PCs) are concatenated only
just before the output. There are **no gene–gene edges** — any gene-pair or gene-gene-correlation
quantity is derived (shared pathways) or noise, never a learned edge.

**Data layout.** `processed_data/seed_<N>/` holds `subjects.csv` (the only thing that differs per
seed: a different stratified 65/20/15 split), a copy of `topology.csv`, and a symlink to the
shared `genotype.h5`. Genotypes are streamed from HDF5 in minibatches (`GenNet_utils/Dataloader.py`),
so RAM stays modest.

**Results layout.** `GenNet.py` always writes to `results/GenNet_experiment_<ID>_/` at top level;
runs are then **moved by hand** into `results/{tanh,relu,softplus}/`. Experiment IDs namespace the
activation: 10x/14x = tanh, 20x/24x = relu, 30x/34x = softplus. `results/` and `processed_data/`
are gitignored (weights, `.npy`, multi-GB CSVs); `reports/` **is** committed — the small CSVs there
are the publishable artefacts.

**Three importance methods, then consensus.** 2A weight-based (`interpretation/weight_importance.py`),
2B SHAP (`GradientExplainer` in `GenNet_utils/Interpret.py`), 2C node ablation
(`interpretation/ablation.py`). Each is run per seed, collapsed across the 5 seeds by **mean rank**
(`pipeline/04_report/seed_consensus.py`), then compared across methods
(`compare_methods.py`). Scores from different methods are never averaged — only ranks.

**ISN stage** (`pipeline/06_isn/`) turns consensus gene sets → per-patient gene-node activations →
LIONESS individual-specific networks → per-edge variance (the HotZone input).

Gene sets follow the supervisor's step 1: **INPUT_0** = the *union* of the top-N sets of
2A/2B/2C (78 genes at N=50, 486 at N=250 — never confuse it with `combined`, which is the
top N of the mean rank), **INPUT_B** = `gene_set_2B_top<N>.csv`. Two reference strategies:
`--reference 01` = all test subjects (ISN_01,i, the default, unsuffixed filenames) and
`--reference 1` = cases only (ISN_1,i, `_ref1` suffix). **The reference is fixed in
`make_isn_input.py`, never downstream** — LIONESS derives it from every sample in the
matrix it is handed, so netZooPy's `subset_numbers` or `--max-patients` would give
ISN_01,i evaluated on cases, not ISN_1,i.

## Fork-specific changes to upstream files

- `GenNet.py` — `-hidden_activation` flag (default: tanh for classification, relu for regression);
  new `interpret -type` values.
- `GenNet_utils/Create_network.py` — `hidden_activation` plumbed through `layer_block`;
  `genetic_logit_model()` (genotype-only, pre-sigmoid, **BatchNorm kept**) alongside upstream's
  `remove_batchnorm_model` / `remove_cov`.
- `GenNet_utils/Interpret.py` — `get_gene_scores` (2A), `get_GradientExplainer_scores` (2B),
  `get_ablation_scores` (2C), and `_atomic_to_csv` (tmp + `os.replace`, so a pre-empted job cannot
  destroy a good previous result).
- `interpretation/weight_importance.py` — gene/pathway/pair importance tables;
  `interpretation/ablation.py` — closed-form gene ablation with a forward-pass self-check.

Keep upstream-touching edits minimal and additive; the fork still merges from `upstream/master`.

## Gotchas that have already cost time

- **Weight↔edge alignment.** The kernel rows are in the mask's *native* COO order. Never sort or
  reorder a mask before attaching weights — that silently mis-assigns every gene→pathway weight.
- **Duplicate COO entries.** The layer-1 mask has one entry per topology row, and TF **sums**
  duplicates; collapse kernel entries for the same connection by SUM at layer 1. At the
  gene/pathway aggregation level, collapse duplicate `(gene, pathway)` weights by **mean of the
  signed weights** — a gene name spans several node indices and each connection repeats per SNP.
  Rule: *aggregate duplicates, never `drop_duplicates`*.
- **Use `GradientExplainer`, not `DeepExplainer`.** DeepExplainer needs BatchNorm stripped, and
  `remove_batchnorm_model` collapses the genetic logit to std ≈ 1e-4 (values become dust).
  GradientExplainer differentiates through BN, so run it on `genetic_logit_model`.
- **Resolve Keras layers structurally, not by name.** Keras uniquifies names per process
  (`batch_normalization_2` on the second model built in one run), so index off the
  `LocallyDirected1D` positions and assert the class of what you land on.
- **Seeds have near-disjoint test cohorts** (~15% pairwise overlap; 6 patients in all five), and
  hidden nodes have no canonical sign, so activations anti-correlate across seeds. Combine across
  seeds at the **edge/gene** level, never patient-by-patient and never by averaging activations.
- **Most gene nodes are dead** — median activation sd ≈ 5.5e-6. Large gene-set cutoffs (150–250)
  are a density-sensitivity arm, not primary results.
- **netZooPy/LIONESS**: never let it `pip install` on a compute node, silence its stdout (417k
  lines/run once filled the disk quota and killed a 40-task array), permute the TF axis, and
  symmetrise by averaging — dropping one direction discards ~half the values non-randomly.
