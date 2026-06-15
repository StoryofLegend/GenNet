# Project Context — GenNet → ISN → HotZone

**Author:** Kristian Kovacev (Erasmus student, ~4 months)
**Supervisors / collaborators:** Kristel (PI), Ahmed (EADB GenNet code), Kavya (folder/access oversight)
**Prior work referenced:** Gaia (ISN / HotZone detection), Giulia
**Companion doc:** [`pipeline.md`](pipeline.md) — the concrete IBD data pipeline being built.

This file records the *why* behind the project — the strategy, the methodological
questions, and the scoring framework — so the context does not have to be
re-explained each session. It is distilled from three communications from the
supervisor (project-plan email, GenNet analysis guidelines, HotZone scoring note).

---

## 1. Big picture

GenNet (Gennady Roshchupkin) is a neural network that bakes **biological priors**
into its architecture: a `SNP → gene → pathway → phenotype` hierarchy where edges
exist only where biology says they should. It is **not** a GNN and does **not**
model true gene–gene interactions.

The overall research arc is to turn GenNet outputs into **Individual-Specific
Networks (ISNs)** and then quantify the "hotness" of variable subnetworks
(**HotZones**):

1. **Nodes / node-pairs** selected by an importance criterion from GenNet runs.
2. **ISN construction** over the collective of nodes, then HotZone identification
   (Gaia's technique).
3. **Hotness metrics** for the HotZones — which design choices (GenNet level,
   gene-set definition, ISN construction) give the best, most interpretable
   hotness.

Strategic motivation:
- Demonstrate the promise of GenNet-linked post-GWAS analyses to IIBDGC.
- Multiple applications → multiple insights / papers.
- Ahmed's current code focuses on **pathways**; here we additionally introduce
  **protein complexes (CORUM)**, enabling a comparative study.

---

## 2. Four-month plan

### Month 1 — Reproduce and adapt
- Get Ahmed's EADB GenNet pipeline running (do **not** start from scratch).
- Adapt and validate on the **IBD dataset** with SNP-linked samples.
- Fix folder structure, access, and code-sharing. Work under the same mother
  folder used by Gaia and Giulia (BIO3 access arranged with Gaelle).

### Month 2 — Controlled methodological comparison
Compare a small, **predefined** set of model choices:
- activation functions
- number / type of inner layers
- pathway layer vs **protein-complex layer (CORUM)**

Define evaluation metrics **in advance**:
- predictive performance
- stability of gene rankings
- stability of pairwise importance / activation measures
- reproducibility across splits / seeds

> The main scientific focus of the whole project is the **stability of
> ranking / importance / activation-score measures** for genes and gene pairs.
> Stability measures should be grounded in the ML literature.

### Month 3 — Connect GenNet outputs to ISN design
- Select only **1–2 node definitions** and **1–2 edge definitions**.
- Edge definitions may make more/less sense depending on the activation-function
  choice.
- Test whether these choices improve **HotZone-related signal** (link to HotZone
  proneness or a related phenotype).

### Month 4 — One downstream proof of concept (pick one)
1. **HotZone prediction** — predict HotZone regions via the combined step 1 + 2
   approach (reduces ISN construction to nodes that matter in a HotZone sense), **or**
2. **ISN embedding for clustering** — prediction-oriented framework for
   unsupervised clustering.

Not both. The third, more ambitious direction (HotZone-aware generative
foundational models) is **probably out of scope** but discussable.

> Preference: work with samples that **also have SNP data**. (Open question
> whether Gaia's TCGA samples had SNP data.)

---

## 3. GenNet analysis guidelines (importance + ISN construction)

### 3.1 Design choices that shape "importance"

| Aspect | Why it matters | Task |
|---|---|---|
| **Network structure** (primary driver) | Genes gain importance only through pathway connections; hub genes look artificially important | Map gene→pathway degrees, normalise importance by connectivity |
| **Biological priors (edges)** | Pairwise importance = 0 where no edge exists | Extract edge list; compute pair importance only on existing edges |
| **Regularisation (L1/L2)** | Strong L1 → fewer important genes (good for ISNs) | Check hyperparameters; affects node-selection threshold |
| **Input encoding** | SNP→gene aggregation, one-hot vs dosage changes what "gene importance" means | Confirm input layer; document interpretation |

**Key open question:** how does the **correlation threshold** for reference-network
edges affect reverse-engineered ISNs? Test sensitivity to edge density; consider
alternative association measures.

### 3.2 Quantifying node & pair importance — start simple, validate rigorously

- **A. Weight-based (GenNet-native, fastest)**
  - Node: `Importance(g) = Σ|w_{g→pathways}|`
  - Pair: `Importance(g_i,g_j) = |w_ij|` (only if edge exists)
  - From `model.get_weights()`.
- **B. SHAP (gold standard)** — `DeepExplainer`; `shap_values` for nodes,
  `shap_interaction_values` for epistatic pairs. Theoretically sound, handles
  the GenNet hierarchy.
- **C. Perturbation (causal)** — node ablation `Δy_g = y_full − y_{g=0}`; pair
  interaction `Δy_ij − (Δy_i + Δy_j)`.
- **D. Validation strategy** — weights → candidates; SHAP → rank by true
  contribution; perturbation → confirm causal effect; compare across **5 random
  seeds** to assess stability.

### 3.3 Building ISNs from GenNet outputs
1. **Reference network** — prioritise edges by `weight(i,j) × SHAP_interaction(i,j)`;
   threshold to top ~10% edge weights.
2. **Patient activation scores** — extract intermediate-layer activations:
   `Node_k = ReLU(W_ref × input_k)`.
3. **ISN construction** — `ISN_k(i,j) = reference_weight(i,j) × (Node_k(i) × Node_k(j))`
   (edge strength = biology × patient state).
4. **Validation** — ISN_k predictive performance; compare high-risk vs low-risk.

**Edge-density investigation:** sparse (~5%) → focused ISNs; dense (~20%) →
diffuse, harder to interpret. Working hypothesis: **5–10% edge density** is
optimal for ISNs.

### 3.4 Intended notebook structure
- `notebook_1_gennet_importance.ipynb` — load model+data → weight importance →
  SHAP (node + interaction) → perturbation → reference network → parameter
  sensitivity.
- `notebook_2_isn_construction.ipynb` — patient activation extraction → ISN edge
  weighting → visualisation (NetworkX/Plotly) → ISN→prediction.

---

## 4. Scoring HotZone "hotness"

**Setup:** many ISNs over the same node set, but edge weights vary across
individuals. A **HotZone** is a subnetwork whose wiring varies most strongly
across individuals. Earlier (Gaia's) work showed HotZones are more enriched for
significant pathways than random subnetworks of the same size.

A good hotness score should capture four properties, each preferably
**standardised against matched null subnetworks** (same size, ideally similar
density/degree):

| Component | Symbol | What it measures | Reasonable options |
|---|---|---|---|
| **Differential wiring** | D | How much internal wiring varies across individuals (core reason a zone is found) | mean/median edge-weight variance; pairwise adjacency-submatrix distance; contrast-subgraph vs background; multiscale dissimilarity |
| **Enrichment** | E | Biological coherence, not just variability | max enrichment (e.g. −log p); geometric-mean enrichment; weighted pathway coverage; enrichment-excess vs matched random; robust aggregate across Reactome/KEGG/GO/disease sets |
| **Stability** | S | Reappearance under perturbation (resampling, bootstrap) | selection frequency of nodes/edges; mean Jaccard overlap vs bootstrap HotZones; rank stability (Spearman); consensus compactness |
| **Clustering / discrimination** | C | Whether the zone improves subtype discovery vs explicit baselines | Δ silhouette / Calinski-Harabasz; Δ cluster stability; Δ ARI/NMI (if labels); Δ held-out AUC/balanced accuracy |

> **For Kristian's scope:** a HotZone can be scored with **simple versions of D
> and E**, consistent with the data generated in Gaia's thesis. The full
> S and C machinery and the superhotzone extension below are context /
> future directions.

### 4.1 Superhotzone (merged HotZones) — future direction

When HotZones are merged, the union may not be complete or well-connected.
Open question to check with Gaia's work: were the three HotZones merged
**node-wise** (re-establish all pairwise connections → complete network per
individual) or **edge-wise** (keep only HotZone edges, others set to zero →
incomplete graphs)?

A superhotzone score aggregates component scores over constituent HotZones and
adds a **merge-coherence term M**:
- **Aggregation:** default = **size- and stability-weighted mean** (avoids
  over-weighting small unstable zones or large diffuse ones). Alternatives:
  simple mean, robust/trimmed mean, max-plus-support.
- **Merge-coherence M** penalises two failure modes:
  - **Fragmentation** (zone breaks into disconnected pieces) — penalty `λ_frag`.
  - **Redundancy** (HotZones repeat the same information) — penalty `λ_red`.
- **Penalty calibration (data-driven, not ad hoc):**
  - `λ_frag` — tune against resampling stability; pick smallest value at the
    elbow / near-maximal stability (stability-selection / consensus-clustering
    logic).
  - `λ_red` — fix `λ_frag`, then tune against held-out utility and null
    comparisons; stop adding HotZones once the composite score / held-out
    clustering no longer improves.
  - Jointly tunable via nested resampling; prefer a **Pareto front** if a single
    scalar optimum is unstable.

### 4.2 Reporting
Report the **full component vector** (D, E, S, C, M) as well as the composite
score — not only the final scalar.

---

## 5. Glossary

- **GenNet** — biologically-constrained neural net (`SNP→gene→pathway→phenotype`).
- **ISN** — Individual-Specific Network; one network per subject over a shared
  node set, with subject-specific edge weights.
- **HotZone** — subnetwork whose wiring varies most across individuals.
- **Superhotzone** — merged union of several HotZones.
- **CORUM** — protein-complex database (alternative inner layer to pathways).
- **CPDB / ConsensusPathDB** — pathway database used in the current IBD pipeline.
- **EADB** — Ahmed's existing GenNet application (the code to reproduce/adapt).
- **IBD** — Inflammatory Bowel Disease (CD + UC + controls); the target dataset.
