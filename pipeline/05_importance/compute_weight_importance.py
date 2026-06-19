#!/usr/bin/env python3
"""Weight-based (GenNet-native) node & pair importance for one trained model.

This is step 2A of the gene-importance pipeline (see docs/project_context.md §3):
the simplest, deterministic, GenNet-native importance, used as the baseline that
SHAP (2B) and perturbation (2C) later validate.

It sources directly from the trained **weights** (not the multi-GB
``connection_weights.csv``): it rebuilds the model with GenNet's own
``load_trained_network`` and reuses GenNet's weight<->edge alignment (the same one
``create_importance_csv`` uses), but keeps only the small per-layer edge tables
instead of the combinatorial SNP->gene->pathway join that makes the CSV huge.

Model hierarchy (classification): SNP -> gene -> pathway -> phenotype.

Outputs (into ``reports/importance/<name>/``):
  * ``gene_importance.csv``    — per gene: local importance Sum/Mean(|w_gene->pathway|)
  * ``pathway_importance.csv`` — per pathway: |w_pathway->output| and incoming gene stats
  * ``pair_importance.csv``    — gene pairs sharing >=1 pathway: Sum_p |w_i->p|*|w_j->p|

Two importance views per node (guideline 1A — hub genes look artificially big):
  * ``importance_sum``  = Sum |w|          (raw)
  * ``importance_mean`` = Sum|w| / degree  (connectivity-normalised)

Duplicate handling (supervisor's rule): when one gene/pathway *name* spans several
node indices, collapse by **mean/median** (``--name-agg``), never ``drop_duplicates``.

Usage (inside conda env_GenNet, from the repo root):
    python pipeline/05_importance/compute_weight_importance.py results/tanh/GenNet_experiment_145_
    python pipeline/05_importance/compute_weight_importance.py results/tanh/GenNet_experiment_145_ \
        --name tanh_seed45 --name-agg median
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

# Keep TensorFlow quiet; we only need weights, never a forward pass / GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_model_and_topology(exp_dir: Path):
    """Rebuild the trained model + coo masks via GenNet, plus its topology table.

    Returns (model, masks, topology_df, datapath). ``args`` is filled in place by
    GenNet's load_train_arguments from the experiment's train_args.json.
    """
    from types import SimpleNamespace

    from GenNet_utils.Train_network import load_trained_network

    args = SimpleNamespace(resultpath=str(exp_dir))
    model, masks = load_trained_network(args)
    topology = pd.read_csv(Path(args.datapath) / "topology.csv")
    return model, masks, topology, args.datapath


def gene_pathway_edges(model, masks, topology: pd.DataFrame,
                       edge_agg: str = "mean") -> pd.DataFrame:
    """Per UNIQUE (gene, pathway) edge with its effective weight.

    GenNet assigns ``get_weights()[0]`` to the mask coordinates *after sorting by
    the source node*, so we reproduce that exact order. The topology repeats each
    gene->pathway connection once per SNP under the gene, so the raw mask holds
    duplicate ``(gene_node, pathway_node)`` coordinates (each with its own weight).
    We collapse those duplicates to one effective weight per connection with
    ``edge_agg`` (default ``mean`` — the supervisor's rule: aggregate duplicates,
    never ``drop_duplicates``; ``sum`` reproduces the forward-pass contribution).
    """
    mask = masks[1]  # LocallyDirected_1 : gene (row) -> pathway (col)
    edges = pd.DataFrame({"gene_node": mask.row, "pathway_node": mask.col})
    edges = edges.sort_values("gene_node").reset_index(drop=True)
    edges["weight"] = model.get_layer("LocallyDirected_1").get_weights()[0].flatten()

    # node index -> name is 1:1, so drop_duplicates here is safe (matches GenNet).
    gene_names = (topology[["layer1_node", "layer1_name"]].drop_duplicates()
                  .rename(columns={"layer1_node": "gene_node", "layer1_name": "gene"}))
    pw_names = (topology[["layer2_node", "layer2_name"]].drop_duplicates()
                .rename(columns={"layer2_node": "pathway_node", "layer2_name": "pathway"}))
    edges = edges.merge(gene_names, on="gene_node").merge(pw_names, on="pathway_node")

    # collapse SNP-duplicated connections -> one effective weight per (gene,pathway)
    n_raw = len(edges)
    edges = edges.groupby(["gene_node", "pathway_node", "gene", "pathway"],
                          as_index=False).agg(weight=("weight", edge_agg),
                                              n_snp=("weight", "size"))
    if len(edges) < n_raw:
        print(f"  collapsed {n_raw} raw edges -> {len(edges)} unique "
              f"gene->pathway connections (edge_agg={edge_agg})")
    edges["abs_w"] = edges["weight"].abs()
    return edges


def collapse_by_name(df: pd.DataFrame, name_col: str, sum_cols, mean_cols,
                     count_col: str, name_agg: str) -> pd.DataFrame:
    """Collapse rows that share a *name* across node indices.

    Per the supervisor's rule we aggregate duplicates by mean/median (never drop).
    ``sum_cols`` are additive (degrees), ``mean_cols`` use ``name_agg``.
    """
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: name_agg for c in mean_cols})
    agg[count_col] = "nunique"
    out = df.groupby(name_col).agg(agg).reset_index()
    return out.rename(columns={count_col: "n_nodes"})


def node_importance(edges: pd.DataFrame, name_agg: str):
    """Gene- and pathway-level node importance (sum + degree-normalised mean)."""
    # --- gene level: importance flows out of the gene to its pathways ---
    per_gene_node = edges.groupby(["gene_node", "gene"]).agg(
        importance_sum=("abs_w", "sum"),
        importance_mean=("abs_w", "mean"),   # degree-normalised
        degree=("abs_w", "size"),
    ).reset_index()
    genes = collapse_by_name(
        per_gene_node, "gene",
        sum_cols=["degree"], mean_cols=["importance_sum", "importance_mean"],
        count_col="gene_node", name_agg=name_agg,
    ).sort_values("importance_sum", ascending=False).reset_index(drop=True)

    # --- pathway level: incoming from genes (degree) summarised here ---
    per_pw_node = edges.groupby(["pathway_node", "pathway"]).agg(
        in_importance_sum=("abs_w", "sum"),
        in_importance_mean=("abs_w", "mean"),
        degree=("abs_w", "size"),
    ).reset_index()
    pathways = collapse_by_name(
        per_pw_node, "pathway",
        sum_cols=["degree"], mean_cols=["in_importance_sum", "in_importance_mean"],
        count_col="pathway_node", name_agg=name_agg,
    ).sort_values("in_importance_sum", ascending=False).reset_index(drop=True)

    return genes, pathways


def pair_importance(edges: pd.DataFrame, name_agg: str,
                    max_genes_per_pathway: int) -> pd.DataFrame:
    """Shared-pathway gene-pair importance: Sum_p |w_i->p| * |w_j->p|.

    Genes are 'connected' only through a shared pathway (GenNet has no gene-gene
    edge), so this is non-zero exactly on existing co-memberships (guideline 1B).
    Pathways with more than ``max_genes_per_pathway`` member genes are skipped:
    they are hubs (C(k,2) explodes and the pairs are uninformative).
    """
    score: dict[tuple, float] = {}
    n_shared: dict[tuple, int] = {}
    skipped = 0
    for _pnode, grp in edges.groupby("pathway_node"):
        members = list(zip(grp["gene_node"].values, grp["abs_w"].values))
        if len(members) > max_genes_per_pathway:
            skipped += 1
            continue
        for (a, wa), (b, wb) in combinations(members, 2):
            key = (a, b) if a < b else (b, a)
            score[key] = score.get(key, 0.0) + wa * wb
            n_shared[key] = n_shared.get(key, 0) + 1
    if skipped:
        print(f"  (skipped {skipped} hub pathways with >{max_genes_per_pathway} genes)")

    if not score:
        return pd.DataFrame(columns=["gene_i", "gene_j", "pair_importance", "n_shared_pathways"])

    node2name = (edges[["gene_node", "gene"]].drop_duplicates()
                 .set_index("gene_node")["gene"].to_dict())
    rows = []
    for (a, b), s in score.items():
        gi, gj = node2name[a], node2name[b]
        if gi == gj:
            continue  # two nodes of the same gene name are not a real pair
        # canonicalise the *name* pair so duplicates collapse symmetrically
        gi, gj = (gi, gj) if gi <= gj else (gj, gi)
        rows.append((gi, gj, s, n_shared[(a, b)]))
    pairs = pd.DataFrame(rows, columns=["gene_i", "gene_j", "pair_importance", "n_shared_pathways"])

    # collapse pairs that map to the same name pair (genes split across nodes)
    pairs = pairs.groupby(["gene_i", "gene_j"]).agg(
        pair_importance=("pair_importance", name_agg),
        n_shared_pathways=("n_shared_pathways", "sum"),
    ).reset_index().sort_values("pair_importance", ascending=False).reset_index(drop=True)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="A GenNet_experiment_<ID>_ folder (has bestweights_job.h5)")
    ap.add_argument("--name", default=None,
                    help="Output subfolder name (default: experiment folder name)")
    ap.add_argument("--name-agg", choices=["mean", "median"], default="mean",
                    help="How to collapse duplicate gene/pathway names (default: mean)")
    ap.add_argument("--edge-agg", choices=["mean", "median", "sum"], default="mean",
                    help="How to collapse SNP-duplicated gene->pathway weights "
                         "(default: mean; 'sum' = forward-pass faithful)")
    ap.add_argument("--max-genes-per-pathway", type=int, default=200,
                    help="Skip hub pathways larger than this for pair importance (default: 200)")
    ap.add_argument("--out-dir", default="reports/importance",
                    help="Base output folder (default: reports/importance)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not (exp_dir / "bestweights_job.h5").exists():
        raise SystemExit(f"No bestweights_job.h5 in {exp_dir}")

    print(f"Loading model from {exp_dir} ...")
    model, masks, topology, _ = load_model_and_topology(exp_dir)
    edges = gene_pathway_edges(model, masks, topology, args.edge_agg)
    print(f"  {len(edges)} gene->pathway edges, "
          f"{edges['gene'].nunique()} genes, {edges['pathway'].nunique()} pathways")

    genes, pathways = node_importance(edges, args.name_agg)
    pairs = pair_importance(edges, args.name_agg, args.max_genes_per_pathway)

    name = args.name or exp_dir.name.strip("_")
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    genes.to_csv(out_dir / "gene_importance.csv", index=False)
    pathways.to_csv(out_dir / "pathway_importance.csv", index=False)
    pairs.to_csv(out_dir / "pair_importance.csv", index=False)

    print(f"Wrote -> {out_dir}/")
    print(f"  genes={len(genes)}  pathways={len(pathways)}  pairs={len(pairs)}")
    top = genes.head(5)[["gene", "importance_sum"]].to_string(index=False)
    print("Top genes by importance_sum:\n" + top)


if __name__ == "__main__":
    main()
