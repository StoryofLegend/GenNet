#!/usr/bin/env python3
"""Step 2A — weight-based gene / pathway / pair importance for one trained GenNet model.

Sources from the trained weights (``model.get_weights()``), not the multi-GB
``connection_weights.csv``. Writes ranked CSVs + plots to ``reports/importance/<name>/``.
Full rationale and definitions: docs/pipeline.md Step 6.

Usage (conda env_GenNet, from the repo root):
    python pipeline/05_importance/compute_weight_importance.py results/tanh/GenNet_experiment_145_
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                                  # headless cluster: render straight to file
import matplotlib.pyplot as plt  # noqa: E402          (must follow matplotlib.use)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for GenNet_utils
from GenNet_utils.Train_network import load_trained_network  # noqa: E402


def load_model_and_topology(exp_dir: Path):
    """Rebuild the trained model + coo masks via GenNet, plus the topology table."""
    args = SimpleNamespace(resultpath=str(exp_dir))
    model, masks = load_trained_network(args)            # fills args from train_args.json
    topology = pd.read_csv(Path(args.datapath) / "topology.csv")
    return model, masks, topology


def _names(topology: pd.DataFrame, layer: int) -> pd.DataFrame:
    """node index -> name for a layer (1-to-1, so drop_duplicates is lossless)."""
    return (topology[[f"layer{layer}_node", f"layer{layer}_name"]].drop_duplicates()
            .rename(columns={f"layer{layer}_node": "node", f"layer{layer}_name": "name"}))


def gene_pathway_edges(model, masks, topology: pd.DataFrame, edge_agg: str) -> pd.DataFrame:
    """One effective weight per (gene, pathway), collapsing duplicate edges by edge_agg."""
    mask = masks[1]                                       # gene (row) -> pathway (col)
    e = pd.DataFrame({"gene_node": mask.row, "pathway_node": mask.col})
    e = e.sort_values("gene_node").reset_index(drop=True)
    # weights align to the mask coords sorted by source node (GenNet's own convention)
    e["weight"] = model.get_layer("LocallyDirected_1").get_weights()[0].flatten()
    e = e.merge(_names(topology, 1).rename(columns={"node": "gene_node", "name": "gene"}),
                on="gene_node")
    e = e.merge(_names(topology, 2).rename(columns={"node": "pathway_node", "name": "pathway"}),
                on="pathway_node")

    n_raw = len(e)
    edges = e.groupby(["gene", "pathway_node", "pathway"], as_index=False).agg(
        weight=("weight", edge_agg))
    edges["abs_w"] = edges["weight"].abs()
    print(f"  {n_raw} raw -> {len(edges)} gene->pathway connections "
          f"({edges['gene'].nunique()} genes, {edges['pathway_node'].nunique()} pathways, "
          f"edge_agg={edge_agg})")
    return edges


def gene_importance(edges: pd.DataFrame) -> pd.DataFrame:
    """importance_sum = Sum|w_gene->pathway|; importance_mean = sum/degree (1A)."""
    g = edges.groupby("gene").agg(importance_sum=("abs_w", "sum"),
                                  degree=("pathway_node", "nunique")).reset_index()
    g["importance_mean"] = g["importance_sum"] / g["degree"]
    return g.sort_values("importance_sum", ascending=False).reset_index(drop=True)


def pathway_importance(model, topology: pd.DataFrame, edges: pd.DataFrame,
                       edge_agg: str) -> pd.DataFrame:
    """importance = |w_pathway->output| (outgoing weight, like the gene view)."""
    w_out = model.get_layer("output_layer").get_weights()[0].flatten()
    pw = pd.DataFrame({"pathway_node": np.arange(len(w_out)), "importance": np.abs(w_out)})
    pw = pw.merge(_names(topology, 2).rename(columns={"node": "pathway_node", "name": "pathway"}),
                  on="pathway_node")
    pw = pw.merge(edges.groupby("pathway_node")["gene"].nunique().rename("degree"),
                  on="pathway_node")
    p = pw.groupby("pathway").agg(importance=("importance", edge_agg),
                                  degree=("degree", "sum")).reset_index()
    return p.sort_values("importance", ascending=False).reset_index(drop=True)


def pair_importance(edges: pd.DataFrame, max_genes_per_pathway: int) -> pd.DataFrame:
    """Shared-pathway gene-pair importance: Sum_p |w_i->p| * |w_j->p|.

    Genes connect only through a shared pathway, so this is non-zero exactly on
    existing co-memberships (1B). Hub pathways above the size cap are skipped
    because the C(k,2) pair count explodes and those pairs are uninformative.
    """
    score: dict[tuple, float] = {}
    shared: dict[tuple, int] = {}
    for _, grp in edges.groupby("pathway_node"):
        if len(grp) > max_genes_per_pathway:
            continue
        for (gi, wi), (gj, wj) in combinations(zip(grp["gene"], grp["abs_w"]), 2):
            key = (gi, gj) if gi < gj else (gj, gi)
            score[key] = score.get(key, 0.0) + wi * wj
            shared[key] = shared.get(key, 0) + 1
    pairs = pd.DataFrame([(a, b, s, shared[(a, b)]) for (a, b), s in score.items()],
                         columns=["gene_i", "gene_j", "pair_importance", "n_shared_pathways"])
    return pairs.sort_values("pair_importance", ascending=False).reset_index(drop=True)


def make_plots(genes: pd.DataFrame, pathways: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    """Top-N gene bars, top-N pathway bars, and the hub-bias diagnostic (1A)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    g = genes.head(top_n)[::-1]
    ax.barh(g["gene"], g["importance_sum"], color="teal")
    ax.set_title(f"Top {top_n} genes  (Sum|w_gene->pathway|)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_genes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    p = pathways.head(top_n)[::-1]
    ax.barh(p["pathway"].str.slice(0, 45), p["importance"], color="indianred")
    ax.set_title(f"Top {top_n} pathways  (|w_pathway->output|)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_pathways.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(genes["degree"], genes["importance_sum"], s=10, alpha=0.3)
    r_raw = genes["degree"].corr(genes["importance_sum"])
    r_norm = genes["degree"].corr(genes["importance_mean"])
    ax.set(xlabel="gene degree (# pathways)", ylabel="raw importance  Sum|w|",
           title=f"Hub-bias check: degree vs importance\n"
                 f"raw r={r_raw:.2f}   normalised r={r_norm:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "hub_bias_diagnostic.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="A GenNet_experiment_<ID>_ folder (has bestweights_job.h5)")
    ap.add_argument("--name", default=None, help="Output subfolder (default: experiment folder name)")
    ap.add_argument("--edge-agg", choices=["mean", "median", "sum"], default="mean",
                    help="Collapse duplicate gene->pathway weights (default: mean; "
                         "'sum' = forward-pass faithful)")
    ap.add_argument("--max-genes-per-pathway", type=int, default=200,
                    help="Skip hub pathways larger than this for pairs (default: 200)")
    ap.add_argument("--top-n", type=int, default=20, help="Bars per plot (default: 20)")
    ap.add_argument("--no-plots", action="store_true", help="Skip the PNG plots")
    ap.add_argument("--out-dir", default="reports/importance", help="Base output folder")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not (exp_dir / "bestweights_job.h5").exists():
        raise SystemExit(f"No bestweights_job.h5 in {exp_dir}")

    print(f"Loading model from {exp_dir} ...")
    model, masks, topology = load_model_and_topology(exp_dir)
    edges = gene_pathway_edges(model, masks, topology, args.edge_agg)

    genes = gene_importance(edges)
    pathways = pathway_importance(model, topology, edges, args.edge_agg)
    pairs = pair_importance(edges, args.max_genes_per_pathway)

    out_dir = Path(args.out_dir) / (args.name or exp_dir.name.strip("_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    genes.to_csv(out_dir / "gene_importance.csv", index=False)
    pathways.to_csv(out_dir / "pathway_importance.csv", index=False)
    pairs.to_csv(out_dir / "pair_importance.csv", index=False)
    if not args.no_plots:
        make_plots(genes, pathways, out_dir, args.top_n)

    print(f"Wrote -> {out_dir}/  (genes={len(genes)}, pathways={len(pathways)}, pairs={len(pairs)})")
    print("Top genes: " + ", ".join(genes.head(5)["gene"]))


if __name__ == "__main__":
    main()
