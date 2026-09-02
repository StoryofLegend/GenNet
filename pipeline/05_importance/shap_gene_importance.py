#!/usr/bin/env python3
"""Aggregate per-SNP SHAP (method 2B) into per-gene importance.

GradientExplain writes one array per seed, ``GradientExplain_test_meanabs.npy``,
of length ``inputsize`` (38225) holding mean_patients |SHAP| for each SNP. That is
a SNP-level score; the guidelines rank GENES, so it has to be lifted through the
SNP -> gene map in topology.csv. This writes ``shap_gene_importance.csv`` into each
experiment folder with the same column conventions as the weight-based (2A) and
ablation (2C) tables, so seed_consensus.py and the 2D comparison can consume all
three the same way.

Columns:
    gene              gene name (layer1_name)
    shap_sum          sum of mean|SHAP| over the gene's SNPs  <- the ranking score,
                      the direct analogue of 2A's importance_sum
    shap_per_snp      shap_sum / n_snps
    shap_per_degree   shap_sum / degree, the Sec 1A connectivity normalisation
                      (2A's importance_mean and 2C's ablation_meanabs_per_degree)
    shap_max          largest single-SNP score in the gene
    n_snps            SNPs wired to this gene
    degree            pathways the gene connects to
    n_gene_nodes      layer1 nodes this gene name is split over

--- Two things to be aware of when reporting this ---

1. shap_sum is an UPPER BOUND, not the gene's additive SHAP value. SHAP is additive
   over features, so the exact gene attribution is |sum_s phi_s| averaged over
   patients. The saved array is already mean|phi_s| per SNP, so summing it adds
   magnitudes that could have cancelled. Getting the exact figure means re-running
   GradientExplain to save the signed (n_cases, n_snps) matrix (~1.4 GB/seed) --
   worth doing only if the ranking here looks driven by within-gene cancellation.

2. SNPs are not partitioned between genes. In this topology 4497 of 25188 mapped
   SNPs sit in more than one gene (up to 9), so their score is counted once per
   gene and the column totals exceed the array total. That is the topology's own
   annotation, not a bug, but it means shap_sum is not a decomposition of the
   prediction. Inputs absent from topology.csv contribute to no gene; the script
   prints how many and how much score they carry.

Usage:
    python pipeline/05_importance/shap_gene_importance.py results/tanh
    python pipeline/05_importance/shap_gene_importance.py results/tanh --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHAP_NAME = "GradientExplain_test_meanabs.npy"
OUT_NAME = "shap_gene_importance.csv"


def topology_for(exp: Path, root: Path) -> Path:
    """Resolve the topology.csv the model was actually built from.

    Read it from train_args.json rather than guessing a sibling path: the seeds
    live in different processed_data/seed_*/ folders, and silently picking the
    wrong one would misalign every SNP index.
    """
    args_path = exp / "train_args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"{args_path} missing - cannot resolve the topology")
    datapath = json.loads(args_path.read_text()).get("datapath")
    if not datapath:
        raise ValueError(f"{args_path}: no 'datapath' key")
    topo = (root / datapath / "topology.csv")
    if not topo.exists():
        raise FileNotFoundError(f"{topo} missing (datapath={datapath!r})")
    return topo


def gene_table(topo_path: Path) -> pd.DataFrame:
    topo = pd.read_csv(topo_path,
                       usecols=["layer0_node", "layer1_node", "layer1_name", "layer2_node"])

    # One row per (SNP, gene): the topology repeats a SNP-gene pair once per pathway.
    snp_gene = topo[["layer0_node", "layer1_name"]].drop_duplicates()
    # degree / n_gene_nodes are gene-level, defined exactly as in 2A and 2C.
    degree = (topo[["layer1_name", "layer2_node"]].drop_duplicates()
              .groupby("layer1_name").size().rename("degree"))
    nodes = (topo[["layer1_name", "layer1_node"]].drop_duplicates()
             .groupby("layer1_name").size().rename("n_gene_nodes"))
    return snp_gene, degree, nodes


def make_shap_importance(exp: Path, root: Path) -> pd.DataFrame:
    shap_path = exp / SHAP_NAME
    if not shap_path.exists():
        raise FileNotFoundError(f"{shap_path} missing - run pipeline/05_importance/run_gradexplain.sh")

    scores = np.load(shap_path)
    if scores.ndim != 1:
        raise ValueError(f"{shap_path}: expected a 1-D per-SNP array, got shape {scores.shape}")

    snp_gene, degree, nodes = gene_table(topology_for(exp, root))

    n_inputs = scores.size
    max_node = int(snp_gene.layer0_node.max())
    if max_node >= n_inputs:
        raise ValueError(f"topology references SNP index {max_node} but {shap_path.name} "
                         f"has only {n_inputs} entries - array and topology disagree")

    mapped = np.zeros(n_inputs, dtype=bool)
    mapped[snp_gene.layer0_node.unique()] = True
    print(f"  {n_inputs} inputs, {mapped.sum()} in topology, {(~mapped).sum()} unmapped "
          f"carrying {scores[~mapped].sum() / scores.sum():.1%} of the total score")

    snp_gene = snp_gene.assign(score=scores[snp_gene.layer0_node.to_numpy()])
    grouped = snp_gene.groupby("layer1_name")["score"]
    out = pd.DataFrame({"shap_sum": grouped.sum(),
                        "shap_max": grouped.max(),
                        "n_snps": grouped.size()})
    out = out.join(degree).join(nodes)
    out["shap_per_snp"] = out.shap_sum / out.n_snps
    # degree is >=1 for every gene in the topology (a gene row exists only via a
    # pathway edge), but guard anyway rather than emitting inf.
    out["shap_per_degree"] = np.where(out.degree > 0, out.shap_sum / out.degree, 0.0)

    out = (out.rename_axis("gene").reset_index()
           .sort_values("shap_sum", ascending=False, kind="stable"))
    return out[["gene", "shap_sum", "shap_per_snp", "shap_per_degree",
                "shap_max", "n_snps", "degree", "n_gene_nodes"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding GenNet_experiment_<ID>_/ dirs, e.g. results/tanh")
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="repo root that datapath in train_args.json is relative to "
                        "(default: cwd)")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and summarise, but write nothing")
    args = p.parse_args()

    exps = sorted(d for d in args.results_dir.glob("GenNet_experiment_*_")
                  if (d / SHAP_NAME).exists())
    if not exps:
        print(f"no experiment in {args.results_dir} has {SHAP_NAME}", file=sys.stderr)
        return 1

    for exp in exps:
        print(f"=== {exp.name}")
        table = make_shap_importance(exp, args.root)
        print(f"  {len(table)} genes, top-10 by shap_sum: {list(table.gene[:10])}")
        if args.dry_run:
            continue
        out_path = exp / OUT_NAME
        tmp = out_path.with_suffix(".csv.tmp")
        table.to_csv(tmp, index=False)
        tmp.replace(out_path)          # atomic: never leave a half-written CSV
        print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
