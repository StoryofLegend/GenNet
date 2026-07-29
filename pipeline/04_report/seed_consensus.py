#!/usr/bin/env python3
"""Aggregate a per-seed gene importance score into one consensus ranking.

Point this at a folder of ``GenNet_experiment_<ID>_/`` directories from a
multi-seed run (e.g. ``results/tanh``). Works for any per-seed CSV with a gene
column and a score column, so the same step serves weight-based (2A), SHAP (2B)
and perturbation (2C) importance.

Ranking by mean rank rather than mean score is deliberate: scores are not on a
comparable scale between seeds, ranks are. Single-seed importance is not
reproducible at the head of the ranking (tanh 2A: pairwise Spearman ~0.7, top-20
overlap ~2/20), which is why downstream steps should consume this file rather
than one experiment's CSV. Read the stability columns: a low mean_rank with a
large spread or cv means one seed is carrying the gene.

Columns:
    gene             gene name
    rank             consensus rank, 1 = strongest
    mean_rank        mean of the per-seed ranks (the sort key)
    best / worst     best and worst single-seed rank
    spread           worst - best
    n_top20          number of seeds ranking this gene in the top --top-n
    mean_importance  mean of the per-seed score
    cv               std/mean of the score across seeds (sample std)
    degree           pathway degree, if present in the input
    n_gene_nodes     gene nodes this gene name is split over, if present

Usage:
    python pipeline/04_report/seed_consensus.py results/tanh
    python pipeline/04_report/seed_consensus.py results/tanh \
        --input-name shap_importance.csv --score-column mean_abs_shap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FIELDS = [
    "gene",
    "rank",
    "mean_rank",
    "best",
    "worst",
    "n_top20",
    "mean_importance",
    "cv",
    "degree",
    "n_gene_nodes",
    "spread",
]

# Carried through unchanged if present; must be identical across seeds.
TOPOLOGY_COLUMNS = ("degree", "n_gene_nodes")


def load_seed(path: Path, gene_column: str, score_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {gene_column, score_column} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    df = df.set_index(gene_column)
    # method="average" so the rare exact ties do not depend on file order.
    df["_rank"] = df[score_column].rank(ascending=False, method="average")
    return df


def build_consensus(paths: list[Path], gene_column: str, score_column: str,
                    top_n: int) -> pd.DataFrame:
    seeds = [load_seed(p, gene_column, score_column) for p in paths]

    ranks = pd.concat([s["_rank"] for s in seeds], axis=1, join="inner")
    scores = pd.concat([s[score_column] for s in seeds], axis=1, join="inner")

    out = pd.DataFrame(index=ranks.index)
    out["mean_rank"] = ranks.mean(axis=1)
    out["best"] = ranks.min(axis=1)
    out["worst"] = ranks.max(axis=1)
    out["n_top20"] = (ranks <= top_n).sum(axis=1)
    out["mean_importance"] = scores.mean(axis=1)
    out["cv"] = scores.std(axis=1, ddof=1) / scores.mean(axis=1)
    out["spread"] = out["worst"] - out["best"]

    for col in TOPOLOGY_COLUMNS:
        if col not in seeds[0].columns:
            continue
        ref = seeds[0][col].reindex(out.index)
        for other in seeds[1:]:
            if not ref.equals(other[col].reindex(out.index)):
                raise ValueError(f"'{col}' differs between seeds - they do not share a "
                                 "topology, so one consensus table is not meaningful")
        out[col] = ref

    # Ties go to the better worst-case rank: never falling far beats one lucky seed.
    out = out.sort_values(["mean_rank", "worst"], kind="stable").reset_index()
    out = out.rename(columns={gene_column: "gene"})
    out["rank"] = out.index + 1
    return out[[c for c in FIELDS if c in out.columns]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding GenNet_experiment_<ID>_/ dirs, e.g. results/tanh")
    p.add_argument("--input-name", default="gene_importance.csv",
                   help="per-seed file to aggregate (default: %(default)s)")
    p.add_argument("--score-column", default="importance_sum",
                   help="column to rank on (default: %(default)s)")
    p.add_argument("--gene-column", default="gene",
                   help="identifier column (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV (default: <results_dir>/<input stem>_consensus.csv)")
    p.add_argument("--top-n", type=int, default=20,
                   help="cutoff for the n_top20 column (default: %(default)s)")
    args = p.parse_args()

    paths = sorted(d / args.input_name for d in args.results_dir.glob("GenNet_experiment_*_")
                   if (d / args.input_name).exists())
    if len(paths) < 2:
        print(f"need >=2 experiments with {args.input_name} in {args.results_dir}, "
              f"found {len(paths)}", file=sys.stderr)
        return 1

    print(f"aggregating {len(paths)} seeds on '{args.score_column}' "
          f"from {args.results_dir}/*/{args.input_name}")
    for path in paths:
        print(f"  {path.parent.name}")

    consensus = build_consensus(paths, args.gene_column, args.score_column, args.top_n)

    out_path = args.out or args.results_dir / f"{Path(args.input_name).stem}_consensus.csv"
    consensus.to_csv(out_path, index=False)
    print(f"wrote {len(consensus)} genes -> {out_path}")

    stable = consensus[consensus.n_top20 == len(paths)]
    print(f"in the top-{args.top_n} of all {len(paths)} seeds: "
          f"{len(stable)} gene(s) {list(stable.gene)}")
    print("consensus top-15:", list(consensus.gene[:15]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
