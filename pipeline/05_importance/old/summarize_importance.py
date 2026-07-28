#!/usr/bin/env python3
"""Cross-seed stability of weight-based gene importance (guideline 2D.4).

Takes several per-seed importance folders (each with a ``gene_importance.csv``
written by ``compute_weight_importance.py``) and reports how stable the gene
ranking is across the data splits:

  * a merged CSV with each seed's importance, the across-seed mean/std/CV, and the
    mean rank — sorted by mean importance;
  * the pairwise **Spearman** rank correlation between seeds (the headline
    stability number is the mean off-diagonal correlation).

Pure pandas/scipy (no model load). Usage (from repo root):
    python pipeline/05_importance/summarize_importance.py \
        reports/importance/GenNet_experiment_105 \
        reports/importance/GenNet_experiment_143 \
        reports/importance/GenNet_experiment_144 \
        reports/importance/GenNet_experiment_145 \
        reports/importance/GenNet_experiment_146 \
        --name tanh
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_seed(folder: Path, metric: str) -> pd.Series:
    df = pd.read_csv(folder / "gene_importance.csv")
    return df.set_index("gene")[metric].rename(folder.name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="Per-seed importance folders")
    ap.add_argument("--metric", default="importance_sum",
                    choices=["importance_sum", "importance_mean"],
                    help="Importance column to compare (default: importance_sum)")
    ap.add_argument("--name", default="importance",
                    help="Output basename (default: importance)")
    ap.add_argument("--out-dir", default="reports/importance",
                    help="Folder for the stability CSV (default: reports/importance)")
    args = ap.parse_args()

    series = [load_seed(Path(f), args.metric) for f in args.folders]
    wide = pd.concat(series, axis=1)  # genes x seeds; NaN where a gene is absent
    cols = list(wide.columns)

    # --- pairwise Spearman on genes present in both members of each pair ---
    print(f"Pairwise Spearman rank correlation ({args.metric}):")
    corrs = []
    for a, b in combinations(cols, 2):
        pair = wide[[a, b]].dropna()
        rho = spearmanr(pair[a], pair[b]).correlation
        corrs.append(rho)
        print(f"  {a} vs {b}: rho={rho:.3f}  (n={len(pair)})")
    if corrs:
        print(f"Mean off-diagonal Spearman: {np.mean(corrs):.3f} "
              f"(min {np.min(corrs):.3f}, max {np.max(corrs):.3f})")

    # --- merged table with across-seed stats ---
    stats = pd.DataFrame(index=wide.index)
    stats["mean"] = wide.mean(axis=1)
    stats["std"] = wide.std(axis=1)
    stats["cv"] = stats["std"] / stats["mean"]
    stats["n_seeds"] = wide.notna().sum(axis=1)
    stats["mean_rank"] = wide.rank(ascending=False).mean(axis=1)
    out = pd.concat([wide, stats], axis=1).sort_values("mean", ascending=False)

    out_path = Path(args.out_dir) / f"{args.name}_gene_stability.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(f"\nWrote {len(out)} genes -> {out_path}")
    print("Top 10 by mean importance (rank stability = low std of mean_rank):")
    print(out.head(10)[["mean", "std", "n_seeds", "mean_rank"]].to_string())


if __name__ == "__main__":
    main()
