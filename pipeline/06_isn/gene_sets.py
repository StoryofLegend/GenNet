#!/usr/bin/env python3
"""Define the ISN gene sets (step 1 of the ISN/HotZone flow) and measure their overlap.

The supervisor wants ISN + HotZone computed for every importance method -- 2A
(weight), 2B (SHAP), 2C (ablation) and a combined ranking -- at several gene-count
cutoffs. This script is the single place those gene sets are defined, so every
downstream step (make_isn_input -> lioness -> hotzone) consumes the same files
rather than re-deriving a ranking and silently disagreeing.

It also answers a question worth settling BEFORE spending the compute: how
different are the four sets? The three methods already agree on 31 of their top
50 genes, so four near-identical gene sets would give four near-identical
HotZones. The Jaccard tables printed here say how much separation to expect.

Input
-----
<results_dir>/method_comparison.csv, written by pipeline/04_report/compare_methods.py.
It carries one rank column per method plus the combined rank, all over the same
6014 genes, so the sets are guaranteed to be aligned.

Output (into <results_dir>/isn_gene_sets/)
-----------------------------------------
gene_set_<method>_top<N>.csv   the selected genes, in rank order, with degree and
                               n_gene_nodes carried through. This is what
                               make_isn_input.py reads.
gene_set_membership_top<N>.csv one row per gene in the UNION of the four sets at
                               cutoff N, with a boolean column per method and
                               n_methods -- shows which genes drive the differences.
gene_set_overlap.csv           long-format pairwise overlap: cutoff, method_a,
                               method_b, n_shared, jaccard.

On the 'combined' ranking
-------------------------
Combined is the mean of the three primary per-method ranks (rank_combined in
method_comparison.csv), not a mean of scores: the three scores are on different
scales and are not averageable, ranks are. It gives a full 6014-gene ordering, so
any cutoff works -- unlike the 3-method intersection, which is a fixed set of 31
genes and cannot be varied by N.

A note on the per-degree variants
---------------------------------
--variant per_degree is supported for completeness, but 2A_per_degree is not a
usable ranking: dividing the weight sum by pathway degree overcorrects (Spearman
vs degree flips from +0.36 to -0.78) and its top genes share only 2-4 of the top
50 with any other method. Use raw unless you are specifically studying the
normalisation.

Usage
-----
    python pipeline/06_isn/gene_sets.py results/tanh
    python pipeline/06_isn/gene_sets.py results/tanh --cutoffs 50 100 250
    python pipeline/06_isn/gene_sets.py results/tanh --variant per_degree
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

# method label -> rank column in method_comparison.csv, per variant
RANK_COLUMNS = {
    "raw": {
        "2A": "rank_2A_weight",
        "2B": "rank_2B_shap",
        "2C": "rank_2C_ablation",
        "combined": "rank_combined",
    },
    "per_degree": {
        "2A": "rank_2A_per_degree",
        "2B": "rank_2B_per_degree",
        "2C": "rank_2C_per_degree",
        "combined": "rank_combined",
    },
}
CARRY = ("degree", "n_gene_nodes")


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


def print_matrix(frame: pd.DataFrame, title: str, fmt: str) -> None:
    width = max(10, max(len(str(c)) for c in frame.columns) + 2)
    print(title)
    print(" " * 10 + "".join(f"{c:>{width}}" for c in frame.columns))
    for label, row in frame.iterrows():
        cells = "".join(f"{fmt.format(v):>{width}}" for v in row)
        print(f"{label:>10}{cells}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding method_comparison.csv, e.g. results/tanh")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[50, 100, 150, 200, 250],
                   help="gene-count cutoffs to emit (default: %(default)s)")
    p.add_argument("--variant", choices=sorted(RANK_COLUMNS), default="raw",
                   help="raw scores or the connectivity-normalised ones "
                        "(default: %(default)s; see the note in the docstring)")
    p.add_argument("--methods", nargs="+", default=["2A", "2B", "2C", "combined"],
                   help="subset of methods to emit (default: %(default)s)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output folder (default: <results_dir>/isn_gene_sets)")
    args = p.parse_args()

    comparison = args.results_dir / "method_comparison.csv"
    if not comparison.exists():
        print(f"{comparison} not found - run pipeline/04_report/compare_methods.py first",
              file=sys.stderr)
        return 1
    df = pd.read_csv(comparison)

    columns = RANK_COLUMNS[args.variant]
    unknown = [m for m in args.methods if m not in columns]
    if unknown:
        print(f"unknown method(s) {unknown}; choose from {sorted(columns)}", file=sys.stderr)
        return 1
    missing = [columns[m] for m in args.methods if columns[m] not in df.columns]
    if missing:
        print(f"{comparison} has no column(s) {missing} - regenerate it with "
              "compare_methods.py so every method is present", file=sys.stderr)
        return 1

    n_genes = len(df)
    too_big = [n for n in args.cutoffs if n > n_genes]
    if too_big:
        print(f"cutoff(s) {too_big} exceed the {n_genes} ranked genes - dropping",
              file=sys.stderr)
    cutoffs = sorted({n for n in args.cutoffs if 0 < n <= n_genes})
    if not cutoffs:
        print("no usable cutoffs", file=sys.stderr)
        return 1

    outdir = args.outdir or args.results_dir / "isn_gene_sets"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{n_genes} ranked genes | variant={args.variant} | cutoffs={cutoffs}")
    print(f"writing to {outdir}\n")

    overlap_rows = []
    for n in cutoffs:
        sets = {}
        for method in args.methods:
            col = columns[method]
            # nsmallest on the rank column rather than head(n) on file order: the
            # file is sorted by the COMBINED rank, so head(n) would return the same
            # genes for every method.
            sel = df.nsmallest(n, col).sort_values(col, kind="stable")
            keep = ["gene", col] + [c for c in CARRY if c in df.columns]
            out = sel[keep].rename(columns={col: "rank"})
            out.to_csv(outdir / f"gene_set_{method}_top{n}.csv", index=False)
            sets[method] = set(out.gene)

        # membership table over the union: which genes are shared, which are unique
        union = sorted(set().union(*sets.values()))
        membership = pd.DataFrame({"gene": union})
        for method in args.methods:
            membership[method] = membership.gene.isin(sets[method])
        membership["n_methods"] = membership[args.methods].sum(axis=1)
        membership = membership.sort_values(["n_methods", "gene"],
                                            ascending=[False, True], kind="stable")
        membership.to_csv(outdir / f"gene_set_membership_top{n}.csv", index=False)

        jac = pd.DataFrame(
            {a: {b: jaccard(sets[a], sets[b]) for b in args.methods} for a in args.methods}
        ).loc[args.methods, args.methods]
        shared = pd.DataFrame(
            {a: {b: len(sets[a] & sets[b]) for b in args.methods} for a in args.methods}
        ).loc[args.methods, args.methods]

        core = int((membership.n_methods == len(args.methods)).sum())
        print(f"--- top {n}: union {len(union)} genes, "
              f"{core} in all {len(args.methods)} methods")
        print_matrix(jac, "  Jaccard", "{:.3f}")
        print_matrix(shared, "  shared genes", "{:.0f}")
        print()

        for a, b in itertools.combinations(args.methods, 2):
            overlap_rows.append({"cutoff": n, "method_a": a, "method_b": b,
                                 "n_shared": len(sets[a] & sets[b]),
                                 "jaccard": jaccard(sets[a], sets[b])})

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(outdir / "gene_set_overlap.csv", index=False)
    print(f"wrote {len(overlap)} pairwise comparisons -> {outdir/'gene_set_overlap.csv'}")
    print(f"wrote {len(cutoffs) * len(args.methods)} gene-set files + "
          f"{len(cutoffs)} membership tables")

    mean_j = overlap.groupby("cutoff").jaccard.mean()
    print("\nmean pairwise Jaccard by cutoff (higher = the four sets are more alike):")
    for n, v in mean_j.items():
        print(f"  top {n:4d}: {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
