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
gene_set_INPUT_0_top<N>.csv    the supervisor's INPUT_0: the UNION of the top-N sets
                               of 2A, 2B and 2C. Carries each method's rank, the best
                               of the three, and n_methods. Same `gene` column as the
                               others, so it is a drop-in for make_isn_input.py.
gene_set_membership_top<N>.csv one row per gene in the union of every emitted set at
                               cutoff N, with a boolean column per method, n_methods
                               (counted over 2A/2B/2C only) and in_INPUT_0.
gene_set_overlap.csv           long-format pairwise overlap: cutoff, method_a,
                               method_b, n_shared, jaccard. Ranked methods only --
                               INPUT_0 is a superset of three of them, so its overlaps
                               are arithmetic, not information.
gene_set_sizes.csv             cutoff, method, n_genes for every set written. This is
                               the table to look at when choosing x (see below).

On the 'combined' ranking
-------------------------
Combined is the mean of the three primary per-method ranks (rank_combined in
method_comparison.csv), not a mean of scores: the three scores are on different
scales and are not averageable, ranks are. It gives a full 6014-gene ordering, so
any cutoff works -- unlike the 3-method intersection, which is a fixed set of 31
genes and cannot be varied by N.

On INPUT_0 and INPUT_B
----------------------
The supervisor's step 1 asks for INPUT_0 = the union of the top-x genes of methods A,
B and C, and INPUT_B = the top-x of method B alone. Method A/B/C are 2A (weight), 2B
(SHAP) and 2C (ablation), so INPUT_B *is* gene_set_2B_top<N>.csv -- it is not written
a second time under another name.

INPUT_0 is NOT the same object as `combined`, and the two must not be substituted for
each other: `combined` takes the top N of the mean rank, INPUT_0 takes the union of
three top-N sets, so |INPUT_0| > N by construction (78 genes at N=50, 181 at N=100,
486 at N=250 on results/tanh). At N=50 the two share only 45 genes.

That size blow-up is the reason `x` has to be tuned rather than chosen: the quantity
to match against Gaia's HotZone gene-set size is |INPUT_0|, not N. gene_set_sizes.csv
gives |INPUT_0| for every cutoff, so run this over a range of cutoffs first and pick x
from that table. Note also that ISN cost scales with the SQUARE of the gene count, so
INPUT_0 at cutoff N costs far more than any single method at the same cutoff -- see
the memory note in run_lioness.sh.

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

# The three primary importance methods (the supervisor's A/B/C). INPUT_0 is their
# union and `combined` is a mean-rank ordering over them; neither is primary, so
# neither is counted in n_methods.
PRIMARY = ("2A", "2B", "2C")
UNION_SET = "INPUT_0"


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
    p.add_argument("--methods", nargs="+",
                   default=["2A", "2B", "2C", "combined", UNION_SET],
                   help=f"subset of methods to emit (default: %(default)s). "
                        f"{UNION_SET} is derived, not ranked: it is the union of the "
                        f"top-N sets of {'/'.join(PRIMARY)}, so those three must be "
                        f"emitted alongside it.")
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
    unknown = [m for m in args.methods if m not in columns and m != UNION_SET]
    if unknown:
        print(f"unknown method(s) {unknown}; choose from "
              f"{sorted(set(columns) | {UNION_SET})}", file=sys.stderr)
        return 1

    # UNION_SET has no rank column of its own; it is assembled from the primaries,
    # so refuse rather than silently emit a partial union.
    ranked = [m for m in args.methods if m != UNION_SET]
    want_union = UNION_SET in args.methods
    if want_union and (absent := [m for m in PRIMARY if m not in ranked]):
        print(f"{UNION_SET} is the union of {'/'.join(PRIMARY)} but {absent} "
              "was not requested - add it to --methods", file=sys.stderr)
        return 1

    missing = [columns[m] for m in ranked if columns[m] not in df.columns]
    if missing:
        print(f"{comparison} has no column(s) {missing} - regenerate it with "
              "compare_methods.py so every method is present", file=sys.stderr)
        return 1
    primary = [m for m in PRIMARY if m in ranked]

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

    overlap_rows, size_rows = [], []
    for n in cutoffs:
        sets = {}
        for method in ranked:
            col = columns[method]
            # nsmallest on the rank column rather than head(n) on file order: the
            # file is sorted by the COMBINED rank, so head(n) would return the same
            # genes for every method.
            sel = df.nsmallest(n, col).sort_values(col, kind="stable")
            keep = ["gene", col] + [c for c in CARRY if c in df.columns]
            out = sel[keep].rename(columns={col: "rank"})
            out.to_csv(outdir / f"gene_set_{method}_top{n}.csv", index=False)
            sets[method] = set(out.gene)

        # INPUT_0 = the union of the primary top-N sets. A union has no rank of its
        # own, so it carries all three ranks plus the best of them and how many sets
        # the gene came from; `gene` is the only column anything downstream reads.
        if want_union:
            members = set().union(*(sets[m] for m in primary))
            u = df[df.gene.isin(members)].copy()
            for m in primary:
                u[f"rank_{m}"] = u[columns[m]]
                u[f"in_{m}"] = u.gene.isin(sets[m])
            u["best_rank"] = u[[f"rank_{m}" for m in primary]].min(axis=1)
            u["n_methods"] = u[[f"in_{m}" for m in primary]].sum(axis=1)
            keep = (["gene", "best_rank", "n_methods"]
                    + [f"rank_{m}" for m in primary] + [f"in_{m}" for m in primary]
                    + [c for c in CARRY if c in df.columns])
            u = u[keep].sort_values(["n_methods", "best_rank"],
                                    ascending=[False, True], kind="stable")
            u.to_csv(outdir / f"gene_set_{UNION_SET}_top{n}.csv", index=False)
            sets[UNION_SET] = set(u.gene)
            if len(sets[UNION_SET]) != len(members):
                raise ValueError(f"{comparison} lists a gene more than once - the "
                                 "union would be mis-sized")

        # membership table: one row per gene in ANY emitted set. n_methods counts the
        # PRIMARIES only -- `combined` is a mean-rank ordering over them and INPUT_0
        # is their union, so counting either would double-count the same evidence.
        union = sorted(set().union(*sets.values()))
        membership = pd.DataFrame({"gene": union})
        for method in ranked:
            membership[method] = membership.gene.isin(sets[method])
        membership["n_methods"] = membership[primary].sum(axis=1) if primary else 0
        membership[f"in_{UNION_SET}"] = membership.n_methods > 0
        membership = membership.sort_values(["n_methods", "gene"],
                                            ascending=[False, True], kind="stable")
        membership.to_csv(outdir / f"gene_set_membership_top{n}.csv", index=False)

        # Ranked methods only: INPUT_0 contains three of them whole, so its Jaccard
        # against them is a function of the set sizes and carries no information.
        jac = pd.DataFrame(
            {a: {b: jaccard(sets[a], sets[b]) for b in ranked} for a in ranked}
        ).loc[ranked, ranked]
        shared = pd.DataFrame(
            {a: {b: len(sets[a] & sets[b]) for b in ranked} for a in ranked}
        ).loc[ranked, ranked]

        core = int((membership.n_methods == len(primary)).sum()) if primary else 0
        print(f"--- top {n}: {len(union)} genes in at least one set, "
              f"{core} in all {len(primary)} of {'/'.join(primary)}")
        if want_union:
            print(f"    |{UNION_SET}| = {len(sets[UNION_SET])} genes")
        print_matrix(jac, "  Jaccard", "{:.3f}")
        print_matrix(shared, "  shared genes", "{:.0f}")
        print()

        for method in ranked + ([UNION_SET] if want_union else []):
            size_rows.append({"cutoff": n, "method": method,
                              "n_genes": len(sets[method])})
        for a, b in itertools.combinations(ranked, 2):
            overlap_rows.append({"cutoff": n, "method_a": a, "method_b": b,
                                 "n_shared": len(sets[a] & sets[b]),
                                 "jaccard": jaccard(sets[a], sets[b])})

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(outdir / "gene_set_overlap.csv", index=False)
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(outdir / "gene_set_sizes.csv", index=False)
    print(f"wrote {len(overlap)} pairwise comparisons -> {outdir/'gene_set_overlap.csv'}")
    print(f"wrote {len(sizes)} gene-set files + {len(cutoffs)} membership tables")

    if not overlap.empty:
        mean_j = overlap.groupby("cutoff").jaccard.mean()
        print(f"\nmean pairwise Jaccard by cutoff (higher = the {len(ranked)} ranked "
              "sets are more alike):")
        for n, v in mean_j.items():
            print(f"  top {n:4d}: {v:.3f}")

    if want_union:
        print(f"\n{UNION_SET} size by cutoff. THIS, not the cutoff, is the number to "
              "match against\nGaia's gene-set size - and ISN cost scales with its "
              "square:")
        for _, r in sizes[sizes.method == UNION_SET].iterrows():
            print(f"  top {int(r.cutoff):4d}: {int(r.n_genes):5d} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
