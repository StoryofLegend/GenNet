#!/usr/bin/env python
"""
Write the list of *live* gene nodes -- the ones whose GenNet gene-layer activation
actually varies between patients.

Most gene nodes are dead: the median activation sd over the 506-gene pool is
4.7e-06, i.e. numerical dust. LIONESS still computes edges for them and still
assigns those edges a variance, so they compete on equal footing in the ranking
that hotzone.py thresholds on. Filtering them out is a correctness measure, not a
cosmetic one.

The filter is unsupervised (labels are never read), so it is not label leakage.

The sd is taken as the MEDIAN ACROSS SEEDS of the per-gene sd, so every seed ends
up with the same gene set. That matters downstream: cross-seed edge consensus can
only average ranks over edges that exist in all five seeds, and a per-seed filter
would give five different node sets.

Slots are collapsed to one value per gene by mean before the sd is taken, matching
what make_isn_input.py does with --agg mean.

    <results_dir>/isn_gene_sets/live_genes_sd<thr>.csv    gene, sd_median, sd_<seed>...

Usage
-----
    python pipeline/07_hotzone/live_genes.py results/tanh --min-sd 1e-4
    python pipeline/07_hotzone/live_genes.py results/tanh --min-sd 1e-3 --restrict-to INPUT_0 --cutoff 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_gene_sd(exp: Path) -> pd.Series | None:
    """Per-gene activation sd for one experiment, slots collapsed by mean."""
    act_f, slot_f = exp / "isn_gene_act.npy", exp / "isn_gene_slots.csv"
    if not (act_f.exists() and slot_f.exists()):
        return None
    act = np.load(act_f)
    slots = pd.read_csv(slot_f)
    if len(slots) != act.shape[1]:
        raise ValueError(f"{exp.name}: {len(slots)} slots vs {act.shape[1]} activation columns")
    df = pd.DataFrame(act, columns=slots.gene.values)
    return df.T.groupby(level=0).mean().T.std(axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="e.g. results/tanh -- holds GenNet_experiment_*_/ and isn_gene_sets/")
    p.add_argument("--min-sd", type=float, default=1e-4,
                   help="keep genes whose across-seed median activation sd exceeds this "
                        "(default 1e-4; the pool median is 4.7e-06)")
    p.add_argument("--restrict-to", default=None,
                   help="also require membership of gene_set_<this>_top<cutoff>.csv")
    p.add_argument("--cutoff", type=int, default=250,
                   help="cutoff of the gene set named by --restrict-to (default 250)")
    p.add_argument("--gene-sets-dir", type=Path, default=None,
                   help="default: <results_dir>/isn_gene_sets")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <gene_sets_dir>/live_genes_sd<min-sd>.csv")
    args = p.parse_args()

    exps = sorted(args.results_dir.glob("GenNet_experiment_*_"))
    sds = {}
    for e in exps:
        s = per_gene_sd(e)
        if s is not None:
            sds[e.name] = s
    if not sds:
        print(f"no isn_gene_act.npy under {args.results_dir} - run "
              f"pipeline/06_isn/run_extract_activations.sh first", file=sys.stderr)
        return 1

    S = pd.DataFrame(sds)
    if S.isna().any().any():
        # a gene missing from one seed means the seeds disagree on the slot table
        missing = S.index[S.isna().any(axis=1)].tolist()
        raise ValueError(f"{len(missing)} genes absent from some seed, e.g. {missing[:5]}")
    S["sd_median"] = S.median(axis=1)

    keep = S.sd_median > args.min_sd
    out = S[keep].sort_values("sd_median", ascending=False).reset_index()
    out = out.rename(columns={"index": "gene"})

    gs_dir = args.gene_sets_dir or args.results_dir / "isn_gene_sets"
    if args.restrict_to:
        f = gs_dir / f"gene_set_{args.restrict_to}_top{args.cutoff}.csv"
        if not f.exists():
            print(f"missing {f}", file=sys.stderr)
            return 1
        members = set(pd.read_csv(f).gene)
        before = len(out)
        out = out[out.gene.isin(members)]
        print(f"restricted to {args.restrict_to}_top{args.cutoff}: {before} -> {len(out)}")

    dest = args.out or gs_dir / f"live_genes_sd{args.min_sd:g}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(dest)

    print(f"seeds        : {len(sds)}")
    print(f"gene pool    : {len(S)} (median sd {S.sd_median.median():.2e})")
    print(f"live         : {len(out)} at sd > {args.min_sd:g}")
    print(f"wrote        : {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
