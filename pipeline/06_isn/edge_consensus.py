#!/usr/bin/env python
"""
Collapse the five seeds into one edge ranking, per (method, cutoff, reference).

Why this exists
---------------
The gene set is already a 5-seed consensus, but the ISN itself is not: the seed
selects which trained model's activations fill the input matrix, so each seed gives
a different edge variance. Without this step you have five separate answers and no
statement of which edges are reproducible.

Seeds are combined by MEAN RANK of the per-edge variance, never by averaging the
variances. Two reasons, both already established in this project:
  * the five seeds have near-disjoint test cohorts (~15% pairwise overlap), so the
    variances are computed over different people and are not on a common scale;
  * hidden nodes have no canonical sign, so magnitudes are not comparable between
    training runs even when the cohorts coincide.
Rank is the only quantity that survives both. This mirrors what
pipeline/04_report/seed_consensus.py does for gene importance.

Edges are canonicalised as a sorted (gene, gene) pair before matching. lioness.py
takes the upper triangle in the matrix's COLUMN order, which is gene-rank order, so
the same undirected edge can appear as (A, B) in one file and (B, A) in another as
soon as the ranking differs. Matching on the raw columns would silently drop those.

Agreement between seeds is measured, not assumed: pairwise Spearman of the variance
ranking and the top-K Jaccard go into the summary. A group where the seeds disagree
is a result, not a failure -- it says the HotZone would not have reproduced.

Outputs (into <results_dir>/isn_consensus/)
------------------------------------------
    edge_consensus_<method>_top<N>[_ref1].csv
        source, target, mean_rank, rank_sd, n_seeds,
        rank_seed<S>..., var_seed<S>..., var_mean, var_median
    edge_consensus_summary.csv
        one row per group: sizes, mean pairwise Spearman, top-K Jaccard

Usage
-----
    python pipeline/06_isn/edge_consensus.py results/tanh
    python pipeline/06_isn/edge_consensus.py results/tanh --methods INPUT_0 --cutoffs 250
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIR_RE = re.compile(r"^(?P<method>.+)_top(?P<cutoff>\d+)_seed(?P<seed>\d+)(?P<ref>_ref\d+)?$")


def scan(isn_root: Path) -> pd.DataFrame:
    """Every completed LIONESS run under <results>/isn/, parsed into a table."""
    rows = []
    for d in sorted(p for p in isn_root.iterdir() if p.is_dir()):
        m = DIR_RE.match(d.name)
        if not m or not (d / "isn_edge_stats.csv").exists():
            continue
        rows.append({"method": m["method"], "cutoff": int(m["cutoff"]),
                     "seed": int(m["seed"]),
                     "reference": (m["ref"] or "_ref01").removeprefix("_ref"),
                     "suffix": m["ref"] or "", "path": d})
    return pd.DataFrame(rows)


def load_ranked(path: Path) -> pd.DataFrame:
    """Edge stats with a canonical undirected key and a variance rank (1 = highest)."""
    df = pd.read_csv(path / "isn_edge_stats.csv")
    a = df[["source", "target"]].to_numpy()
    lo = np.minimum(a[:, 0], a[:, 1])
    hi = np.maximum(a[:, 0], a[:, 1])
    df["edge"] = pd.Index(lo) + "|" + pd.Index(hi)
    if df.edge.duplicated().any():
        dup = df.edge[df.edge.duplicated()].iloc[0]
        raise ValueError(f"{path}: duplicate undirected edge {dup} - the ISN was "
                         "expected to be an upper triangle")
    df["rank"] = df["var"].rank(ascending=False, method="average")
    return df.set_index("edge")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path, help="e.g. results/tanh")
    p.add_argument("--methods", nargs="+", default=None, help="default: all found")
    p.add_argument("--cutoffs", type=int, nargs="+", default=None)
    p.add_argument("--refs", nargs="+", default=None, help="e.g. 01 1")
    p.add_argument("--min-seeds", type=int, default=2,
                   help="skip groups with fewer completed seeds (default 2)")
    p.add_argument("--topk", type=int, default=100,
                   help="K for the top-K Jaccard agreement diagnostic (default 100)")
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    isn_root = args.results_dir / "isn"
    if not isn_root.is_dir():
        print(f"no {isn_root}", file=sys.stderr)
        return 1
    runs = scan(isn_root)
    if runs.empty:
        print(f"no completed runs under {isn_root}", file=sys.stderr)
        return 1
    for col, want in (("method", args.methods), ("cutoff", args.cutoffs),
                      ("reference", args.refs)):
        if want:
            runs = runs[runs[col].isin(want)]
    if runs.empty:
        print("nothing matches those filters", file=sys.stderr)
        return 1

    outdir = args.outdir or args.results_dir / "isn_consensus"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for (method, cutoff, ref), grp in runs.groupby(["method", "cutoff", "reference"],
                                                   sort=True):
        grp = grp.sort_values("seed")
        if len(grp) < args.min_seeds:
            print(f"{method}_top{cutoff}_ref{ref}: only {len(grp)} seed(s), skipped")
            continue

        per_seed = {int(r.seed): load_ranked(r.path) for r in grp.itertuples()}
        common = set.intersection(*(set(d.index) for d in per_seed.values()))
        if not common:
            print(f"{method}_top{cutoff}_ref{ref}: no edges shared by all seeds, skipped")
            continue
        edges = sorted(common)

        ranks = pd.DataFrame({s: d.loc[edges, "rank"] for s, d in per_seed.items()},
                             index=edges)
        variances = pd.DataFrame({s: d.loc[edges, "var"] for s, d in per_seed.items()},
                                 index=edges)

        # pairwise agreement between seeds, on the shared edges only
        rhos = [spearmanr(ranks[a], ranks[b]).statistic
                for a, b in itertools.combinations(ranks.columns, 2)]
        topsets = [set(ranks[s].nsmallest(args.topk).index) for s in ranks.columns]
        jacs = [jaccard(a, b) for a, b in itertools.combinations(topsets, 2)]

        out = pd.DataFrame(index=edges)
        out["source"] = [e.split("|")[0] for e in edges]
        out["target"] = [e.split("|")[1] for e in edges]
        out["mean_rank"] = ranks.mean(axis=1)
        out["rank_sd"] = ranks.std(axis=1, ddof=0)
        out["n_seeds"] = len(ranks.columns)
        for s in ranks.columns:
            out[f"rank_seed{s}"] = ranks[s]
        for s in variances.columns:
            out[f"var_seed{s}"] = variances[s]
        out["var_mean"] = variances.mean(axis=1)
        out["var_median"] = variances.median(axis=1)
        out = out.sort_values("mean_rank", kind="stable").reset_index(drop=True)

        suffix = "" if ref == "01" else f"_ref{ref}"
        dest = outdir / f"edge_consensus_{method}_top{cutoff}{suffix}.csv"
        tmp = dest.with_suffix(".tmp.csv")
        out.to_csv(tmp, index=False)
        tmp.replace(dest)

        n_union = len(set().union(*(set(d.index) for d in per_seed.values())))
        summary.append({
            "method": method, "cutoff": cutoff, "reference": ref,
            "n_seeds": len(per_seed), "seeds": ",".join(map(str, sorted(per_seed))),
            "n_edges_common": len(edges), "n_edges_union": n_union,
            "mean_spearman": float(np.mean(rhos)), "min_spearman": float(np.min(rhos)),
            f"mean_jaccard_top{args.topk}": float(np.mean(jacs)),
            "file": dest.name,
        })
        print(f"{method}_top{cutoff}_ref{ref:<2}: {len(per_seed)} seeds, "
              f"{len(edges):>6} common edges (union {n_union}), "
              f"rho={np.mean(rhos):.3f} [min {np.min(rhos):.3f}], "
              f"J@{args.topk}={np.mean(jacs):.3f}")

    if not summary:
        print("no group had enough seeds", file=sys.stderr)
        return 1
    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "edge_consensus_summary.csv", index=False)
    print(f"\nwrote {len(summary)} consensus file(s) + summary to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
