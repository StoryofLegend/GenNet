#!/usr/bin/env python3
"""Build the patients x genes matrices LIONESS consumes (step 3).

Takes the per-patient node activations written by extract_activations.py and the
gene sets written by gene_sets.py, and emits one matrix per
(method, cutoff, seed) -- the input `lioness.py` reads with
`pd.read_csv(..., index_col=0)` followed by `.corr()`.

    <results_dir>/isn_input/isn_input_<method>_top<N>_seed<S>.csv
        rows    = patient_id (all 9942 test subjects, cases and controls)
        columns = the top-N genes of that method, in rank order

Collapsing slots to one value per gene
--------------------------------------
A gene occupies several gene-layer nodes (mean 3.5, max 103 here), so the slots
have to be reduced to a single per-patient number. --agg mean is the default: it
is independent of how many slots a gene happens to own. --agg sum makes a gene
with more slots systematically larger, which is the same connectivity artefact
Sec 1A warns about, so it is offered but not recommended.

The old implementation also had a max_abs option, picking whichever slot had the
largest magnitude FOR EACH PATIENT. That is not implemented here on purpose: it
makes the column a different quantity for different patients, and every
downstream step correlates that column across patients.

Near-constant genes
-------------------
Most of this network is inactive (L1=0.001, L1_act=0.01): the median gene-layer
slot has sd ~5e-6 across individuals, and the methods differ a lot in how many
such genes they select -- in the top 100, 2A picks 47 with sd < 1e-4, 2C only 18.
This matters because a HotZone is by definition a subnetwork whose wiring varies
across individuals, so a flat gene contributes only noise, which clustering can
still assemble into a spurious community.

The dead-gene count is therefore ALWAYS reported, per method and cutoff, and
written to the manifest. But --min-sd defaults to 0, i.e. nothing is dropped.
Filtering by default would be a mistake here: it would leave each method with a
different number of genes (2A ~53, 2C ~82 at top-100), so the resulting networks
would differ in size and their HotZone scores would no longer be comparable --
which is the whole point of running all four methods. Use --min-sd only for an
explicit sensitivity check, where every method is filtered at the same threshold
and the changed sizes are acknowledged.

Usage
-----
    python pipeline/06_isn/make_isn_input.py results/tanh
    python pipeline/06_isn/make_isn_input.py results/tanh --cutoffs 100 250
    python pipeline/06_isn/make_isn_input.py results/tanh --methods 2C --agg sum
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AGGREGATORS = ("mean", "sum", "median")


def seed_of(exp: Path) -> str:
    """Seed label from train_args.json datapath (processed_data/seed_42/ -> '42')."""
    datapath = json.loads((exp / "train_args.json").read_text()).get("datapath", "")
    m = re.search(r"seed_(\w+)", datapath)
    return m.group(1) if m else exp.name


def load_seed(exp: Path):
    """Return (gene_frame, subjects). gene_frame is patients x gene, slots collapsed."""
    for name in ("isn_gene_act.npy", "isn_gene_slots.csv", "isn_subjects.csv"):
        if not (exp / name).exists():
            raise FileNotFoundError(f"{exp/name} missing - run "
                                    "pipeline/06_isn/extract_activations.py first")
    act = np.load(exp / "isn_gene_act.npy")
    slots = pd.read_csv(exp / "isn_gene_slots.csv")
    subjects = pd.read_csv(exp / "isn_subjects.csv")
    if len(slots) != act.shape[1]:
        raise ValueError(f"{exp.name}: {len(slots)} slots vs {act.shape[1]} columns")
    if len(subjects) != act.shape[0]:
        raise ValueError(f"{exp.name}: {len(subjects)} subjects vs {act.shape[0]} rows")
    return act, slots, subjects


def collapse(act: np.ndarray, slots: pd.DataFrame, genes: list[str],
             agg: str) -> np.ndarray:
    """(n_patients, n_slots) -> (n_patients, len(genes)), one column per gene."""
    by_gene = slots.groupby("gene").indices        # gene -> positions in `act`
    out = np.zeros((act.shape[0], len(genes)), dtype=np.float32)
    for j, gene in enumerate(genes):
        cols = by_gene.get(gene)
        if cols is None:
            continue                                # absent: reported by the caller
        block = act[:, cols]
        if agg == "mean":
            out[:, j] = block.mean(axis=1)
        elif agg == "sum":
            out[:, j] = block.sum(axis=1)
        else:
            out[:, j] = np.median(block, axis=1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding GenNet_experiment_<ID>_/ dirs, e.g. results/tanh")
    p.add_argument("--gene-sets", type=Path, default=None,
                   help="folder of gene_set_*.csv (default: <results_dir>/isn_gene_sets)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output folder (default: <results_dir>/isn_input)")
    p.add_argument("--methods", nargs="+", default=["2A", "2B", "2C", "combined"])
    p.add_argument("--cutoffs", type=int, nargs="+", default=None,
                   help="cutoffs to build (default: every cutoff found in --gene-sets)")
    p.add_argument("--agg", choices=AGGREGATORS, default="mean",
                   help="how to collapse a gene's slots (default: %(default)s)")
    p.add_argument("--min-sd", type=float, default=0.0,
                   help="drop genes whose activation sd across patients is below this "
                        "(default: %(default)s = keep everything; see the docstring "
                        "before changing it)")
    p.add_argument("--dead-sd", type=float, default=1e-4,
                   help="threshold for the reported dead-gene count (default: %(default)s)")
    args = p.parse_args()

    exps = sorted(d for d in args.results_dir.glob("GenNet_experiment_*_")
                  if (d / "isn_gene_act.npy").exists())
    if not exps:
        print(f"no experiment in {args.results_dir} has isn_gene_act.npy - run "
              "pipeline/06_isn/extract_activations.py first", file=sys.stderr)
        return 1

    gene_sets_dir = args.gene_sets or args.results_dir / "isn_gene_sets"
    available = sorted({int(m.group(1)) for f in gene_sets_dir.glob("gene_set_*_top*.csv")
                        if (m := re.search(r"_top(\d+)\.csv$", f.name))})
    if not available:
        print(f"no gene_set_*_top*.csv in {gene_sets_dir} - run "
              "pipeline/06_isn/gene_sets.py first", file=sys.stderr)
        return 1
    cutoffs = sorted(set(args.cutoffs) & set(available)) if args.cutoffs else available
    if args.cutoffs and (skipped := sorted(set(args.cutoffs) - set(available))):
        print(f"[warn] no gene set for cutoff(s) {skipped}; available: {available}",
              file=sys.stderr)
    if not cutoffs:
        return 1

    outdir = args.outdir or args.results_dir / "isn_input"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(exps)} seeds | methods={args.methods} | cutoffs={cutoffs} | "
          f"agg={args.agg} | min_sd={args.min_sd}")
    print(f"writing to {outdir}\n")

    manifest, cohorts = [], {}
    for exp in exps:
        seed = seed_of(exp)
        act, slots, subjects = load_seed(exp)
        cohorts[seed] = tuple(subjects.patient_id)
        print(f"=== seed {seed} ({exp.name}): {act.shape[0]} patients, "
              f"{act.shape[1]} slots, {slots.gene.nunique()} genes")

        for method in args.methods:
            for n in cutoffs:
                gs = gene_sets_dir / f"gene_set_{method}_top{n}.csv"
                if not gs.exists():
                    print(f"  [warn] {gs.name} missing - skipping", file=sys.stderr)
                    continue
                genes = pd.read_csv(gs).gene.tolist()
                present = set(slots.gene)
                absent = [g for g in genes if g not in present]
                genes = [g for g in genes if g in present]

                mat = collapse(act, slots, genes, args.agg)
                sd = mat.std(axis=0)
                n_dead = int((sd < args.dead_sd).sum())
                if args.min_sd > 0:
                    keep = sd >= args.min_sd
                    mat, genes = mat[:, keep], [g for g, k in zip(genes, keep) if k]

                df = pd.DataFrame(mat, index=subjects.patient_id, columns=genes)
                df.index.name = "patient_id"
                out = outdir / f"isn_input_{method}_top{n}_seed{seed}.csv"
                tmp = out.with_suffix(".csv.tmp")
                df.to_csv(tmp, float_format="%.6g")
                tmp.replace(out)

                manifest.append({"seed": seed, "method": method, "cutoff": n,
                                 "n_genes_written": len(genes),
                                 "n_absent_from_topology": len(absent),
                                 f"n_dead_sd_lt_{args.dead_sd:g}": n_dead,
                                 "median_sd": float(np.median(sd)) if len(sd) else np.nan,
                                 "agg": args.agg, "min_sd": args.min_sd,
                                 "file": out.name})
                note = f" ({len(absent)} absent)" if absent else ""
                print(f"  {method:>8} top{n:<4} -> {len(genes):4d} genes, "
                      f"{n_dead:3d} dead (sd<{args.dead_sd:g}), "
                      f"median sd {np.median(sd):.2e}{note}")

    man = pd.DataFrame(manifest)
    man.to_csv(outdir / "isn_input_manifest.csv", index=False)
    print(f"\nwrote {len(man)} matrices -> {outdir}")

    # Cross-seed cohort check: seeds have their own train/val/test split, so the
    # test cohorts may differ. Downstream comparisons of ISNs across seeds are only
    # patient-wise meaningful if the cohorts coincide.
    uniq = set(cohorts.values())
    if len(uniq) == 1:
        print(f"all {len(cohorts)} seeds share the same {len(next(iter(uniq)))}-patient "
              "test cohort - ISNs are directly comparable across seeds")
    else:
        sizes = {s: len(v) for s, v in cohorts.items()}
        print(f"[warn] seeds do NOT share a test cohort ({sizes}) - compare ISNs across "
              "seeds at the gene/edge level, not patient by patient")

    dead_col = [c for c in man.columns if c.startswith("n_dead")][0]
    print("\ndead genes by method (mean over seeds and cutoffs):")
    for method, grp in man.groupby("method"):
        print(f"  {method:>8}: {grp[dead_col].mean():5.1f} of "
              f"{grp.n_genes_written.mean():5.1f} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
