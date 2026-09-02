#!/usr/bin/env python3
"""Cross-method comparison of gene importance (guidelines Sec 2D).

The guidelines ask for three independent importance methods and a validation step
that compares them and checks stability across the 5 random seeds:

    2A weight-based   sum |w_gene->pathway|          gene_importance_consensus.csv
    2B SHAP           sum mean|SHAP| over the SNPs   shap_gene_importance_consensus.csv
    2C perturbation   mean |y_full - y_{gene=0}|     ablation_importance_consensus.csv

This consumes the seed-consensus tables (not single seeds) and reports:

  * Spearman between the three consensus rankings, and between each raw ranking and
    its connectivity-normalised variant.
  * Top-N overlap between methods.
  * Sec 1A hub diagnostic: Spearman of each method's score against pathway degree.
    A method whose ranking tracks degree is measuring connectivity, not biology.
  * Seed stability per method: how many genes hold a top-20 place in all 5 seeds,
    and the median best-to-worst rank spread.
  * A per-gene table with every method's rank side by side, so a gene supported by
    all three can be told from one carried by a single method.

The three methods are on different scales and are NOT averaged into a single score;
they are combined by mean RANK, for the same reason seed_consensus.py ranks that way.

Usage:
    python pipeline/04_report/compare_methods.py results/tanh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# label -> (consensus filename, is the score connectivity-normalised?)
METHODS = {
    "2A_weight":       ("gene_importance_consensus.csv", False),
    "2B_shap":         ("shap_gene_importance_consensus.csv", False),
    "2C_ablation":     ("ablation_importance_consensus.csv", False),
    "2A_per_degree":   ("gene_importance_per_degree_consensus.csv", True),
    "2B_per_degree":   ("shap_gene_importance_per_degree_consensus.csv", True),
    "2C_per_degree":   ("ablation_importance_per_degree_consensus.csv", True),
}
PRIMARY = ["2A_weight", "2B_shap", "2C_ablation"]


def load(results_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tables, missing = {}, []
    for label, (name, _) in METHODS.items():
        path = results_dir / name
        if path.exists():
            tables[label] = pd.read_csv(path).set_index("gene")
        else:
            missing.append(f"{label} ({name})")
    return tables, missing


def matrix(frame: pd.DataFrame, title: str, fmt: str = "{:.3f}") -> str:
    lines = [title, "  " + " ".join(f"{c:>14}" for c in frame.columns)]
    for gene, row in frame.iterrows():
        cells = " ".join(f"{fmt.format(v):>14}" if pd.notna(v) else f"{'-':>14}"
                         for v in row)
        lines.append(f"{gene:>14} {cells}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding the *_consensus.csv files, e.g. results/tanh")
    p.add_argument("--top-n", type=int, default=50,
                   help="cutoff for the overlap matrix and the n_methods_top column "
                        "(default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="per-gene CSV (default: <results_dir>/method_comparison.csv)")
    args = p.parse_args()

    tables, missing = load(args.results_dir)
    if missing:
        print(f"note: no consensus file for {', '.join(missing)}", file=sys.stderr)
    have_primary = [m for m in PRIMARY if m in tables]
    if len(have_primary) < 2:
        print(f"need at least 2 of {PRIMARY} in {args.results_dir}", file=sys.stderr)
        return 1

    # inner join: only genes every method scored, so the correlations compare like
    # with like. All methods here derive from the same topology, so this should not
    # drop anything -- it is reported so a silent mismatch cannot pass unnoticed.
    ranks = pd.concat({k: v["mean_rank"] for k, v in tables.items()}, axis=1, join="inner")
    sizes = {k: len(v) for k, v in tables.items()}
    print(f"genes per method: {sizes}")
    print(f"genes common to all: {len(ranks)}\n")

    print(matrix(ranks.corr(method="spearman"),
                 "Spearman between consensus rankings (1.0 = identical order)"))

    top = {label: set(tables[label].nsmallest(args.top_n, "mean_rank").index)
           for label in tables}
    overlap = pd.DataFrame({a: {b: len(top[a] & top[b]) for b in tables} for a in tables})
    print("\n" + matrix(overlap.loc[list(tables), list(tables)],
                        f"Top-{args.top_n} overlap (genes shared)", fmt="{:.0f}"))

    # --- Sec 1A: is the ranking just measuring connectivity? ---
    print("\nHub diagnostic (Sec 1A) - Spearman of consensus score vs pathway degree,")
    print("and seed stability. A high |rho| means the method rewards hub genes.")
    print(f"{'method':>14} {'rho(score,degree)':>18} {'top20 in all seeds':>20} "
          f"{'median spread':>14}")
    for label, table in tables.items():
        rho = (table["mean_importance"].corr(table["degree"], method="spearman")
               if "degree" in table else float("nan"))
        n_all = int((table["n_top20"] == table["n_top20"].max()).sum())
        print(f"{label:>14} {rho:>18.3f} {n_all:>20} {table['spread'].median():>14.0f}")

    # --- per-gene table ---
    out = pd.DataFrame(index=ranks.index)
    for label in tables:
        out[f"rank_{label}"] = tables[label]["rank"].reindex(ranks.index)
    out["mean_rank_primary"] = out[[f"rank_{m}" for m in have_primary]].mean(axis=1)
    out[f"n_methods_top{args.top_n}"] = sum(
        (out[f"rank_{m}"] <= args.top_n).astype(int) for m in have_primary)
    for col in ("degree", "n_gene_nodes"):
        if col in tables[have_primary[0]]:
            out[col] = tables[have_primary[0]][col].reindex(ranks.index)
    out = out.sort_values("mean_rank_primary", kind="stable").reset_index()
    out.insert(1, "rank_combined", out.index + 1)

    out_path = args.out or args.results_dir / "method_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} genes -> {out_path}")

    agreed = out[out[f"n_methods_top{args.top_n}"] == len(have_primary)]
    print(f"\nin the top-{args.top_n} of ALL {len(have_primary)} primary methods: "
          f"{len(agreed)} gene(s)")
    print("  " + ", ".join(agreed.gene) if len(agreed) else "  (none)")
    print(f"\ncombined top-20: {list(out.gene[:20])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
