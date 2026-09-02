"""
Clean LIONESS labeled_lioness_data.csv files by:
  1. Removing self-loops (tf == gene)
  2. Removing bidirectional duplicates: for each row, check if its reverse
     (gene, tf) actually exists in the data. If it does, drop the row where
     tf > gene. If the reverse does not exist, keep the row regardless.

Output: labeled_lioness_data_clean.csv alongside the original in each directory.

Usage:
  python clean_lioness.py                          # all dirs under results/lioness/
  python clean_lioness.py results/lioness/top50_seed42
  python clean_lioness.py results/lioness/top50_seed42 results/lioness/top100_seed42
"""

import sys
import os
import pandas as pd


def clean_lioness(input_csv: str) -> pd.DataFrame:
    # Pass 1: read only tf/gene to determine which rows to keep (memory-efficient).
    pairs = pd.read_csv(input_csv, usecols=["tf", "gene"])
    n_before = len(pairs)

    # Remove self-loops
    pairs = pairs[pairs["tf"] != pairs["gene"]]
    n_after_selfloops = len(pairs)

    # Remove bidirectional duplicates: drop a row only if its reverse exists AND tf > gene.
    edge_keys = set(pairs["tf"] + "||" + pairs["gene"])
    has_reverse = (pairs["gene"] + "||" + pairs["tf"]).isin(edge_keys)
    is_duplicate = has_reverse & (pairs["tf"] > pairs["gene"])
    keep_idx = pairs.index[~is_duplicate]
    n_after_bidir = len(keep_idx)

    print(
        f"  {os.path.basename(os.path.dirname(input_csv))}: "
        f"{n_before} -> -{n_before - n_after_selfloops} self-loops "
        f"-> -{n_after_selfloops - n_after_bidir} bidir dups "
        f"= {n_after_bidir} edges"
    )

    # Pass 2: read full CSV, keep only the rows identified above (skiprows discards the rest).
    all_rows = set(keep_idx + 1)  # +1 because row 0 in file is the header
    n_total = n_before + 1        # header + data rows
    skip = [i for i in range(1, n_total + 1) if i not in all_rows]
    df = pd.read_csv(input_csv, skiprows=skip)
    return df


def process_dir(directory: str):
    input_csv = os.path.join(directory, "labeled_lioness_data.csv")
    output_csv = os.path.join(directory, "labeled_lioness_data_clean.csv")
    if not os.path.exists(input_csv):
        print(f"  Skipping {directory} — labeled_lioness_data.csv not found")
        return
    df_clean = clean_lioness(input_csv)
    df_clean.to_csv(output_csv, index=False)
    print(f"  Saved -> {output_csv}")


def main():
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        base = os.path.join(os.path.dirname(__file__), "results", "lioness")
        if not os.path.isdir(base):
            print(f"Default directory not found: {base}")
            sys.exit(1)
        dirs = sorted(
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )

    print(f"Processing {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}...")
    for d in dirs:
        process_dir(d)
    print("Done.")


if __name__ == "__main__":
    main()
