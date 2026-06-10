"""
Step 2e — Fix ambiguous layer1_node IDs.

After step 04, intergenic SNPs that map to two genes have both genes sharing
the same layer1_node ID. This causes two distinct genes to be the same network
node, making weights uninterpretable.

Fix: for every (chr, layer1_node) that maps to more than one gene name,
genes after the first are assigned a new unique node ID.

Input:  processed_data/topology_with_intergenic.csv
Output: processed_data/topology_fixed.csv
        processed_data/topology_fix_report.csv  (audit log)
"""

import pandas as pd
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "processed_data", "topology_with_intergenic.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "topology_fixed.csv")
REPORT_PATH = os.path.join(BASE_DIR, "processed_data", "topology_fix_report.csv")

print(f"Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, index_col=0)
print(f"  Rows: {len(df):,}  |  Cols: {list(df.columns)}")

def count_ambiguous(df):
    return (
        df.groupby(["chr", "layer1_node"])["layer1_name"]
        .nunique()
        .reset_index()
        .query("layer1_name > 1")
    )

ambiguous_before = count_ambiguous(df)
print(f"\nAmbiguous (chr, layer1_node) pairs before fix: {len(ambiguous_before):,}")

df_fixed = df.copy()
next_node_id = int(df["layer1_node"].max()) + 1
reassignments = []

for (chr_val, node_val), group in df.groupby(["chr", "layer1_node"]):
    unique_genes = group["layer1_name"].unique()
    if len(unique_genes) <= 1:
        continue
    for gene in unique_genes[1:]:
        new_id = next_node_id
        next_node_id += 1
        mask = (
            (df_fixed["chr"] == chr_val) &
            (df_fixed["layer1_node"] == node_val) &
            (df_fixed["layer1_name"] == gene)
        )
        df_fixed.loc[mask, "layer1_node"] = new_id
        reassignments.append({
            "chr": chr_val,
            "layer1_node_old": node_val,
            "layer1_node_new": new_id,
            "layer1_name": gene,
            "n_rows_updated": mask.sum(),
        })

ambiguous_after = count_ambiguous(df_fixed)
print(f"Ambiguous pairs after fix: {len(ambiguous_after):,}  ← must be 0")
assert len(ambiguous_after) == 0, "ERROR: ambiguous pairs still present"
assert set(df["layer1_name"]) == set(df_fixed["layer1_name"]), "ERROR: gene names changed"
assert len(df) == len(df_fixed), f"ERROR: row count changed {len(df)} → {len(df_fixed)}"
print("OK: all checks passed.")

df_fixed.to_csv(OUTPUT_PATH)
report_df = pd.DataFrame(reassignments)
report_df.to_csv(REPORT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")
print(f"Audit: {REPORT_PATH}")
print(f"Genes reassigned: {len(reassignments):,}")
print("\nNext: run 06_add_cpdb_pathways.py")
