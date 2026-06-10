"""
Step 2d — Fix intergenic multi-gene annotations.

GenNet's create_gene_network is supposed to keep only the first nearest gene for
intergenic SNPs, but due to a regex behaviour change in newer pandas versions the
full ANNOVAR annotation is preserved in layer1_name (e.g.
    'TTLL10(dist=1927),TNFRSF18(dist=3646)').

This script:
  1. Splits layer1_name on commas → one row per gene
  2. Strips any remaining (dist=...) or similar annotations from gene names
  3. Saves the expanded topology

After this step, the two genes of an intergenic SNP share the same layer1_node ID.
Run 05_fix_topo_id.py next to assign unique IDs.

Input:  processed_data/topology.csv
Output: processed_data/topology_with_intergenic.csv
"""

import pandas as pd
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "processed_data", "topology.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "topology_with_intergenic.csv")

print(f"Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, index_col=0)
print(f"  Rows before: {len(df):,}")

intergenic_mask = df["layer1_name"].str.contains(",", na=False)
print(f"  Rows with multi-gene annotation: {intergenic_mask.sum():,}")

def clean_gene(name):
    """Strip (dist=...) and similar annotations, return gene symbol."""
    return re.sub(r"\(.*?\)", "", str(name)).strip()

df["layer1_name"] = df["layer1_name"].apply(
    lambda x: [clean_gene(g) for g in str(x).split(",")] if "," in str(x) else clean_gene(x)
)
df = df.explode("layer1_name")
df["layer1_name"] = df["layer1_name"].str.strip()
df = df[df["layer1_name"].notna() & (df["layer1_name"] != "")].reset_index(drop=True)

print(f"  Rows after:  {len(df):,}")

df.to_csv(OUTPUT_PATH, index_label="Unnamed: 0")
print(f"Saved: {OUTPUT_PATH}")
print("\nNext: run 05_fix_topo_id.py")
