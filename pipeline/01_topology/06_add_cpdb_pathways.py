"""
Step 2f — Add pathway layer from ConsensusPathDB (CPDB).

Joins the gene layer (layer1) to CPDB pathways (layer2) and removes genes
that have no pathway annotation (layer2_node = -1 after left join).

PREREQUISITES:
  Download CPDB_pathways_genes.tab from ConsensusPathDB:
    http://cpdb.molgen.mpg.de/  →  Downloads → Gene sets (HGNC) → tab-separated
  Save to: processed_data/CPDB_pathways_genes.tab

Input:  processed_data/topology_fixed.csv
        processed_data/CPDB_pathways_genes.tab
Output: processed_data/topology_final.csv   (ready for training)

Stats from the IBD run (for reference):
  Before CPDB join: ~35,000 rows
  After removing unannotated genes: 567,575 rows
  Unique pathways: 4,513
"""

import pandas as pd
import html
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOPO_FILE = os.path.join(BASE_DIR, "processed_data", "topology_fixed.csv")
CPDB_FILE = os.path.join(BASE_DIR, "processed_data", "CPDB_pathways_genes.tab")
OUT_FILE  = os.path.join(BASE_DIR, "processed_data", "topology_final.csv")

print("Loading topology and CPDB...")
topo = pd.read_csv(TOPO_FILE, index_col=0)
cpdb = pd.read_csv(CPDB_FILE, sep="\t")
print(f"  Topology rows: {len(topo):,}")
print(f"  CPDB pathways: {len(cpdb):,}")

# Decode HTML entities in pathway names (e.g. 'na&#xef;ve' → 'naïve')
cpdb = cpdb[["pathway", "hgnc_symbol_ids"]].copy()
cpdb["pathway"] = cpdb["pathway"].apply(html.unescape)

# Explode gene list (comma-separated in CPDB)
cpdb["gene"] = cpdb["hgnc_symbol_ids"].str.split(",")
cpdb = cpdb.explode("gene")
cpdb["gene"] = cpdb["gene"].str.strip()

# Join topology genes to pathways
df = pd.merge(
    topo,
    cpdb[["gene", "pathway"]],
    left_on="layer1_name",
    right_on="gene",
    how="left",
)
df["layer2_name"] = df["pathway"]
df["layer2_node"] = df["layer2_name"].astype("category").cat.codes
df = df.drop(columns=["pathway", "gene"])

unannotated = (df["layer2_node"] == -1).sum()
print(f"\n  Unannotated rows (no CPDB pathway): {unannotated:,} → removed")

df = df[df["layer2_node"] != -1].reset_index(drop=True)

# Final column order
cols = ["chr", "layer0_node", "layer0_name", "layer1_node", "layer1_name", "layer2_node", "layer2_name"]
df = df[cols]

df.to_csv(OUT_FILE, index_label="Unnamed: 0")
print(f"  Final rows: {len(df):,}")
print(f"  Unique genes: {df['layer1_name'].nunique():,}")
print(f"  Unique pathways: {df['layer2_name'].nunique():,}")
print(f"\nSaved: {OUT_FILE}")
print("\nNext: copy topology_final.csv to each processed_data/seed_N/ directory.")
