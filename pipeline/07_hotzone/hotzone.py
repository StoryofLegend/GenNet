#!/usr/bin/env python
"""
HotZone -- variance-thresholded community detection on LIONESS individual-specific
networks.

This is a port of Gaia's Colab notebook (pipeline/07_hotzone/PipelineHot.ipynb) to
something that runs on the cluster over the full cohort. The method is hers; the
implementation is rewritten. Her project was a different one (X/Y-linked
methylation), so none of her gene names, her 359-gene network or her 50 individuals
carry over -- those were Colab data-size ceilings, not choices.

Method
------
1. Rank every edge by its variance across individuals (already computed by
   lioness.py into isn_edge_stats.csv). A high-variance edge is one that differs
   between patients, which is the whole point of building ISNs.
2. Keep the top --top-frac of edges (0.10, hers).
3. Sweep a threshold over that subset and, at each level, run Leiden on the
   surviving graph.
4. Score each community by mean edge variance and by the Fiedler value of its
   induced subgraph -- how variable it is, and how tightly it holds together.
5. The HotZone is the community structure at the strictest usable threshold.

Two threshold modes
-------------------
    --threshold-mode quantile   quantiles WITHIN the top --top-frac (Gaia's)
    --threshold-mode topk       keep exactly K edges

quantile keeps a fixed FRACTION, so a bigger gene set survives with proportionally
more edges: at Q98 an N-gene set retains ~0.001*N*(N-1) edges -- 6 edges at N=78 but
128 at N=358. Any comparison of two differently sized input sets under `quantile` is
therefore confounded by size. Use `topk` to compare INPUT_0 against INPUT_B, and
`quantile` for continuity with Gaia.

Differences from the notebook, all deliberate
---------------------------------------------
* Reads isn_edge_stats.csv / isn_weights.npy, never the N^2 labeled_lioness_data.csv.
  The ISNs in results/*/isn/ are already symmetrised by averaging both directions and
  reduced to the upper triangle, so self-loops and reversed duplicates are gone before
  this script sees them -- no clean_lioness.py step, and no `nx.Graph` edge collapse
  silently keeping whichever direction was added last.
* Leiden is seeded through find_partition(seed=...). `random.seed()` does not reach
  leidenalg's own RNG, so the notebook's results were not actually reproducible.
* n_iterations=-1: iterate to convergence rather than stopping after one pass.
* Cluster mean variance is reported two ways. The notebook used an OR mask
  (`tf in cluster | gene in cluster`), which counts every edge LEAVING the community
  as well; that is `mean_var_incident` here. `mean_var_internal` uses only edges with
  both endpoints inside, which is what "how hot is this community" should mean.
* The Fiedler value is taken on each community's largest connected component. A
  disconnected subgraph has algebraic connectivity 0 by definition, so computing it on
  the whole subgraph reports 0 for a graph that is really two healthy pieces; n_comp is
  reported alongside so the case is visible.
* Vectorised graph construction instead of iterrows().

Outputs (into --outdir, default <isn_dir>/hotzone[_<tag>])
----------------------------------------------------------
    hotzone_clusters.csv       threshold, cluster, n_genes, n_edges, mean_var_*, fiedler, n_comp
    hotzone_gene_clusters.csv  threshold, gene, cluster        (the Sankey's input)
    hotzone_edges.csv          the surviving edges at --extract, with cluster labels
    hotzone_genes.csv          the HotZone gene list at --extract
    hotzone_sankey.html        cluster evolution across thresholds   [--sankey]
    hotzone_run_info.json      settings, sizes, modularity per threshold

Usage
-----
    python pipeline/07_hotzone/hotzone.py results/tanh/isn/INPUT_0_top250_seed42
    python pipeline/07_hotzone/hotzone.py results/tanh/isn/INPUT_0_top250_seed42 \
        --genes results/tanh/isn_gene_sets/live_genes_sd0.0001.csv \
        --threshold-mode topk --topk 500 300 200 128 --sankey
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import laplacian

DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.97, 0.98)


def load_edges(isn_dir: Path, genes: set[str] | None) -> pd.DataFrame:
    f = isn_dir / "isn_edge_stats.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} - run pipeline/06_isn/run_lioness.sh for this config")
    df = pd.read_csv(f)
    need = {"source", "target", "var"}
    if not need.issubset(df.columns):
        raise ValueError(f"{f} lacks {need - set(df.columns)}")
    if (df.source == df.target).any():
        raise ValueError(f"{f} contains self-loops; expected an upper-triangle edge list")
    if genes is not None:
        before = len(df)
        df = df[df.source.isin(genes) & df.target.isin(genes)].reset_index(drop=True)
        print(f"gene filter  : {before} -> {len(df)} edges "
              f"({df[['source', 'target']].stack().nunique()} genes)")
    return df


def build_graph(sub: pd.DataFrame) -> ig.Graph:
    """igraph from an edge table, weights = edge variance. No iterrows()."""
    names = pd.unique(pd.concat([sub.source, sub.target], ignore_index=True))
    idx = pd.Series(np.arange(len(names)), index=names)
    g = ig.Graph(n=len(names),
                 edges=list(zip(idx[sub.source].to_numpy(), idx[sub.target].to_numpy())))
    g.vs["name"] = list(names)
    g.es["weight"] = sub["var"].to_numpy(dtype=float)
    return g


def fiedler(sub_g: ig.Graph) -> tuple[float, int]:
    """Algebraic connectivity of the largest component, and the component count.

    Normalised Laplacian, so the value is comparable across communities of
    different size and edge weight.
    """
    comps = sub_g.connected_components()
    n_comp = len(comps)
    if sub_g.vcount() < 2:
        return 0.0, n_comp
    lcc = sub_g.subgraph(max(comps, key=len))
    if lcc.vcount() < 2:
        return 0.0, n_comp
    A = np.array(lcc.get_adjacency(attribute="weight").data, dtype=float)
    A = np.abs(A)                      # variance is non-negative, but be explicit
    np.fill_diagonal(A, 0.0)
    ev = np.sort(eigvalsh(laplacian(A, normed=True)))
    return float(ev[1]), n_comp


def cluster_stats(g: ig.Graph, part, sub: pd.DataFrame, label: str) -> list[dict]:
    name = np.array(g.vs["name"])
    rows = []
    for cid, members in enumerate(part):
        cluster_genes = set(name[members])
        inside = sub.source.isin(cluster_genes) & sub.target.isin(cluster_genes)
        incident = sub.source.isin(cluster_genes) | sub.target.isin(cluster_genes)
        sub_g = g.subgraph(members)
        fied, n_comp = fiedler(sub_g)
        rows.append({
            "threshold": label,
            "cluster": cid,
            "n_genes": len(members),
            "n_edges": int(inside.sum()),
            "mean_var_internal": float(sub.loc[inside, "var"].mean()) if inside.any() else 0.0,
            "mean_var_incident": float(sub.loc[incident, "var"].mean()) if incident.any() else 0.0,
            "fiedler": fied,
            "n_comp": n_comp,
        })
    return rows


def sankey(gene_cluster: dict[str, dict[str, int]], labels: list[str], dest: Path) -> None:
    import plotly.graph_objects as go

    # plotly's Sankey rejects 8-digit hex, so links need a real rgba() string
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
               (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
               (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
               (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
               (0, 0, 128), (128, 128, 0), (255, 215, 180), (128, 64, 0),
               (255, 100, 255)]

    def rgb(cid):
        return "rgb(%d, %d, %d)" % palette[cid % len(palette)]

    def rgba(cid, a=0.4):
        return "rgba(%d, %d, %d, %s)" % (*palette[cid % len(palette)], a)
    node_labels, node_index, node_colors = [], {}, []
    for lab in labels:
        for cid in sorted(set(gene_cluster[lab].values())):
            node_index[(lab, cid)] = len(node_labels)
            node_labels.append(f"{lab}_C{cid}")
            node_colors.append(rgb(cid))

    src, tgt, val, col = [], [], [], []
    for a, b in zip(labels, labels[1:]):
        m1, m2 = gene_cluster[a], gene_cluster[b]
        flow: dict[tuple[int, int], int] = {}
        for gene in m1.keys() & m2.keys():
            flow[(m1[gene], m2[gene])] = flow.get((m1[gene], m2[gene]), 0) + 1
        for (c1, c2), n in flow.items():
            src.append(node_index[(a, c1)])
            tgt.append(node_index[(b, c2)])
            val.append(n)
            col.append(rgba(c1))

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=node_labels, color=node_colors,
                  line=dict(color="black", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=col)))
    fig.update_layout(title_text="HotZone cluster evolution across thresholds",
                      font_size=12, width=1200, height=800, title_x=0.5)
    fig.write_html(str(dest))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("isn_dir", type=Path, help="a results/*/isn/<method>_top<N>_seed<S>/ folder")
    p.add_argument("--genes", type=Path, default=None,
                   help="CSV with a 'gene' column (e.g. live_genes.py output); "
                        "edges are kept only when BOTH endpoints are in it")
    p.add_argument("--top-frac", type=float, default=0.10,
                   help="fraction of highest-variance edges entering the sweep (default 0.10)")
    p.add_argument("--threshold-mode", choices=("quantile", "topk"), default="quantile")
    p.add_argument("--quantiles", type=float, nargs="+", default=list(DEFAULT_QUANTILES))
    p.add_argument("--topk", type=int, nargs="+", default=[1000, 500, 300, 200, 128])
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Leiden RBConfiguration resolution parameter (default 1.0)")
    p.add_argument("--seed", type=int, default=42, help="Leiden RNG seed")
    p.add_argument("--extract", default=None,
                   help="threshold label whose graph is written out (default: the last)")
    p.add_argument("--sankey", action="store_true", help="also write hotzone_sankey.html")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    if not 0 < args.top_frac <= 1:
        print("--top-frac must be in (0, 1]", file=sys.stderr)
        return 1

    genes = None
    if args.genes:
        genes = set(pd.read_csv(args.genes).gene)
        print(f"gene list    : {len(genes)} from {args.genes}")

    df = load_edges(args.isn_dir, genes)
    if df.empty:
        print("no edges survive the gene filter", file=sys.stderr)
        return 1

    ranked = df.sort_values("var", ascending=False, kind="stable").reset_index(drop=True)
    n_top = max(1, int(len(ranked) * args.top_frac))
    top = ranked.iloc[:n_top].reset_index(drop=True)
    print(f"edges        : {len(df)} -> top {args.top_frac:.0%} = {len(top)}")

    if args.threshold_mode == "quantile":
        levels = [(f"Q{q * 100:g}", float(top["var"].quantile(q))) for q in sorted(args.quantiles)]
    else:
        levels = [(f"K{k}", k) for k in sorted(args.topk, reverse=True)]

    gene_cluster: dict[str, dict[str, int]] = {}
    rows: list[dict] = []
    modularity: dict[str, float] = {}
    kept: dict[str, pd.DataFrame] = {}

    for label, cut in levels:
        sub = top[top["var"] >= cut] if args.threshold_mode == "quantile" else top.iloc[:int(cut)]
        sub = sub.reset_index(drop=True)
        if len(sub) < 2:
            print(f"{label:>8}: {len(sub)} edges - too few to cluster, skipped")
            continue
        g = build_graph(sub)
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=args.resolution, n_iterations=-1, seed=args.seed)
        gene_cluster[label] = {n: c for n, c in zip(g.vs["name"], part.membership)}
        rows += cluster_stats(g, part, sub, label)
        modularity[label] = float(part.modularity)
        kept[label] = sub
        print(f"{label:>8}: {len(sub):>6} edges  {g.vcount():>4} genes  "
              f"{len(part):>3} clusters  Q={part.modularity:.3f}")

    if not gene_cluster:
        print("nothing survived any threshold", file=sys.stderr)
        return 1

    labels = list(gene_cluster)
    extract = args.extract or labels[-1]
    if extract not in gene_cluster:
        print(f"--extract {extract!r} not among {labels}", file=sys.stderr)
        return 1

    outdir = args.outdir or args.isn_dir / (f"hotzone_{args.tag}" if args.tag else "hotzone")
    outdir.mkdir(parents=True, exist_ok=True)

    clusters = pd.DataFrame(rows).sort_values(
        ["threshold", "mean_var_internal"], ascending=[True, False], kind="stable")
    clusters.to_csv(outdir / "hotzone_clusters.csv", index=False)

    pd.DataFrame([{"threshold": t, "gene": gn, "cluster": c}
                  for t, m in gene_cluster.items() for gn, c in m.items()]
                 ).to_csv(outdir / "hotzone_gene_clusters.csv", index=False)

    cmap = gene_cluster[extract]
    e = kept[extract].copy()
    e["cluster_source"] = e.source.map(cmap)
    e["cluster_target"] = e.target.map(cmap)
    e["internal"] = e.cluster_source == e.cluster_target
    e.to_csv(outdir / "hotzone_edges.csv", index=False)
    (pd.Series(sorted(cmap), name="gene").to_frame()
       .assign(cluster=lambda d: d.gene.map(cmap))
       .to_csv(outdir / "hotzone_genes.csv", index=False))

    if args.sankey:
        sankey(gene_cluster, labels, outdir / "hotzone_sankey.html")

    info = {
        "isn_dir": str(args.isn_dir),
        "gene_list": str(args.genes) if args.genes else None,
        "n_genes_in_list": len(genes) if genes else None,
        "n_edges_total": int(len(df)),
        "n_edges_top": int(len(top)),
        "top_frac": args.top_frac,
        "threshold_mode": args.threshold_mode,
        "levels": {l: c for l, c in levels},
        "resolution": args.resolution,
        "seed": args.seed,
        "modularity": modularity,
        "extract": extract,
        "n_hotzone_genes": len(cmap),
        "n_hotzone_clusters": len(set(cmap.values())),
    }
    (outdir / "hotzone_run_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nHotZone at {extract}: {len(cmap)} genes in {len(set(cmap.values()))} clusters")
    print(clusters[clusters.threshold == extract].to_string(index=False))
    print(f"\nwrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
