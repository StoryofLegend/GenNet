#!/usr/bin/env python3
"""LIONESS individual-specific networks from a GenNet activation matrix (step 4).

Reads one patients x genes matrix from make_isn_input.py, builds a PANDA network
from the gene-gene correlation structure, and runs LIONESS to get one network per
individual. Same method as the original Colab-derived script; this version fixes
the correctness and scale problems that made it unusable for a 4-method x 5-seed
grid.

What changed, and why
---------------------
1. SYMMETRISATION, not arbitrary dropping. PANDA's message passing is directional
   (TF -> gene), so with tf_names == gene_names every pair appears twice with
   DIFFERENT values: measured on top50_seed42, W(a,b) vs W(b,a) has Pearson
   r = 0.57, 19.3% of pairs differ in SIGN, and 89.7% differ by >10%. The old
   clean_lioness.py called these "bidirectional duplicates" and kept whichever had
   tf < gene lexicographically -- discarding 1225 of 2450 real values by alphabetical
   accident (for CD40/IL23R it kept -0.044 and threw away +1.146). Here the two
   orientations are AVERAGED and the upper triangle kept, which is a stated
   assumption rather than a silent one. The raw asymmetry is measured and reported
   so the assumption stays auditable.

2. SELF-LOOPS DROPPED. They are meaningless as network edges and dominate by
   magnitude (mean |w| 2.12 vs 1.78 off-diagonal).

3. BINARY OUTPUT, WRITTEN ONCE. The original wrote every (tf, gene) ordered pair
   plus self-loops as CSV text: 1.9 GB at top100, 12 GB at top250, then
   clean_lioness.py read the whole thing back and wrote a second copy. Here the
   symmetrisation happens in memory and only the upper triangle is saved, as
   float32 .npy -- 197 MB at top100, ~10x smaller, with no second pass.

4. PATIENT IDS KEPT. The original renamed the sample columns to Sample_1..N,
   destroying the link to labels and covariates and making the Sec 3.4 case/control
   comparison impossible. netZooPy already carries the real ids through
   `expression_samples`; they are preserved here.

5. NO pip install AT IMPORT TIME. The original ran `pip install` for netZooPy and
   friends on import, which must not happen on a compute node. netZooPy 0.11.0 is
   in env_GenNet already.

6. PER-EDGE STATISTICS ALWAYS WRITTEN. HotZone detection only needs the variance
   of each edge across individuals -- hotzone_score.py drops the sample columns
   immediately. That summary is a few hundred KB, so it is always emitted, and
   --no-full lets you skip the big per-patient matrix when only HotZones are wanted.

A caveat to carry into the interpretation
-----------------------------------------
The PANDA prior here is the SIGN of the gene-gene correlation of GenNet
activations. Those correlations are at noise level in this model: over the genes
active in all 5 seeds the median |r| is 0.008 against a sampling SE of 0.010 at
n=9942, and the only pair above |r|=0.1 is HLA-DQA1/HLA-DQB1 (r=0.132), which is
linkage disequilibrium between adjacent MHC class II genes. This is architectural
-- GenNet has no gene-gene edges, so a gene's activation depends only on its own
SNPs and two genes can correlate only through LD. Treat structure found downstream
accordingly, and compare against a permuted-input null before claiming a HotZone
is real.

Outputs (into --outdir)
-----------------------
isn_edges.csv       source, target for each kept edge (upper triangle, no self-loops)
isn_weights.npy     (n_edges, n_patients) float32, symmetrised   [unless --no-full]
isn_patients.csv    patient_id in the column order of the above
isn_edge_stats.csv  source, target, mean, sd, var across individuals -- HotZone input
isn_run_info.json   shapes, timings, asymmetry diagnostics, settings, and which
                    reference population the ISNs came from (ISN_01,i or ISN_1,i)

Usage
-----
    python pipeline/06_isn/lioness.py \
        --input results/tanh/isn_input/isn_input_2C_top100_seed42.csv \
        --outdir results/tanh/isn/2C_top100_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import contextlib
import os
import shutil

import numpy as np
import pandas as pd


@contextlib.contextmanager
def quiet_stdout(enabled=True):
    """Swallow netZooPy's own prints.

    Panda and Lioness print a progress line per solver step per patient: 417,608
    lines / 17 MB for one 9,942-patient run, against 88 lines of ours. A 40-task
    array writing that much to the shared log directory is what exhausted the disk
    quota and killed the jobs mid-write. Redirect at the file-descriptor level, not
    by rebinding sys.stdout, so the LIONESS worker processes are covered too.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    with open(os.devnull, "w") as devnull:
        saved = os.dup(1)          # after the open, so a failed open cannot leak it
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="patients x genes CSV from make_isn_input.py (index = patient_id)")
    p.add_argument("--outdir", required=True, type=Path,
                   help="output folder (one per method/cutoff/seed combination)")
    p.add_argument("--reference", choices=("01", "1"), default=None,
                   help="metadata only: which reference population the input matrix "
                        "holds (01 = all subjects -> ISN_01,i, 1 = cases only -> "
                        "ISN_1,i). Inferred from the input filename's _ref suffix if "
                        "omitted. It does NOT subset anything - the reference is "
                        "whatever make_isn_input.py wrote into the file.")
    p.add_argument("--ncores", type=int, default=1, help="LIONESS worker processes")
    p.add_argument("--alpha", type=float, default=0.1, help="PANDA learning rate")
    p.add_argument("--no-full", action="store_true",
                   help="skip isn_weights.npy; write only the per-edge statistics")
    p.add_argument("--max-patients", type=int, default=0,
                   help="use only the first N patients (0 = all); for smoke tests")
    p.add_argument("--verbose-netzoo", action="store_true",
                   help="let netZooPy print its per-step progress (17 MB per run)")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep lioness_raw/, netZooPy's full non-symmetrised N x N x "
                        "patients array (192 MB at top50, 5 GB at top250). Everything "
                        "downstream reads isn_weights.npy and it is reproducible by "
                        "rerunning, so it is deleted by default.")
    args = p.parse_args()

    from netZooPy.panda import Panda
    from netZooPy.lioness import Lioness

    args.outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- input ---------------------------------------------------------------
    if args.reference is None:
        args.reference = "1" if args.input.stem.endswith("_ref1") else "01"
    exp = pd.read_csv(args.input, index_col=0)
    if args.max_patients:
        exp = exp.iloc[:args.max_patients]
    if exp.isna().any().any():
        raise ValueError(f"{args.input} contains NaNs")
    genes = list(exp.columns)
    print(f"input {args.input.name}: {exp.shape[0]} patients x {len(genes)} genes")

    # Genes with no variance have an undefined correlation; PANDA would propagate
    # NaN through the whole network, so refuse rather than emit a silently broken ISN.
    # exp.std() is ddof=1, so it is NaN (not 0) for a single patient; test for
    # "not usefully variable" rather than exact zero, which also catches sd ~ 1e-15.
    sd = exp.std()
    flat = [g for g, v in sd.items() if not np.isfinite(v) or v < 1e-12]
    if flat:
        hint = ("this is an artefact of --max-patients, not of the data; rerun "
                "without it" if args.max_patients else
                "regenerate the matrix - make_isn_input.py drops constant genes")
        raise ValueError(f"{len(flat)} gene(s) are constant across these "
                         f"{len(exp)} patients (e.g. {flat[:5]}) - {hint}")

    # --- PANDA prior ---------------------------------------------------------
    # Same construction as the original script: the motif is every ordered gene
    # pair, weighted 1 where the correlation is positive. See the caveat above.
    corr = exp.corr()
    pairs = pd.MultiIndex.from_product([genes, genes], names=["source", "target"])
    motif = pd.DataFrame(index=pairs)
    motif["weight"] = (corr.values.flatten() > 0).astype(int)
    motif.reset_index(inplace=True)

    off = ~np.eye(len(genes), dtype=bool)
    print(f"correlation prior: median |r| = {np.median(np.abs(corr.values[off])):.4f}, "
          f"{100*(np.abs(corr.values[off]) > 0.1).mean():.2f}% of pairs |r| > 0.1")

    expression = exp.T          # PANDA wants genes x samples
    t0 = time.time()
    with quiet_stdout(not args.verbose_netzoo):
        # alpha must be passed here too. Panda defaults to alpha=0.1 independently of
        # Lioness, so passing --alpha only to Lioness would make it subtract a PANDA
        # model fitted at a different learning rate from the one it built - netZooPy's
        # own docstring warns the two have to be kept in step by hand.
        panda_obj = Panda(expression, motif, None, computing="cpu", remove_missing=False,
                          keep_expression_matrix=True, save_memory=False,
                          modeProcess="legacy", alpha=args.alpha)
    t_panda = time.time() - t0
    print(f"PANDA done in {t_panda:.1f}s")

    # --- LIONESS -------------------------------------------------------------
    # netZooPy gates its whole compute loop on n_conditions >= n_cores and otherwise
    # never assigns total_lioness_network, failing later with a bare AttributeError.
    ncores = max(1, min(args.ncores, len(exp)))
    if ncores != args.ncores:
        print(f"ncores {args.ncores} > {len(exp)} patients; using {ncores}")

    t0 = time.time()
    raw_dir = args.outdir / "lioness_raw"

    def drop_raw():
        """Remove netZooPy's raw dump. Called from a finally: the steps after LIONESS
        are the peak-memory point of the run, and a crash there used to strand up to
        5 GB per job - which is what exhausted the quota across the grid."""
        if args.keep_raw or not raw_dir.exists():
            return
        freed = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"removed {raw_dir.name}/ ({freed / 1e6:.0f} MB); pass --keep-raw to retain")

    with quiet_stdout(not args.verbose_netzoo):
        lio = Lioness(panda_obj, computing="cpu", ncores=ncores, alpha=args.alpha,
                      save_dir=str(raw_dir), save_fmt="npy",
                      output="network", ignore_final=False)
    t_lioness = time.time() - t0
    print(f"LIONESS done in {t_lioness:.1f}s")
    try:
        return assemble_and_write(args, lio, exp, ncores, raw_dir, corr, off,
                                  t_panda, t_lioness, t_start)
    finally:
        drop_raw()


def assemble_and_write(args, lio, exp, ncores, raw_dir, corr, off,
                       t_panda, t_lioness, t_start):
    """Symmetrise netZooPy's directed output and write the ISN files."""

    # netZooPy builds export_lioness_results by concatenating two rows of strings onto
    # the weights, which upcasts the entire edges-by-patients block to object dtype -
    # several times the float payload. Nothing here reads it; drop it before we make
    # our own float32 copies, so the two peaks do not overlap.
    if hasattr(lio, "export_lioness_results"):
        del lio.export_lioness_results

    # --- assemble ------------------------------------------------------------
    # total_lioness_network is (n_samples, n_edges); edges run with the TF index
    # fastest, i.e. edge e = (tf_names[e % n_tf], gene_names[e // n_tf]).
    W = np.asarray(lio.total_lioness_network, dtype=np.float32)
    tf_names, gene_names = list(lio.tf_names), list(lio.gene_names)
    n_tf, n_gene = len(tf_names), len(gene_names)
    if W.ndim != 2 or W.shape[1] != n_tf * n_gene:
        raise ValueError(f"unexpected LIONESS shape {W.shape} for "
                         f"{n_tf}x{n_gene} edges")
    if set(tf_names) != set(gene_names):
        raise ValueError("tf and gene name sets differ; the symmetrisation below "
                         "assumes a square gene-by-gene network")
    n_samples = W.shape[0]
    M = W.reshape(n_samples, n_gene, n_tf)        # M[k, gene_j, tf_i]

    # netZooPy sorts the TF axis (unique_tfs = sorted(...)) but leaves the gene axis
    # in expression order, so the two axes of M are the same genes in DIFFERENT
    # orders. Permute the TF axis onto the gene order before averaging, otherwise
    # M.transpose() would pair up unrelated genes and silently corrupt every edge.
    if tf_names != gene_names:
        pos = {name: i for i, name in enumerate(tf_names)}
        perm = [pos[name] for name in gene_names]
        M = M[:, :, perm]

    # asymmetry diagnostic, on the first patient, before we average it away
    a = M[0]
    offd = ~np.eye(n_gene, dtype=bool)
    x, y = a[offd], a.T[offd]
    denom = np.maximum(np.abs(x), np.abs(y))
    # json.dumps writes a bare NaN token, which Python reads back but jq and
    # jsonlite reject - and this file is the run's only machine-readable record.
    def _finite(v):
        v = float(v)
        return v if np.isfinite(v) else None

    asym = {
        "pearson_r_forward_vs_reverse": _finite(np.corrcoef(x, y)[0, 1]),
        "frac_opposite_sign": _finite(np.mean(np.sign(x) != np.sign(y))),
        "frac_rel_diff_gt_10pct": _finite(np.mean(
            np.divide(np.abs(x - y), denom, out=np.zeros_like(x), where=denom > 0) > 0.10)),
    }
    print(f"raw asymmetry (patient 0): r={asym['pearson_r_forward_vs_reverse'] or float('nan'):.3f}, "
          f"{100*asym['frac_opposite_sign']:.1f}% opposite sign, "
          f"{100*asym['frac_rel_diff_gt_10pct']:.1f}% differ >10% -> averaging")

    # symmetrise and keep the upper triangle (this also removes self-loops)
    M = 0.5 * (M + M.transpose(0, 2, 1))
    iu = np.triu_indices(n_gene, k=1)
    edge_weights = np.ascontiguousarray(M[:, iu[0], iu[1]].T)   # (n_edges, n_samples)
    del M, W

    edges = pd.DataFrame({"source": [gene_names[i] for i in iu[0]],
                          "target": [gene_names[j] for j in iu[1]]})
    samples = getattr(lio, "expression_samples", None)
    idx = getattr(lio, "indexes", None)
    try:
        ids = list(np.asarray(samples)[idx]) if idx is not None else list(samples)
    except (TypeError, IndexError, KeyError) as exc:
        # Falling back to input order is only correct if netZooPy did not reorder or
        # subset the samples. The length check below cannot detect a reordering, so
        # make the substitution loud rather than silent.
        print(f"[warn] could not read sample labels from netZooPy ({exc!r}); falling "
              "back to input order - verify isn_patients.csv before trusting it")
        ids = list(exp.index)
    patients = pd.DataFrame({"patient_id": ids})
    if len(patients) != edge_weights.shape[1]:
        raise ValueError(f"{len(patients)} sample labels vs "
                         f"{edge_weights.shape[1]} weight columns")

    # --- write ---------------------------------------------------------------
    edges.to_csv(args.outdir / "isn_edges.csv", index=False)
    patients.to_csv(args.outdir / "isn_patients.csv", index=False)

    stats = edges.copy()
    stats["mean"] = edge_weights.mean(axis=1)
    stats["sd"] = edge_weights.std(axis=1)
    stats["var"] = stats["sd"] ** 2
    stats.to_csv(args.outdir / "isn_edge_stats.csv", index=False)

    if not args.no_full:
        tmp = args.outdir / "isn_weights.tmp.npy"
        np.save(tmp, edge_weights)
        tmp.replace(args.outdir / "isn_weights.npy")

    info = {"input": str(args.input), "reference": args.reference,
            "isn_label": f"ISN_{args.reference},i",
            "n_patients": int(edge_weights.shape[1]),
            "n_genes": n_gene, "n_edges": int(edge_weights.shape[0]),
            "seconds_panda": round(t_panda, 1), "seconds_lioness": round(t_lioness, 1),
            "seconds_total": round(time.time() - t_start, 1),
            "alpha": args.alpha, "ncores": ncores,
            "wrote_full_matrix": not args.no_full,
            "raw_asymmetry": asym,
            "prior_median_abs_corr": _finite(np.median(np.abs(corr.values[off])))}
    (args.outdir / "isn_run_info.json").write_text(json.dumps(info, indent=2))

    size = 0 if args.no_full else edge_weights.nbytes / 1e6
    print(f"wrote {edge_weights.shape[0]} edges x {edge_weights.shape[1]} patients "
          f"({size:.0f} MB) -> {args.outdir}")
    print(f"total {info['seconds_total']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
