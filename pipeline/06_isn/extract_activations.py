#!/usr/bin/env python3
"""Extract per-patient node activations for ISN construction (step 2).

The ISN formula in the guidelines is

    ISN_k(i, j) = reference_weight(i, j) * (Node_k(i) * Node_k(j))

so every downstream step needs Node_k -- the activation of each node for each
individual. This runs one forward pass per seed over the WHOLE test set (cases and
controls) and saves the gene-layer and pathway-layer activations, aligned to real
patient_ids.

Why the whole test set, not just cases
--------------------------------------
ablation_per_patient.csv from method 2C covers cases only, and its columns are
positional because get_data() was called with num_sample_pat > 0, which permutes
via .sample(random_state=1). Neither is usable here: the Sec 3.4 validation
compares ISNs between high- and low-risk individuals, so controls are required,
and the patient axis has to carry real ids. Calling get_data(sample_pat=0) skips
the permutation, so rows follow subjects.csv order -- and the script asserts that
rather than assuming it (see --no-verify).

Which layer to take
-------------------
Default is the post-BatchNorm tensor, the value that actually propagates into the
next LocallyDirected layer, and the same point method 2C ablates at. --layer
activation takes the pre-BatchNorm tanh output instead.

For the correlation/LIONESS route the choice is irrelevant: these BatchNorm layers
have center=False, scale=False, so at inference each node is transformed by
x -> (x - moving_mean)/sqrt(moving_var + eps), a POSITIVE affine map per node, and
Pearson correlation across individuals is invariant to it. It matters only for the
multiplicative GenNet-native ISN, where the scale of Node_k is real.

Outputs (per experiment folder)
-------------------------------
isn_gene_act.npy       (n_patients, n_slots)    float32, gene-layer activations
isn_gene_slots.csv     slot -> layer1_node, gene name (column order of the above)
isn_pathway_act.npy    (n_patients, n_pathways) float32, pathway-layer activations
isn_pathway_nodes.csv  slot -> layer2_node, pathway name
isn_subjects.csv       patient_id, labels, cov_* in row order of both arrays

By default only the gene slots belonging to the ISN gene sets are kept (the union
over every gene_set_*.csv), which is ~2k of 44,090 slots -- tens of MB instead of
1.75 GB per seed. Pass --all-genes to keep the full layer. The pathway layer is
small (4,510) and is always kept whole.

Usage
-----
    python pipeline/06_isn/extract_activations.py results/tanh
    python pipeline/06_isn/extract_activations.py results/tanh --all-genes
    python pipeline/06_isn/extract_activations.py results/tanh --layer activation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

# Offset from each LocallyDirected1D layer to the tensor we want. Layer NAMES cannot
# be used: Keras uniquifies them per process, so the second model built in one run gets
# 'batch_normalization_2' instead of 'batch_normalization'. The block structure
# (LocallyDirected1D -> Activation -> BatchNormalization) is stable, so resolve by it.
LAYER_OFFSET = {"activation": 1, "batchnorm": 2}
EXPECTED_CLASS = {"activation": "Activation", "batchnorm": "BatchNormalization"}


def resolve_layers(model, which):
    """Return (gene_layer, pathway_layer) for the two LocallyDirected blocks."""
    ld = [i for i, l in enumerate(model.layers)
          if l.__class__.__name__ == "LocallyDirected1D"]
    if len(ld) < 2:
        raise ValueError(f"expected 2 LocallyDirected1D layers, found {len(ld)}")
    offset = LAYER_OFFSET[which]
    gene, pathway = model.layers[ld[0] + offset], model.layers[ld[1] + offset]
    for layer in (gene, pathway):
        if layer.__class__.__name__ != EXPECTED_CLASS[which]:
            raise ValueError(f"expected {EXPECTED_CLASS[which]} at offset {offset} after "
                             f"LocallyDirected1D, found {layer.__class__.__name__} "
                             f"({layer.name}) - the network structure has changed")
    return gene, pathway


def topology_for(exp: Path, root: Path) -> Path:
    """Resolve the topology the model was built from, via train_args.json."""
    args_path = exp / "train_args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"{args_path} missing - cannot resolve the topology")
    datapath = json.loads(args_path.read_text()).get("datapath")
    if not datapath:
        raise ValueError(f"{args_path}: no 'datapath' key")
    topo = root / datapath / "topology.csv"
    if not topo.exists():
        raise FileNotFoundError(f"{topo} missing (datapath={datapath!r})")
    return topo


def wanted_genes(gene_sets_dir: Path) -> set[str] | None:
    """Union of every gene_set_*.csv in the folder, or None if there are none."""
    files = sorted(gene_sets_dir.glob("gene_set_*_top*.csv"))
    if not files:
        return None
    genes: set[str] = set()
    for f in files:
        genes |= set(pd.read_csv(f, usecols=["gene"]).gene)
    print(f"  gene sets: {len(files)} files -> {len(genes)} distinct genes")
    return genes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="folder holding GenNet_experiment_<ID>_/ dirs, e.g. results/tanh")
    p.add_argument("--gene-sets", type=Path, default=None,
                   help="folder of gene_set_*.csv (default: <results_dir>/isn_gene_sets)")
    p.add_argument("--all-genes", action="store_true",
                   help="keep every gene slot (1.75 GB/seed) instead of the gene-set union")
    p.add_argument("--layer", choices=sorted(LAYER_OFFSET), default="batchnorm",
                   help="which tensor to save (default: %(default)s; see docstring)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="forward-pass batch size (default: %(default)s)")
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="repo root that datapath is relative to (default: cwd)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if outputs already exist")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the patient-alignment and BatchNorm sanity checks")
    args = p.parse_args()

    # Imported here so --help works without paying TensorFlow's import cost.
    # The repo root is not on sys.path when this runs from pipeline/06_isn/.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import tensorflow as tf
    from GenNet_utils.Train_network import load_trained_network
    from GenNet_utils.Create_network import remove_cov
    from GenNet_utils.Dataloader import EvalGenerator

    exps = sorted(d for d in args.results_dir.glob("GenNet_experiment_*_")
                  if (d / "bestweights_job.h5").exists())
    if not exps:
        print(f"no trained experiment in {args.results_dir}", file=sys.stderr)
        return 1

    gene_sets_dir = args.gene_sets or args.results_dir / "isn_gene_sets"
    keep_genes = None
    if not args.all_genes:
        keep_genes = wanted_genes(gene_sets_dir)
        if keep_genes is None:
            print(f"no gene_set_*.csv in {gene_sets_dir} - run pipeline/06_isn/gene_sets.py "
                  "first, or pass --all-genes", file=sys.stderr)
            return 1

    for exp in exps:
        print(f"\n=== {exp.name}")
        out_gene = exp / "isn_gene_act.npy"
        if out_gene.exists() and not args.overwrite:
            print("  outputs exist - skipping (use --overwrite)")
            continue

        # --- topology -> which slots to keep -------------------------------------
        topo = pd.read_csv(topology_for(exp, args.root),
                           usecols=["layer1_node", "layer1_name",
                                    "layer2_node", "layer2_name"])
        gene_slots = (topo[["layer1_node", "layer1_name"]].drop_duplicates()
                      .sort_values("layer1_node", kind="stable"))
        if keep_genes is not None:
            gene_slots = gene_slots[gene_slots.layer1_name.isin(keep_genes)]
            found = set(gene_slots.layer1_name)
            if missing := (keep_genes - found):
                print(f"  [warn] {len(missing)} requested gene(s) absent from the "
                      f"topology, e.g. {sorted(missing)[:5]}")
        pathway_nodes = (topo[["layer2_node", "layer2_name"]].drop_duplicates()
                         .sort_values("layer2_node", kind="stable"))
        slot_idx = gene_slots.layer1_node.to_numpy()
        print(f"  keeping {len(slot_idx)} gene slots "
              f"({gene_slots.layer1_name.nunique()} genes), "
              f"{len(pathway_nodes)} pathways")

        # --- model ---------------------------------------------------------------
        t0 = time.time()
        tf.keras.backend.clear_session()   # free the previous seed's graph
        model, masks = load_trained_network(types.SimpleNamespace(resultpath=str(exp) + "/"))
        targs = json.loads((exp / "train_args.json").read_text())
        model = remove_cov(model, masks)   # single genotype input; BatchNorm retained
        gene_layer, pathway_layer = resolve_layers(model, args.layer)
        sub = tf.keras.Model(inputs=model.input,
                             outputs=[gene_layer.output, pathway_layer.output])
        n_gene_units = gene_layer.output_shape[1]
        n_path_units = pathway_layer.output_shape[1]
        if slot_idx.max(initial=-1) >= n_gene_units:
            raise ValueError(f"topology references gene slot {slot_idx.max()} but the "
                             f"layer has {n_gene_units} units")
        if len(pathway_nodes) != n_path_units:
            raise ValueError(f"{len(pathway_nodes)} pathways in topology vs "
                             f"{n_path_units} units in {pathway_layer.name}")
        print(f"  model loaded in {time.time()-t0:.1f}s "
              f"({gene_layer.name} {n_gene_units}, {pathway_layer.name} {n_path_units})")

        # --- data: whole test set, unpermuted ------------------------------------
        xtest, ytest = EvalGenerator(datapath=targs["datapath"],
                                     genotype_path=targs["genotype_path"],
                                     batch_size=64, setsize=-1,
                                     one_hot=targs["onehot"], inputsize=-1,
                                     evalset="test").get_data(sample_pat=0)
        x = xtest[0]
        ytest = np.asarray(ytest).flatten()
        subjects = pd.read_csv(Path(args.root) / targs["datapath"] / "subjects.csv")
        subjects = subjects[subjects["set"] == 3].reset_index(drop=True)
        if len(subjects) != len(x):
            raise ValueError(f"{len(subjects)} test subjects in subjects.csv vs "
                             f"{len(x)} rows from the loader - refusing to guess "
                             "an alignment")
        if not args.no_verify:
            # The real check: labels must agree row for row. Truncating or assuming
            # order (as the old make_isn_input did) can misalign patients silently.
            if not np.array_equal(subjects["labels"].to_numpy(), ytest):
                raise ValueError("subjects.csv labels do not match the loader's y - "
                                 "the patient axis is NOT aligned; aborting")
            print(f"  alignment verified: {len(x)} subjects "
                  f"({int((ytest==1).sum())} cases, {int((ytest==0).sum())} controls)")

        # --- forward pass, slicing columns per batch to bound memory -------------
        t0 = time.time()
        n = len(x)
        gene_act = np.empty((n, len(slot_idx)), dtype=np.float32)
        path_act = np.empty((n, n_path_units), dtype=np.float32)
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            g, q = sub(x[start:stop], training=False)
            g = np.asarray(g)
            q = np.asarray(q)
            if g.ndim == 3:
                g = g[..., 0]
                q = q[..., 0]
            gene_act[start:stop] = g[:, slot_idx]
            path_act[start:stop] = q
        print(f"  forward pass over {n} subjects in {time.time()-t0:.1f}s")

        if not args.no_verify and args.layer == "batchnorm":
            # center=False, scale=False -> the layer divides by the training moving
            # std after subtracting the moving mean, so the test cohort should sit
            # near 0 with sd ~1. Large drift means train/test distribution shift.
            print(f"  BN check  gene: mean %+.4f sd %.4f | pathway: mean %+.4f sd %.4f"
                  % (gene_act.mean(), gene_act.std(), path_act.mean(), path_act.std()))

        # --- write ---------------------------------------------------------------
        for path, arr in ((out_gene, gene_act), (exp / "isn_pathway_act.npy", path_act)):
            # np.save appends '.npy' unless the name already ends in it, so the temp
            # name has to keep that suffix or the rename below chases a missing file.
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)
        gene_slots.reset_index(drop=True).rename_axis("slot").reset_index().rename(
            columns={"layer1_node": "layer1_node", "layer1_name": "gene"}
        ).to_csv(exp / "isn_gene_slots.csv", index=False)
        pathway_nodes.reset_index(drop=True).rename_axis("slot").reset_index().rename(
            columns={"layer2_name": "pathway"}
        ).to_csv(exp / "isn_pathway_nodes.csv", index=False)
        keep = ["patient_id", "labels"] + [c for c in subjects.columns if c.startswith("cov_")]
        subjects[keep].to_csv(exp / "isn_subjects.csv", index=False)

        print(f"  wrote {gene_act.shape} gene + {path_act.shape} pathway activations "
              f"({(gene_act.nbytes + path_act.nbytes)/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
