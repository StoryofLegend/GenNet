"""Method 2C -- perturbation (node ablation) importance for GenNet.

    Delta_y(g) = y_full - y_{g ablated}

WHERE THE ABLATION HAPPENS
--------------------------
A gene is ablated at the GENE-LAYER ACTIVATION, not at the genotype input.

Two reasons:

1. Zeroing the genotype (``X[:, snp] = 0``) is not a neutral baseline -- dosage 0
   is homozygous-reference, a real genotype with a real effect. Delta_y would then
   mix "this gene is gone" with "everyone is ref/ref".
2. ``layer_block`` (Create_network.py) builds every block as
   ``LocallyDirected_i -> Activation -> BatchNormalization(center=False, scale=False)``.
   The gene-layer output is therefore already standardised to mean ~0 across the
   cohort, so setting it to 0 means exactly "replace this gene by the population
   average gene" -- the neutral reference the ablation formula asks for.

A gene NAME is split over several ``layer1_node`` s (see make_gene_importance);
all of its nodes are zeroed together, so the score is per gene, not per node.

WHAT IS MEASURED
----------------
Delta on the GENETIC LOGIT (pre-sigmoid ``output_layer``), never on the probability:
the sigmoid saturates and squashes the very differences we are trying to rank. The
covariate branch is not involved -- it sits downstream of the genetic head and only
adds a constant shift per subject.

HOW IT IS COMPUTED
------------------
Zeroing gene g changes only the pathway nodes g connects to. With
    z  = pre-activation of the pathway layer   (LocallyDirected_1 output, bias included)
    d  = A[:, nodes(g)] @ W1[nodes(g), :]      (what g contributes to z)
    m  = gamma / sqrt(moving_var + eps)        (the pathway BatchNorm, elementwise)
    c  = w_out * m
the ablation delta is exactly

    Delta_y(g)_i = sum_p c_p * ( act(z_ip) - act(z_ip - d_ip) )

restricted to the pathways p that g actually touches. The BatchNorm mean and both
biases cancel in the difference. This is not an approximation: ``ablation_scores``
checks it against a real forward pass through the reused Keras layers
(``verify_genes``) and raises if they disagree.

Duplicate COO entries (one topology row per SNP, so a gene->pathway edge appears
many times) are summed, matching what tf.sparse.sparse_dense_matmul does inside
LocallyDirected1D. The kernel is in the mask's native COO order -- the mask must
never be sorted before attaching weights.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from scipy.sparse import coo_matrix

GENE_LAYER = 1  # SNP(0) -> Gene(1) -> Pathway(2): gene nodes feed LocallyDirected_1


def genetic_chain(model):
    """The genotype branch of a trained GenNet model: input_layer .. output_layer."""
    names = [layer.name for layer in model.layers]
    if "output_layer" not in names:
        raise ValueError("No 'output_layer' in the model; not a GenNet network?")
    return model.layers[:names.index("output_layer") + 1]


def split_at_gene_layer(model):
    """Split the genetic branch into (head, tail_layers) at the gene activation.

    head        : genotype -> gene-layer activation, shape (n, n_gene_nodes, 1).
                  This is the post-BatchNorm tensor feeding LocallyDirected_1.
    tail_layers : LocallyDirected_1 .. output_layer, the original layer objects
                  (reusing them keeps the trained weights, no copying).
    """
    chain = genetic_chain(model)
    ld_name = "LocallyDirected_" + str(GENE_LAYER)
    names = [layer.name for layer in chain]
    if ld_name not in names:
        raise ValueError("No %s in the model; expected a SNP->Gene->Pathway network" % ld_name)
    idx = names.index(ld_name)

    head = tf.keras.Model(inputs=model.inputs[0], outputs=chain[idx].get_input_at(0),
                          name="gene_activation_model")
    return head, chain[idx:]


def build_tail_models(tail_layers, n_gene_nodes):
    """Reuse the trained tail layers on a fresh gene-activation input.

    Returns (tail_model, z_model): gene activation -> genetic logit, and gene
    activation -> pathway pre-activation z. Reusing the layer objects shares the
    trained weights, so no copying and no chance of a stale duplicate.
    """
    inputs = tf.keras.Input(shape=(n_gene_nodes, 1), name="gene_activation")
    x = tail_layers[0](inputs)
    z_model = tf.keras.Model(inputs=inputs, outputs=x, name="gene_to_pathway_preact")
    for layer in tail_layers[1:]:
        x = layer(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="gene_to_logit"), z_model


def tail_parameters(model, masks, tail_layers):
    """Everything the analytic delta needs from the pathway layer and the readout.

    Returns dict with:
        W1        csr (n_gene_nodes x n_pathway_nodes), duplicate COO entries SUMMED
        act       the pathway activation function (tanh / relu / softplus / ...)
        c         w_out * gamma / sqrt(moving_var + eps), one per pathway node
    """
    ld, activation_layer, bn = tail_layers[0], tail_layers[1], tail_layers[2]
    dense = tail_layers[-1]

    if not isinstance(activation_layer, tf.keras.layers.Activation):
        raise ValueError("Expected an Activation after %s, got %s"
                         % (ld.name, type(activation_layer).__name__))
    if not isinstance(bn, tf.keras.layers.BatchNormalization):
        raise ValueError("Expected a BatchNormalization after %s, got %s"
                         % (activation_layer.name, type(bn).__name__))
    if not isinstance(dense, tf.keras.layers.Dense):
        raise ValueError("Expected output_layer to be Dense, got %s" % type(dense).__name__)

    mask = masks[GENE_LAYER]
    kernel = ld.get_weights()[0]
    if kernel.shape[0] != len(mask.data):
        raise ValueError("Kernel/mask mismatch: %d weights vs %d mask entries"
                         % (kernel.shape[0], len(mask.data)))
    # Native COO order, duplicates summed by tocsr() -- exactly what the layer does.
    W1 = coo_matrix((kernel[:, 0], (mask.row, mask.col)), shape=mask.shape).tocsr()

    var = bn.moving_variance.numpy().reshape(-1)
    gamma = bn.gamma.numpy().reshape(-1) if bn.scale else np.ones_like(var)
    w_out = dense.get_weights()[0].reshape(-1)

    return {"W1": W1,
            "act": activation_layer.activation,
            "c": (w_out * gamma / np.sqrt(var + bn.epsilon)).astype(np.float64)}


def gene_node_map(datapath):
    """gene name -> array of layer1_node indices, from topology.csv."""
    topology = pd.read_csv(datapath + "/topology.csv")
    nodes = (topology[["layer1_node", "layer1_name"]].drop_duplicates()
             .rename(columns={"layer1_node": "gene_node", "layer1_name": "gene"}))
    return {gene: group["gene_node"].to_numpy()
            for gene, group in nodes.groupby("gene", sort=True)}


def _apply(act, x):
    """Run a Keras activation on a numpy array."""
    return np.asarray(act(tf.convert_to_tensor(x, dtype=tf.float32)), dtype=np.float64)


def ablation_delta(gene_nodes, A, z, params):
    """Delta_y = y_full - y_ablated on the genetic logit, per subject. Exact.

    gene_nodes : gene-layer node indices to zero together (one gene)
    A          : (n_subjects, n_gene_nodes) gene activations
    z          : (n_subjects, n_pathway_nodes) pathway pre-activations
    """
    W_sub = params["W1"][gene_nodes, :]
    pathways = np.unique(W_sub.indices)
    if pathways.size == 0:                    # gene wired to nothing
        return np.zeros(A.shape[0]), pathways

    contribution = A[:, gene_nodes] @ W_sub[:, pathways].toarray()   # (n, |P|)
    z_p = z[:, pathways]
    delta = _apply(params["act"], z_p) - _apply(params["act"], z_p - contribution)
    return delta @ params["c"][pathways], pathways


def verify_genes(genes, gene_nodes, A, z, params, tail_model, batch_size=256, tol=1e-3):
    """Check the analytic delta against a real forward pass through the trained layers.

    Returns the largest absolute disagreement over the checked genes. Ablation deltas
    live on the logit scale (order 1e-2..1e1), so tol=1e-3 is a tight float32 bound.
    """
    logit_full = tail_model.predict(A[:, :, None], batch_size=batch_size,
                                    verbose=0).reshape(-1).astype(np.float64)
    worst = 0.0
    for gene in genes:
        nodes = gene_nodes[gene]
        A_ablated = A.copy()
        A_ablated[:, nodes] = 0.0
        logit_ablated = tail_model.predict(A_ablated[:, :, None], batch_size=batch_size,
                                           verbose=0).reshape(-1).astype(np.float64)
        analytic, _ = ablation_delta(nodes, A, z, params)
        difference = np.max(np.abs(analytic - (logit_full - logit_ablated)))
        print("  verify %-15s max|analytic - forward pass| = %.3e" % (gene, difference))
        worst = max(worst, difference)

    if worst > tol:
        raise AssertionError("Analytic ablation disagrees with the forward pass "
                             "(max %.3e > tol %.3e)" % (worst, tol))
    return worst


def make_ablation_importance(datapath, model, masks, x, batch_size=256,
                             n_verify=3, per_patient=False, seed=1):
    """Node-ablation importance (method 2C), one row per gene.

    x        : genotype matrix of the subjects to ablate over, shape (n, inputsize)
    per_patient : also return the full (n_genes x n_subjects) delta matrix -- the
                  per-subject signal ISNs need later.

    Columns: gene, ablation_meanabs (mean_i |Delta_i|, the ranking score),
    ablation_meanabs_per_degree (the same score divided by the pathway degree --
    the connectivity normalisation the guidelines ask for, see below),
    ablation_mean (signed, direction of effect), ablation_absmean, ablation_std,
    degree (pathways touched), n_gene_nodes.

    On the normalisation: zeroing the whole gene at once already measures its joint
    effect through all its pathways, so ablation_meanabs is far less hub-driven than a
    sum over edges (Spearman vs degree: 0.23 here, 0.32 for the weight-based 2A sum).
    The per-degree column is therefore a robustness check, not a correction -- report
    both rather than replacing one with the other.
    """
    head, tail_layers = split_at_gene_layer(model)
    n_gene_nodes = head.output_shape[1]
    tail_model, z_model = build_tail_models(tail_layers, n_gene_nodes)
    params = tail_parameters(model, masks, tail_layers)

    print("Computing gene activations for %d subjects x %d gene nodes" % (len(x), n_gene_nodes))
    A = head.predict(x, batch_size=batch_size, verbose=0)[:, :, 0]

    # Pathway pre-activations: LocallyDirected_1 has no activation of its own, so its
    # output IS z (bias included).
    z = z_model.predict(A[:, :, None], batch_size=batch_size, verbose=0)[:, :, 0]
    print("gene activations %s, pathway pre-activations %s" % (A.shape, z.shape))

    gene_nodes = gene_node_map(datapath)
    genes = sorted(gene_nodes)

    if n_verify > 0:
        checked = list(np.random.RandomState(seed).choice(genes, size=min(n_verify, len(genes)),
                                                          replace=False))
        print("Verifying the analytic delta against a full forward pass on %d genes:" % len(checked))
        verify_genes(checked, gene_nodes, A, z, params, tail_model, batch_size=batch_size)
        print("Verified.")

    rows = []
    deltas = np.zeros((len(genes), A.shape[0]), dtype=np.float32) if per_patient else None
    for i, gene in enumerate(tqdm.tqdm(genes, desc="ablating genes")):
        nodes = gene_nodes[gene]
        delta, pathways = ablation_delta(nodes, A, z, params)
        if per_patient:
            deltas[i] = delta
        meanabs = np.mean(np.abs(delta))
        rows.append({"gene": gene,
                     "ablation_meanabs": meanabs,
                     # degree is 0 only for a gene wired to no pathway, whose delta is
                     # identically 0 -- keep the score at 0 rather than dividing by 0.
                     "ablation_meanabs_per_degree": (meanabs / pathways.size
                                                     if pathways.size else 0.0),
                     "ablation_mean": np.mean(delta),
                     "ablation_absmean": np.abs(np.mean(delta)),
                     "ablation_std": np.std(delta),
                     "degree": pathways.size,
                     "n_gene_nodes": nodes.size})

    importance = (pd.DataFrame(rows)
                  .sort_values("ablation_meanabs", ascending=False)
                  .reset_index(drop=True))

    if per_patient:
        return importance, pd.DataFrame(deltas, index=genes)
    return importance
