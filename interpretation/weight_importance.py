import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as skm
import scipy
import sys
import pandas as pd
import numpy as np
import itertools

from scipy.sparse import coo_matrix



def make_importance_values_input(model, masks):
    
    '''
    Regular importance for the input based on a multiplication over all the
    weights propogated to the input
    '''
    mask_count = 0
    previous_layer_type = ""
    importance = pd.DataFrame([])

    inputsize = model.layers[0].input_shape[0][1]
    importance["key_0"] = np.arange(inputsize)
    importance["importance_0"] = np.ones(inputsize)

    for i, layer in enumerate(model.layers):
        importance_new = pd.DataFrame([])
        layer_type = model.layers[i].__class__.__name__ 

        if "LocallyDirected1D" in layer_type:
            mask = masks[mask_count]
            mask_count +=1 

            weights = layer.get_weights()[0]
            names = ['weights_'+str(i)+ "_"+str(j) for j in range(weights.shape[1])]

            importance_layer = pd.DataFrame(weights, columns= names)
            importance_layer["key_0"] = mask.row
            importance_layer["key_1"] = mask.col


            importance = importance.merge(importance_layer, on="key_0")
            importance = importance.drop("key_0", axis=1)
            importance = importance.rename({"key_1":"key_0"}, axis=1)

            num_filter= 0
            for importance_col in importance.filter(like="importance_").columns:
                for weight_col  in importance_layer.filter(like="weights_").columns:
                    importance_new["importance_"+str(num_filter)] = importance[importance_col]*importance[weight_col]
                    num_filter+=1

            importance_new["key_0"] = importance_layer["key_1"]
            importance = importance_new
            previous_layer_type = layer_type
            print(layer_type, importance.shape)


        elif "LocallyConnected1D" in layer_type :

            importance_new = pd.DataFrame([])
            importance_temp = pd.DataFrame([])

            layer_type = model.layers[i].__class__.__name__ 

            if len(importance) > 1:
                pass
            else:
                print("Had expected a different layer before, not", previous_layer_type)
                print("asume 1x1")


            weights = np.reshape(layer.get_weights()[0], layer.input_shape[1:3] )
            names = ['weights_'+str(i)+ "_"+str(j) for j in range(weights.shape[1])]
            importance_layer = pd.DataFrame(weights, columns= names)
            importance_layer["key_0"] = np.arange(importance_layer.shape[0]) 

            importance = importance.merge(importance_layer, on="key_0")

            importance_new["importance_0"] = (importance.filter(like="importance").prod(axis=1)* 
                                              importance.filter(like="weights_").prod(axis=1))

            importance_new["key_0"] = importance["key_0"]
            importance = importance_new
            previous_layer_type = layer_type    
            print(layer_type, importance.shape)

        elif "Dense" in layer_type :

            importance_new = pd.DataFrame([])
            importance_temp = pd.DataFrame([])

            layer_type = model.layers[i].__class__.__name__ 

            if len(importance) > 1:
                pass
            else:
                print("Had expected a different layer before, not", previous_layer_type)
                print("asume 1x1")


            weights = np.reshape(layer.get_weights()[0], layer.input_shape[1:3] )
            names = ['weights_dense']
            importance_layer = pd.DataFrame(weights, columns= names)
            importance_layer["key_0"] = np.arange(importance_layer.shape[0]) 

            importance = importance.merge(importance_layer, on="key_0")

            importance_new["importance_0"] = (importance.filter(like="importance").prod(axis=1)* 
                                              importance.filter(like="weights_").prod(axis=1))

            importance_new["key_0"] = importance["key_0"]
            importance = importance_new
            previous_layer_type = layer_type    

            print(layer_type, importance.shape)        


        else:
            print(" -- Skipping", layer_type, '--')


        importance["percentage"] = abs(importance["importance_0"]) / sum(abs(importance["importance_0"]))
    return importance

def make_gene_importance(datapath, model, masks):
    '''
    Node importance (weight-based): sum of the absolute gene->pathway weights per gene,
    collapsing the SNP-duplicated edges by the mean of their signed weights.
    Returns a per-gene DataFrame ranked by importance_sum.
    '''
    GENE_LAYER = 1  # SNP(0) -> Gene(1) -> Pathway(2): gene is the source of layer 1
    if len(masks) <= GENE_LAYER:
        raise ValueError("Expected a SNP->Gene->Pathway network (>=2 masks), got %d" % len(masks))

    mask = masks[GENE_LAYER]
    weights = model.get_layer(name="LocallyDirected_" + str(GENE_LAYER)).get_weights()[0]

    edges = pd.DataFrame({
        "gene_node": mask.row,
        "pathway_node": mask.col,
        "weight": weights.mean(axis=1),
    })

    pair = edges.groupby(["gene_node", "pathway_node"], as_index=False)["weight"].mean()
    pair["abs_weight"] = pair["weight"].abs()

    # Gene importance: sum over the gene's pathways + hub-normalised mean.
    gene_importance = (pair
                       .groupby("gene_node", as_index=False)
                       .agg(importance_sum=("abs_weight", "sum"),
                            degree=("pathway_node", "nunique")))
    gene_importance["importance_mean"] = gene_importance["importance_sum"] / gene_importance["degree"]

    network_csv = pd.read_csv(datapath + "/topology.csv")
    gene_names = (network_csv[["layer1_node", "layer1_name"]].drop_duplicates()
                  .rename(columns={"layer1_node": "gene_node", "layer1_name": "gene"}))

    gene_importance = gene_importance.merge(gene_names, on="gene_node", how="left")

    gene_importance = (gene_importance[["gene", "gene_node", "importance_sum", "degree", "importance_mean"]]
                       .sort_values("importance_sum", ascending=False).reset_index(drop=True))

    return gene_importance


def make_pair_importance(datapath, model, masks):
    '''
    Pair importance (weight-based): absolute gene->pathway edge weight per (gene, pathway),
    collapsing the SNP-duplicated edges by the mean of their signed weights.
    Returns a per-edge DataFrame ranked by abs_weight.

    NOTE: this is currently the gene->pathway edge magnitude, not a true gene-gene
    pairwise importance -- the pair logic still needs review (kept as-is for now).
    '''
    GENE_LAYER = 1  # SNP(0) -> Gene(1) -> Pathway(2): gene is the source of layer 1
    if len(masks) <= GENE_LAYER:
        raise ValueError("Expected a SNP->Gene->Pathway network (>=2 masks), got %d" % len(masks))

    mask = masks[GENE_LAYER]
    weights = model.get_layer(name="LocallyDirected_" + str(GENE_LAYER)).get_weights()[0]

    edges = pd.DataFrame({
        "gene_node": mask.row,
        "pathway_node": mask.col,
        "weight": weights.mean(axis=1),
    })

    pair_importance = edges.groupby(["gene_node", "pathway_node"], as_index=False)["weight"].mean()
    pair_importance["abs_weight"] = pair_importance["weight"].abs()

    network_csv = pd.read_csv(datapath + "/topology.csv")
    gene_names = (network_csv[["layer1_node", "layer1_name"]].drop_duplicates()
                  .rename(columns={"layer1_node": "gene_node", "layer1_name": "gene"}))
    pathway_names = (network_csv[["layer2_node", "layer2_name"]].drop_duplicates()
                     .rename(columns={"layer2_node": "pathway_node", "layer2_name": "pathway"}))

    pair_importance = (pair_importance.merge(gene_names, on="gene_node", how="left")
                                      .merge(pathway_names, on="pathway_node", how="left"))

    pair_importance = (pair_importance[["gene", "pathway", "gene_node", "pathway_node", "abs_weight"]]
                       .sort_values("abs_weight", ascending=False).reset_index(drop=True))

    return pair_importance


