import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import torch
from pynndescent import NNDescent

from scipy import sparse
from typing import Optional, Union, Mapping, Literal
import warnings
import sys
import os
import importlib.util
import argparse
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")



def nn2adj(nn, n1=None, n2=None):
    if n1 is None:
        n1 = nn[1].shape[0]
    if n2 is None:
        n2 = np.max(nn[1].flatten())

    df = pd.DataFrame(
        {
            "i": np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
            "j": nn[0].flatten(),
            "x": nn[1].flatten(),
        }
    )
    adj = sparse.csr_matrix(
        (np.repeat(1, df.shape[0]), (df["i"], df["j"])), shape=(n1, n2)
    )
    return adj


def build_nn(
    ref,
    query=None,
    k=100,
    weight: Literal["unweighted", "dist", "gaussian_kernel"] = "unweighted",
    sigma=None,
):
    if query is None:
        query = ref

    if torch.cuda.is_available() and importlib.util.find_spec("cuml"):
        print("GPU detected and cuml installed. Use cuML for neighborhood estimation.")
        from cuml.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k)
        model.fit(ref)
        knn = model.kneighbors(query)
        
    else:
        print(
            "Failed calling cuML. Falling back to neighborhood estimation using CPU with pynndescent."
        )
        index = NNDescent(ref, n_neighbors=k)
        knn = index.query(query, k=k)

    adj = nn2adj(knn, n1=query.shape[0], n2=ref.shape[0])
    return adj


def build_mutual_nn(dat1, dat2=None, k1=15, k2=None):
    if dat2 is None:
        dat2 = dat1
    if k2 is None:
        k2 = k1

    adj_12 = build_nn(dat1, dat2, k=k2)
    adj_21 = build_nn(dat2, dat1, k=k1)

    adj_mnn = adj_12.multiply(adj_21.T)
    return adj_mnn

def percent_query_with_anchor(ref_adata, query_adata):
    ref = ref_adata.obsm["latent_rep"]
    query = query_adata.obsm["latent_rep"]
    adj_mnn=build_mutual_nn(ref,query)
    has_anchor=adj_mnn.sum(0)>0 #all query cells that have an anchor (output dim: no query cells)
    percentage = (has_anchor.sum()/adj_mnn.shape[1])*100
    return round(percentage, 2)


def get_transition_prob_mat(dat, k=50, symm=True):
    adj = build_nn(dat, k=k)
    if symm:
        adj = adj + adj.transpose()
    prob = sparse.diags(1 / np.array(adj.sum(1)).flatten()) @ adj.transpose()
    return prob


def random_walk_with_restart(init, transition_prob, alpha=0.5, num_rounds=100):
    init = np.array(init).flatten()
    heat = init[:, None]
    for i in range(num_rounds):
        heat = init[:, None] * alpha + (1 - alpha) * (
            transition_prob.transpose() @ heat
        )
    return heat


def get_wknn(
    ref,
    query,
    ref2=None,
    k: int = 100,
    query2ref: bool = True,
    ref2query: bool = True,
    weighting_scheme: Literal[
        "n", "top_n", "jaccard", "jaccard_square", "gaussian", "dist"
    ] = "jaccard_square",
    top_n: Optional[int] = None,
    return_adjs: bool = False,
):
    """
    Compute the weighted k-nearest neighbors graph between the reference and query datasets

    Parameters
    ----------
    ref : np.ndarray
        The reference representation to build ref-query neighbor graph
    query : np.ndarray
        The query representation to build ref-query neighbor graph
    ref2 : np.ndarray
        The reference representation to build ref-ref neighbor graph
    k : int
        Number of neighbors per cell
    query2ref : bool
        Consider query-to-ref neighbors
    ref2query : bool
        Consider ref-to-query neighbors
    weighting_scheme : str
        How to weight edges in the ref-query neighbor graph
    top_n : int
        The number of top neighbors to consider
    return_adjs : bool
        Whether to return the adjacency matrices of ref-query, query-ref, and ref-ref for weighting
    """
    adj_q2r = build_nn(ref=ref, query=query, k=k)

    adj_r2q = None
    if ref2query:
        adj_r2q = build_nn(ref=query, query=ref, k=k)

    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn = ((adj_r2q + adj_q2r.T) > 0) + 0
    else:
        warnings.warn(
            "At least one of query2ref and ref2query should be True. Reset to default with both being True."
        )
        adj_knn = ((adj_r2q + adj_q2r.T) > 0) + 0 # 1 if either R_i or Q_j are considered a nn of the other 

    if ref2 is None:
        ref2 = ref
    adj_ref = build_nn(ref=ref2, k=k)


    num_shared_neighbors = adj_q2r @ adj_ref # no. neighbours that Q_i and R_j have in common 
    num_shared_neighbors_nn = num_shared_neighbors.multiply(adj_knn.T) # only keep weights if q and r are both nearest neigbours of eachother

    del num_shared_neighbors

    wknn = num_shared_neighbors_nn.copy()
    if weighting_scheme == "top_n":
        if top_n is None:
            top_n = k // 4 if k > 4 else 1
        wknn = (wknn > top_n) * 1
    elif weighting_scheme == "jaccard":
        wknn.data = wknn.data / (k + k - wknn.data)
    elif weighting_scheme == "jaccard_square":
        wknn.data = (wknn.data / (k + k - wknn.data)) ** 2

    if return_adjs:
        adjs = {"q2r": adj_q2r, "r2q": adj_r2q, "knn": adj_knn, "r2r": adj_ref}
        return (wknn, adjs)
    else:
        return wknn

#estimate presence of reference cell in query based on how many neighbours a query and ref cell shares
# uncertainty: How far is a query cell from the nearest ref cells.
def estimate_presence_score(
    ref_adata,
    query_adata,
    wknn=None,
    use_rep_ref_wknn="latent_rep",
    use_rep_query_wknn="latent_rep",
    k_wknn=15,
    query2ref_wknn=True,
    ref2query_wknn=False,
    weighting_scheme_wknn="jaccard_square",
    ref_trans_prop=None,
    use_rep_ref_trans_prop=None,
    k_ref_trans_prop=15,
    symm_ref_trans_prop=True,
    split_by=None,
    do_random_walk=True,
    alpha_random_walk=0.1,
    num_rounds_random_walk=100,
    log=True,
):
    if wknn is None:
        ref = ref_adata.obsm[use_rep_ref_wknn]
        query = query_adata.obsm[use_rep_query_wknn]
        wknn = get_wknn(
            ref=ref,
            query=query,
            k=k_wknn,
            query2ref=query2ref_wknn,
            ref2query=ref2query_wknn,
            weighting_scheme=weighting_scheme_wknn,
        )

    if ref_trans_prop is None and do_random_walk:
        if use_rep_ref_trans_prop is None:
            use_rep_ref_trans_prop = use_rep_ref_wknn
        ref = ref_adata.obsm[use_rep_ref_trans_prop]
        ref_trans_prop = get_transition_prob_mat(ref, k=k_ref_trans_prop)

    if split_by and split_by in query_adata.obs.columns:
        presence_split = [
            np.array(wknn[query_adata.obs[split_by] == x, :].sum(axis=0)).flatten()
            for x in query_adata.obs[split_by].unique()
        ]
    else:
        presence_split = [np.array(wknn.sum(axis=0)).flatten()]
    if do_random_walk:
        presence_split_sm = [
            random_walk_with_restart(
                init=x,
                transition_prob=ref_trans_prop,
                alpha=alpha_random_walk,
                num_rounds=num_rounds_random_walk,
            )
            for x in presence_split
        ]
    else:
        presence_split_sm = [x[:, None] for x in presence_split]

    columns = (
        query_adata.obs[split_by].unique()
        if split_by and split_by in query_adata.obs.columns
        else ["query"]
    )
    if len(columns) > 1:
        df_presence = pd.DataFrame(
            np.concatenate(presence_split_sm, axis=1),
            columns=columns,
            index=ref_adata.obs_names,
        )
    else:
        df_presence = pd.DataFrame({columns[0]: presence_split_sm[0].flatten()}).set_index(
            ref_adata.obs_names
        )

    if log:
        df_presence = df_presence.apply(lambda x: np.log1p(x), axis=0)
    df_presence_norm = df_presence.apply(
        lambda x: np.clip(x, np.percentile(x, 1), np.percentile(x, 99)), axis=0
    ).apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)
    max_presence = df_presence_norm.max(1)


    return {
        "max": max_presence,
        "per_group": df_presence_norm,
        "ref_trans_prop": ref_trans_prop,
    }


# def transfer_labels(ref_adata, query_adata, wknn, label_key="celltype"):
#     scores = pd.DataFrame(
#         wknn @ pd.get_dummies(ref_adata.obs[label_key]),
#         columns=pd.get_dummies(ref_adata.obs[label_key]).columns,
#         index=query_adata.obs_names,
#     )
#     scores["best_label"] = scores.idxmax(1)
#     scores["best_score"] = scores.max(1)
#     return scores


# cluster preservation score

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.special import rel_entr
import pandas as pd

def cluster_preservation_score(adata, ds_amount=5000, type='standard'):
    """
    Calculate the cluster preservation score for a query after mapping.
    Parameters:
    - adata: AnnData object containing single-cell data.
    - ds_amount: Maximum number of cells to include.
    Returns:
    - score: Cluster preservation score.
    """
    dims = min(50, adata.uns.get('Azimuth_map_ndims', 50))

    if type == 'standard':
        # Following the standard preprocessing workflow
        # sc.pp.scale(adata, zero_center=True)
        if adata.n_obs > ds_amount:
            adata = adata[np.random.choice(adata.obs_names, ds_amount, replace=False), :]
        sc.tl.pca(adata, svd_solver='arpack', n_comps=dims)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=dims, method='umap', metric='euclidean')
        sc.tl.leiden(adata, resolution=0.6)
        # sc.pp.neighbors(adata, use_rep='latent_rep', key_added="integrated_neighbors")

    elif type == 'bridge':
        # Bridge-specific preprocessing
        sc.pp.scale(adata, zero_center=True)
        if adata.n_obs > ds_amount:
            adata = adata[np.random.choice(adata.obs_names, ds_amount, replace=False), :]
        sc.pp.svd(adata, n_comps=dims)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_svd', method='umap', metric='euclidean')
        sc.tl.leiden(adata, resolution=0.6)
        # sc.pp.neighbors(adata, use_rep='latent_rep', key_added="integrated_neighbors")

    else:
        print("Incorrect type: Must be either 'standard' or 'bridge'")
        return None

    # Entropy calculations for neighborhood preservation
    nn_orig = NearestNeighbors(n_neighbors=15)
    nn_orig.fit(adata.obsm['X_pca'] if type == 'standard' else adata.obsm['X_svd'])
    _, orig_indices = nn_orig.kneighbors()

    nn_integrated = NearestNeighbors(n_neighbors=15)
    nn_integrated.fit(adata.obsm['latent_rep'])
    _, integrated_indices = nn_integrated.kneighbors()

    # Calculate entropy for each set of neighbors
    def entropy_of_labels(indices):
        labels = adata.obs['leiden'][indices].to_numpy()
        _, counts = np.unique(labels, return_counts=True)
        return rel_entr(counts, np.full_like(counts, fill_value=1/len(counts))).sum()

    orig_ent = np.array([entropy_of_labels(idx) for idx in orig_indices])
    integrated_ent = np.array([entropy_of_labels(idx) for idx in integrated_indices])

    # Calculate the cluster preservation statistic
    ids = adata.obs['leiden'].to_numpy()
    orig_means = pd.Series(orig_ent).groupby(ids).mean()
    integrated_means = pd.Series(integrated_ent).groupby(ids).mean()
    stat = np.median(orig_means - integrated_means)

    if stat <= 0:
        return 5.00
    else:
        stat = -1 * np.log2(stat)
        stat = np.clip(stat, 0.00, 5.00)
        return stat

def percentage_unknown(query, prediction_label, uncertainty_threshold=0.5):
    query.obs[f"{prediction_label}_filtered_by_uncert>0.5"] = query.obs[
    prediction_label
    ].mask(
        query.obs["uncertainty_mahalanobis"] > uncertainty_threshold,
        "Unknown",
    )

    number_unknown = (query.obs["uncertainty_mahalanobis"] > uncertainty_threshold).sum()

    return number_unknown/len(query)*100
