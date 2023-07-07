import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scarches as sca
from scipy.stats import entropy
import anndata as ad
import milopy


def mahalanobis(v, data):
    """Computes the Mahalanobis distance from a query cell to all the centroids

    Returns:
        vector: All the distances to the centroids
    """
    vector = np.zeros(len(data))
    for centroid_index in range(len(data)):
        v_mu = v - np.mean(data[centroid_index])
        inv_cov = np.eye(len(data[centroid_index]))
        left = np.dot(v_mu, inv_cov)
        mahal = np.dot(left, v_mu.T)
        vector[centroid_index] = mahal
    return vector


def classification_uncert_mahalanobis(
        adata_ref_latent,
        adata_query_latent):
    """ Computes classification uncertainty, based on the Mahalanobis distance of each cell
    to the cell cluster centroids

    Args:
        adata_ref_latent (AnnData): Latent representation of the reference
        adata_query_latent (AnnData): Latent representation of the query

    Returns:
        uncertainties (pandas DataFrame): Classification uncertainties for all of the query cell types
    """    
    num_clusters = adata_ref_latent.n_vars
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(adata_ref_latent.X)
    uncertainties = pd.DataFrame(columns=["uncertainty"], index=adata_query_latent.obs_names)
    centroids = kmeans.cluster_centers_
    adata_query = kmeans.transform(adata_query_latent.X)

    for query_cell_index in range(len(adata_query)):
        query_cell = adata_query_latent.X[query_cell_index]
    
        distance = mahalanobis(query_cell, centroids) 
        weighted_distance = np.average(distance/ np.linalg.norm(distance))
        uncertainties.iloc[query_cell_index]['uncertainty'] = weighted_distance

        adata_query_latent.obsm['uncertainty_mahalanobis'] = uncertainties
    return uncertainties

def classification_uncert_euclidean(
        adata_ref_latent,
        adata_query_latent,
        cell_type_key,
        n_neighbors = 15):
    """Computes classification uncertainty, based on the Euclidean distance of each cell
    to its k-nearest neighbors. Additional adjustment by a Gaussian kernel is made

    Args:
        adata_ref_latent (AnnData): Latent representation of the reference
        adata_query_latent (AnnData): Latent representation of the query
        cell_type_key (String): cell type key
        n_neighbors (int, optional): The amount of nearest neighbors. Defaults to 15.

    Returns:
        uncertainties (pandas DataFrame): Classification uncertainties for all of query cell_types
    """    
    trainer = sca.utils.weighted_knn_trainer(
    adata_ref_latent,
    "X",
    n_neighbors = n_neighbors
    )

    _, uncertainties = sca.utils.weighted_knn_transfer(
        adata_query_latent,
        "X",
        adata_ref_latent.obs,
        cell_type_key,
        trainer
    )

    adata_query_latent.obsm['uncertainty_euclidean'] = uncertainties
    return uncertainties

# Test differential abundance analysis on neighbourhoods with Milo.
# Works only when X_scvi is present
def classification_uncert_milo(
        adata_all_latent,
        cell_type_key,
        ref_or_query_key = "ref_or_query",
        ref_key = "ref",
        query_key = "query",
        n_neighbors = 15,
        sample_col = "",
        d = 30):
        
    if "X_scVI" not in adata_all_latent.obsm:
        return

    sc.pp.neighbors(adata_all_latent, n_neighbors=n_neighbors)

    milopy.core.make_nhoods(adata_all_latent, use_rep="X_scVI", prop=0.1)
    milopy.core.count_nhoods(adata_all_latent, sample_col=sample_col)
    milopy.utils.annotate_nhoods(adata_all_latent[adata_all_latent.obs[ref_or_query_key] == ref_key], cell_type_key)
    adata_all_latent.obs["is_query"] = adata_all_latent.obs[ref_or_query_key] == query_key
    milopy.core.DA_nhoods(adata_all_latent, design="is_query")

    return 

def integration_uncertain(
        adata_latent,
        batch_key,
        n_neighbors = 15):
    """Computes the integration uncertainty per batch based on its entropy.
    The uncertainty is computed 1 - batch_entropy

    Args:
        adata_ref_latent (AnnData): Latent representation of the reference
        adata_query_latent (AnnData): Latent representation of the query
        n_neighbors (int, optional): _description_. Defaults to 15.

    Returns: 
    uncertainties (pandas DataFrame): Integration uncertainties for all batches
        
    """    
    adata = sca.dataset.remove_sparsity(adata_latent)
    batches = adata.obs[batch_key].nunique()
    uncertainty = pd.DataFrame(columns=["uncertainty"], index=adata_latent.obs_names)
    uncertainty = adata_latent.obs[[batch_key]].copy()

    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(adata_latent.X)

    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    
    batch_indices = adata.obs[batch_key].values[indices]

    entropies = np.array([entropy(np.unique(row, return_counts=True)[1].astype(np.int64), base=batches)
                          for row in batch_indices])

    uncertainty["uncertainty"] = 1 - entropies
    uncert_by_batch = uncertainty.groupby(batch_key)['uncertainty'].mean().reset_index()

    return uncert_by_batch


def uncert_diagram(uncertainties, cell_type_key):
    """Creates a plot for classification uncertainty per cell type

    Args:
        uncertainties (pandas DataFrame): Classification uncertainty per cell type
        cell_type_key (String):  cell type key
    """    
    data = []
    uncertainties["uncertainty"].plot(kind='box')
    labels = uncertainties[cell_type_key].unique()
    
    for cell_type in labels:
        pl = uncertainties[uncertainties[cell_type_key] == cell_type]["uncertainty"]
        data.append(pl)

    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title('Uncertainty based on Cell Type')
    plt.savefig('class_uncert_boxplot.png')
    plt.show()

def integration_uncert_diagram(uncertainties, batch_key):
    y_axis = uncertainties["uncertainty"].tolist()
    x_axis = uncertainties[batch_key].tolist()

    plt.bar(x_axis, y_axis)
    plt.title('Batch integration uncertainty')
    plt.xlabel('Batch')
    plt.ylabel('Uncertainty')
    plt.savefig('integration_uncertainty.png')
    plt.show()


# Creates a UMAP Diagram for the given uncertainty
def uncert_umap_diagram(
        adata_ref,
        adata_query,
        uncertainties,
        batch_key,
        cell_type_key,
        n_neighbors = 15):
    adata_ref.obs["uncertainty"] = 0
    adata_query.obs["uncertainty"] = uncertainties

    combined_emb = ad.concat([adata_ref, adata_query])

    sc.pp.neighbors(combined_emb, n_neighbors=n_neighbors)
    sc.tl.umap(combined_emb)
    sc.pl.umap(combined_emb,
               color=["uncertainty", batch_key, cell_type_key],
               frameon=False,
               wspace=0.6)

# Plots a precision recall curve
def precision_recall_curve(precision, recall):
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.savefig('precision_recall.png')
    plt.show()
     