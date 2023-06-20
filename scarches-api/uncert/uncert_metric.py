import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scarches as sca
from scipy.stats import entropy

def mahalanobis(v, data):
    vector = np.zeros(len(data))
    for centroid_index in range(len(data)):
        v_mu = v - np.mean(data[centroid_index])
        inv_cov = np.eye(len(data[centroid_index]))
        left = np.dot(v_mu, inv_cov)
        mahal = np.dot(left, v_mu.T)
        vector[centroid_index] = mahal
    return vector

def classification_uncert_mahalanobis(adata_ref_latent, adata_query_latent):
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

        adata_query_latent.obsm['uncertainty'] = uncertainties
    return uncertainties

def classification_uncert_euclidean(adata_ref_latent, adata_query_latent):
    trainer = sca.utils.weighted_knn_trainer(
    adata_ref_latent,
    "X",
    n_neighbors = 15
    )

    _, uncertainties = sca.utils.weighted_knn_transfer(
        adata_query_latent,
        "X",
        adata_ref_latent.obs,
        "cell_type",
        trainer
    )

    adata_query_latent.obsm['uncertainty'] = uncertainties
    return uncertainties

def integration_uncertain(
        adata_latent,
        batch_key,
        n_neighbors = 15):
    
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


def uncert_diagram(uncertainties):
    data = []
    uncertainties["uncertainty"].plot(kind='box')
    labels = uncertainties["cell_type"].unique()
    
    for cell_type in labels:
        pl = uncertainties[uncertainties["cell_type"] == cell_type]["uncertainty"]
        data.append(pl)

    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title('Uncertainty based on Cell Type')
    plt.show()
     