import scanpy as sc
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scarches as sca

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

def classification_uncert_euclideaan(adata_ref_latent, adata_query_latent):
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

def diagram(uncertainties):
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
     