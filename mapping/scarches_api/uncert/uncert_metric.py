import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scarches as sca
from scipy.stats import entropy
import anndata as ad
#import milopy
from matplotlib.lines import Line2D

from sklearn.mixture import GaussianMixture

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


# def classification_uncert_mahalanobis2(
#         adata_ref_latent,
#         adata_query_latent,
#         cell_type_key):
#     """ Computes classification uncertainty, based on the Mahalanobis distance of each cell
#     to the cell cluster centroids

#     Args:
#         adata_ref_latent (AnnData): Latent representation of the reference
#         adata_query_latent (AnnData): Latent representation of the query

#     Returns:
#         uncertainties (pandas DataFrame): Classification uncertainties for all of the query cell types
#     """    
#     num_clusters = adata_ref_latent.obs[cell_type_key].nunique()
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(adata_ref_latent.X)
#     uncertainties = pd.DataFrame(columns=["uncertainty"], index=adata_query_latent.obs_names)
#     centroids = kmeans.cluster_centers_
#     adata_query = kmeans.transform(adata_query_latent.X)

#     for query_cell_index in range(len(adata_query)):
#         query_cell = adata_query_latent.X[query_cell_index]
#         distance = mahalanobis(query_cell, centroids)
#         uncertainties.iloc[query_cell_index]['uncertainty'] = np.mean(distance)
        
#     max_distance = np.max(uncertainties["uncertainty"])
#     min_distance = np.min(uncertainties["uncertainty"])
#     uncertainties["uncertainty"] = (uncertainties["uncertainty"] - min_distance) / (max_distance - min_distance + 1e-8)
#     adata_query_latent.obsm['uncertainty_mahalanobis'] = uncertainties
    
#     return uncertainties, centroids

def classification_uncert_mahalanobis(adata_ref_latent, embedding_name, adata_query_latent, cell_type_key):
    num_clusters = adata_ref_latent.obs[cell_type_key].nunique()

    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(adata_ref_latent.X.toarray())
    centroids = gmm.means_
    cluster_membership = gmm.predict_proba(adata_query_latent[embedding_name])

    uncertainties = pd.DataFrame(columns=["uncertainty"], index=adata_query_latent.obs_names)
    for query_cell_index, query_cell in enumerate(adata_query_latent.X):
        distance = mahalanobis(query_cell, centroids)
        weighed_distance = np.multiply(cluster_membership[query_cell_index], distance)
        uncertainties.iloc[query_cell_index]['uncertainty'] = np.mean(weighed_distance)
        
    max_distance = np.max(uncertainties["uncertainty"])
    min_distance = np.min(uncertainties["uncertainty"])
    uncertainties["uncertainty"] = (uncertainties["uncertainty"] - min_distance) / (max_distance - min_distance + 1e-8)
    adata_query_latent.obsm['uncertainty_mahalanobis'] = uncertainties
    return uncertainties, centroids

def classification_uncert_euclidean(
        adata_ref_latent,
        embedding_name,
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
    embedding_name,
    n_neighbors = n_neighbors
    )

    _, uncertainties = sca.utils.weighted_knn_transfer(
        adata_query_latent,
        embedding_name,
        adata_ref_latent.obs,
        cell_type_key,
        trainer
    )

    #Important to store as numpy array in obs for cellbygene visualization
    if(len(uncertainties.columns) > 1):
        for entry in uncertainties.columns:
            name = str(entry + '_uncertainty_euclidean')
            adata_query_latent.obs[name] = uncertainties[entry].to_numpy(dtype="float32")
    else:
        adata_query_latent.obs['uncertainty_euclidean'] = uncertainties.to_numpy(dtype="float32")
    
    return uncertainties

# Test differential abundance analysis on neighbourhoods with Milo.
# def classification_uncert_milo(
#         adata_latent,
#         cell_type_key,
#         ref_or_query_key="ref_or_query",
#         ref_key="ref",
#         query_key="query",
#         n_neighbors=15,
#         sample_col="batch",
#         red_name = "X_trVAE",
#         d=30):

#     adata_all_latent = adata_latent.copy()
#     x = pd.DataFrame(adata_latent.X, index=adata_latent.obs_names)
#     adata_all_latent.obsm[red_name] = x.values
#     print(adata_all_latent)
#     sc.pp.neighbors(adata_all_latent, n_neighbors=n_neighbors, use_rep=red_name)

#     milopy.core.make_nhoods(adata_all_latent, prop=0.1)

#     adata_all_latent[adata_all_latent.obs['nhood_ixs_refined'] != 0].obs[['nhood_ixs_refined', 'nhood_kth_distance']]

#     milopy.core.count_nhoods(adata_all_latent, sample_col=sample_col)
#     print(adata_all_latent[adata_all_latent.obs['nhood_ixs_refined'] != 0].obs[['nhood_ixs_refined', 'nhood_kth_distance']])
#     # milopy.utils.annotate_nhoods(adata_all_latent[adata_all_latent.obs[ref_or_query_key] == ref_key], cell_type_key)
#     adata_all_latent.obs["is_query"] = adata_all_latent.obs[ref_or_query_key] == query_key
#     milopy.core.DA_nhoods(adata_all_latent, design="is_query")

#     results = adata_all_latent.uns["nhood_adata"].obs
#     adata_latent.obsm["logFC"] = results["logFC"]
#     return results["logFC"], results["PValue"], results["SpatialFDR"]

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
    

def centroid_map(adata_ref, adata_query, centroids, cell_type_key, n_neighbors=15):
    combined_emb = ad.concat([adata_query, adata_ref])
    combined_emb.obs["centroid"] = "non centroid"

    centroid_adata = ad.AnnData(centroids)
    centroid_adata.obs["centroid"] = "centroid"

    combined_emb_with_centroids = ad.concat([combined_emb, centroid_adata], join="outer", fill_value="")
    sc.pp.neighbors(combined_emb_with_centroids, n_neighbors=n_neighbors)
    sc.tl.umap(combined_emb_with_centroids)

    fig,ax=plt.subplots(figsize=(3,3))
    location_cells = combined_emb_with_centroids[combined_emb_with_centroids.obs["centroid"] == "centroid"].obsm["X_umap"]
    centroid_x=location_cells[:,0]
    centroid_y=location_cells[:,1]
    combined_emb_with_centroids = combined_emb_with_centroids[combined_emb_with_centroids.obs[cell_type_key].notna()]
    sc.pl.umap(combined_emb_with_centroids, color=[cell_type_key],ax=ax,show=False)

    size=0.50

    for (x, y) in zip(centroid_x, centroid_y):
        circle = plt.Circle((x, y), size, color='black', fill=True)
        ax.add_patch(circle)

    l1=ax.get_legend()
    l1.set_title('Cell type')
    # Make a new Legend for the centroids
    l2=ax.legend(handles=[Line2D([0],[0],marker='o', color='purple',  markerfacecolor='none', 
                            markersize=12,markeredgecolor='k',lw=0,label='Centroid')], 
            frameon=False, bbox_to_anchor=(3,1),title='Cluster')

    _=plt.gca().add_artist(l1)


def map_right_wrong(original_labels, predicted_labels, cell_type_key, n_neighbors):
    color_map = {True: 'green', False: 'red'}
    original_labels_array = original_labels.obs[cell_type_key].values

    # Check if the labels in predicted_labels match the original_labels
    predicted_labels_array = predicted_labels.obs[cell_type_key].values
    match = predicted_labels_array == original_labels_array

    # Create a new column in predicted_labels indicating whether labels match or not
    predicted_labels.obs['right_prediction'] = match
    sc.pp.neighbors(predicted_labels, n_neighbors=n_neighbors)
    sc.tl.umap(predicted_labels)
    sc.pl.umap(predicted_labels,
                color=['right_prediction'],
               color_map = color_map,
               frameon=False,
               wspace=0.6)

def benchmark_uncertainty(uncertainty_list, x_labels, dataset_name):
    fig, ax = plt.subplots()
    ax.set_title(f'Benchmarking {dataset_name}')
    ax.boxplot(uncertainty_list)
    ax.set_xticklabels(x_labels)
    plt.show()
    return