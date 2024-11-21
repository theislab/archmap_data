import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scarches as sca
import pickle
import os


from sklearn.mixture import GaussianMixture

def train_mahalanobis(atlas, adata_ref, embedding_name, cell_type_key, pretrained=True):


    num_clusters = adata_ref.obs[cell_type_key].nunique()
    print(num_clusters)

    train_emb = adata_ref.obsm[embedding_name]

    #Required too much RAM
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(train_emb)

    #Less RAM alternative
    # kmeans = KMeans(n_clusters=num_clusters)
    # kmeans.fit(train_emb)

    #Save or return model
    if pretrained:
        
        with open("models_uncert/" + atlas + "/" + cell_type_key + "_mahalanobis_distance.pickle", "wb") as file:
            pickle.dump(gmm, file, pickle.HIGHEST_PROTOCOL)
    else:
        return gmm
    
def train_euclidian(atlas, adata_ref, embedding_name, pretrained =True, n_neighbors = 15):

    trainer = sca.utils.weighted_knn_trainer(
    adata_ref,
    embedding_name,
    n_neighbors = n_neighbors
    )

    #Save model
    if pretrained:
        with open("models_uncert/" + atlas + "/" + "euclidian_distance.pickle", "wb") as file:
            pickle.dump(trainer, file, pickle.HIGHEST_PROTOCOL)
    else:
        return trainer


def main(atlas, adata_ref, cell_type_key_list=None, is_scpoli = False):

    if is_scpoli:
        embedding_name = "X_latent_qzm_scpoli"
    else:
        embedding_name = "X_latent_qzm"

    train_euclidian(atlas, adata_ref, embedding_name)

    if isinstance(cell_type_key_list,str):
        cell_type_key_list = [cell_type_key_list]

    for cell_type_key in cell_type_key_list:
        print(cell_type_key)
        train_mahalanobis(atlas, adata_ref, embedding_name, cell_type_key)

    

if __name__ == "__main__":


    
    # cell_type_key_list = ['annot_level_1',
    #                     'annot_level_2',
    #                     'annot_level_3_rev2',
    #                     'annot_level_4_rev2',
    #                     'annot_region_rev2',
    #                     'annot_ntt_rev2',]
    
    atlas_dict = {"hlca": "scanvi_label",
                  #"gb": "CellID",
                #   "fetal_immune":"celltype_annotation",
                #   "nsclc": "cell_type",
                #   "retina": "CellType",
                #   "pancreas_scanvi":"cell_type",
                #   "hypomap": "Author_CellType",
                  }
    atlas_dict_scpoli = {
                #   "hlca_retrained": "ann_finest_level",           
                #   "pancreas_scpoli":"cell_type",
                #   "pbmc":"cell_type_for_integration",
                #   "hnoca":"annot_level_2",
                #   "heoca": "cell_type"
     
                }

    for atlas, cell_type_key_list in atlas_dict_scpoli.items():

        directory = "models_uncert/" + atlas + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        adata_ref = sc.read(f"data/{atlas}.h5ad")
        main(atlas, adata_ref, cell_type_key_list, is_scpoli = True)

    for atlas, cell_type_key_list in atlas_dict.items():

        directory = "models_uncert/" + atlas + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        adata_ref = sc.read(f"data/{atlas}.h5ad")
        main(atlas, adata_ref, cell_type_key_list, is_scpoli = False)