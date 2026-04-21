import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import scarches as sca
from classifiers.classifiers import Classifiers 
import os
import pandas as pd
from sklearn.metrics import classification_report


def main():

    cell_type_key_list=["ann_level_3"]

    queries = ["most_50", "least_50", "Meyer_2021_5prime_100"]

    for query_name in queries:

        classify(cell_type_key_list, query_name)

def classify(cell_type_key_list, query_name):

    path = "mapping/hlca_tutorial/"

    adata = sc.read(f"hlca_combined_embeddings_gene_overlap.h5ad")
    adata.obs['type'] = np.where(adata.obs['dataset'] == 'Meyer_2021_5prime', 'query', 'ref')

    for cell_type_key in cell_type_key_list: 
        if "Unknown" not in adata.obs[cell_type_key].cat.categories:
            adata.obs[cell_type_key] = adata.obs[cell_type_key].cat.add_categories(['Unknown'])
        adata.obs[cell_type_key] = adata.obs[cell_type_key].fillna("Unknown")

    reference_adata = adata[adata.obs["type"]=="ref"]
    query_adata = adata[adata.obs["type"]=="query"]


    clf_xgb=False
    clf_knn=True
    #Create AnnData objects off the latent representation
    query_latent = sc.AnnData(query_adata.obsm["X_scanvi_emb"])
    reference_latent = sc.AnnData(reference_adata.obsm["X_scanvi_emb"])
    reference_latent.obs = reference_adata.obs

    print(reference_latent)
    print(query_latent)

    model = sca.models.SCANVI.load(path+"model_hlca_scanvi/")

    clf = Classifiers(clf_xgb, clf_knn, None, model.__class__)

    for cell_type_key in cell_type_key_list:
    
        clf_encoding_path = path + query_name + "_" + cell_type_key + "/classifier_encoding.pickle"

        clf_model_path = path + query_name + "_" + cell_type_key + "/classifier_knn.pickle"

        clf.create_classifier(reference_latent, reference_adata, True, "", cell_type_key, path + query_name + "_" + cell_type_key)

        percent_unknown = clf.predict_labels(query_adata, query_latent, clf_model_path, clf_encoding_path, cell_type_key)
        clf_report = classification_report(y_true=query_adata.obs[cell_type_key], y_pred=query_adata.obs[f"{cell_type_key}_prediction_knn"], output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df.to_csv(f"{query_name}_{cell_type_key}")
        # evaluate label transfer

if __name__  == "__main__":
    main()


