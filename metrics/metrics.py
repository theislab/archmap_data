import scib_metrics

from scib_metrics.benchmark import Benchmarker
from scib_metrics.benchmark import BioConservation
from scib_metrics.benchmark import BatchCorrection

import scanpy
import anndata
import numpy
import scvi
import scanorama
import xgboost as xgb
import scarches as sca
import pickle
import scib.preprocessing as pp
import scib.integration as ig
import scib.metrics as me

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def metrics():
    ### Theislab scib
    # adata = scanpy.read_h5ad("atlas_626ea3311d7d1a27de465b64_data.h5ad")

    # pp.normalize(adata)
    # # pp.scale_batch(adata, "batch")
    # # pp.hvg_batch(adata, "batch")
    # # pp.hvg_intersect(adata, "batch")
    # # pp.reduce_data(adata, "batch")

    # scanpy.pp.normalize_total()

    # print(adata)


    # integration = reference_integration(adata)
    # print(integration["combat"])


    # metrics_space = ["feature", "embedding", "kNN_graph"]

    # if(metrics_space == "feature"):
    #     return
    # elif(metrics_space == "embedding"):
    #     return
    #     #embedding_space(unintegraded, integrated);
    # elif(metrics_space == "kNN_graph"):
    #     return

    ### Yoseflab scib
    '''
    scib.metrics.metrics_fast include:
    Biological conservation:
    1. HVG overlap
    2. Cell type ASW
    3. Isolated labels ASW

    Batch correction:
    1. Graph connectivity
    2. Batch ASW
    3. Principal component regression
    '''

    adata = scanpy.read("data/atlases/Full_obj_log_counts_soupx_v2.h5ad")
    query_adata_emb = scanpy.read("Duong_lungMAP_unpubl_emb_LCAv2.h5ad")
    source_adata_emb = scanpy.read("HLCA_emb_and_metadata.h5ad")
    label_key = "ann_level_3"

    scanpy.pp.subsample(adata, 0.1)

    reference_mappability(query_adata_emb, source_adata_emb, label_key)

    integration_adata, integration_methods = reference_integration(adata, True, False, False, False)

    benchmarking(integration_adata, integration_methods)
    
    return

def embedding_space(unintegrated = anndata.AnnData(), integrated = anndata.AnnData()):
    #Reduce unintegrated adata to embedding space
    batch_key = "batch"

    pp.reduce_data(unintegrated, batch_key)

    me.metrics_fast(unintegrated, integrated, batch_key)

def reference_integration(adata, Combat = False, Scanorama = False, scVI = False, scANVI = False):
    batch_key = "study"
    labels_key = "scanvi_label"

    integration_methods = []

    scanpy.tl.pca(adata, n_comps=30)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    integration_methods.append("Unintegrated")

    if(Combat):
        adata.obsm["Combat"] = scanpy.pp.combat(adata, batch_key, inplace=False)
        integration_methods.append("Combat")

        #TODO: Error with Scanorama as Combat changing adata in place
        # adata.obsm["Combat"] = ig.combat(adata, batch_key)
        # integration_methods.append("Combat")

    if(Scanorama):
        adata.obsm["Scanorama"] = ig.scanorama(adata, batch_key).obsm["X_emb"]
        integration_methods.append("Scanorama")

    if(scVI):
        adata.obsm["scVI"] = ig.scvi(adata, batch_key).obsm["X_emb"]
        integration_methods.append("scVI")

    if(scANVI):
        # adata.obsm["scANVI"] = ig.scanvi(adata, batch_key, labels_key, max_epochs=20).obsm["X_emb"]
        adata.obsm["scANVI"] = adata.obsm["X_scanvi_emb"]

        integration_methods.append("scANVI")

    return adata, integration_methods

    # if(combat):
    #     combat_adata = ig.combat(adata, batch_key)
    # else:
    #     combat_adata = numpy.nan

    # if(scanorama):
    #     scanorama_adata = ig.scanorama(adata, batch_key)
    # else:
    #     scanorama_adata = numpy.nan

    # if(scvi):
    #     scvi_adata = ig.scvi(adata, batch_key)
    # else:
    #     scvi_adata = numpy.nan

    # if(scanvi):
    #     scanvi_adata = ig.scanvi(adata, batch_key, labels_key)
    # else:
    #     scanvi_adata = numpy.nan

    # output = {
    #     "combat": combat_adata,
    #     "scanorama": scanorama_adata,
    #     "scvi": scvi_adata,
    #     "scanvi": scanvi_adata
    # }

    # return output

def reference_mappability(query_adata_emb, source_adata_emb, label_key, XGBoost = False, kNN = True):
    if(XGBoost):
        with open("XGBoost_Encoding.pickle", "rb") as file:
            labels_encoder = pickle.load(file)

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("XGBoost_Classifier.ubj")
        

        query_adata_emb.obs["prediction"] = labels_encoder.inverse_transform(xgb_model.predict(query_adata_emb.X))

        print("Accuracy score XGBoost: ", accuracy_score(query_adata_emb.obs[label_key], query_adata_emb.obs["prediction"]))

    if(kNN):
        with open("kNN_Classifier.pickle", "rb") as file:
            knn_model = pickle.load(file)

        query_adata_emb.obs["prediction"], query_adata_emb.obs["uncertainties"] = sca.utils.weighted_knn_transfer(query_adata_emb, "X", source_adata_emb.obs, label_key, knn_model)

        print("Accuracy score kNN: ", accuracy_score(source_adata_emb.obs[label_key], query_adata_emb.obs["prediction"]))


    return

def benchmarking(integration_adata, integration_methods):
    adata = integration_adata
    batch_key = "study"
    label_key = "scanvi_label"
    embedding_obsm_keys = integration_methods
    # n_jobs = len(integration_methods)
    
    bm = Benchmarker(
        adata,
        batch_key,
        label_key,
        embedding_obsm_keys,
        BioConservation(True, True, True, True, True),
        BatchCorrection(True, True, True, True, True)        
    )

    bm.benchmark()

    bm.plot_results_table(save_dir="")

    from rich import print

    df = bm.get_results(min_max_scale=False)
    print(df)
    

if __name__ == "__main__":
    metrics()
