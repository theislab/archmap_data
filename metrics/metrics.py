import scib_metrics

from scib_metrics.benchmark import Benchmarker
from scib_metrics.benchmark import BioConservation
from scib_metrics.benchmark import BatchCorrection

import scanpy
import anndata
import numpy
import pandas as pd
import scvi
import scanorama

from xgboost import XGBClassifier

import scarches as sca
from scvi.model.base import _utils
import pickle
import scib.preprocessing as pp
import scib.integration as ig
import scib.metrics as me

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import KNeighborsClassifier

class Classifiers:
    def __init__(self, adata, latent_rep=False, model_path="", label_key="CellType", classifier_xgb=True, classifier_knn=True) -> None:
        self.adata = adata
        self.latent_rep = latent_rep
        self.model_path = model_path
        self.label_key = label_key
        self.classifier_xgb = classifier_xgb
        self.classifier_knn = classifier_knn

        Classifiers.create_classifier(self)

    def create_classifier(self):
        train_data = Classifiers.__get_train_data(adata=self.adata, latent_rep=self.latent_rep, model_path=self.model_path)
        X_train, X_test, y_train, y_test = Classifiers.__split_train_data(train_data=train_data, input_adata=self.adata, label_key=self.label_key)
        xgbc, knnc = Classifiers.__train_classifier(X_train=X_train, y_train=y_train, xgb=self.classifier_xgb, kNN=self.classifier_xgb)
        Classifiers.eval_classifier(X_test=X_test, y_test=y_test, xgbc=xgbc, knnc=knnc)

    def __get_train_data(adata, latent_rep=True, model_path=None):
        if latent_rep:
            latent_rep = adata
        else:
            var_names = _utils._load_saved_files(model_path, False, None,  "cpu")[1]
            adata_subset = adata[:,var_names].copy()

            scvi.model.SCVI.setup_anndata(adata_subset)

            model = scvi.model.SCVI.load(model_path, adata_subset)

            latent_rep = scanpy.AnnData(model.get_latent_representation(), adata_subset.obs)

        train_data = pd.DataFrame(
            data = latent_rep.X,
            index = latent_rep.obs_names
        )

        return train_data

    '''
    Parameters
    ----------
    input_adata: adata to read the labels from
    '''
    def __split_train_data(train_data, input_adata, label_key):
        train_data['cell_type'] = input_adata.obs[label_key]

        le = LabelEncoder()
        le.fit(train_data["cell_type"])
        train_data['cell_type'] = le.transform(train_data["cell_type"])

        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns='cell_type'), train_data['cell_type'], test_size=0.2, random_state=42, stratify=train_data['cell_type'])

        return X_train, X_test, y_train, y_test
    
    def __train_classifier(X_train, y_train, xgb=True, kNN=True):
        xgbc = None
        knnc = None

        if xgb:
            xgbc = XGBClassifier(tree_method = "hist", objective = 'multi:softprob', verbosity=3)
            xgbc.fit(X_train, y_train)

        if kNN:
            knnc = KNeighborsClassifier()
            knnc.fit(X_train, y_train)

        return xgbc, knnc
    
    def eval_classifier(X_test, y_test, xgbc=XGBClassifier, knnc=KNeighborsClassifier):
        if xgbc is not None:
            preds = xgbc.predict(X_test)
        
            xgbc_report = Classifiers.__eval_classification_report(y_test, preds)

        if knnc is not None:
            preds = knnc.predict(X_test)
        
            knnc_report = Classifiers.__eval_classification_report(y_test, preds)
            

        print("XGBoost classifier report:")
        print(xgbc_report)

        print("kNN classifier report:")
        print(knnc_report)

        #TODO: Save the reports

    def __eval_classification_report(y_true, y_pred):
        clf_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()

        return clf_report_df

    def __eval_accuracy(y_true, y_pred):
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def __eval_precision(y_true, y_pred):
        return precision_score(y_true=y_true, y_pred=y_pred)

    def __eval_roc_auc(y_true, predict_proba):
        #predict_proba = xgb.XGBClassifier.predict_proba()

        return roc_auc_score(y_true=y_true, y_score=predict_proba, multi_class="ovr")


class Metrics:
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
    
def Scanorama(adata, batch_key, labels_key):
    import scanorama

    # List of adata per batch
    batch_cats = adata.obs[batch_key].cat.categories
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adata_list)

    adata.obsm["Scanorama"] = numpy.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm["Scanorama"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

if __name__ == "__main__":
    adata = scanpy.read_h5ad("NSCLC_reduced.h5ad")
    # embedding = scanpy.read_h5ad("HLCA_emb_and_metadata.h5ad")

    # Classifiers(adata, False, "", "CellType", True, True)

    import scipy

    # adata.X = scipy.sparse.csr_matrix(adata.X)
    scanpy.pp.log1p(adata)
    # scanpy.pp.filter_genes(adata, min_cells=1)
    scanpy.pp.highly_variable_genes(adata)
    scanpy.tl.pca(adata, n_comps=30, use_highly_variable=True)

    # adata = adata[:, adata.var.highly_variable]

    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    #adata.obsm["scVI"] = adata.obsm["X_scvi"]
    adata.obsm["Combat"] = scanpy.pp.combat(adata, "dataset", inplace=False)

    #scanorama = ig.scanorama(adata, "dataset")


    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="CellType",
        embedding_obsm_keys=["Unintegrated", "scVI"],
        n_jobs=2,
        batch_correction_metrics=BatchCorrection(True, True, True, True, False)
    )
    bm.benchmark()

    bm.plot_results_table(min_max_scale=False, save_dir="")
