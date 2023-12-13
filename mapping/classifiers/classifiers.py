# import scib_metrics

# from scib_metrics.benchmark import Benchmarker
# from scib_metrics.benchmark import BioConservation
# from scib_metrics.benchmark import BatchCorrection

import scanpy
import anndata
import numpy
import pandas as pd
import scvi
# import scanorama

from process.processing import Preprocess

from xgboost import XGBClassifier

import scarches as sca
from scvi.model.base import _utils
import pickle
import gzip
# import scib.preprocessing as pp
# import scib.integration as ig
# import scib.metrics as me

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import KNeighborsClassifier

class Classifiers:
    def __init__(self, classifier_xgb=False, classifier_knn=False, classifier_scanvi=sca.models.SCANVI, model_type="scVI") -> None:
        self.__classifier_xgb = classifier_xgb
        self.__classifier_knn = classifier_knn
        self.__classifier_scanvi = classifier_scanvi
        self.__model_type = model_type

    '''
    Parameters
    ----------
    query: adata to save labels to
    query_latent: adata to read .X from for label prediction
    '''
    def predict_labels(self, query=scanpy.AnnData(), query_latent=scanpy.AnnData(), classifier_path="/path/to/classifier", encoding_path="/path/to/encoding"):
        le = LabelEncoder()

        if self.__classifier_xgb:
            with open(encoding_path, "rb") as file:
                le = pickle.load(file)

            xgb_model = XGBClassifier()
            xgb_model.load_model(classifier_path)
            
            query.obs["prediction_xgb"] = le.inverse_transform(xgb_model.predict(query_latent.X))

        if self.__classifier_knn:
            with open(encoding_path, "rb") as file:
                le = pickle.load(file)

            with open(classifier_path, "rb") as file:
                knn_model = pickle.load(file)

            query.obs["prediction_knn"] = le.inverse_transform(knn_model.predict(query_latent.X))

        if self.__classifier_scanvi is not None:
            #scanvi_model = sca.models.SCANVI.load_query_data(adata=query, reference_model="assets/scANVI/" + self.__atlas_name)

            query.obs["prediction_scanvi"] = self.__classifier_scanvi.predict(query)

    def create_classifier(self, adata, latent_rep=False, model_path="", label_key="CellType", classifier_directory="path/to/classifier_output"):
        train_data = Classifiers.__get_train_data(
            self,
            adata=adata,
            latent_rep=latent_rep,
            model_path=model_path
        )
        
        X_train, X_test, y_train, y_test = Classifiers.__split_train_data(
            self,
            train_data=train_data,
            input_adata=adata,
            label_key=label_key,
            classifier_directory=classifier_directory
        )
        
        xgbc, knnc = Classifiers.__train_classifier(
            self,
            X_train=X_train,
            y_train=y_train,
            xgb=self.__classifier_xgb,
            kNN=self.__classifier_knn,
            classifier_directory=classifier_directory
        )
        
        reports = Classifiers.eval_classifier(
            self,
            X_test=X_test,
            y_test=y_test,
            xgbc=xgbc,
            knnc=knnc,
            classifier_directory=classifier_directory
        )
        
        Classifiers.__plot_eval_metrics(
            self,
            reports,
            classifier_directory=classifier_directory
        )

        Classifiers.__save_eval_metrics_csv(
            self,
            reports,
            classifier_directory=classifier_directory
        )

    def __get_train_data(self, adata, latent_rep=True, model_path=None):
        if latent_rep:
            latent_rep = adata
        else:
            # var_names = _utils._load_saved_files(model_path, False, None,  "cpu")[1]
            # adata_subset = adata[:,var_names].copy()

            # Preprocess.conform_vars(model_path=model_path, adata=adata)

            # scvi.model.SCVI.setup_anndata(adata_subset)

            if self.__model_type == "scVI":
                model = scvi.model.SCVI.load(model_path, adata)
            elif self.__model_type == "scANVI":
                model = scvi.model.SCANVI.load(model_path, adata)
            else:
                raise Exception("Choose model type 'scVI' or 'scANVI'")

            latent_rep = scanpy.AnnData(model.get_latent_representation(), adata.obs)

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
    def __split_train_data(self, train_data, input_adata, label_key, classifier_directory):
        train_data['cell_type'] = input_adata.obs[label_key]

        #Enable if at least one class has only 1 sample -> Error in stratification for validation set
        train_data = train_data.groupby('cell_type').filter(lambda x: len(x) > 1)

        le = LabelEncoder()
        le.fit(train_data["cell_type"])
        train_data['cell_type'] = le.transform(train_data["cell_type"])        

        X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns='cell_type'), train_data['cell_type'], test_size=0.2, random_state=42, stratify=train_data['cell_type'])

        #Save label encoder
        with open(classifier_directory + "/classifier_encoding.pickle", "wb") as file:
                        pickle.dump(le, file, pickle.HIGHEST_PROTOCOL)

        return X_train, X_test, y_train, y_test
    
    def __train_classifier(self, X_train, y_train, xgb=True, kNN=True, classifier_directory="path/to/classifier"):
        xgbc = None
        knnc = None

        if xgb:
            xgbc = XGBClassifier(tree_method = "hist", objective = 'multi:softprob', verbosity=3)
            xgbc.fit(X_train, y_train)

            #Save classifier
            xgbc.save_model(classifier_directory + "/classifier_xgb.ubj")

        if kNN:
            knnc = KNeighborsClassifier()
            knnc.fit(X_train, y_train)

            #Save classifier
            with open(classifier_directory + "/classifier_knn.pickle", "wb") as file:
                pickle.dump(knnc, file, pickle.HIGHEST_PROTOCOL)

        return xgbc, knnc
    
    def eval_classifier(self, X_test, y_test, xgbc=XGBClassifier, knnc=KNeighborsClassifier, classifier_directory="path/to/classifier"):
        reports = {}

        #Load label encoder to get classes with real names in report
        le = LabelEncoder()

        with open(classifier_directory + "/classifier_encoding.pickle", "rb") as file:
                le = pickle.load(file)
        
        if xgbc is not None:
            preds = xgbc.predict(X_test)
        
            xgbc_report = Classifiers.__eval_classification_report(le.inverse_transform(y_test), le.inverse_transform(preds))

            reports["xgb"] = xgbc_report

            print("XGBoost classifier report:")
            print(xgbc_report)

        if knnc is not None:
            preds = knnc.predict(X_test)
        
            knnc_report = Classifiers.__eval_classification_report(le.inverse_transform(y_test), le.inverse_transform(preds))

            reports["knn"] = knnc_report
            
            print("kNN classifier report:")
            print(knnc_report)

        if self.__classifier_scanvi is not None:
            preds = self.__classifier_scanvi.predict(self.__classifier_scanvi, X_test)

            scanvic_report = Classifiers.__eval_classification_report(le.inverse_transform(y_test), preds)

            reports["scanvi"] = scanvic_report
            
            print("scanVI classifier report:")
            print(scanvic_report)

        return reports

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

    def __plot_eval_metrics(self, reports, classifier_directory):
        import seaborn
        import matplotlib.pyplot as plt

        for key in reports:
            #Drop "support" from classification report
            reports[key] = reports[key].drop("support", axis=1)

            #Calculate size of plot depending on rows and columns of report
            size_y = len(reports[key].index) * 0.5
            size_x = (len(reports[key].columns) + 2) * 2.5

            #Set figure size, title and remove grid
            plt.figure(figsize=(size_x, size_y), layout="tight")
            plt.grid(False)
            plt.title('Classification report: ' + key);

            seaborn.heatmap(reports[key], cmap="viridis", annot=True)   

            plt.savefig(classifier_directory + "/classifier_" + key + "_report.png")

    def __save_eval_metrics_csv(self, reports, classifier_directory):
        for key in reports:
            out = pd.DataFrame(reports[key])

            out.to_csv(classifier_directory + "/classifier_" + key + "_report.csv")

if __name__ == "__main__":
    adata = scanpy.read_h5ad("scEiaD_all_anndata_mini_ref.h5ad")
    # embedding = scanpy.read_h5ad("HLCA_emb_and_metadata.h5ad")

    # Classifiers(adata, False, "", "CellType", True, True)

    import scipy

    # adata.X = scipy.sparse.csr_matrix(adata.X)
    # scanpy.pp.log1p(adata)
    # scanpy.pp.filter_genes(adata, min_cells=1)
    scanpy.pp.highly_variable_genes(adata)
    scanpy.tl.pca(adata, n_comps=30, use_highly_variable=True)

    adata = adata[:, adata.var.highly_variable].copy()

    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    adata.obsm["scVI"] = adata.obsm["X_scvi"]


    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="CellType",
        embedding_obsm_keys=["Unintegrated", "scVI"],
        n_jobs=2,
        batch_correction_metrics=BatchCorrection(True, True, True, True, False)
    )
    bm.benchmark()

    bm.plot_results_table(save_dir="")
