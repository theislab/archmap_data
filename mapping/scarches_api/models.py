import scarches
import scanpy
import pandas
import numpy
import tempfile
import os

import utils.parameters as parameters
from utils.utils import get_from_config
from utils.utils import fetch_file_from_s3
from utils.utils import read_h5ad_file_from_s3

from process.processing import Preprocess
from process.processing import Postprocess

from uncert.uncert_metric import classification_uncert_euclidean
from uncert.uncert_metric import classification_uncert_mahalanobis

from classifiers.classifiers import Classifiers

class ArchmapBaseModel():
    def __init__(self, configuration) -> None:
        self._configuration = configuration

        self._atlas = get_from_config(configuration=configuration, key=parameters.ATLAS)
        self._model_type = get_from_config(configuration=configuration, key=parameters.MODEL)
        #self._model_path = "assets/" + self._model_type + "/" + self._atlas + "/"
        self._model_path = get_from_config(configuration=configuration, key=parameters.PRETRAINED_MODEL_PATH)
        self._reference_adata_path = get_from_config(configuration=configuration, key=parameters.REFERENCE_DATA_PATH)
        self._query_adata_path = get_from_config(configuration=configuration, key=parameters.QUERY_DATA_PATH)
        self._max_epochs = 1
        self._use_gpu = get_from_config(configuration=configuration, key=parameters.USE_GPU)

        #Has to be empty for the load_query_data function to work properly (looking for "model.pt")
        self._temp_model_path = ""
        self._model = None
        self._temp_clf_model_path = None
        self._temp_clf_encoding_path = None
        self._query_adata = None
        self._reference_adata = None
        self._combined_adata = None

        #Load and process required data
        self._acquire_data()

        #Set respective keys coherent to chosen atlas
        self._cell_type_key = None
        self._batch_key = None
        self._unlabeled_key = None

        self._cell_type_key, self._batch_key, self._unlabeled_key = Preprocess.get_keys(self._atlas, self._query_adata)        

        self._clf_native = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("Native")
        self._clf_xgb = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("XGBoost")
        self._clf_knn = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("kNN")
        self._clf_model_path = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_PATH)
        self._clf_encoding_path = get_from_config(configuration=configuration, key=parameters.ENCODING_PATH)

    def run(self):
        self._map_query()
        self._eval_mapping()
        self._transfer_labels()
        self._concat_data()
        self._save_data()
        self._cleanup()

    def _map_query(self):
        #Map the query onto reference
        self._model.train(
            max_epochs=self._max_epochs,
            plan_kwargs=dict(weight_decay=0.0),
            check_val_every_n_epoch=10,
            use_gpu=self._use_gpu
        )

        #Save out the latent representation
        self._compute_latent_representation(explicit_representation=self._reference_adata)
        self._compute_latent_representation(explicit_representation=self._query_adata)

    def _acquire_data(self):
        #Download query and reference from GCP
        print("Download reference")
        self._reference_adata = read_h5ad_file_from_s3(self._reference_adata_path)
        self._reference_adata.obs["type"] = "reference"

        print("Download query")
        self._query_adata = read_h5ad_file_from_s3(self._query_adata_path) 
        self._query_adata.obs["type"] = "query"

        # #Check if cell_type_key exists in query
        # if self._cell_type_key not in self._query_adata.obs.columns:
        #     self._query_adata.obs[self._cell_type_key] = "Unlabeled"

        # #Check if batch_key exists in query
        # if self._batch_key not in self._query_adata.obs.columns:
        #     self._query_adata.obs[self._batch_key] = "query_batch"

        #Store counts in layer if not stored already (atlas should already have counts stored in layers)
        if "counts" not in self._query_adata.layers.keys():
            self._query_adata.layers['counts'] = self._query_adata.X

        #Convert bool to categorical to avoid write error during concatenation
        Preprocess.bool_to_categorical(self._reference_adata)
        Preprocess.bool_to_categorical(self._query_adata)

        #Download model from GCP
        fetch_file_from_s3(self._model_path, "./model.pt")




        #Remove later - for testing only
        # self._reference_adata = scanpy.pp.subsample(self._reference_adata, 0.1, copy=True)

    def _eval_mapping(self):
        #Create AnnData objects off the latent representation
        query_latent = scanpy.AnnData(self._query_adata.obsm["latent_rep"])
        reference_latent = scanpy.AnnData(self._reference_adata.obsm["latent_rep"])
        reference_latent.obs = self._reference_adata.obs

        #Calculate mapping uncertainty and write into .obs
        classification_uncert_euclidean(self._configuration, reference_latent, query_latent, self._query_adata, "X", self._cell_type_key, False)
        classification_uncert_mahalanobis(self._configuration, reference_latent, query_latent, self._query_adata, "X", self._cell_type_key, False)

    def _transfer_labels(self):
        #Return if no classifier chosen
        if not self._clf_xgb and not self._clf_knn and not self._clf_native:
            return

        #If native classifier chosen set model as classifier
        if self._clf_native:
            clf = Classifiers(self._clf_xgb, self._clf_knn, self._model.__class__.__name__)
        else:
            clf = Classifiers(self._clf_xgb, self._clf_knn, None, self._model.__class__.__name__)

        #Download classifier and encoding from GCP
        if self._clf_xgb:
            self._temp_clf_model_path = tempfile.mktemp(suffix=".ubj")
            fetch_file_from_s3(self._clf_model_path, self._temp_clf_model_path)
        elif self._clf_knn:
            self._temp_clf_model_path = tempfile.mktemp(suffix=".pickle")
            fetch_file_from_s3(self._clf_model_path, self._temp_clf_model_path)

        self._temp_clf_encoding_path = tempfile.mktemp(suffix=".pickle")
        fetch_file_from_s3(self._clf_encoding_path, self._temp_clf_encoding_path)

        query_latent = scanpy.AnnData(self._query_adata.obsm["latent_rep"])

        #Compute label transfer and save to respective .obs
        clf.predict_labels(self._query_adata, query_latent, self._temp_clf_model_path, self._temp_clf_encoding_path)

        return clf, query_latent

    def _concat_data(self):
        self._reference_adata.X = self._reference_adata.layers["counts"]

        #Added because concat_on_disk only allows inner joins
        self._reference_adata.obs[self._cell_type_key + '_uncertainty_euclidean'] = pandas.Series(dtype="float32")
        self._reference_adata.obs['uncertainty_mahalanobis'] = pandas.Series(dtype="float32")
        self._reference_adata.obs['prediction_xgb'] = pandas.Series(dtype="category")
        self._reference_adata.obs['prediction_knn'] = pandas.Series(dtype="category")

        #Added because concat_on_disk only allows csr concat
        if self._query_adata.X.format == "csc":
            #self._query.X = csr_matrix(self._query.X)
            self._query_adata.X = self._query_adata.X.tocsr()

        if self._reference_adata.X.format == "csc":
            #self._reference.X = csr_matrix(self._reference.X)
            self._reference_adata.X = self._reference_adata.X.tocsr()

        #Create temp files on disk
        temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

        #Write data to temp files
        scanpy.write(temp_reference.name, self._reference_adata)
        scanpy.write(temp_query.name, self._query_adata)

        del self._reference_adata
        del self._query_adata

        #Concatenate on disk to save memory
        from anndata import experimental
        experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)

        #Read concatenated data back in
        self._combined_adata = scanpy.read_h5ad(temp_combined.name)

        #Store latent representation of combined adata (query, reference)
        self._compute_latent_representation(explicit_representation=self._combined_adata)

    def _compute_latent_representation(self, explicit_representation):
        #Store latent representation of combined adata (query, reference)
        explicit_representation.obsm["latent_rep"] = self._model.get_latent_representation(explicit_representation)

    def _save_data(self):
        #Save output
        Postprocess.output(None, self._combined_adata, self._configuration)

    def _cleanup(self):
        #Remove all temp files
        os.remove(os.path.join(self._temp_model_path, "model.pt"))
        os.remove(self._temp_clf_model_path)
        os.remove(self._temp_clf_encoding_path)

class ScVI(ArchmapBaseModel):
    def _map_query(self):
        #Align genes and gene order to model
        scarches.models.SCVI.prepare_query_anndata(self._query_adata, self._temp_model_path)

        #Setup adata internals for mapping
        scarches.models.SCVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key)

        #Load scvi model with query
        model = scarches.models.SCVI.load_query_data(
            self._query_adata,
            self._temp_model_path,
            freeze_dropout=True,
        )

        self._model = model
        self._max_epochs = get_from_config(configuration=self._configuration, key=parameters.SCVI_QUERY_MAX_EPOCHS)

        super()._map_query()

    def _compute_latent_representation(self, explicit_representation):
        #Setup adata before quering model for latent representation
        scarches.models.SCVI.setup_anndata(explicit_representation, batch_key=self._batch_key)

        super()._compute_latent_representation(explicit_representation=explicit_representation)

class ScANVI(ArchmapBaseModel):
    def _map_query(self, supervised=False):
        #Align genes and gene order to model
        scarches.models.SCANVI.prepare_query_anndata(self._query_adata, self._temp_model_path)

        #Setup adata internals for mapping
        scarches.models.SCANVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key, unlabeled_category=self._unlabeled_key)

        #Load scanvi model with query
        model = scarches.models.SCANVI.load_query_data(
            self._query_adata,
            self._temp_model_path,
            freeze_dropout=True,
        )

        #Check if mapping supervised, unsupervised or semi-supervised
        if supervised:
            model._unlabeled_indices = []
            model._labeled_indices = self._query_adata.n_obs
        else:
            model._unlabeled_indices = numpy.arange(self._query_adata.n_obs)
            model._labeled_indices = []

        self._model = model
        self._max_epochs = get_from_config(configuration=self._configuration, key=parameters.SCANVI_MAX_EPOCHS_QUERY)

        super()._map_query()

    def _compute_latent_representation(self, explicit_representation):
        #Setup adata before quering model for latent representation
        scarches.models.SCANVI.setup_anndata(explicit_representation, labels_key=self._cell_type_key, unlabeled_category="Unlabeled", batch_key=self._batch_key)

        super()._compute_latent_representation(explicit_representation=explicit_representation)

class ScPoli(ArchmapBaseModel):
    def __init__(self, configuration) -> None:
        self._configuration = configuration
        #self._query = query
        #self._temp_model_path = model_path
        self._model = None

    def _map_query(self, query):
        model_path = get_from_config(self._configuration, parameters.RESULTING_MODEL_PATH)

        scpoli_query = scarches.models.scPoli.load_query_data(
            adata=query,
            reference_model=self._temp_model_path,
            labeled_indices=[]
        )

        scpoli_query.train(
            n_epochs=50,
            pretraining_epochs=40,
            eta=10
        )

        self._model = scpoli_query

    def label_transfer(self, query):
        results_dict = self._model.classify(query, scale_uncertainties=True)

    def sample_embeddings(self):
        sample_embedding = self._model.get_conditional_embeddings()

        from sklearn.decomposition import KernelPCA
        import matplotlib.pyplot as plt
        import seaborn as sns

        pca = KernelPCA(n_components=2, kernel='linear')
        emb_pca = pca.fit_transform(sample_embedding.X)
        conditions = self._model.conditions_['study']
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.scatterplot(x=emb_pca[:, 0], y=emb_pca[:, 1], hue=conditions, ax=ax)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        for i, c in enumerate(conditions):
            ax.plot([0, emb_pca[i, 0]], [0, emb_pca[i, 1]])
            ax.text(emb_pca[i, 0], emb_pca[i, 1], c)
        sns.despine()