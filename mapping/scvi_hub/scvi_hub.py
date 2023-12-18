import scarches

import scarches_api.utils.utils as utils
import scarches_api.utils.parameters as parameters
from huggingface_hub import hf_hub_download, HfApi, ModelFilter, snapshot_download, login
from anndata import experimental
from pathlib import Path
import json
import scanpy
import scvi
import gdown
import os
import tempfile

from process.processing import Postprocess

from classifiers.classifiers import Classifiers

class ScviHub:
    def __init__(self, configuration) -> None:
        self.__configuration = configuration
        self.__training_data_url = None
        self.__model_parent_module = None
        self.__model_cls_name = None
        self.__model = None
        self.__batch_key = None
        self.__labels_key = None

        #Classifier setup
        self._clf_native = None
        self._clf_xgb = None
        self._clf_knn = None

        self.__download_data()

    def map_query(self):
        reference = scanpy.read_h5ad("../scvi_hub/atlas/atlas.h5ad")
        query = scanpy.read_h5ad("../scvi_hub/query/query.h5ad")

        del reference.obsm
        del reference.uns
        del query.obsm
        del query.uns

        reference.obs["type"] = "reference"
        query.obs["type"] = "query"

        #Conform vars to model for query and reference
        if(self.__model_cls_name == "SCVI"):
            scvi.model.SCVI.prepare_query_anndata(reference, "../scvi_hub/model/")
            scvi.model.SCVI.setup_anndata(reference)

            try:
                scvi.model.SCVI.prepare_query_anndata(query, "../scvi_hub/model/")
            except:
                var_names = scvi.model.SCVI.prepare_query_anndata(query, "../scvi_hub/model/", return_reference_var_names=True)
                raise ValueError(f"Make sure query var_names match the one of the chosen atlas: \n {var_names}")
            scvi.model.SCVI.setup_anndata(query)

            self.__model = scvi.model.SCVI.load_query_data(
                        query,
                        "../scvi_hub/model/",
                        freeze_dropout=True,
                    )
        
        if(self.__model_cls_name == "SCANVI"):
            scvi.model.SCANVI.prepare_query_anndata(reference, "../scvi_hub/model/")
            scvi.model.SCANVI.setup_anndata(reference, labels_key=self.__labels_key, unlabeled_category="Unlabeled")

            try:
                scvi.model.SCANVI.prepare_query_anndata(query, "../scvi_hub/model/")
            except:
                var_names = scvi.model.SCANVI.prepare_query_anndata(query, "../scvi_hub/model/", return_reference_var_names=True)
                raise ValueError("Make sure query var_names match the one of the chosen atlas: " + var_names)
            scvi.model.SCANVI.setup_anndata(query, labels_key=self.__labels_key, unlabeled_category="Unlabeled")

            self.__model = scvi.model.SCANVI.load_query_data(
                        query,
                        "../scvi_hub/model/",
                        freeze_dropout=True,
                    )

        self.__model.train(
            max_epochs=10,
            plan_kwargs=dict(weight_decay=0.0),
            check_val_every_n_epoch=10,
            use_gpu=False
        )

        #Query model and store respective latent representation
        reference.obsm["latent_rep"] = self.__model.get_latent_representation(reference)
        query.obsm["latent_rep"] = self.__model.get_latent_representation(query)

        #Set up classification
        #self.classification(atlas=reference, query=query, query_latent=scanpy.AnnData(query.obsm["latent_rep"]))

        #Concatenate reference and query
        temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

        scanpy.write(temp_reference.name, reference)
        scanpy.write(temp_query.name, query)

        del reference
        del query
        
        experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)
        combined_adata = scanpy.read_h5ad(temp_combined.name)

        #Save output
        Postprocess.output(None, combined_adata, self.__configuration)

        #Remove created tmp files
        temp_reference.close()
        temp_query.close()
        temp_combined.close()

        #Clean up local directories
        self.__cleanup()

    def classification(self, atlas, query, query_latent):
        #Create directory
        cache_path = Path("../scvi_hub/classifiers/")
        cache_path.mkdir(parents=True, exist_ok=True)

        #Set class for respective cls_name
        if self.__model_cls_name == "SCVI":
            model_class = scarches.models.SCVI.__class__
        if self.__model_cls_name == "SCANVI":
            model_class = scarches.models.SCANVI.__class__

        #Initialize and create classifier
        if self._clf_native:
            clf = Classifiers(self._clf_xgb, self._clf_knn, self.__model, model_class)
        else:
            clf = Classifiers(self._clf_xgb, self._clf_knn, None, model_class)

        clf.create_classifier(adata=atlas, latent_rep=False, model_path="../scvi_hub/model", label_key=self.__labels_key, classifier_directory="../scvi_hub/classifiers")
        
        #Predict the labels
        if self._clf_xgb:
            clf.predict_labels(query=query, query_latent=query_latent, classifier_path="../scvi_hub/classifiers/classifier_xgb.ubj", encoding_path="../scvi_hub/classifiers/classifier_encoding.pickle")
        elif self._clf_knn:
            clf.predict_labels(query=query, query_latent=query_latent, classifier_path="../scvi_hub/classifiers/classifier_knn.pickle", encoding_path="../scvi_hub/classifiers/classifier_encoding.pickle")
        else:
            clf.predict_labels(query=query, query_latent=query_latent, classifier_path=None, encoding_path=None)

    def __download_data(self):
        scvi_hub_id = utils.get_from_config(self.__configuration, parameters.SCVI_HUB_ID)
        # metadata_path = utils.get_from_config(self.__configuration, parameters.META_DATA_PATH)
        query_download_path = utils.get_from_config(self.__configuration, parameters.QUERY_DATA_PATH)

        #Create directories
        cache_path = Path("../scvi_hub/cache/")
        cache_path.mkdir(parents=True, exist_ok=True)
        download_path = Path("../scvi_hub/download/")
        download_path.mkdir(parents=True, exist_ok=True)
        atlas_path = Path("../scvi_hub/atlas/")
        atlas_path.mkdir(parents=True, exist_ok=True)
        query_path = Path("../scvi_hub/query/")
        query_path.mkdir(parents=True, exist_ok=True)
        metadata_path = Path("../scvi_hub/metadata/")
        metadata_path.mkdir(parents=True, exist_ok=True)
        model_path = Path("../scvi_hub/model/")
        model_path.mkdir(parents=True, exist_ok=True)

        folder_path = snapshot_download(repo_id=scvi_hub_id, allow_patterns=["*.h5ad","*.pt","*.json","*.md"], cache_dir="../scvi_hub/cache/", local_dir="../scvi_hub/download/")

        self.__read_metadata()

        #Try to download if training_data_url exists in metadata else use the smaller existing one
        if self.__training_data_url is not None:
            gdown.download(self.__training_data_url, "../scvi_hub/atlas/atlas.h5ad")
        else:
            os.replace("../scvi_hub/download/adata.h5ad", "../scvi_hub/atlas/atlas.h5ad")
        utils.fetch_file_from_s3(query_download_path, "../scvi_hub/query/query.h5ad")
        os.replace("../scvi_hub/download/model.pt", "../scvi_hub/model/model.pt")

    def __read_metadata(self):
        f = open("../scvi_hub/download/_scvi_required_metadata.json")
        metadata = json.load(f)
        
        self.__training_data_url = metadata.pop("training_data_url")
        self.__model_parent_module = metadata.pop("model_parent_module")
        self.__model_cls_name = metadata.pop("model_cls_name")

        self.__batch_key = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.SCVI_HUB_ARGS).pop("batch_key")
        self.__labels_key = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.SCVI_HUB_ARGS).pop("labels_key")

        self._clf_native = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.CLASSIFIER_TYPE).pop("Native")
        self._clf_xgb = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.CLASSIFIER_TYPE).pop("XGBoost")
        self._clf_knn = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.CLASSIFIER_TYPE).pop("kNN")

    def __cleanup(self):
        import shutil

        #Remove directories
        shutil.rmtree("../scvi_hub/cache/")
        shutil.rmtree("../scvi_hub/download/")
        shutil.rmtree("../scvi_hub/atlas/")
        shutil.rmtree("../scvi_hub/query/")
        shutil.rmtree("../scvi_hub/metadata/")
        shutil.rmtree("../scvi_hub/model/")

        shutil.rmtree("../scvi_hub/classifiers/")