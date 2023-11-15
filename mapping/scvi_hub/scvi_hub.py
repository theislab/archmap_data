import scarches_api.utils.utils as utils
import scarches_api.utils.parameters as parameters
from huggingface_hub import hf_hub_download, HfApi, ModelFilter, snapshot_download, login
from anndata import experimental
import json
import scanpy
import scvi
import gdown
import os
import tempfile

from process.processing import Postprocess

class ScviHub:
    def __init__(self, configuration) -> None:
        self.__configuration = configuration
        self.__training_data_url = None
        self.__model_parent_module = None
        self.__model_cls_name = None
        self.__batch_key = None
        self.__labels_key = None

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

        model = None

        #Conform vars to model for query and reference
        if(self.__model_cls_name == "SCVI"):
            scvi.model.SCVI.prepare_query_anndata(reference, "../scvi_hub/model/")
            scvi.model.SCVI.setup_anndata(reference)

            scvi.model.SCVI.prepare_query_anndata(query, "../scvi_hub/model/")
            scvi.model.SCVI.setup_anndata(query)

            model = scvi.model.SCVI.load_query_data(
                        query,
                        "../scvi_hub/model/",
                        freeze_dropout=True,
                    )
        
        if(self.__model_cls_name == "SCANVI"):
            scvi.model.SCANVI.prepare_query_anndata(reference, "../scvi_hub/model/")
            scvi.model.SCANVI.setup_anndata(reference, labels_key=self.__labels_key, unlabeled_category="Unlabeled")

            scvi.model.SCANVI.prepare_query_anndata(query, "../scvi_hub/model/")
            scvi.model.SCANVI.setup_anndata(query, labels_key=self.__labels_key, unlabeled_category="Unlabeled")

            model = scvi.model.SCANVI.load_query_data(
                        query,
                        "../scvi_hub/model/",
                        freeze_dropout=True,
                    )

        model.train(
            max_epochs=10,
            plan_kwargs=dict(weight_decay=0.0),
            check_val_every_n_epoch=10,
            use_gpu=False
        )

        #Query model and store respective latent representation
        reference.obsm["latent_rep"] = model.get_latent_representation(reference)
        query.obsm["latent_rep"] = model.get_latent_representation(query)

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

    def __download_data(self):
        scvi_hub_id = utils.get_from_config(self.__configuration, parameters.SCVI_HUB_ID)
        # metadata_path = utils.get_from_config(self.__configuration, parameters.META_DATA_PATH)
        query_path = utils.get_from_config(self.__configuration, parameters.QUERY_DATA_PATH)

        cache_path = "../scvi_hub/cache/"
        download_path = "../scvi_hub/download/"
        folder_path = snapshot_download(repo_id=scvi_hub_id, allow_patterns=["*.h5ad","*.pt","*.json","*.md"], cache_dir=cache_path, local_dir=download_path)

        self.__read_metadata()

        gdown.download(self.__training_data_url, "../scvi_hub/atlas/atlas.h5ad")
        utils.fetch_file_from_s3(query_path, "../scvi_hub/query/query.h5ad")
        os.replace("../scvi_hub/download/model.pt", "../scvi_hub/model/model.pt")

    def __read_metadata(self):
        f = open("../scvi_hub/download/_scvi_required_metadata.json")
        metadata = json.load(f)
        
        self.__training_data_url = metadata.pop("training_data_url")
        self.__model_parent_module = metadata.pop("model_parent_module")
        self.__model_cls_name = metadata.pop("model_cls_name")

        if(self.__model_cls_name != "SCVI"):
            self.__batch_key = utils.get_from_config(self.__configuration, utils.parameters.SCVI_HUB_ARGS).pop("batch_key")
            self.__labels_key = utils.get_from_config(self.__configuration, utils.parameters.SCVI_HUB_ARGS).pop("labels_key")