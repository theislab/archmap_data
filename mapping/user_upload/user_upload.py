import scarches

import scarches_api.utils.utils as utils
import scarches_api.utils.parameters as parameters

from scvi.model.base import _utils

from process.processing import Preprocess

from anndata import experimental
from pathlib import Path
import scanpy
import scvi
import gdown
import os
import tempfile

class UserUpload:
    def __init__(self, configuration) -> None:
        self.__configuration = configuration

        self.__model_path = utils.get_from_config(configuration=configuration, key=parameters.PRETRAINED_MODEL_PATH)
        self.__model_type = utils.get_from_config(configuration=configuration, key=parameters.MODEL)

        self.__reference_data = utils.get_from_config(configuration=configuration, key=parameters.REFERENCE_DATA_PATH)

        self.__cell_type_key = utils.get_from_config(configuration=configuration, key=parameters.CELL_TYPE_KEY)
        self.__batch_key = utils.get_from_config(configuration=configuration, key=parameters.CONDITION_KEY)
        self.__unlabeled_key = utils.get_from_config(configuration=configuration, key=parameters.UNLABELED_KEY)

        self.__result = {
            "atlas": "verified",
            "model": "verified",
            "classifier": "verified",
            "errors": {
                "atlas": [],
                "model": [],
                "classifier": []
            }
        }

    def __load_file_tmp(self, key, filename):
        utils.delete_file(file="/app/user_upload/tmp/" + filename)
        
        file = open("/app/user_upload/tmp/" + filename, "x")
        utils.fetch_file_from_s3(key, file.name)
        
        return "/app/user_upload/tmp/" + filename

    def check_upload(self):
        local_model_path = self.__load_file_tmp(self.__model_path, "model.pt")
        local_reference_data_path = self.__load_file_tmp(self.__reference_data, "reference.h5ad")

        self.__check_model_version()
        self.__check_model_registry()

        self.__check_atlas_labels(local_model_path)
        self.__check_atlas_genes(local_model_path, local_reference_data_path)

        self.__check_classifier()

        self.__share_results()

    def __check_model_version(self):
        pass

    def __check_model_registry(self):
        pass

    def __check_atlas_labels(self, local_model_path):
        tuple = os.path.split(local_model_path)
        cell_type_key, batch_key, unlabeled_key = Preprocess.get_keys_model(tuple[0])

        #Cell type keys only relevant to scANVI models
        if self.__model_type == "scANVI":
            if self.__cell_type_key != cell_type_key:
                #raise ValueError(f"Provided cell type key \"{self.__cell_type_key}\" is different from model \"{cell_type_key}\"")
                self.__result["errors"]["atlas"].append(f"Provided cell type key \"{self.__cell_type_key}\" is different from model \"{cell_type_key}\"")
                self.__result["atlas"] = "not_verified"

        if self.__batch_key != batch_key:
            #raise ValueError(f"Provided batch key \"{self.__batch_key}\" is different from model \"{batch_key}\"")
            self.__result["errors"]["atlas"].append(f"Provided batch key \"{self.__batch_key}\" is different from model \"{batch_key}\"")
            self.__result["atlas"] = "not_verified"

        #Unlabeled keys only relevant if not None
        if unlabeled_key is not None:
            if self.__unlabeled_key != unlabeled_key:
                #raise ValueError(f"Provided unlabeled key \"{self.__unlabeled_key}\" is different from model \"{unlabeled_key}\"")
                self.__result["errors"]["atlas"].append(f"Provided unlabeled key \"{self.__unlabeled_key}\" is different from model \"{unlabeled_key}\"")
                self.__result["atlas"] = "not_verified"

    def __check_atlas_genes(self, local_model_path, local_reference_data_path):
        tuple = os.path.split(local_model_path)
        var_names = _utils._load_saved_files(tuple[0], False, None,  "cpu")[1]
        reference_data = scanpy.read_h5ad(local_reference_data_path, backed="r")

        try:
            reference_data_sub = reference_data[:,var_names]
        except:
            #raise ValueError("var_names from reference are different to the model")
            self.__result["errors"]["atlas"].append("var_names from reference are different to the model")
            self.__result["atlas"] = "not_verified"
        
    def __check_classifier(self):
        pass 

    def __share_results(self):
        webhook = utils.get_from_config(self.__configuration, parameters.WEBHOOK)

        if webhook is not None and len(webhook) > 0:
            utils.notify_backend(webhook, self.__result)