import os
import time
import psutil

startTime = time.time()

from scANVI.scANVI import compute_scANVI
from scVI.scVI import compute_scVI
from totalVI.totalVI import computeTotalVI
from utils import utils, parameters

from scvi_hub.scvi_hub import ScviHub
from models import ScANVI
from models import ScVI
from models import ScPoli

from utils.utils import read_h5ad_file_from_s3
import scanpy as sc
import tempfile
import gc
from anndata import experimental

from process.processing import Preprocess
from utils.utils import read_h5ad_file_from_s3
import scanpy as sc
import tempfile
import gc


def default_config():
    """
    returns the default config combined for all the models
    :return: dict containing all the default values
    """
    return {
        parameters.SCVI_HUB_ID: None,
        parameters.SCVI_HUB_ARGS: {},
        parameters.MODEL: 'scVI',
        parameters.MINIFICATION: True,
        parameters.CLASSIFIER_TYPE: {"XGBoost":False, "KNN":False, "scANVI":False},
        parameters.ATLAS: 'Pancreas',

        parameters.REFERENCE_DATA_PATH: 'pancreas_source.h5ad',
        parameters.USE_REFERENCE_EMBEDDING: False,
        parameters.QUERY_DATA_PATH: 'pancreas_query.h5ad',
        parameters.OUTPUT_PATH: 'query.csv',
        parameters.OUTPUT_TYPE: ["csv", "cxg"],

        parameters.USE_PRETRAINED_SCVI_MODEL: True,
        parameters.USE_PRETRAINED_TOTALVI_MODEL: True,
        parameters.USE_PRETRAINED_SCANVI_MODEL: True,
        parameters.USE_GPU: False,

        # scANVI stuff
        # parameters.SCANVI_COMPARE_REFERENCE_AND_QUERY: False,
        # parameters.SCANVI_COMPARE_OBSERVED_AND_PREDICTED_CELLTYPES: False,
        # parameters.SCANVI_PREDICT_CELLTYPES: True,

        parameters.CONDITION_KEY: None,
        parameters.CELL_TYPE_KEY: None,
        parameters.PRETRAINED_MODEL_PATH: '',
        parameters.NUMBER_OF_LAYERS: 2,
        parameters.ENCODE_COVARIATES: True,
        parameters.DEEPLY_INJECT_COVARIATES: False,
        parameters.USE_LAYER_NORM: 'both',
        parameters.USE_BATCH_NORM: 'none',
        parameters.UNLABELED_KEY: None,
        parameters.SCANVI_MAX_EPOCHS: 20,
        parameters.SCANVI_MAX_EPOCHS_QUERY: 100,
        parameters.SCVI_MAX_EPOCHS: 400,
        parameters.SCVI_QUERY_MAX_EPOCHS: 200,
        parameters.SCPOLI_MAX_EPOCHS: 50,
        parameters.NUMBER_OF_NEIGHBORS: 8,
        parameters.MAX_EPOCHS: 100,
        parameters.UNWANTED_LABELS: ['leiden'],
        parameters.DEBUG: False,
        parameters.RUN_ASYNCHRONOUSLY: False,
        parameters.ATTRIBUTES: None,

        # totalVI stuff
        parameters.TOTALVI_MAX_EPOCHS_1: 400,
        parameters.TOTALVI_MAX_EPOCHS_2: 200,

        parameters.DEV_DEBUG: False,
    }


def get_from_config(configuration, key):
    """
    returns the config with key value if the key is in the config, otherwise return none
    :param configuration:
    :param key: key values to be checked in the config
    :return: dict with the parsed key values or none
    """
    if key in configuration:
        return configuration[key]
    return None


def merge_configs(user_config):
    """
    overwrites the default config with the input from the rest api
    :param user_config: input from the rest api
    :return: dict
    """
    return default_config() | user_config


# def query(reference_dataset, query_dataset, model_path, surgery_path,  model_type):
def query(user_config):
    """
    sets model, atlas, attributes with input from the rest api and returns config
    :param user_config: keys of config parsed from the rest api
    :return: config
    """

    print("got config " + str(user_config))
    start_time = time.time()
    configuration = merge_configs(user_config)
    #Sets the correct condition and cell_type key
    #configuration = utils.set_keys(configuration)

    ### NEW dynamic set_key function
    #Preprocess.set_keys_dynamic(configuration)


    scvi_hub_id = utils.get_from_config(configuration, parameters.SCVI_HUB_ID)

    if scvi_hub_id:
        sh = ScviHub(configuration=configuration)

        sh.map_query()
    else:
        model = utils.get_from_config(configuration, parameters.MODEL)
        configuration['atlas'] = utils.translate_atlas_to_directory(configuration)

        if model == 'scVI':
            mapping = ScVI(configuration=configuration)
            mapping.run()
        elif model == 'scANVI':
            mapping = ScANVI(configuration=configuration)
            mapping.run()
        elif model == "scPoli":
            mapping = ScPoli(configuration=configuration)
            mapping.run()


        # if model == 'scVI':
        #     attributes = compute_scVI(configuration)
        # elif model == 'scANVI':
        #     attributes = compute_scANVI(configuration)
        # elif model == 'totalVI':
        #     attributes = computeTotalVI(configuration)
        # elif model == "scPoli":
        #     pass
        #     # mapping = ScPoli(configuration=configuration)

        #     # from models import ScANVI
        #     # scanvi = ScANVI()
        # else:
        #     raise ValueError(model + ' is not one of the supported models')
        # configuration["attributes"] = attributes
        # run_time = (time.time() - start_time)
        # print('completed query in ' + str(run_time) + 's and stored it in: ' + get_from_config(configuration,
        #                                                                                     parameters.OUTPUT_PATH))
        if get_from_config(configuration, parameters.WEBHOOK) is not None and len(
                get_from_config(configuration, parameters.WEBHOOK)) > 0:
            utils.notify_backend(get_from_config(configuration, parameters.WEBHOOK), configuration)
            if ("counts" not in mapping._combined_adata.layers or mapping._combined_adata.layers["counts"].size == 0):
                if not mapping._reference_adata_path.endswith("data.h5ad"):
                    raise ValueError("The reference data should be named data.h5ad")
                else:
                    count_matrix_path = mapping._reference_adata_path[:-len("data.h5ad")] + "data_only_count.h5ad"
                combined_adata = mapping._combined_adata

                count_matrix = read_h5ad_file_from_s3(count_matrix_path)


                #Added because concat_on_disk only allows csr concat
                if count_matrix.X.format == "csc" or mapping.adata_query_X.X.format == "csc":

                    combined_data_X = count_matrix.concatenate(mapping.adata_query_X)

                    del count_matrix
                    del mapping.adata_query_X

                else:

                    #Create temp files on disk
                    temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
                    temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
                    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

                    #Write data to temp files
                    sc.write(temp_reference.name, count_matrix)
                    sc.write(temp_query.name, mapping.adata_query_X)

                    del count_matrix
                    del mapping.adata_query_X
                
                    experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)
                    combined_data_X = sc.read_h5ad(temp_combined.name)
                
                combined_adata.X = combined_data_X.X
            
                cxg_with_count_path = get_from_config(configuration, parameters.OUTPUT_PATH)[:-len("cxg.h5ad")] + "cxg_with_count.h5ad"
                
                
                filename = tempfile.mktemp( suffix=".h5ad")
                sc.write(filename, combined_adata)
                
                print("cxg_with_count_path written to: " + filename)
                print("storing cxg_with_count_path to gcp with output path: " + cxg_with_count_path)
                utils.store_file_in_s3(filename, cxg_with_count_path)
                utils.notify_backend(get_from_config(configuration, parameters.WEBHOOK), configuration)

        return configuration
    



if __name__ == "__main__":
    """
    sets endpoint and fetches input from rest api
    
    """
    # os.environ["AWS_BUCKET"] = 'minio-bucket'
    # os.environ['AWS_ENDPOINT'] = 'http://127.0.0.1:9000'
    # os.environ['AWS_ACCESS_KEY'] = 'minioadmin'
    # os.environ['AWS_SECRET_KEY'] = 'minioadmin'

    query({})

