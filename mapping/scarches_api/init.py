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

from utils.utils import read_h5ad_file_from_s3, get_file_size_in_gb, fetch_file_to_temp_path_from_s3
import scanpy as sc
import tempfile
import gc
from anndata import experimental

from process.processing import Preprocess
from utils.utils import read_h5ad_file_from_s3
import scanpy as sc
import tempfile
import gc
import h5py
from anndata.experimental import write_elem, read_elem


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

        if True or get_from_config(configuration, parameters.WEBHOOK) is not None and len(
                get_from_config(configuration, parameters.WEBHOOK)) > 0:
            # utils.notify_backend(get_from_config(configuration, parameters.WEBHOOK), configuration)
            if ("counts" not in mapping._combined_adata.layers or mapping._combined_adata.layers["counts"].size == 0):
                if not mapping._reference_adata_path.endswith("data.h5ad"):
                    raise ValueError("The reference data should be named data.h5ad")
                else:
                    count_matrix_path = mapping._reference_adata_path[:-len("data.h5ad")] + "data_only_count.h5ad"
                combined_adata = mapping._combined_adata

                cxg_with_count_path = get_from_config(configuration, parameters.OUTPUT_PATH)[:-len("cxg.h5ad")] + "cxg_with_count.h5ad"
                count_matrix_size_gb = get_file_size_in_gb(count_matrix_path)
                temp_output = tempfile.mktemp( suffix=".h5ad")
                print("1")
                if count_matrix_size_gb < 2:
                    print("2")
                    count_matrix = read_h5ad_file_from_s3(count_matrix_path)
                    #Added because concat_on_disk only allows csr concat
                    if count_matrix.X.format == "csc" or mapping.adata_query_X.X.format == "csc":
                        print("3")
                        combined_data_X = count_matrix.concatenate(mapping.adata_query_X)

                        del count_matrix
                        del mapping.adata_query_X

                    else:
                        print("4")
                        #Create temp files on disk
                        temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
                        temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
                        temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

                        #Write data to temp files
                        count_matrix.write_h5ad(temp_reference.name)
                        mapping.adata_query_X.write_h5ad(temp_query.name)

                        del count_matrix
                        del mapping.adata_query_X
                    
                        experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)
                        combined_data_X = sc.read_h5ad(temp_combined.name)

                    combined_adata.X = combined_data_X.X
                    sc.write(temp_output, combined_adata)

                else:
                    print("5")
                    temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
                    mapping.adata_query_X.write_h5ad(temp_query.name)
                    del mapping.adata_query_X
                    temp_output=replace_X_on_disk(combined_adata,temp_output, temp_query.name, count_matrix_path)
                
                print("11")
                print("cxg_with_count_path written to: " + temp_output)
                print("storing cxg_with_count_path to gcp with output path: " + cxg_with_count_path)
                utils.store_file_in_s3(temp_output, cxg_with_count_path)
                print("12")
                # utils.notify_backend(get_from_config(configuration, parameters.WEBHOOK), configuration)

        return configuration
    

def replace_X_on_disk(combined_adata,temp_output, query_X_file, ref_count_matrix_path):
    """
    Writes combined_adata to disk, fetches another .h5ad file specified by new_adata_key,
    and replaces the .X of combined_adata with the .X of the new file on disk.
    :param combined_adata: The Anndata object in memory.
    :param new_adata_key: The S3 key for the .h5ad file to be fetched.

    Returns: File path to saved adata with concatenated metadata and .X
    """
    print("6")
    # Write combined_adata to disk
    combined_adata.write(temp_output)
    print(f"combined_adata written to {temp_output}")
    print("7")
    # Fetch the new file and get its path
    temp_ref_count_matrix_path = fetch_file_to_temp_path_from_s3(ref_count_matrix_path)
    if temp_ref_count_matrix_path is None:
        print("No file fetched. Exiting.")
        return
    print("8")
    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    
    experimental.concat_on_disk([temp_ref_count_matrix_path, query_X_file], temp_combined.name)
    print("9")

    # write concatenated adata metadata (eg .obs and .vars and .varm and .obsm and uns) to adata with .X on disk
    with h5py.File(temp_output, "r") as f:
        with h5py.File(temp_combined.name, 'r+') as target_file:

            elems=list(f.keys())
            if "X" in elems:
                elems.remove("X")
                
            for elem in elems:
                v=read_elem(f[elem])
                if isinstance(v,dict) and not bool(v):
                    continue

                write_elem(target_file, f"{elem}", v)
        print("10")
        print("Added concatenated metadata to anndata with full .X on disk")

    return temp_combined.name




if __name__ == "__main__":
    """
    sets endpoint and fetches input from rest api
    
    """
    # os.environ["AWS_BUCKET"] = 'minio-bucket'
    # os.environ['AWS_ENDPOINT'] = 'http://127.0.0.1:9000'
    # os.environ['AWS_ACCESS_KEY'] = 'minioadmin'
    # os.environ['AWS_SECRET_KEY'] = 'minioadmin'

    query({})

