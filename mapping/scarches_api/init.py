import os
import time
import tempfile
import scanpy as sc

from scarches_api.utils import utils, parameters

from scvi_hub.scvi_hub import ScviHub
from models import ScANVI
from models import ScVI
from models import ScPoli


# from process.processing import Preprocess


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
        # parameters.USE_GPU: False,

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
    start_time2 = time.time()
    print("got config " + str(user_config))
    start_time = time.time()
    configuration = merge_configs(user_config)

    print("running dev branch")
    #Sets the correct condition and cell_type key
    #configuration = utils.set_keys(configuration)

    ### NEW dynamic set_key function
    #Preprocess.set_keys_dynamic(configuration)


    scvi_hub_id = utils.get_from_config(configuration, parameters.SCVI_HUB_ID)

    if scvi_hub_id:
        mapping = ScviHub(configuration=configuration)

        mapping.map_query()
    
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

        end_time2 = time.time()
        

        print(f"time end: {end_time2}-{start_time2}")

        
        #TODO: add obsm and other keys from query_adata_raw to adata_combined for download

        # atlas_name = utils.get_from_config(configuration, parameters.ATLAS)
        
        # sc.AnnData(mapping._combined_adata.obsm["latent_rep"], mapping._combined_adata.obs).write(f"results/{atlas_name}.h5ad")

    if get_from_config(configuration, parameters.WEBHOOK) is not None and len(
            get_from_config(configuration, parameters.WEBHOOK)) > 0:
        
        utils.notify_backend(get_from_config(configuration, parameters.WEBHOOK), configuration)

 
        output_model_path = get_from_config(configuration, parameters.OUTPUT_PATH)

        #Save as .h5ad
        data_cxg = mapping.data_cxg
        output_path = get_from_config(configuration, parameters.OUTPUT_CXG_PATH)

        # store file for cxg
        filename = tempfile.mktemp( suffix=".h5ad")
        sc.write(filename, data_cxg)
        print("file written to: " + filename)
        print("Now storing cxg data to gcp with output path: " + output_path)
        utils.store_file_in_s3(filename, output_path)
        
        #save model and adata as tar file
        import tarfile
        os.makedirs("finetuned_model/", exist_ok=True)
        mapping._combined_adata.write("finetuned_model/adata.h5ad")
        mapping._model.save("finetuned_model/", save_anndata=False, overwrite=True)
        output_filename="query_model.tar.gz"
        source_dir="finetuned_model/"
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        print(f"Created tar archive: {output_filename}")

        #store model to gcp
        print("storing model to gcp with output path: " + output_model_path)
        utils.store_file_in_s3("query_model.tar.gz", output_model_path)
        # print("Stored adata with counts on cloud")
        # print("storing fine-tuned model to gcp with output path: " + output_model_path)
        # utils.store_file_in_s3(mapping._model, output_model_path)
        print("Stored finetuned model on cloud")
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

