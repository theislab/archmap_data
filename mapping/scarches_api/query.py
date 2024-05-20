import argparse
import os
import init as scarches
from threading import Thread
from utils import parameters
import ast

def get_from_config(configuration, key):
    if key in configuration:
        return configuration[key]
    return None

def query(config):
    run_async = get_from_config(config, parameters.RUN_ASYNCHRONOUSLY)
    if run_async is not None and run_async:
        actual_config = scarches.merge_configs(config)
        thread = Thread(target=scarches.query, args=(config,))
        thread.start()
        return actual_config, 200
    else:
        actual_configuration = scarches.query(config)
        return actual_configuration, 200
    
if __name__ == "__main__":

    configuration = {
        "model": "scANVI",
        "atlas": "HLCA",
        "output_path": "test_output/HLCA",
        "output_type": {
            "csv": False,
            "cxg": True,
        },
        "classifier_type": {
            "XGBoost": False,
            "kNN": False,
            "Native": True
        },
        "classifier_path": "classifiers/628668716f930d8b7f44d575/classifier_knn.pickle",
        "encoder_path": "classifiers/628668716f930d8b7f44d575/classifier_encoding.pickle",
        "model_path": "models/655b580a0c9e68011f3a9ea3/model.pt",
        "reference_data": "atlas/628668716f930d8b7f44d575/data.h5ad",
        "query_data": "query_test_data/hlca/HLCA_delorey.h5ad",
        "scanvi_max_epochs_query": 1
    }

    

    queryinfo = ast.literal_eval(os.environ["QUERY"])
    query(queryinfo)

    # parser = argparse.ArgumentParser(description='Train model')
    # parser.add_argument('--query', type=dict, default=configuration, help='Query input for training')
    # args = parser.parse_args()
    
    # query(args.query)
