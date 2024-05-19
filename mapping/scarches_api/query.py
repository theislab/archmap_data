import argparse
import os
import init as scarches
from threading import Thread
from utils import parameters

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
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--query', type=dict, help='Query input for training')
    args = parser.parse_args()
    
    query(args.query)
