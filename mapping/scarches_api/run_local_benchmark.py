import os
import boto3
from benchmark_atlas_upload_local import benchmark, benchmark_plot, minify, classify, uncertainty_train, store_results, subset_data
import scanpy as sc
import requests
import ast

def fetch_file_from_s3(key, path):
    """
    downloads a file identified by a given key to a given path
    :param key: key in s3
    :param path: desired path
    :return:
    """
    client = boto3.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT'),
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
    print("fetching file from s3 with Bucket. key. path ", os.getenv('AWS_BUCKET'), key, path)
    client.download_file(os.getenv('AWS_BUCKET'), key, path)

def notify_backend(endpoint, payload):
    """
    makes a post request to an endpoint specified by backend to notify them about the computed results
    :param endpoint: url
    :param payload: configuration initially specified from backend, allows them to identify which result is ready
    :return:
    """
    print("notifying backend with endpoint and payload ", endpoint, payload)
    print("\n")
    requests.post(endpoint, data=payload)


def main():

    print(os.environ) # show all environment variables and their values.


    modelName = "scPoli"
    atlasName = "HNOCA"
    classifierLabels = ['annot_level_1',
                        'annot_level_2',
                        'annot_level_3_rev2',
                        'annot_level_4_rev2',
                        'annot_region_rev2',
                        'annot_ntt_rev2',]


    modelpath_local = "model"
    adatafile_local = "model/adata.h5ad"
    batchkey = "batch"
    celltypekey = "annot_level_1"

    #subset adata if needed
    subset_data(adatafile_local, modelpath_local, celltypekey, batchkey)
    # benchmark integration
    benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey)
    benchmark_plot(atlasName, modelName, batchkey, celltypekey)

    # minify
    minify(modelName, atlasName, modelpath_local)

    # get classifiers and uncert
    adata = classify(atlasName, modelName, classifierLabels)

    uncertainty_train(atlasName, adata, modelName, classifierLabels)

if __name__ == "__main__":
    main()
