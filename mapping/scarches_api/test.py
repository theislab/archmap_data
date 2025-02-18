import os
import boto3
from benchmark_atlas_upload import benchmark, benchmark_plot, minify, classify, uncertainty_train, store_results, subset_data
import scanpy as sc
import requests

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

    modelPath = os.getenv('modelPath')
    atlasPath = os.getenv('atlasPath')
    modelName = os.getenv('modelName')
    batchkey = os.getenv('batchKey')
    celltypekey = os.getenv('cellTypeKey')
    atlasName = os.getenv('atlasName')
    webhook = os.getenv('webhook')
    classifierLabels = os.getenv('classifierLabels')

    print(f"modelpath: {modelPath}")
    print(f"atlaspath: {atlasPath}")

    #Get model and data
    modelfile_gcp = f"models/{modelPath}/model.pt"
    adatafile_gcp = f"atlas/{atlasPath}/data.h5ad"

    directory="model/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    modelfile_local = "model/model.pt"
    adatafile_local = "model/adata.h5ad"

    fetch_file_from_s3(modelfile_gcp, modelfile_local)
    fetch_file_from_s3(adatafile_gcp, adatafile_local)

    modelpath_local = "model"

    #subset adata if needed
    subset_data(adatafile_local, modelpath_local, celltypekey, batchkey)
    # benchmark integration
    benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey, modelPath)
    benchmark_plot(atlasName, modelName, batchkey, celltypekey)

    # recollect the full data after benchmarking
    fetch_file_from_s3(adatafile_gcp, adatafile_local)
    
    # minify
    minify(modelName, atlasName, modelpath_local, modelPath, atlasPath)

    # get classifiers and uncert
    adata = classify(atlasName, modelName, classifierLabels, modelPath)
    store_results(atlasName, classifierLabels, modelPath)


    uncertainty_train(atlasName, adata, modelName, classifierLabels, modelPath)

    # notify backend that benchmarking completed
    notify_backend(webhook, {})

    # TODO: Store classifiers, uncert in GCP


if __name__ == "__main__":
    main()
