import os
import boto3
from benchmark_atlas_upload import benchmark, benchmark_plot, minify, classify, uncertainty_train
import scanpy as sc

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



def main():

    print(os.environ) # show all environment variables and their values.

    modelPath = os.getenv('modelpath')
    atlasPath = os.getenv('atlaspath')
    modelName = os.getenv('modelname')
    batchkey = os.getenv('batchkey')
    # celltypekey = os.getenv('celltypekey')
    celltypekey ="cell_type_level1"
    atlasName = os.getenv('atlasname')

    print(f"modelpath: {modelPath}")
    print(f"atlaspath: {atlasPath}")

    #Get model and data
    modelfile_gcp = f"models/{modelPath}/model.pt"
    adatafile_gcp = f"atlas/{atlasPath}/data.h5ad"

    modelfile_local = "model/model.pt"
    adatafile_local = "model/data.h5ad"

    fetch_file_from_s3(modelfile_gcp, modelfile_local)
    fetch_file_from_s3(adatafile_gcp, adatafile_local)

    modelpath_local = "model"

    # TODO: 
    # Check that data is not minified

    # benchmark integration
    benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey)
    benchmark_plot(atlasName, batchkey, celltypekey)

    # minify
    minify(modelName, atlasName, modelpath_local)

    # get classifiers and uncert
    adata = classify(atlasName, modelName, celltypekey)

    uncertainty_train(atlasName, adata, modelName, celltypekey)


if __name__ == "__main__":
    main()
