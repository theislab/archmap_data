
import scanpy as sc
from scarches.models.scpoli import scPoli
import scvi
import pickle
import scarches as sca
from scib_metrics.benchmark import Benchmarker
import pickle
import torch 
import os
import boto3
from aiohttp import ClientError
import h5py
from anndata.experimental import write_elem, read_elem
from scipy import sparse
from mapping.classifiers.classifiers import Classifiers
import pandas as pd
import numpy as np


def store_file_in_s3(path, key):
    """
    stores a file in the given path in an s3 bucket
    :param path: path in our filesystem
    :param key: key to where to store the file in s3
    :return: returns ContentLength if successfully uploaded, 0 otherwise
    """
    try:
        bucket = os.getenv('AWS_BUCKET')
        client = boto3.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT'),
                              aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                              aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
        print("Uploading file:: Bucket. key. path ", bucket, key, path)
        client.upload_file(path, bucket, key)
        response = client.head_object(Bucket=bucket, Key=key)
        print("Response from the upload: ", response)
        return response['ContentLength']
    except ClientError as e:
        print(e)
    return 0


def convert_scpoli(input_path, output_path):
    model = torch.load(f"{input_path}/model.pt")

    torch.save(model["model_state_dict"],f"{output_path}/model_params.pt")

    with open(f"{output_path}/attr.pkl","wb") as f:
            pickle.dump(model["attr_dict"], f, pickle.HIGHEST_PROTOCOL)

    pd.Series(model["var_names"]).to_csv(f"{output_path}/var_names.csv")

#minify
def minify(modelName, atlasName, modelpath_local):
     
    model_type = modelName
    model_name = modelpath_local
    model_minified_path = "model_minified"
    atlas = atlasName.replace(" ", "_").lower()

    # save count data
    with h5py.File(f"{model_name}/adata.h5ad", mode="r") as store1:
        X=read_elem(store1['X'])
        print(X)
        var_names=read_elem(store1['var']).index
        adata_count = sc.AnnData(X)
        adata_count.var_names =var_names
        adata_count.write(f"data_only_count_{atlas}.h5ad")
        print(adata_count.X)

     # minify
    if model_type=="scpoli":
            tm = sca.models.scPoli.load(model_name)
            tm.minify_adata()
    elif model_type=="scanvi":
        tm = scvi.model.SCANVI.load(model_name)
        qzm, qzv = tm.get_latent_representation(give_mean=False, return_dist=True)
        tm.adata.obsm["X_latent_qzm"] = qzm
    else:
        tm = scvi.model.SCVI.load(model_name)
        qzm, qzv = tm.get_latent_representation(give_mean=False, return_dist=True)
        tm.adata.obsm["X_latent_qzm"] = qzm

    tm.save(model_minified_path, save_anndata=True, overwrite=True)

    # zero out counts for minified version
    with h5py.File(f"{model_minified_path}/adata.h5ad", mode="r") as store1:
        all_zeros = sparse.csr_matrix(X.shape)
        write_elem(store1, "X", all_zeros)


# benchmark atlas integration
def benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey):

    modelName = modelName.lower()
    
    # read model and get embedding
    if modelName == "scpoli":
        convert_scpoli(modelpath_local,modelpath_local)
        model = sca.models.scPoli.load(modelpath_local)
        model.adata.obsm["X_user_integrated"] = scpoli_model.get_latent(model.adata, mean=True)

    elif modelName == "scvi":
        model = scvi.model.SCVI.load(modelpath_local)
        model.adata.obsm["X_user_integrated"] = model.get_latent_representation()

    elif modelName == "scanvi":
        model = scvi.model.SCVI.load(modelpath_local)
        model.adata.obsm["X_user_integrated"] = model.get_latent_representation()
    else:
        raise ValueError(f"The model '{modelName}' is not available.")


    # get cell type key
    cell_type_key = celltypekey
    # get condition key
    condition_key = batchkey

    atlas = atlasName.replace(" ", "_").lower()

    adata = model.adata
    

    # run pca
    sc.pp.pca(adata)

    # run scvi
    if modelName!="scvi":
        scvi.model.SCVI.setup_anndata(adata, batch_key=condition_key)
        vae = scvi.model.SCVI(adata, gene_likelihood="nb")
        vae.train(max_epochs=500)
        adata.obsm["X_scvi"] = vae.get_latent_representation()

    # run scanvi
    if modelName!="scanvi":
        scvi.model.SCANVI.setup_anndata(adata, batch_key=condition_key)
        vae = scvi.model.SCANVI(adata, gene_likelihood="nb")
        vae.train(max_epochs=500)
        adata.obsm["X_scanvi"] = vae.get_latent_representation()

    # run scpoli
    if modelName!="scpoli":
        early_stopping_kwargs = {
            "early_stopping_metric": "val_prototype_loss",
            "mode": "min",
            "threshold": 0,
            "patience": 20,
            "reduce_l": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }

        scpoli_model = scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            embedding_dims=5,
            recon_loss='nb',
        )
        scpoli_model.train(
            n_epochs=200,
            pretraining_epochs=40,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=0, #prototype loss weight -> higher means more clustering of each ct towards its avg latent score.
        )


        adata.obsm["X_scpoli_no_prototype"] = scpoli_model.get_latent(adata, mean=True)
        print("first scpoli trained")

        scpoli_model = scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            embedding_dims=5,
            recon_loss='nb',
        )
        scpoli_model.train(
            n_epochs=200,
            pretraining_epochs=40,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5, #prototype loss weight -> higher means more clustering of each ct towards its avg latent score.
        )

        adata.obsm["X_scpoli_with_prototype"] = scpoli_model.get_latent(adata, mean=True)
        print("2nd scpoli trained")


        from pathlib import Path
        benchmark_results ="benchmark_results"
        path = Path(benchmark_results)
        path.mkdir(parents=True, exist_ok=True)
        adata.write(f"benchmark_results/adata_{atlas}_{cell_type_key}_integrated.h5ad")



# plot benchmarking results
def benchmark_plot(atlasName, batchkey, celltypekey):

    cell_type_key = celltypekey
    
    atlas = atlasName.replace(" ", "_").lower()

    condition_key = batchkey

    adata = sc.read(f"benchmark_results/adata_{atlas}_{cell_type_key}_integrated.h5ad")


    # run scib metrics
    bm = Benchmarker(
        adata,
        batch_key=condition_key,
        label_key=cell_type_key,
        embedding_obsm_keys=["X_pca", "X_scvi", "X_scpoli_no_prototype","X_scpoli_with_prototype", "X_user_integrated"],
        n_jobs=4,
    )


    bm.benchmark()

    with open("benchmark_results/results.pickle","wb") as f:
        pickle.dump(bm, f, pickle.HIGHEST_PROTOCOL)
    
    with open("benchmark_results/results.pickle","rb") as f:
        bm = pickle.load(f)
    store_file_in_s3("benchmark_results/results.pickle", "benchmark_results/results.pickle")
        


    df = bm.get_results(min_max_scale=False)
    df_t = df.transpose()
    df_t.to_csv("benchmark_results/integration_comparison.csv")
    store_file_in_s3("benchmark_results/integration_comparison.csv", "benchmark_results/integration_comparison.csv")

    bm.plot_results_table(save_dir=f"benchmark_results/results_min_max_scale.png")
    store_file_in_s3("benchmark_results/results_min_max_scale.png", "benchmark_results/results_min_max_scale.png")

    bm.plot_results_table(min_max_scale=False, save_dir=f"benchmark_results/results.png")
    store_file_in_s3("benchmark_results/results.png", "benchmark_results/results.png")
        



def classify(atlas, modelName, label):

    model_minified_path = "model_minified"
    modelName = modelName.lower()
    

    if modelName=="scpoli":
        model = sca.models.scPoli.load(f"{model_minified_path}/", map_location="cpu")
        adata=model.adata
        reference_latent=sc.AnnData(adata.obsm["X_latent_qzm_scpoli"], adata.obs)

    elif modelName=="scanvi":
        model = sca.models.SCANVI.load(f"{model_minified_path}/")
        adata=model.adata
        reference_latent=sc.AnnData(adata.obsm["X_latent_qzm"], adata.obs)
    else:
        model = sca.models.SCVI.load(f"{model_minified_path}/")
        adata=model.adata
        reference_latent=sc.AnnData(adata.obsm["X_latent_qzm"], adata.obs)

    if not isinstance(label, list):
        label = [label]

    for l in label:
        #create knn classifier
        clf = Classifiers(False, True, None)
        clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}/{atlas}_{l}")

        #create xgb classifier
        clf = Classifiers(True, False, None)
        clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}/{atlas}_{l}")

        #create native
        clf = Classifiers(False, False, model, model.__class__)
        clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}/{atlas}_{l}")


