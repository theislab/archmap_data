
import scanpy as sc
from scarches.models.scpoli import scPoli
import scvi
import pickle
import scarches as sca
from scib_metrics.benchmark import Benchmarker
import torch 
import os
import boto3
from aiohttp import ClientError
import h5py
from anndata.experimental import write_elem, read_elem
from scipy import sparse
from classifiers import Classifiers
import pandas as pd
import shutil
from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np
import gc

def sample_cells(adata, celltype_key):
    

    total_ref_cells_to_sample=200000

    celltypes = adata.obs[celltype_key].unique()

    # Calculate the proportion of each cell type in the reference data
    celltype_proportions = {celltype: np.sum(adata.obs[celltype_key] == celltype) / len(adata) for celltype in celltypes}

    # Sample cells from each cell type according to its proportion
    sampled_cell_index = []
    for celltype, proportion in celltype_proportions.items():
        cell_indices = np.where(adata.obs[celltype_key] == celltype)[0]
        sample_size = int(total_ref_cells_to_sample * proportion)
        
        # Adjust sample size if it exceeds the number of available cells
        if sample_size > len(cell_indices):
            sample_size = len(cell_indices)
        
        sampled_cells = np.random.choice(cell_indices, size=sample_size, replace=False)
        sampled_cell_index.extend(sampled_cells)

    return sampled_cell_index



def subset_data(adatafile_local, modelpath_local, celltype_key, batch_key):
        
        
        adata = sc.read(f"{adatafile_local}")

        model = torch.load(f"{modelpath_local}/model.pt", map_location="cpu")

        # check that adata is not already minified
        if (adata.X is None or not adata.X.sum()>0):
            raise ValueError(f"The uploaded h5ad file does not have count data saved in the .X attribute. Please reupload your atlas with count data in .X.")
        

        adata=adata[:,pd.Series(model["var_names"]).values]

        del adata.uns
        del adata.obsm
        del adata.obsp
        del adata.varm
        del adata.layers
        del adata.varp
        del adata.raw

        # delete adata file to save memory
        os.remove(adatafile_local)

        
        if adata.n_obs>200000:

            # subset adata and make a new copy of the model to new directory

            sampled_cell_index = sample_cells(adata, celltype_key)

            # Create downsampled AnnData object
            adata_downsample = adata[sampled_cell_index].copy()


            # check if all batches are present
            batches = adata.obs[batch_key].unique()
            batches_sub = adata_downsample.obs[batch_key].unique()

            missing_batches = set(batches).difference(set(batches_sub))
            missing_batches_len = len(missing_batches)
            if missing_batches_len>0:
                print("missing batches in adata downsample. sampling missing batches")

            adata_downsample.write(f"{modelpath_local}/adata.h5ad")


            #     for batch in list(missing_batches):

            #         adata_batch_sub = adata[adata.obs[batch_key]==batch]

            #         sampled_cell_index_sub = sample_cells(adata_downsample, celltype_key)
        
        else:

            adata.write(f"{modelpath_local}/adata.h5ad")




def convert_scpoli(input_path, output_path):
    model = torch.load(f"{input_path}/model.pt", map_location="cpu")

    torch.save(model["model_state_dict"],f"{output_path}/model_params.pt")

    with open(f"{output_path}/attr.pkl","wb") as f:
            pickle.dump(model["attr_dict"], f, pickle.HIGHEST_PROTOCOL)

    pd.Series(model["var_names"]).to_csv(f"{output_path}/var_names.csv", header=False, index=False)


#minify
def minify(modelName, atlasName, modelpath_local):

     
    model_type = modelName.lower()
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

    adata = sc.read(f"{model_name}/adata.h5ad", backed="r+")

    del adata.uns
    del adata.obsm
    del adata.obsp
    del adata.varm
    del adata.layers
    del adata.varp
    del adata.raw

    adata.write()
    adata.file.close()

     # minify
    if model_type=="scpoli":
            tm = sca.models.scPoli.load(model_name, map_location="cpu")
            tm.get_latent(tm.adata)
            tm.minify_adata()
    elif model_type=="scanvi":
        tm = scvi.model.SCANVI.load(model_name)
        qzm, qzv = tm.get_latent_representation(give_mean=False, return_dist=True)
        tm.adata.obsm["X_latent_qzm"] = qzm
    else:
        tm = scvi.model.SCVI.load(model_name)
        qzm, qzv = tm.get_latent_representation(give_mean=False, return_dist=True)
        tm.adata.obsm["X_latent_qzm"] = qzm

    # delete files to save memory
    os.remove(f"{model_name}/adata.h5ad")
    os.remove(f"{model_name}/model.pt")

    tm.save(model_minified_path, save_anndata=True, overwrite=True)

    del tm
    del adata
    gc.collect()

    # zero out counts for minified version
    with h5py.File(f"{model_minified_path}/adata.h5ad", mode="r+") as store1:
        all_zeros = sparse.csr_matrix(X.shape)
        write_elem(store1, "X", all_zeros)


# benchmark atlas integration
def benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey):

    modelName = modelName.lower()
    
    # read model and get embedding
    if modelName == "scpoli":
        convert_scpoli(modelpath_local,modelpath_local)
        model = sca.models.scPoli.load(modelpath_local, map_location="cpu")
        model.adata.obsm["X_user_integrated"] = model.get_latent(model.adata, mean=True)

    elif modelName == "scvi":
        model = scvi.model.SCVI.load(modelpath_local)
        model.adata.obsm["X_user_integrated"] = model.get_latent_representation()

    elif modelName == "scanvi":
        model = scvi.model.SCANVI.load(modelpath_local)
        model.adata.obsm["X_user_integrated"] = model.get_latent_representation()
    else:
        raise ValueError(f"The model '{modelName}' is not available.")
    
    # delete files to save memory
    os.remove(f"{modelpath_local}/adata.h5ad")


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
        for key in ["unlabeled", "unknown"]:
            if key in adata.obs[cell_type_key].str.lower().values:
                unlabeled_key = key
                break
            else:
                unlabeled_key = "unknown"
            
        scvi.model.SCANVI.setup_anndata(adata, batch_key=condition_key, labels_key=cell_type_key, unlabeled_category=unlabeled_key)
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
            "reduce_lr": True,
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

    return adata


# plot benchmarking results
def benchmark_plot(atlasName, modelName, batchkey, celltypekey):

    cell_type_key = celltypekey
    modelName = modelName.lower()
    
    atlas = atlasName.replace(" ", "_").lower()

    condition_key = batchkey

    adata = sc.read(f"benchmark_results/adata_{atlas}_{cell_type_key}_integrated.h5ad")

    # delete adata file to save memory
    os.remove(f"benchmark_results/adata_{atlas}_{cell_type_key}_integrated.h5ad")


    # run scib metrics
    if modelName=="scpoli":
        bm = Benchmarker(
            adata,
            batch_key=condition_key,
            label_key=cell_type_key,
            embedding_obsm_keys=["X_pca", "X_scvi","X_scanvi", "X_user_integrated"],
            n_jobs=4,
        )
    elif modelName=="scanvi":
        bm = Benchmarker(
            adata,
            batch_key=condition_key,
            label_key=cell_type_key,
            embedding_obsm_keys=["X_pca", "X_scvi", "X_scpoli_no_prototype","X_scpoli_with_prototype", "X_user_integrated"],
            n_jobs=4,
        )
    else:
        bm = Benchmarker(
            adata,
            batch_key=condition_key,
            label_key=cell_type_key,
            embedding_obsm_keys=["X_pca", "X_scanvi", "X_scpoli_no_prototype","X_scpoli_with_prototype", "X_user_integrated"],
            n_jobs=4,
        )


    bm.benchmark()

    with open("benchmark_results/results.pickle","wb") as f:
        pickle.dump(bm, f, pickle.HIGHEST_PROTOCOL)
    
    with open("benchmark_results/results.pickle","rb") as f:
        bm = pickle.load(f)


    df = bm.get_results(min_max_scale=False)
    df_t = df.transpose()
    df_t.to_csv("benchmark_results/integration_comparison.csv")

    
    benchmark_results_min_max ="benchmark_results/scib_min_max_scale/"
    path = Path(benchmark_results_min_max)
    path.mkdir(parents=True, exist_ok=True)

    bm.plot_results_table(save_dir=benchmark_results_min_max)

    bm.plot_results_table(min_max_scale=False, save_dir=f"benchmark_results/")
    
        


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
        clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}_{l}")

        #create xgb classifier
        clf = Classifiers(True, False, None)
        clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}_{l}")

        # #create native
        # clf = Classifiers(False, False, model, model.__class__)
        # clf.create_classifier(reference_latent, adata, True, "", l, f"classifier_models/{atlas}_{l}")


    return adata

def store_results(atlas, label, modelPath):
    results_path = f"models/{modelPath}/"


    files = [
        "classifier_knn_report.csv",
        "classifier_knn_report.png",
        "classifier_xgb_report.csv",
        "classifier_xgb_report.png"]
    
    if not isinstance(label, list):
        label = [label]

    results_dir = "benchmark_results"
    for l in label:
        for file in files:
            shutil.copy(f"classifier_models/{atlas}_{l}/{file}", results_dir)

    
    # create tar.gz from results_dir
    import tarfile
    output_filename=f"{results_dir}.tar.gz"
    source_dir=f"{results_dir}/"
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print(f"Created tar archive: {output_filename}")




def train_mahalanobis(atlas, adata_ref, embedding_name, cell_type_key, pretrained=True):


    num_clusters = adata_ref.obs[cell_type_key].nunique()
    print(num_clusters)

    train_emb = adata_ref.obsm[embedding_name]

    #Required too much RAM
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(train_emb)

    #Less RAM alternative
    # kmeans = KMeans(n_clusters=num_clusters)
    # kmeans.fit(train_emb)

    #Save or return model
    if pretrained:

        directory="models_uncert/" + atlas + "/"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(directory + cell_type_key + "_mahalanobis_distance.pickle", "wb") as file:
            pickle.dump(gmm, file, pickle.HIGHEST_PROTOCOL)
    else:
        return gmm
    
def train_euclidian(atlas, adata_ref, embedding_name, pretrained =True, n_neighbors = 15):

    trainer = sca.utils.weighted_knn_trainer(
    adata_ref,
    embedding_name,
    n_neighbors = n_neighbors
    )

    #Save model
    if pretrained:

        directory="models_uncert/" + atlas + "/"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(directory + "euclidian_distance.pickle", "wb") as file:
            pickle.dump(trainer, file, pickle.HIGHEST_PROTOCOL)
    else:
        return trainer


def uncertainty_train(atlas, adata_ref, modelName, cell_type_key_list):

    modelName=modelName.lower()

    if modelName=="scpoli":
        embedding_name = "X_latent_qzm_scpoli"
    else:
        embedding_name = "X_latent_qzm"


    if isinstance(cell_type_key_list,str):
        cell_type_key_list = [cell_type_key_list]

    files = ["euclidian_distance.pickle"]

    for cell_type_key in cell_type_key_list:
        print(cell_type_key)
        train_euclidian(atlas, adata_ref, embedding_name)
        train_mahalanobis(atlas, adata_ref, embedding_name, cell_type_key)

        files.append(cell_type_key + "_mahalanobis_distance.pickle")

