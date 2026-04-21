
import scarches
import scanpy
import pandas
import numpy as np
import tempfile
import gc
import numpy as np
from scipy.sparse import csr_matrix
from anndata import experimental
import scanpy as sc
import os
from scarches_api.utils import parameters
from scarches_api.utils.utils import get_from_config
from scarches_api.utils.utils import check_h5ad_format
import pandas as pd

from process.processing import Preprocess

import torch


def compute_latent_representation(explicit_representation, cell_type_key, model, batch_key):
    #Setup adata before quering model for latent representation
    scarches.models.SCANVI.setup_anndata(explicit_representation, labels_key=cell_type_key, unlabeled_category="unlabeled", batch_key=batch_key)
    explicit_representation.obsm["latent_rep"] = model.get_latent_representation(explicit_representation)



def acquire_data(path, query_adata_path):


    reference_adata.obs["type"] = "reference"
    del reference_adata.layers

    try:
        query_adata_raw = sc.read(path + query_adata_path) 
        print("Data successfully loaded.")
    except Exception as e:
        raise RuntimeError(f"Error message: {e}, There is likely an issue with the way your data (anndata object) is formatted upon upload. Please reach out to ArchMap (archmap.bio@gmail.com) with a screenshot of this error and we can help resolve this.")

    check_h5ad_format(query_adata_raw)



    query_adata_raw.obs["type"] = "query"

    # Check if var_names of ref and query match. Convert var_names of query if not.
    ensembl_ref = True
    for var_name in reference_adata.var_names[:5]:
        if "ENS" in var_name: 
            continue
        else:
            ensembl_ref = False
            break

    ensembl_query = True
    for var_name in query_adata_raw.var_names[:5]:
        if "ENS" in var_name: 
            continue
        else:
            ensembl_query = False
            break


    if ensembl_query != ensembl_ref: 
        import pickle
        # convert query var_names to match ref

        if ensembl_ref == True:
            if "ENSMUS" in reference_adata.var_names[0]:

                with open(path + "gene_conversions/genesymbol_to_ensembl_mouse.pkl", "rb") as file:
                    dict_conversions = pickle.load(file)

            else:
                with open(path + "gene_conversions/genesymbol_to_ensembl_human.pkl", "rb") as file:
                    dict_conversions = pickle.load(file)
                
    
        else:
            if "ENSMUS" in query_adata_raw.var_names[0]:
                #fetch mouse conversions
                with open(path + "gene_conversions/ensembl_to_genesymbol_mouse.pkl", "rb") as file:
                    dict_conversions = pickle.load(file)

            else:
                with open(path + "gene_conversions/ensembl_to_genesymbol_human.pkl", "rb") as file:
                    dict_conversions = pickle.load(file)

        query_adata_raw.var_names = pd.Index([dict_conversions.get(item, item) for item in query_adata_raw.var_names])

    ref_vars = reference_adata.var_names
    query_vars = query_adata_raw.var_names
    
    intersection = ref_vars.intersection(query_vars)
    inter_len = len(intersection)
    ratio = (inter_len / len(ref_vars))*100

    print(f"{ratio}% of genes in your query overlap with the reference data")


    query_adata_raw.obs_names_make_unique()
    query_adata_raw.var_names_make_unique()

    #Convert bool to categorical to avoid write error during concatenation
    Preprocess.bool_to_categorical(reference_adata)
    Preprocess.bool_to_categorical(query_adata_raw)

    
    # save only necessary data for mapping to new adata
    query_adata = query_adata_raw.copy()
    del query_adata.varm
    del query_adata.obsm
    del query_adata.layers
    del query_adata.uns
    del query_adata.obsp
    del query_adata.varp

    query_adata.layers['counts'] = query_adata.X

    return query_adata, reference_adata


def main(query_name):

    path = "mapping/hlca_tutorial/"
    atlas = "HLCA"
    model_type = "scANVI"
    max_epochs = 100

    query_path = "queries/" + query_name + ".h5ad"

    configuration = {
        "model": model_type,
        "atlas": atlas,
        "output_type": {
            "csv": False,
            "cxg": True,
        },
        "classifier_type": {
            "XGBoost": False,
            "kNN": True,
        },
        "n_neighbors": 15,
    }

    atlas = atlas.lower()
    model_type = model_type.lower()

    # Intialize variables
    model_file = path + "model_" + atlas + "_" + model_type + "/model.pt"
    model_path = path + "model_" + atlas 
    model = None
    temp_clf_model_path = None
    temp_clf_encoding_path = None
    query_adata = None
    reference_adata = None
    combined_adata = None
    percent_unknown = "n/a"

    #Load and process required data
    query_adata, reference_adata = acquire_data(path, query_path)

    #Set respective keys coherent to chosen atlas
    cell_type_key = None
    batch_key = None
    unlabeled_key = None
    cell_type_key_input = "user_cell_type"
    batch_key_input = "batch"


    query_adata.X=query_adata.X.tocsr()

    cell_type_key, cell_type_key_classifier, cell_type_key_list, batch_key, unlabeled_key, uploaded = Preprocess.get_keys(atlas, query_adata, configuration, model_file) 

    if isinstance(cell_type_key,list):
        for key in cell_type_key:
            query_adata.obs[key] = [unlabeled_key]*len(query_adata) 
    else:
        query_adata.obs[cell_type_key] = [unlabeled_key]*len(query_adata)

    if cell_type_key_classifier is None:
        cell_type_key_classifier = cell_type_key

    if cell_type_key_list is None:
        if isinstance(cell_type_key_classifier,list):
            cell_type_key_list = cell_type_key_classifier
        else:
            cell_type_key_list = [cell_type_key_classifier]

    if batch_key_input != batch_key:
        query_adata.obs[batch_key] = query_adata.obs[batch_key_input].copy()
        del query_adata.obs[batch_key_input]

    #### MAP QUERY ####

    supervised=False
    model_path = path + "model_" + atlas + "_" + model_type 
    query_adata.var_names_make_unique()

    if model_type!= "scpoli":
        if model_type == "scanvi":
            scarches.models.SCANVI.prepare_query_anndata(query_adata, model_path)

            #Setup adata internals for mapping
            scarches.models.SCANVI.setup_anndata(query_adata, batch_key=batch_key, labels_key=cell_type_key, unlabeled_category=unlabeled_key)

            #Load scanvi model with query
            model = scarches.models.SCANVI.load_query_data(
                query_adata,
                model_path,
                freeze_dropout=True,
            )

            #Check if mapping supervised, unsupervised or semi-supervised
            if supervised:
                model._unlabeled_indices = []
                model._labeled_indices = query_adata.n_obs
            else:
                model._unlabeled_indices = np.arange(query_adata.n_obs)
                model._labeled_indices = []

        elif model_type == "scvi":
            scarches.models.SCVI.prepare_query_anndata(query_adata, model_path)

            #Setup adata internals for mapping
            scarches.models.SCVI.setup_anndata(query_adata, batch_key=batch_key, labels_key=cell_type_key)

            #Load scvi model with query
            model = scarches.models.SCVI.load_query_data(
                query_adata,
                model_path,
                freeze_dropout=True,
            )

        #Map the query onto reference
        lr=0.001

        try: 
            model.train(
                max_epochs=max_epochs,
                plan_kwargs=dict(weight_decay=0.0,lr=lr),
                check_val_every_n_epoch=10,
            )
        except ValueError as e:
            if "Expected parameter loc" in str(e):
                raise ValueError("Please check that your anndata object has raw counts (not normalized) saved in adata.X. Mapping can only occur with raw count data.") from e
            else:
                raise

        if "X_scanvi_emb" in reference_adata.obsm:
            print("__________getting X_latent_qzm from minified atlas for scvi-tools models___________")
            qzm = reference_adata.obsm["X_scanvi_emb"]
            reference_adata.obsm["latent_rep"] = qzm

        else:
            raise ValueError("No embedding saved in the reference adata object. Please make sure you save the latent representation of the reference atlas under 'X_latent_qzm' in the reference .obsm attribute.")


        #Save out the latent representation for QUERY
        compute_latent_representation(explicit_representation=query_adata, cell_type_key=cell_type_key, model=model, batch_key=batch_key)

    else:
        model = scarches.models.scPoli.load_query_data(
                adata=query_adata,
                reference_model=model_path,
                labeled_indices=[],
                map_location=torch.device("cpu")
            )

        query_adata = model.adata

        
        try:
            model.train(
                n_epochs=max_epochs,
                pretraining_epochs=40,
                eta=10
            )
        except ValueError as e:
            if "Expected parameter loc" in str(e):
                raise ValueError("Please check that your anndata object has raw counts (not normalized) saved in adata.X. Mapping can only occur with raw count data.") from e
            else:
                raise
        

        if "X_latent_qzm_scpoli" in reference_adata.obsm:

            qzm = reference_adata.obsm["X_latent_qzm_scpoli"]
            reference_adata.obsm["latent_rep"] = qzm

            #Save out the latent representation for QUERY
            compute_latent_representation(explicit_representation=query_adata, cell_type_key=cell_type_key, model=model, batch_key=batch_key)

        else:
            raise ValueError("No embedding saved in the reference adata object. Please make sure you save the latent representation of the reference atlas under 'X_latent_qzm_scpoli' in the reference .obsm attribute.")


    # TODO: Add ref counts and concatenate query and reference, save results for each query

    #save .X and var_names of query in new adata for later concatenation after cellxgene
    adata_query_X = scanpy.AnnData(query_adata.X.copy())
    adata_query_X.var_names = query_adata.var_names
    #we can then zero out .X in original query
    all_zeros = csr_matrix(query_adata.X.shape)

    query_adata.X = all_zeros.copy()


    latent_full_from_mean_var = np.concatenate((reference_adata.obsm["latent_rep"], query_adata.obsm["latent_rep"]))

    query_adata.obs["query"]=["1"]*query_adata.n_obs
    reference_adata.obs["query"]=["0"]*reference_adata.n_obs


    print("concatenating on disk")
    #Added because concat_on_disk only allows inner joins  
    for cell_type_key in cell_type_key_list:
        reference_adata.obs[cell_type_key + '_uncertainty_euclidean'] = pandas.Series(dtype="float32")
        reference_adata.obs[cell_type_key + '_uncertainty_mahalanobis'] = pandas.Series(dtype="float32")
        reference_adata.obs[cell_type_key + '_prediction_xgb'] = reference_adata.obs[cell_type_key]
        reference_adata.obs[cell_type_key + '_prediction_knn'] = reference_adata.obs[cell_type_key]
        reference_adata.obs[cell_type_key + "_prediction_scanvi"] = reference_adata.obs[cell_type_key]
        reference_adata.obs[cell_type_key + "_prediction_scpoli"] = reference_adata.obs[cell_type_key]

        query_adata.obs[cell_type_key] = pandas.Series(dtype="category")

    #Create temp files on disk
    temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
    temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")


    #Write data to temp files
    reference_adata.write_h5ad(temp_reference.name)
    query_adata.write_h5ad(temp_query.name)

    #Concatenate on disk to save memory
    experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)

    query_obs_columns=set(query_adata.obs.columns)
    ref_obs_columns=set(reference_adata.obs.columns)
    columns_only_query = query_obs_columns.difference(ref_obs_columns)
    query_obs = query_adata.obs[columns_only_query].copy()

    del query_adata
    gc.collect()

    print("successfully concatenated")

    #Read concatenated data back in
    combined_adata = scanpy.read_h5ad(temp_combined.name)

    # combined_adata.obs=combined_adata.obs[list(new_columns)]

    print("read concatenated file")

    combined_adata.obsm["latent_rep"] = latent_full_from_mean_var

    combined_adata.obs_names_make_unique()

    combined_adata.obs=pd.concat([combined_adata.obs,query_obs], axis=1)

    print("added latent rep to adata")

    # save results
    combined_adata.write(f"{query_name}_mappped.h5ad")

            




if __name__  == "__main__":

    queries = ["most_80", "least_80","most_90", "least_90"]

    path = "mapping/hlca_tutorial/"
    atlas = "HLCA"
    model_type = "scANVI"
    max_epochs = 100

    reference_adata = sc.read(path + "model_" + atlas.lower() + "_" + model_type.lower() + "/adata.h5ad")
    for query_name in queries:
        print(f"mapping {query_name} file")

        main(query_name)


