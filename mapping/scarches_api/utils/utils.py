import os
import tempfile

import pynndescent
import numpy
import numba

from utils import parameters
import requests
import boto3
from aiohttp import ClientError
import scanpy
from pathlib import Path
import pandas as pd
# from scarches.dataset.trvae.data_handling import remove_sparsity
import traceback
from anndata import experimental
from anndata.experimental import write_elem, read_elem
import h5py

UNWANTED_LABELS = ['leiden', '', '_scvi_labels', '_scvi_batch']


def get_from_config(configuration, key):
    """
    Gets a key value from the config file 
    """

    if key in configuration:
        return configuration[key]
    return None

def to_drop(adata_obs):
    """
    Checks the adata.obs and makes a list of the labels to be dropped as columns.
    Only "helper" labels are removed, such as leiden, _scvi_batch etc.
    """
    drop_list = []
    print(adata_obs)
    for attr in UNWANTED_LABELS:
        if attr in adata_obs:
            drop_list.append(attr)
    print(drop_list)
    return drop_list


def write_latent_csv(latent, key=None, filename=tempfile.mktemp(), drop_columns=None, predictScanvi=False,
                     configuration={}):
    """
    stores a given latent in a file, and if a key is given also in an s3 bucket
    :param latent: data to be saved
    :param key: s3 key
    :param filename: local filename
    :param drop_columns: not needed columns
    :return:
    """

    drop_columns = to_drop(latent.obs_keys())
    #TODO: This is HARDCODING for the lung atlas
    if get_from_config(configuration, parameters.ATLAS) == 'hlca':
        drop_columns = ['sample', 'study_long', 'study', 'last_author_PI', 'subject_ID', 'sex', 'ethnicity',
                        'mixed_ethnicity', 'smoking_status', 'BMI', 'condition', 'subject_type', 'sample_type',
                        'single_cell_platform', "3'_or_5'", 'sequencing_platform', 'cell_ranger_version',
                        'fresh_or_frozen', 'dataset', 'anatomical_region_level_1', 'anatomical_region_level_2',
                        'anatomical_region_level_3', 'anatomical_region_highest_res', 'age', 'ann_highest_res',
                        'n_genes', 'log10_total_counts', 'mito_frac', 'ribo_frac', 'original_ann_level_1',
                        'original_ann_level_2', 'original_ann_level_3', 'original_ann_level_4', 'original_ann_level_5',
                        'scanvi_label', 'leiden_1', 'leiden_2', 'leiden_3', 'anatomical_region_ccf_score',
                        'entropy_study_leiden_3', 'entropy_dataset_leiden_3', 'entropy_subject_ID_leiden_3',
                        'entropy_original_ann_level_1_leiden_3', 'entropy_original_ann_level_2_clean_leiden_3',
                        'entropy_original_ann_level_3_clean_leiden_3', 'entropy_original_ann_level_4_clean_leiden_3',
                        'entropy_original_ann_level_5_clean_leiden_3', 'leiden_4', 'reannotation_type', 'leiden_5',
                        'ann_finest_level', 'ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4', 'ann_level_5']

    final = latent.obs.drop(columns=drop_columns)
    #TODO: HARDCODING for the lung atlas
    if get_from_config(configuration, parameters.ATLAS) == 'hlca':
        final['ann_coarse_for_GWAS_and_modeling'] = final['ann_coarse_for_GWAS_and_modeling'].astype(str).apply(
            lambda cell: '' if cell == 'nan' else cell)
        final['ann_level_1_pred'] = final['ann_level_1_pred'].astype(str).apply(
            lambda cell: '' if cell == 'nan' else cell)
        final['cell_type'] = final['ann_coarse_for_GWAS_and_modeling'] + (final['ann_level_1_pred'])

        final = final.drop(columns=['ann_coarse_for_GWAS_and_modeling', 'ann_level_1_pred', 'ann_level_1_uncertainty',
                                    'ann_level_2_pred', 'ann_level_2_uncertainty', 'ann_level_3_pred',
                                    'ann_level_3_uncertainty',
                                    'ann_level_4_pred', 'ann_level_4_uncertainty', 'ann_level_5_pred',
                                    'ann_level_5_uncertainty',
                                    'ann_finest_level_pred', 'ann_finest_level_uncertainty'])

        final['predicted'] = list(map(lambda p: 'yes' if p == 'query' else 'no', final['type']))

    #Gets the umap coordinates
    final["x"] = list(map(lambda p: p[0], latent.obsm["X_umap"]))
    final["y"] = list(map(lambda p: p[1], latent.obsm["X_umap"]))

    #Merging the columns cell_type and predicted to fill in 'Unknown'(which is our unlabelled key) cell types
    #We make a separate column "predicted" with values yes/no. Wherever a merge is done, it's marked as a yes in the predicted column, else no
    try:
        if predictScanvi and get_from_config(configuration, parameters.ATLAS) != 'hlca':
            cell_types = list(map(lambda p: p, latent.obs['cell_type']))
            predictions = list(map(lambda p: p, latent.obs['predicted']))
            for i in range(len(cell_types)):
                if cell_types[i] == get_from_config(configuration, parameters.UNLABELED_KEY):
                    cell_types[i] = predictions[i]
                    predictions[i] = 'yes'
                else:
                    predictions[i] = 'no'
            final['cell_type'] = cell_types
            final['predicted'] = predictions
    except Exception as e:
        traceback.print_exc()
        print("no prediction column found")

    final.to_csv(filename)

    if key is not None:
        store_file_in_s3(filename, key)
    return filename


def prediction_value(cell_type):
    """
    Decides whether a cell is predicted or not
    """
    if cell_type is None:
        return "yes"
    else:
        return "no"

    
def mark_uncertain_labels(query_adata, obs_col, uncertainty_threshold=0.2):
    """
    Label cells with uncertain predictions as "Unknown"
    :param query_adata: AnnData object with labels and uncertainty
    :param obs_col: Column with labels (probably, cell types) in `query_adata.obs` for which uncertainty is calculated. Uncertainty must be stored in "`obs_col`_uncertainty" .obs column
    :param uncertainty_threshold: Predictions with higher uncertainty will be marked as "Unknown"
    :return: Updated AnnData object with uncertain labels marked as "Unknown"
    """
    mask = query_adata.obs[obs_col + "_uncertainty"] > uncertainty_threshold
    print(f"{obs_col}: {sum(mask) / len(mask)} unknown")
    query_adata.obs.loc[mask, obs_col + "_pred"] = "Unknown"

    return query_adata


def knn_labels_transfer(source_adata, query_adata, n_neighbors=8, uncertainty_threshold=0.2):
    """
    Transfer cell labels using K-Nearest Neighbors classifier
    :param source_adata: AnnData object with reference on which the classifier is trained
    :param query_adata: AnnData object with query for which cell types are predicted
    :param n_neighbors: How many nearest neighbors to use for the prediction
    :param uncertainty_threshold: Predictions with higher uncertainty will be marked as "Unknown"
    :return: Updated AnnData object with predicted labels and uncertainties
    """

    @numba.njit
    def weighted_prediction(weights, ref_cats):
        """Get highest weight category."""
        N = len(weights)
        predictions = numpy.zeros((N,), dtype=ref_cats.dtype)
        uncertainty = numpy.zeros((N,))
        for i in range(N):
            obs_weights = weights[i]
            obs_cats = ref_cats[i]
            best_prob = 0
            for c in numpy.unique(obs_cats):
                cand_prob = numpy.sum(obs_weights[obs_cats == c])
                if cand_prob > best_prob:
                    best_prob = cand_prob
                    predictions[i] = c
                    uncertainty[i] = max(1 - best_prob, 0)

        return predictions, uncertainty


    def distances_to_affinities(distances):
        stds = numpy.std(distances, axis=1)
        stds = (2.0 / stds) ** 2
        stds = stds.reshape(-1, 1)
        distances_tilda = numpy.exp(-numpy.true_divide(distances, stds))

        return distances_tilda / numpy.sum(distances_tilda, axis=1, keepdims=True)


    ref_nn_index = pynndescent.NNDescent(source_adata.X)
    ref_nn_index.prepare()
    ref_neighbors, ref_distances = ref_nn_index.query(query_adata.X, k=n_neighbors)
    weights = distances_to_affinities(ref_distances)

    # for each annotation level, get prediction and uncertainty
    label_keys = [f"ann_level_{i}" for i in range(1, 6)] + ["ann_finest_level"]

    for l in label_keys:
        ref_cats = source_adata.obs[l].cat.codes.to_numpy()[ref_neighbors]
        p, u = weighted_prediction(weights, ref_cats)
        p = numpy.asarray(source_adata.obs[l].cat.categories)[p]
        query_adata.obs[l + "_pred"], query_adata.obs[l + "_uncertainty"] = p, u

        query_adata = mark_uncertain_labels(query_adata, obs_col=l, uncertainty_threshold=uncertainty_threshold)

    return query_adata


def xgboost_labels_transfer(source_adata, query_adata, uncertainty_threshold=0.2, use_gpu=False):
    """
    Transfer cell labels using K-Nearest Neighbors classifier
    :param source_adata: AnnData object with reference on which the classifier is trained
    :param query_adata: AnnData object with query for which cell types are predicted
    :param uncertainty_threshold: Predictions with higher uncertainty will be marked as "Unknown"
    :return: Updated AnnData object with predicted labels and uncertainties
    """
    import xgboost
    from sklearn.preprocessing import LabelEncoder

    X_train = source_adata.X

    label_keys = [f"ann_level_{i}" for i in range(1, 6)] + ["ann_finest_level"]

    for l in label_keys:
        y_train = source_adata.obs[l]
        labels_encoder = LabelEncoder()
        labels_encoder.fit(y_train)

        n_classes = len(numpy.unique(y_train))
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        
        tree_method = "gpu_hist" if use_gpu else "hist"        
        xgb_model = xgboost.XGBClassifier(tree_method=tree_method, objective=objective)
        xgb_model.fit(X_train, labels_encoder.transform(y_train))

        query_adata.obs[l + "_pred"] = labels_encoder.inverse_transform(xgb_model.predict(query_adata.X))
        
        probs = xgb_model.predict_proba(query_adata.X)
        # Convert probabilities to uncertainties
        df_probs = pd.DataFrame(probs, columns=xgb_model.classes_, index=query_adata.obs_names)
        query_adata.obs[l + "_uncertainty"] = 1 - df_probs.max(1)

        query_adata = mark_uncertain_labels(query_adata, obs_col=l, uncertainty_threshold=uncertainty_threshold)

    return query_adata


def transfer_labels(source_adata, query_adata, use_knn=True, n_neighbors=8, uncertainty_threshold=0.2):
    if use_knn:
        return knn_labels_transfer(source_adata, query_adata, n_neighbors=n_neighbors, uncertainty_threshold=uncertainty_threshold)
    else:
        return xgboost_labels_transfer(source_adata, query_adata, uncertainty_threshold=uncertainty_threshold)


def write_full_adata_to_csv(model, source_adata, target_adata, key=None, filename=tempfile.mktemp(), drop_columns=None,
                            unlabeled_category='', cell_type_key='', condition_key='', neighbors=8,
                            predictScanvi=False, configuration=None):
    """
    Concatenates reference and query adata.\n
    (Note:) It's separate from write_adata_to_csv(), in order to be able to use write_adata_to_csv() independently:
    to get only a reference csv or only query csv, mostly for debugging purposes
    """    
    adata_full = source_adata.concatenate(target_adata)
    return write_adata_to_csv(model, adata=adata_full, source_adata=source_adata, target_adata=target_adata, key=key,
                              filename=filename, drop_columns=drop_columns,
                              unlabeled_category=unlabeled_category, cell_type_key=cell_type_key,
                              condition_key=condition_key, neighbors=neighbors, predictScanvi=predictScanvi,
                              configuration=configuration)


def write_adata_to_csv(model, adata=None, source_adata=None, target_adata=None, key=None, filename=tempfile.mktemp(),
                       drop_columns=None,
                       unlabeled_category='Unknown', cell_type_key='cell_type',
                       condition_key='study', use_knn=True, neighbors=8, predictScanvi=False, configuration=None):
    """
    Prepares the adata for csv export:\n
    This function computes the latent representation and
    makes sure that the batch, cell_type(when existent), predicted(when existent) and type key are in place in the adata.obs.
    After that neighbours, leiden and umap are computed.

    Args:
        adata (AnnData): full adata, query and reference concatenation. Defaults to None.
        source_adata (AnnData): reference adata. Defaults to None.
        target_adata (AnnData): query adata. Defaults to None.
        drop_columns (str List): columns to be removed. Defaults to None.
        unlabeled_category (str, optional): scANVI unlabeled category's name. Defaults to 'Unknown'.
        cell_type_key (str, optional): name for the cell type key. Defaults to 'cell_type'.
        condition_key (str, optional): name for the batch key. Defaults to 'study'.
        predictScanvi (bool): tells if the model used is scANVI, it's checked for the "predicted column". Defaults to False.

    """    
    anndata = None
    if adata is None:
        anndata = scanpy.AnnData(model.get_latent_representation())
    else:
        try:
            anndata = scanpy.AnnData(model.get_latent_representation(adata=adata))
        except Exception as e:
            anndata = adata

    #TODO: HARDCODING for the lung atlas------------------------------------------------
    if get_from_config(configuration, parameters.ATLAS) == 'hlca':
        
        query_emb = scanpy.AnnData(model.get_latent_representation())
        query_emb.obs_names = target_adata.obs_names
        query_emb = transfer_labels(source_adata=source_adata, query_adata=query_emb, use_knn=use_knn, n_neighbors=neighbors)
        query_emb.obs["dataset"] = "test_dataset_delorey_regev"
        # adata.obsm["X_mde"] = mde(adata.X, init="random")
        anndata = source_adata.concatenate(query_emb)
    #HARDCODING ends here---------------------------------------------------------------------
    latent = anndata
    latent.obs['cell_type'] = adata.obs[cell_type_key].tolist()
    latent.obs['batch'] = adata.obs[condition_key].tolist()
    latent.obs['type'] = adata.obs['type'].tolist()

    #TODO: If any gene-filtering or reduction of the umap size are to occur, it's most probably supposed to be done below
    print("calculate neighbors")
    scanpy.pp.neighbors(latent)

    if get_from_config(configuration, parameters.ATLAS) != 'hlca':
        print("calculate leiden")
        scanpy.tl.leiden(latent)
    print("create umap")
    scanpy.tl.umap(latent)
    if predictScanvi and get_from_config(configuration, parameters.ATLAS) != 'hlca':
        print("predicting")
        latent.obs['predicted'] = model.predict(adata=adata)
    print("writing csv")
    return write_latent_csv(latent, key, filename, predictScanvi=predictScanvi, configuration=configuration)


# def write_combined_csv(latent_ref, latent_query, key=None, filename=tempfile.mktemp(), drop_columns=None):
#     """
#     stores a given latent in a file, and if a key is given also in an s3 bucket
#     :param latent_ref: reference_data to be saved
#     :param latent_query: query data to be saved
#     :param key: s3 key
#     :param filename: local filename
#     :param drop_columns: not needed columns
#     :return:
#     """
#     if drop_columns is None:
#         drop_columns = []
#     query = latent_query.obs.drop(columns=drop_columns)
#     query["x"] = list(map(lambda p: float(p[0]), latent_query.obsm["X_umap"]))
#     query["y"] = list(map(lambda p: float(p[1]), latent_query.obsm["X_umap"]))
#     query["is_reference"] = ['No'] * len(latent_query.obsm["X_umap"])
#     query.to_csv(filename)
#     reference = latent_ref.obs.drop(columns=drop_columns)
#     reference["x"] = list(map(lambda p: float(p[0]), latent_ref.obsm["X_umap"]))
#     reference["y"] = list(map(lambda p: float(p[1]), latent_ref.obsm["X_umap"]))
#     reference["is_reference"] = ['Yes'] * len(latent_ref.obsm["X_umap"])
#     reference.to_csv(filename, header=False, mode='a')

#     if key is not None:
#         store_file_in_s3(filename, key)


def print_csv(filename):
    """
    Prints out the csv contents. Used for debugging.
    """
    with open(filename, mode='r') as file:
        for row in file.readlines():
            print(row, end='')


def save_umap_as_pdf(latent, filepath, color=None, wspace=0.6):
    """
    Saves the latent adata as a pdf
    """
    if color is None:
        color = []
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    scanpy.pl.umap(latent,
                   color=color,
                   frameon=False,
                   wspace=wspace,
                   show=False,
                   save=True
                   )
    os.rename('figures/umap.pdf', filepath)


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


def delete_file(file):
    """
    deletes a file is found, otherwise does nothing
    :param file: file to delete
    :return:
    """
    if os.path.isfile(file):
        os.remove(file)


def read_atlas_from_s3(key):
    """
    reads an atlas from s3
    param key: key to the atlas
    """
    if key is None or len(key) == 0:
        return None
    # print("Making tmp file for ", key)
    filename = os.path.join("/mnt/gcs", key)
    
    # print("Fetching file from s3 with Bucket. key. path ", os.getenv('AWS_BUCKET'), key, filename )
    # fetch_file_from_s3(key, filename)
    print("Reading file from ", filename)
    data = scanpy.read(filename)
    # delete_file(filename)
    # print("File deleted")
    return data

def read_h5ad_file_from_s3(key):
    """
    downloads an .h5ad file from s3, reads the data and deletes the file
    :param key:
    :return:
    """
    if key is None or len(key) == 0:
        return None
    filename = tempfile.mktemp(suffix=".h5ad")
    print("The key to download is ", key)
    fetch_file_from_s3(key, filename)
    print("Reading file from ", filename)
    
    data = scanpy.read(filename)
    delete_file(filename)
    print("File deleted")
    return data

def check_model_atlas_compatibility(model, atlas):
    """
    Checks for model-atlas compatibility
    """
    compatible_atlases = []
    if model == 'scVI':
        compatible_atlases = ['pancreas', 'heart', 'hlca', 'retina', 'fetal_immune']
    if model == 'scANVI':
        compatible_atlases = ['pancreas', 'heart', 'hlca', 'retina', 'fetal_immune']
    if model == 'totalVI':
        compatible_atlases = ['pmbc', 'bone_marrow']
    return atlas in compatible_atlases


# def pre_process_data(configuration):
#     """
#     Used to pre-process the adata objects.\n
#     After reading the .h5ad files, it makes the distinction ref/query, removes sparsity
#     and reintroduces the counts layer if it has been deleted during sparsity removal.
#     """
#     source_adata = read_h5ad_file_from_s3(get_from_config(configuration, parameters.REFERENCE_DATA_PATH))
#     target_adata = read_h5ad_file_from_s3(get_from_config(configuration, parameters.QUERY_DATA_PATH))
#     source_adata.obs["type"] = "reference"
#     target_adata.obs["type"] = "query"
#     #TODO: HARDCODING---------------------------------------------------
#     if get_from_config(configuration, parameters.ATLAS) == 'human_lung':
#         X_train = source_adata.X
#         ref_nn_index = pynndescent.NNDescent(X_train)
#         ref_nn_index.prepare()
#     # source_adata.raw = source_adata
#     #-------------------------------------------------------------------



#     ### NEW IMPLEMENTATION ###
#     var_names = _utils._load_saved_files("assets/scVI/retina/", False, None,  "cpu")[1]

#     genes = source_adata.var.index[source_adata.var.index.isin(var_names)].tolist()

#     adata_sub = source_adata[:,genes].copy()
#     # Pad object with 0 genes if needed
#     # Genes to pad with
#     genes_to_add = set(var_names).difference(set(adata_sub.var_names))

#     genes_to_add = list(genes_to_add)

#     df_padding = pd.DataFrame(data=numpy.zeros((adata_sub.shape[0],len(genes_to_add))), index=adata_sub.obs_names, columns=genes_to_add)
#     adata_padding = scanpy.AnnData(df_padding)
#     # Concatenate object
#     adata_sub = scanpy.concat([adata_sub, adata_padding], axis=1, join='outer', index_unique=None, merge='unique')

#     # and order:
#     adata_sub = adata_sub[:,var_names].copy()
#     ### NEW IMPLEMENTATION ###
    


#     try:
#         source_adata = remove_sparsity(source_adata)
#     except Exception as e:
#         pass
#     try:
#         target_adata = remove_sparsity(target_adata)
#     except Exception as e:
#         pass
#     try:
#         source_adata.layers['counts']
#     except Exception as e:
#         source_adata.layers['counts'] = source_adata.X.copy()
#         print("counts layer source")

#     try:
#         target_adata.layers['counts']
#     except Exception as e:
#         target_adata.layers['counts'] = target_adata.X.copy()
#         print("counts layer query")

#     return source_adata, target_adata


def translate_atlas_to_directory(configuration):
    """
    Translates the atlas names, received in the request as per the JSON atlas information file,
    into the folder names for the pre-trained atlases. 
    """
    #TODO: get rid of this function!
    atlas = get_from_config(configuration, 'atlas')
    if atlas == 'Pancreas':
        return 'pancreas'
    elif atlas == 'PBMC':
        return 'pbmc'
    elif atlas == 'Heart':
        return 'heart'
    elif atlas == 'HLCA':
        return 'hlca'
    elif atlas == 'HLCA retrained':
        return 'hlca_retrained'
    elif atlas == 'Bone marrow':
        return 'bone_marrow'
    elif atlas == 'Retina':
        return 'retina'
    elif atlas == 'Fetal immune':
        return 'fetal_immune'
    elif atlas == 'Glioblastoma':
        return 'gb'
    elif atlas == 'NSCLC':
        return 'nsclc'
    elif atlas == 'Hypomap':
        return 'hypomap'
    elif atlas == 'Tonsil':
        return 'tonsil'
    elif atlas == 'Gut':
        return 'gut'
    elif atlas == "HRCA":
        return "HRCA"
    elif atlas == "HNOCA":
        return "hnoca"
    elif atlas == "HEOCA":
        return "heoca"
    elif atlas == "Fetal Brain":
        return "fetal_brain"
    elif atlas == "HNOCA Extended":
        return "hnoca_extended"
    elif atlas == "Breast":
        return "breast"
    elif atlas == "Plaque":
        return "plaque"


def rename_duplicate_columns(df):
    # Check if the index has a name and if that name exists as a column in the DataFrame
    if df.index.name in df.columns:
        # Compare the index values with the corresponding column values
        index_values = df.index.to_series()  # Convert index to series
        column_values = df[df.index.name]
        
        # If index values and column values are not the same, rename the column
        if not index_values.equals(column_values):
            new_column_name = df.index.name + '_duplicate'
            
            # Ensure the new column name is unique
            i = 1
            while new_column_name in df.columns:
                new_column_name = df.index.name + f'_duplicate_{i}'
                i += 1
            
            # Rename the column
            df = df.rename(columns={df.index.name: new_column_name})
            
            print(f"Column '{df.index.name}' renamed to '{new_column_name}' to avoid conflict with the index name.")

    return df

def set_keys(configuration):
    """
    Sets the batch(condition) and cell_type keys, according to the atlas chosen.
    This is necessary as the reference files all have these keys under different names,
    although they contain the same information.
    """
    #TODO: Medium hardcoding due to file differences
    atlas = get_from_config(configuration, 'atlas')
    if atlas == 'Pancreas':
        return configuration
    elif atlas == 'PBMC':
        return configuration
    elif atlas == 'Heart cell atlas':
        configuration[parameters.CELL_TYPE_KEY] = 'cell_type'
        configuration[parameters.CONDITION_KEY] = 'source'
        configuration[parameters.USE_PRETRAINED_SCANVI_MODEL] = False
        return configuration
    elif atlas == 'Human lung cell atlas':
        configuration[parameters.CELL_TYPE_KEY] = 'scanvi_label'
        configuration[parameters.CONDITION_KEY] = 'dataset'
        configuration[parameters.UNLABELED_KEY] = 'unlabeled'
        return configuration
    elif atlas == 'Bone marrow':
        # configuration[parameters.CELL_TYPE_KEY] = 'cell_type'
        # configuration[parameters.CONDITION_KEY] = 'source'
        return configuration
    elif atlas == 'Retina atlas':
        configuration[parameters.CELL_TYPE_KEY] = 'CellType'
        configuration[parameters.CONDITION_KEY] = 'batch'
        return configuration
    elif atlas == 'Fetal immune atlas':
        configuration[parameters.CELL_TYPE_KEY] = 'cell_name'
        configuration[parameters.CONDITION_KEY] = 'bbk'
        configuration[parameters.USE_PRETRAINED_SCANVI_MODEL] = False
        return configuration
    
def get_file_size_in_gb(key):
    """
    Fetches the size of a file in S3 and returns it in gigabytes.
    :param key: key in s3
    :return: size in GB
    """
    client = boto3.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT'),
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))

    # Get file metadata
    response = client.head_object(Bucket=os.getenv('AWS_BUCKET'), Key=key)
    size_bytes = response['ContentLength']

    # Convert size to gigabytes
    size_gb = size_bytes / (1024 ** 3)
    return size_gb


def fetch_file_to_temp_path_from_s3(key):
    """
    Downloads an .h5ad file from s3 and saves it to the /tmp directory, returning the file path.
    :param key: The S3 key for the .h5ad file.
    :return: The path to the downloaded file.
    """
    if key is None or len(key) == 0:
        return None
    
    # Create a temporary file path in /tmp
    filename = tempfile.mktemp(suffix=".h5ad", prefix="temp_", dir="/tmp")
    
    # Download the file
    fetch_file_from_s3(key, filename)
    print(f"File downloaded to {filename}")
    
    return filename


def replace_X_on_disk(combined_adata,temp_output, query_X_file, ref_count_matrix_path, use_downsample=False):
    """
    Writes combined_adata to disk, fetches another .h5ad file specified by ref_count_matrix_path.
    Concatenates the .X of the fetched file with query_X_file.
    Writes concatenated adata metadata (eg .obs and .vars and .varm and .obsm and uns) to adata with .X on disk
  
    :param combined_adata: The Anndata object in memory.
    :param temp_output: Temp file to write combined_adata to.
    :param query_X_file: File with .X for query.
    :param ref_count_matrix_path: The S3 key for the .h5ad file to be fetched.

    Returns: File path to saved adata with concatenated metadata and .X
    """

    
    temp_ref_count_matrix_path = fetch_file_to_temp_path_from_s3(ref_count_matrix_path)
    # Fetch the new file and get its path
    if use_downsample:
        # get obs index
        with h5py.File(temp_ref_count_matrix_path, "r") as f:
            obs_df=read_elem(f["obs"])
            obs_index = obs_df.index

            # add query index
            query_index = combined_adata.obs[combined_adata.obs["query"] == "1"].index
            obs_index_combined =obs_index.append(query_index)
            combined_adata = combined_adata[obs_index_combined]

    # Write combined_adata to disk
    combined_adata.write(temp_output)
    print(f"combined_adata written to {temp_output}")
    if temp_ref_count_matrix_path is None:
        print("No file fetched. Exiting.")
        return
    print("Fetched reference adata with counts")
    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    
    experimental.concat_on_disk([temp_ref_count_matrix_path, query_X_file], temp_combined.name)
    print("Concatenated reference and query counts")

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
        print("Added concatenated metadata to anndata with full .X on disk")

    return temp_combined.name

