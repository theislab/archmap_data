import tempfile

import numpy
import scanpy as sc

from utils import utils 


def download_test_file(directory):
    HLCA_REFERENCE_URL = "https://zenodo.org/record/6337966/files/HLCA_emb_and_metadata.h5ad"

    reference_adata = sc.read(f"{directory}/hlca.h5ad", backup_url=HLCA_REFERENCE_URL)

    return reference_adata

def test_knn():
    with tempfile.TemporaryDirectory() as directory:
        reference_adata = download_test_file(directory)
    
    query_adata = reference_adata[: 100, :]

    label_keys = [f"ann_level_{i}" for i in range(1, 6)] + ["ann_finest_level"]

    # Make sure that before the training there are no predicted labels and uncertainties
    for label in label_keys:
        assert label + "_pred" not in query_adata.obs.columns
        assert label + "_uncertainty" not in query_adata.obs.columns 
    
    query_adata = utils.knn_labels_transfer(source_adata=reference_adata, query_adata=query_adata, n_neighbors=3, uncertainty_threshold=0.2)

    # Make sure that the labels are predicted
    for label in label_keys:
        assert label + "_pred" in query_adata.obs.columns
        assert label + "_uncertainty" in query_adata.obs.columns 


def test_xgboost():
    with tempfile.TemporaryDirectory() as directory:
        reference_adata = download_test_file(directory)
    
    query_adata = reference_adata[: 100, :]

    label_keys = [f"ann_level_{i}" for i in range(1, 6)] + ["ann_finest_level"]

    # Make sure that before the training there are no predicted labels and uncertainties
    for label in label_keys:
        assert label + "_pred" not in query_adata.obs.columns
        assert label + "_uncertainty" not in query_adata.obs.columns 
    
    query_adata = utils.xgboost_labels_transfer(source_adata=reference_adata, query_adata=query_adata, uncertainty_threshold=0.2)

    # Make sure that the labels are predicted
    for label in label_keys:
        assert label + "_pred" in query_adata.obs.columns
        assert label + "_uncertainty" in query_adata.obs.columns 
