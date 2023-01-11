from utils.utils import read_h5ad_file_from_s3
from utils.utils import store_file_in_s3
import scanpy
import scarches
import tempfile
import numpy as np
import scvi

from scvi.model.base import _utils
from init import query
import utils.parameters as parameters

#import scib-metrics
#import metrics.metrics as metrics

def main():
    # adata = read_h5ad_file_from_s3("atlas_626ea3311d7d1a27de465b64_data.h5ad")
    # adata2 = read_h5ad_file_from_s3("pbmc_totalVI.h5ad")

    hlca = scanpy.read_h5ad("HLCA_v2_reduced.h5ad")
    # nsclc = scanpy.read_h5ad("NSCLC_core.h5ad")

    scarches.models.SCANVI.load("assets/scANVI/human_lung/v2", hlca)

    # model_state_dict = torch.load("assets/scANVI/human_lung/v2/model_params.pt", "cpu")

    # model = test.BaseMixin.load("assets/scANVI/human_lung/v2/", hlca)
    # test.BaseMixin._load_params("assets/scANVI/human_lung/v2/", "cpu")

    return

    # adata = read_h5ad_file_from_s3("HLCA_v2_reduced.h5ad")

    # print("Loaded data")
    # print("Removing unncessary .obs")

    # print(adata)

    # filename = tempfile.mktemp(suffix=".h5ad")
    # scanpy.write(filename, adata)
    # store_file_in_s3(filename, "HLCA_v2_reduced.h5ad")

    # return

    configuration = {
    "model": "scVI",
    "atlas": "Fetal immune atlas",
    "output_path": "fetal_scvi",
    "output_type": {
        "csv": True,
        "cxg": True,
    },
    "model_path": "model.pt",
    "pre_trained_scANVI": True,
    "reference_data": "atlas_628668f96f930d8b7f44d57b_data.h5ad",
    "query_data": "fetal_scvi.h5ad",
    "ref_path": "model.pt"
    }

    query(configuration)

    return

    # scvi.model.SCANVI.view_setup_args("assets/scVI/pancreas/")
    # scvi.model.TOTALVI.view_setup_args("assets/totalVI/pbmc/")
    # scvi.model.SCVI.view_setup_args("assets/scVI/heart/")
    # scvi.model.SCVI.view_setup_args("assets/scVI/human_lung/")
    # scvi.model.SCANVI.view_setup_args("assets/scANVI/retina/")
    # scvi.model.SCVI.view_setup_args("assets/scVI/fetal_immune/")

    attr_dict = _utils._load_saved_files("assets/totalVI/pbmc/", False, None, "cpu")[0]
    registry = attr_dict.pop("registry_")
    setup_args = registry["setup_args"]

    print(setup_args)

    return

    attr_dict = _utils._load_saved_files("assets/scVI/human_lung/", False, None)[0]
    registry = attr_dict.pop("registry_")
    setup_args = registry["setup_args"]

    print(setup_args["batch_key"])

    print(adata)

    configuration = {
    "model": "scANVI",
    "atlas": "Pancreas",
    "output_path": "query",
    "model_path": "model.pt",
    "pre_trained_scANVI": True,
    "reference_data": "atlas_626ea3311d7d1a27de465b63_data.h5ad",
    "query_data": "pancreas_scANVI.h5ad",
    "ref_path": "model.pt"
    }

    # configuration[parameters.CELL_TYPE_KEY] = setup_args["labels_key"]
    # configuration[parameters.CONDITION_KEY] = setup_args["batch_key"]
    # configuration[parameters.UNLABELED_KEY] = setup_args["unlabeled_category"]

    query(configuration)

    #print(adata)

    #Check following .obs labels
    # print(adata.obs["cultured"])
    # print(adata.obs["cultured"].loc[adata.obs["cultured"] == "nan"])
    # print(adata.obs["age"].compare(adata.obs["age_in_years"]))

    return

    del adata.obs["original_celltype_ann"]
    del adata.obs["study_long"]
    del adata.obs["subject_ID_as_published"]
    del adata.obs["age_range"]
    del adata.obs["cause_of_death"]
    del adata.obs["sequencing_platform"]
    del adata.obs["ensembl_release_reference_genome"]
    del adata.obs["cell_ranger_version"]
    del adata.obs["comments"]
    del adata.obs["total_counts"]
    del adata.obs["ribo_frac"]
    del adata.obs["size_factors"]
    del adata.obs["scanvi_label"]
    del adata.obs["leiden_1"]
    del adata.obs["leiden_2"]
    del adata.obs["leiden_3"]
    del adata.obs["anatomical_region_ccf_score"]
    del adata.obs["leiden_4"]
    del adata.obs["leiden_5"]
    del adata.obs["original_ann_level_1"]
    del adata.obs["original_ann_level_2"]
    del adata.obs["original_ann_level_3"]
    del adata.obs["original_ann_level_4"]
    del adata.obs["original_ann_level_5"]
    del adata.obs["original_ann_highest_res"]
    del adata.obs["original_ann_new"]
    del adata.obs["original_ann_level_1_clean"]
    del adata.obs["original_ann_level_2_clean"]
    del adata.obs["original_ann_level_3_clean"]
    del adata.obs["original_ann_level_4_clean"]
    del adata.obs["original_ann_level_5_clean"]
    del adata.obs["entropy_subject_ID_leiden_3"]
    del adata.obs["entropy_dataset_leiden_3"]
    del adata.obs["entropy_original_ann_level_1_leiden_3"]
    del adata.obs["entropy_original_ann_level_2_clean_leiden_3"]
    del adata.obs["entropy_original_ann_level_3_clean_leiden_3"]

    del adata.obs["pre_or_postnatal"]
    del adata.obs["cells_or_nuclei"]
    del adata.obs["cultured"]
    del adata.obs["age"]

    
    del adata.var
    del adata.uns
    #delete it? gets created in archmap_repo from adata.X
    del adata.layers
    del adata.obsp
    #del adata.varp

    print("Removed unncessary .obs")
    print("Saving...")

    filename = tempfile.mktemp(suffix=".h5ad")
    scanpy.write(filename, adata)
    store_file_in_s3(filename, "HLCA_v2_reduced.h5ad")

if __name__ == "__main__":
    main()