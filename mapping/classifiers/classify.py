import scanpy as sc
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
from classifiers import Classifiers


def main(atlas, label, is_scpoli=False):

    adata=sc.read(f"data/{atlas}.h5ad")

    if is_scpoli:
        reference_latent=sc.AnnData(adata.obsm["X_latent_qzm_scpoli"], adata.obs)

    else:
        reference_latent=sc.AnnData(adata.obsm["X_latent_qzm"], adata.obs)

    if isinstance(label, list):
        for l in label:
            #create knn classifier
            clf = Classifiers(False, True, None)
            clf.create_classifier(reference_latent, True, "", l, f"models_new/{atlas}/{atlas}_{l}")

            #create xgb classifier
            clf = Classifiers(True, False, None)
            clf.create_classifier(reference_latent, True, "", l, f"models_new/{atlas}/{atlas}_{l}")


    else:
        #create knn classifier
        clf = Classifiers(False, True, None)
        clf.create_classifier(reference_latent, True, "", label,f"models_new/{atlas}/{atlas}_{label}")

        #create xgb classifier
        clf = Classifiers(True, False, None)
        clf.create_classifier(reference_latent, True, "", label, f"models_new/{atlas}/{atlas}_{label}")


if __name__ == "__main__":

    # atlas_dict = {"fetal_brain": "subregion_class"}

    atlas_dict = {"hnoca_new":
    ['annot_level_1',
    'annot_level_2',
    'annot_level_3_rev2',
    'annot_level_4_rev2',
    'annot_region_rev2',
    'annot_ntt_rev2',],
    
    "hnoca_ce":  'annot_level_2_ce'}

    # atlas_dict = {"hlca": "ann_level_4",
    #               "hlca_retrained": "ann_finest_level",
    #               "gb": "CellID",
    #               "fetal_immune":"celltype_annotation",
    #               "nsclc": "cell_type",
    #               "retina": "CellType",
    #               "pancreas_scpoli":"cell_type",
    #               "pancreas_scanvi":"cell_type",
    #               "pbmc":"cell_type_for_integration",
    #               "hypomap": "Author_CellType",
    #               }
    # atlas_dict_scpoli = {
    #               "hlca_retrained": "ann_finest_level",           
    #               "pancreas_scpoli":"cell_type",
    #               "pbmc":"cell_type_for_integration",
    #             #   "hnoca":"annot_level_2",
    #             #   "heoca": "cell_type"
    #               }

    for atlas, label in atlas_dict.items():
        main(atlas, label, is_scpoli=True)
        print(f"successfully created classifier for {atlas}")
    # for atlas, label in atlas_dict_scpoli.items():
    #     main(atlas, label, is_scpoli=True) 
    #     print(f"successfully created classifier for {atlas}")