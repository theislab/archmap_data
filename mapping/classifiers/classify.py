import scanpy as sc
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
from classifiers import Classifiers


def main(atlas, label):

    adata=sc.read(f"data/{atlas}.h5ad")

    reference_latent=sc.AnnData(adata.obsm["latent_rep"], adata.obs)

    #create knn classifier
    clf = Classifiers(False, True, None)
    clf.create_classifier(reference_latent, False, "", label, f"models_new/{atlas}_{label}")

    #create xgb classifier
    clf = Classifiers(True, False, None)
    clf.create_classifier(reference_latent, False, "", label, f"models_new/{atlas}_{label}")


if __name__ == "__main__":

    atlas_dict = {"hlca": "ann_level_4",
                  "hlca_retrained": "ann_finest_level",
                  "gb": "CellID",
                  "fetal_immune":"celltype_annotation",
                  "nsclc": "cell_type",
                  "retina": "CellType",
                  "pancreas_scpoli":"cell_type",
                  "pancreas_scanvi":"cell_type",
                  "pbmc":"cell_type_for_integration",
                  "hypomap": "Author_CellType",
                  }
    atlas_dict_scpoli = {
                  "hlca_retrained": "ann_finest_level",           
                  "pancreas_scpoli":"cell_type",
                  "pbmc":"cell_type_for_integration",
                #   "hnoca":"annot_level_2",
                #   "heoca": "cell_type"
                  }

    for atlas, label in atlas_dict.items():
        main(atlas, label)
        print(f"successfully created classifier for {atlas}")
    for atlas, label in atlas_dict_scpoli.items():
        main(atlas, label, is_scpoli=True) 
        print(f"successfully created classifier for {atlas}")