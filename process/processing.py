import tempfile
from scarches_api.utils import parameters
import scarches_api.utils.utils as utils
import scanpy as sc
from scvi.model.base import _utils
import pynndescent
import logging

class Preprocess:
    def __init__(self):
        return

    def reference_shrinking(reference_data, configuration):
        '''
        .X: Data matrix obs x var
        .layers: Created from adata.X downstream if removed here
        .obs: Annotations of observations
            .obsm: 
            .obsp:
        .var: Variables
            .varm:
            .varp:
        .uns: Unstructured annotations
        '''
        UNNECESSARY_ANNOTATION = ["adata.var", "adata.uns", "adata.layers", "adata.obsp"]

        '''
        Unnecessary .obs labels
        '''
        UNNECESSARY_LABELS = ['leiden', '', '_scvi_labels', '_scvi_batch']

        #TODO: Has to be shrinked manually (maybe think about more procedural way)
        #HLCA
        unnecessary_obs = [
            "original_celltype_ann",
            "study_long",
            "subject_ID_as_published",
            "age_range",
            "cause_of_death",
            "sequencing_platform",
            "ensembl_release_reference_genome",
            "cell_ranger_version",
            "comments",
            "total_counts",
            "ribo_frac",
            "size_factors",
            "scanvi_label",
            "leiden_1",
            "leiden_2",
            "leiden_3",
            "anatomical_region_ccf_score",
            "leiden_4",
            "leiden_5",
            "original_ann_level_1",
            "original_ann_level_2",
            "original_ann_level_3",
            "original_ann_level_4",
            "original_ann_level_5",
            "original_ann_highest_res",
            "original_ann_new",
            "original_ann_level_1_clean",
            "original_ann_level_2_clean",
            "original_ann_level_3_clean",
            "original_ann_level_4_clean",
            "original_ann_level_5_clean",
            "entropy_subject_ID_leiden_3",
            "entropy_dataset_leiden_3",
            "entropy_original_ann_level_1_leiden_3",
            "entropy_original_ann_level_2_clean_leiden_3",
            "entropy_original_ann_level_3_clean_leiden_3",

            "pre_or_postnatal",
            "cells_or_nuclei",
            "cultured",
            "age"
        ]

        del reference_data.obs[unnecessary_obs]
        
        del reference_data.var
        del reference_data.uns
        #delete it? gets created in archmap_repo from adata.X
        #del reference_data.layers
        del reference_data.obsp
        del reference_data.varp

        #Save shrinked reference data
        reference_name = utils.get_from_config(configuration, parameters.REFERENCE_DATA_PATH)
        output_name = reference_name + "_shrunk.h5ad"

        filename = tempfile.mktemp(suffix=".h5ad")
        sc.write(filename, reference_data)
        utils.store_file_in_s3(filename, output_name)

        return

    def to_drop(adata_obs):
        """
        Checks the adata.obs and makes a list of the labels to be dropped as columns.
        Only "helper" labels are removed, such as leiden, _scvi_batch etc.
        """
        UNWANTED_LABELS = ['leiden', '', '_scvi_labels', '_scvi_batch']

        drop_list = []
        print(adata_obs)
        for attr in UNWANTED_LABELS:
            if attr in adata_obs:
                drop_list.append(attr)
        print(drop_list)
        return drop_list

    def pre_process_data(configuration):
        """
        Used to pre-process the adata objects.\n
        After reading the .h5ad files, it makes the distinction ref/query, removes sparsity
        and reintroduces the counts layer if it has been deleted during sparsity removal.
        """
        source_adata = utils.read_h5ad_file_from_s3(utils.get_from_config(configuration, parameters.REFERENCE_DATA_PATH))
        target_adata = utils.read_h5ad_file_from_s3(utils.get_from_config(configuration, parameters.QUERY_DATA_PATH))
        source_adata.obs["type"] = "reference"
        target_adata.obs["type"] = "query"
        #TODO: HARDCODING---------------------------------------------------
        if utils.get_from_config(configuration, parameters.ATLAS) == 'human_lung':
            X_train = source_adata.X
            ref_nn_index = pynndescent.NNDescent(X_train)
            ref_nn_index.prepare()
        # source_adata.raw = source_adata
        #-------------------------------------------------------------------

        try:
            source_adata = utils.remove_sparsity(source_adata)
        except Exception as e:
            pass
        try:
            target_adata = utils.remove_sparsity(target_adata)
        except Exception as e:
            pass
        try:
            source_adata.layers['counts']
        except Exception as e:
            source_adata.layers['counts'] = source_adata.X.copy()
            print("counts layer source")

        try:
            target_adata.layers['counts']
        except Exception as e:
            target_adata.layers['counts'] = target_adata.X.copy()
            print("counts layer query")

        return source_adata, target_adata

    def set_keys(configuration):
        """
        Sets the batch(condition) and cell_type keys, according to the atlas chosen.
        This is necessary as the reference files all have these keys under different names,
        although they contain the same information.
        """
        #TODO: Medium hardcoding due to file differences
        atlas = utils.get_from_config(configuration, 'atlas')
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

    def set_keys_alternative(configuration):
        #TODO: Use relative path
        model_path = "assets/scVI/fetal_immune/"

        #Get label names the model was setup with
        attr_dict = _utils._load_saved_files(model_path, False, None)[0]
        registry = attr_dict.pop("registry_")
        setup_args = registry["setup_args"]

        #Set parameters according to model labels if provided
        if setup_args["labels_key"] is not None:
            configuration[parameters.CELL_TYPE_KEY] = setup_args["labels_key"]
        if setup_args["batch_key"] is not None:
            configuration[parameters.CONDITION_KEY] = setup_args["batch_key"]
        if setup_args["unlabeled_category"] is not None:
            configuration[parameters.UNLABELED_KEY] = setup_args["unlabeled_category"]

        return

class Postprocess:
    def __init__(self):
        return

    def output_csv(self, obs_to_drop: list, latent_adata: sc.AnnData, predict_scanvi, filename=tempfile.mktemp()):
        final = latent_adata.obs.drop(columns=obs_to_drop)

        final["x"] = list(map(lambda p: p[0], latent_adata.obsm["X_umap"]))
        final["y"] = list(map(lambda p: p[1], latent_adata.obsm["X_umap"]))

        try:
            if predict_scanvi:
                cell_types = list(map(lambda p: p, latent_adata.obs['cell_type']))
                predictions = list(map(lambda p: p, latent_adata.obs['predicted']))
                for i in range(len(cell_types)):
                    if cell_types[i] == self.config[parameters.UNLABELED_KEY]:
                        cell_types[i] = predictions[i]
                        predictions[i] = 'yes'
                    else:
                        predictions[i] = 'no'
            final['cell_type'] = cell_types
            final['predicted'] = predictions
        except Exception as e:
            logging.warning(msg = e)

        final.to_csv(filename)
        return filename

    def output_cxg(latent_adata, config):
        #Cellxgene data format requirements
        #1. Expression values in adata.X
        if latent_adata.X is None:
            try:
                latent_adata.X = latent_adata.raw
            except Exception as e:
                logging.warning(msg = e)

        #2. Embedding in adata.obsm
        if "X_umap" not in latent_adata.obsm:
            #Get specified amount of neighbours for computation
            n_neighbors=config[parameters.NUMBER_OF_NEIGHBORS]

            sc.pp.neighbors(latent_adata, n_neighbors)
            sc.tl.umap(latent_adata)

        #3. Unique var index identifier
        latent_adata.var_names_make_unique

        #4. Unique obs index identifier
        latent_adata.obs_names_make_unique

        #Save as .h5ad
        #TODO: Check output_path parameter. Still usable if multiple outputs possible?
        output_path = config[parameters.OUTPUT_PATH], "/output_cxg.h5ad"

        filename = tempfile.mktemp(suffix=".h5ad")
        sc.write(filename, latent_adata)
        utils.store_file_in_s3(filename, "output_cxg.h5ad")