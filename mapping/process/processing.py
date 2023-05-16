import tempfile
from scarches_api.utils import parameters
import scarches_api.utils.utils as utils
import scanpy as sc
import scarches as sca
from scvi.model.base import _utils
import pynndescent
import logging
import pandas as pd
import numpy as np

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

    def drop_unknown_batch_labels(configuration, adata):
        #Get relative model path
        model_path = "assets/" + utils.get_from_config(configuration, parameters.MODEL) + "/" + utils.get_from_config(configuration, parameters.ATLAS) + "/"

        #Get label names the model was set up with
        attr_dict = _utils._load_saved_files(model_path, False, None,  "cpu")[0]
        registry = attr_dict.pop("registry_") 

        field_registries = registry["field_registries"]
        for field in field_registries:
            #Filter out all the batches the model doesnt know
            if field == "batch":
                batch_key = utils.get_from_config(configuration, parameters.CONDITION_KEY) 

                state_registry = field_registries[field]["state_registry"]
                categorical_mapping = state_registry["categorical_mapping"] 
                adata = adata[adata.obs[batch_key].isin(categorical_mapping)].copy()

            # #Filter out all the labels the model doesnt know
            # if field == "labels":
            #     labels_key = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)

            #     state_registry = field_registries[field]["state_registry"]
            #     categorical_mapping = state_registry["categorical_mapping"]
            #     adata = adata[adata.obs[labels_key].isin(categorical_mapping)].copy()

        return adata

        # #Get batches the model knows
        # state_registry_batch = model.adata_manager.get_state_registry("batch")
        # batches = state_registry_batch["categorical_mapping"]

        # #Filter out all the batches the model doesnt know
        # adata = adata[adata.obs["batch"].isin(batches)].copy()

    def conform_vars(configuration, adata):
        #Get relative model path
        model_path = "assets/" + utils.get_from_config(configuration, parameters.MODEL) + "/" + utils.get_from_config(configuration, parameters.ATLAS) + "/"

        #Get var_names from model
        var_names = _utils._load_saved_files(model_path, False, None,  "cpu")[1]

        # test if adata.var.index has gene names or ensembl names:
        n_gene_ids = sum(adata.var.index.isin(var_names))

        if "gene_ids" in adata.var.columns:
            adata.var.index = adata.var["gene_ids"]
        elif "ensembl" in adata.var.columns:
            adata.var.index = adata.var["ensembl"]

        #If adata.var equals model.var nothing to conform
        if(adata.n_vars == len(var_names)):
            return adata

        #Start conforming adata
        # delete obsm and varm to enable concatenation
        del adata.obsm
        del adata.varm

        #Get genes from adata that exist in var_names
        genes = adata.var.index[adata.var.index.isin(var_names)].tolist()
        #Get intersection of adata and var_names genes
        adata_sub = adata[:,genes].copy()
        #Pad object with 0 genes if needed
        #Genes to pad with
        genes_to_add = set(var_names).difference(set(adata_sub.var_names))
        #Convert to list
        genes_to_add = list(genes_to_add)

        if len(genes_to_add) == 0:
            return adata_sub

        df_padding = pd.DataFrame(data=np.zeros((adata_sub.shape[0],len(genes_to_add))), index=adata_sub.obs_names, columns=genes_to_add)
        adata_padding = sc.AnnData(df_padding)
        #Concatenate object
        adata_sub = sc.concat([adata_sub, adata_padding], axis=1, join='outer', index_unique=None, merge='unique')
        #and order:
        adata_sub = adata_sub[:,var_names].copy()
        
        return adata_sub

    def conform_obs(configuration, source_adata, target_adata):
        #Conform labels if different among source and query
        cell_type_key_config = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)
        batch_key_config = utils.get_from_config(configuration, parameters.CONDITION_KEY)

        # if cell_type_key_config not in target_adata.obs.columns:

        return
            
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
        # if utils.get_from_config(configuration, parameters.ATLAS) == 'human_lung':
        #     X_train = source_adata.X
        #     ref_nn_index = pynndescent.NNDescent(X_train)
        #     ref_nn_index.prepare()
        # # source_adata.raw = source_adata
        # #-------------------------------------------------------------------

        #Set label keys
        Preprocess.set_keys_dynamic(configuration, target_adata, source_adata)

        #Adjust .var according to model
        source_adata = Preprocess.conform_vars(configuration, source_adata)
        target_adata = Preprocess.conform_vars(configuration, target_adata)

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

        #TODO: Dont preprocess if using embedding
        if utils.get_from_config(configuration, parameters.USE_REFERENCE_EMBEDDING):
            source_adata = utils.read_h5ad_file_from_s3(utils.get_from_config(configuration, parameters.REFERENCE_DATA_PATH))
            source_adata.obs["type"] = "reference"

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

    def __get_keys_model(configuration):
        #Get relative model path
        model_path = "assets/" + utils.get_from_config(configuration, parameters.MODEL) + "/" + utils.get_from_config(configuration, parameters.ATLAS) + "/"

        #Get label names the model was set up with
        attr_dict = _utils._load_saved_files(model_path, False, None,  "cpu")[0]

        # try:
        #     attr_dict = _utils._load_saved_files(model_path, False, None,  "cpu")[0]
        # except:
        #     if utils.get_from_config(configuration, parameters.MODEL) == "scANVI":
        #         sca.models.SCANVI.convert_legacy_save(model_path, model_path, True)
        #     if utils.get_from_config(configuration, parameters.MODEL) == "scVI":
        #         sca.models.SCVI.convert_legacy_save(model_path, model_path, True)

        #Get model data registry and labels
        #Data management can be different among models, no clear indication in docs
        #Docs: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/data_tutorial.html
        cell_type_key_model = None
        condition_key_model = None
        unlabeled_key_model = None

        if("registry_" not in attr_dict):
            data_registry = attr_dict["scvi_setup_dict_"]["categorical_mappings"]

            cell_type_key_model = data_registry["_scvi_labels"]["original_key"]
            condition_key_model = data_registry["_scvi_batch"]["original_key"]
        else:
            data_registry = attr_dict["registry_"]["field_registries"]

            cell_type_key_model = data_registry["labels"]["state_registry"]["original_key"]
            condition_key_model = data_registry["batch"]["state_registry"]["original_key"]

        if "unlabeled_category_" in attr_dict:
            if attr_dict["unlabeled_category_"] is not None:
                unlabeled_key_model = attr_dict["unlabeled_category_"]

        return cell_type_key_model, condition_key_model, unlabeled_key_model

    def __get_keys_user(configuration):
        #Get parameters from user input
        cell_type_key_user = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)
        condition_key_user = utils.get_from_config(configuration, parameters.CONDITION_KEY)
        unlabeled_key_user = utils.get_from_config(configuration, parameters.UNLABELED_KEY)

        return cell_type_key_user, condition_key_user, unlabeled_key_user

    def set_keys_dynamic(configuration, target_adata, source_adata):
        # #Get relative model path
        # model_path = "assets/" + utils.get_from_config(configuration, parameters.MODEL) + "/" + utils.translate_atlas_to_directory(configuration) + "/"

        # #Get label names the model was set up with
        # attr_dict = _utils._load_saved_files(model_path, False, None,  "cpu")[0]

        # #Get model data registry and labels
        # #Data management can be different among models, no clear indication in docs
        # #Docs: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/data_tutorial.html
        # cell_type_key_model = None
        # condition_key_model = None
        # unlabeled_key_model = None

        # if("registry_" not in attr_dict):
        #     data_registry = attr_dict["scvi_setup_dict_"]["categorical_mappings"]

        #     cell_type_key_model = data_registry["_scvi_labels"]["original_key"]
        #     condition_key_model = data_registry["_scvi_batch"]["original_key"]
        # else:
        #     data_registry = attr_dict["registry_"]["field_registries"]

        #     cell_type_key_model = data_registry["labels"]["state_registry"]["original_key"]
        #     condition_key_model = data_registry["batch"]["state_registry"]["original_key"]

        # #Get parameters from config
        # cell_type_key_config = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)
        # condition_key_config = utils.get_from_config(configuration, parameters.CONDITION_KEY)
        # unlabeled_key_config = utils.get_from_config(configuration, parameters.UNLABELED_KEY)

        # #Use manual input if possible, fallback to saved model if necessary
        # if cell_type_key_config == "cell_type":
        #     if cell_type_key_model is not None:
        #         #if setup_args["labels_key"] is not None:
        #         configuration[parameters.CELL_TYPE_KEY] = cell_type_key_model
        # if condition_key_config == "study":
        #     if condition_key_model is not None:
        #         #if setup_args["batch_key"] is not None:
        #         configuration[parameters.CONDITION_KEY] = condition_key_model
        # if unlabeled_key_config == "Unknown":
        #     if unlabeled_key_model is not None:
        #         #if setup_args["unlabeled_category"] is not None:
        #         configuration[parameters.UNLABELED_KEY] = unlabeled_key_model


        # #TODO: Incorporate information that is not stored in model
        # atlas = utils.get_from_config(configuration, 'atlas')
        # if atlas == 'Heart cell atlas':
        #     configuration[parameters.USE_PRETRAINED_SCANVI_MODEL] = False
        # elif atlas == 'Fetal immune atlas':
        #     configuration[parameters.USE_PRETRAINED_SCANVI_MODEL] = False

        # return




        #Get labels stored in model
        cell_type_key_model, condition_key_model, unlabeled_key_model = Preprocess.__get_keys_model(configuration)

        #Get labels from user input
        cell_type_key_user, condition_key_user, unlabeled_key_user = Preprocess.__get_keys_user(configuration)

        #Get model
        model_type = utils.get_from_config(configuration, parameters.MODEL)

        #Preliminary check for unsupervised model (unlabeled data)
        if model_type == "scVI" or model_type == "totalVI":
            if cell_type_key_model == "_scvi_labels":
                cell_type_key_model = None

        #Check for cell type input from model and user
        if cell_type_key_user is None:
            #If user none, model none
            if cell_type_key_model is None:
                #Unsupervised approach with scVI
                print("No cell type input provided")
                target_adata.obs["CellType"] = "Unlabeled"
                configuration[parameters.CELL_TYPE_KEY] = "CellType"
            #If user none, model input
            else:
                #Unsupervised approach with scANVI
                if cell_type_key_model not in target_adata.obs.columns:
                    target_adata.obs[cell_type_key_model] = unlabeled_key_model
                    configuration[parameters.CELL_TYPE_KEY] = cell_type_key_model
        else:
            #If user input, model none
            if cell_type_key_model is None:
                #Unsupervised approach with scVI                
                configuration[parameters.CELL_TYPE_KEY] = cell_type_key_user
            #If user input, model input
            else:
                #Get cell type labels that are unique to target_adata
                cell_type_differences = set(target_adata.obs[cell_type_key_user]).difference(source_adata.obs[cell_type_key_model])

                #If target_adata not a subset of source_adata prepare for unlabeled scANVI
                if len(cell_type_differences) > 0:
                    target_adata.obs['orig_cell_types'] = target_adata.obs[cell_type_key_user].copy()
                    del target_adata.obs[cell_type_key_user]
                    target_adata.obs[cell_type_key_model] = unlabeled_key_model
                else:
                    #target_adata = sc.AnnData()

                    target_adata.obs[cell_type_key_model] = target_adata.obs[cell_type_key_user]
                    del target_adata.obs[cell_type_key_user]

                configuration[parameters.CELL_TYPE_KEY] = cell_type_key_model


        #Check for condition input from model and user
        if condition_key_user is None:
            #If user none, model none
            if condition_key_model is None:
                #Set custom name for custom condition key
                target_adata.obs["batch"] = "query_batch"

                configuration[parameters.CONDITION_KEY] = "batch"
            #If user none, model input
            else:
                # #Set artificial name for model condition key
                # target_adata.obs[condition_key_model] = "query_batch"
                # source_adata.obs[condition_key_model] = "reference_batch"

                configuration[parameters.CONDITION_KEY] = condition_key_model
        else:
            #If user input, model none
            if condition_key_model is None:
                #Set user input for custom condition key
                target_adata.obs["batch"] = condition_key_user

                configuration[parameters.CONDITION_KEY] = condition_key_user
            #If user input, model input
            else:
                #Set custom name for model condition key
                target_adata.obs.rename({condition_key_user: condition_key_model}, axis=1, inplace=True)
                source_adata.obs.rename({condition_key_user: condition_key_model}, axis=1, inplace=True)

                configuration[parameters.CONDITION_KEY] = condition_key_model


        #Check for unlabeled input from model and user
        if unlabeled_key_user is None:
            #If user none, model none
            if unlabeled_key_model is None:
                #Set custom name for custom condition key
                configuration[parameters.UNLABELED_KEY] = "unlabeled"
            #If user none, model input
            else:
                #Set custom name for model condition key
                configuration[parameters.UNLABELED_KEY] = unlabeled_key_model
        else:
            #If user input, model none
            if unlabeled_key_model is None:
                #Set user input for custom condition key
                configuration[parameters.UNLABELED_KEY] = unlabeled_key_user
            #If user input, model input
            else:
                #Get cell type labels that are unique to target_adata
                configuration[parameters.UNLABELED_KEY] = unlabeled_key_model


        # if cell_type_key_model is not cell_type_key_user:
        #     if cell_type_key_user is not None and cell_type_key_model is not None:
        #         target_adata.obs.rename(columns={cell_type_key_user : cell_type_key_model}, inplace=True)
        #         configuration[parameters.CELL_TYPE_KEY] = cell_type_key_model
        #     else:
        #         raise Exception("No cell type key specified")
            
        # if condition_key_model is not condition_key_user:
        #     #If condition key is provided rename to match model else create a new column with input "query_dataset"
        #     if condition_key_user is not None:
        #         target_adata.obs.rename(columns={condition_key_user : condition_key_model}, inplace=True)
        #     else:
        #         target_adata.obs[condition_key_model] = "query_dataset"

        #     configuration[parameters.CONDITION_KEY] = condition_key_model

        # if unlabeled_key_model is not unlabeled_key_user:
        #     configuration[parameters.UNLABELED_KEY] = unlabeled_key_model

        return

    def scANVI_process_labels(configuration, source_adata, target_adata):
        '''
        If the cell types in target_adata are equal to or a subset of the reference data cell types,
        one can just pass the adata without further preprocessing.

        If however there are new cell types in target_adata use scANVI in unsupervised manner

        (https://scarches.readthedocs.io/en/latest/scanvi_surgery_pipeline.html)
        '''

        #Get relative model path
        model_path = "assets/" + utils.get_from_config(configuration, parameters.MODEL) + "/" + utils.get_from_config(configuration, parameters.ATLAS) + "/"

        #Get model configuration
        attr_dict = _utils._load_saved_files(model_path, False, None,  "cpu")[0]

        #Get cell type labels that are unique to target_adata
        cell_type_key = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)
        cell_type_differences = set(target_adata.obs[cell_type_key]).difference(source_adata.obs[cell_type_key])

        #If target_adata not a subset of source_adata prepare for unlabeled scANVI
        if len(cell_type_differences) > 0:
            target_adata.obs['orig_cell_types'] = target_adata.obs[cell_type_key].copy()
            target_adata.obs[cell_type_key] = attr_dict["unlabeled_category_"]

        return target_adata

class Postprocess:
    def __init__(self):
        return

    def __prepare_output(latent_adata: sc.AnnData, combined_adata: sc.AnnData, config):
        #Get labels from config
        cell_type_key = utils.get_from_config(config, parameters.CELL_TYPE_KEY)
        condition_key = utils.get_from_config(config, parameters.CONDITION_KEY)

        latent_adata.obs['cell_type'] = combined_adata.obs[cell_type_key].tolist()
        latent_adata.obs['batch'] = combined_adata.obs[condition_key].tolist()
        latent_adata.obs['type'] = combined_adata.obs['type'].tolist()
        if("uncertainty" in combined_adata.obs):
            latent_adata.obs['uncertainty'] = combined_adata.obs['uncertainty'].tolist()

        if "X_umap" not in latent_adata.obsm:
            #Get specified amount of neighbours for computation
            n_neighbors=config[parameters.NUMBER_OF_NEIGHBORS]

            sc.pp.neighbors(latent_adata, n_neighbors)
            sc.tl.leiden(latent_adata)
            sc.tl.umap(latent_adata)

    def __output_csv(obs_to_drop: list, latent_adata: sc.AnnData, combined_adata: sc.AnnData, config, predict_scanvi):
        Postprocess.__prepare_output(latent_adata, combined_adata, config)
        
        final = latent_adata.obs.drop(columns=obs_to_drop)

        final["x"] = list(map(lambda p: p[0], latent_adata.obsm["X_umap"]))
        final["y"] = list(map(lambda p: p[1], latent_adata.obsm["X_umap"]))

        try:
            if predict_scanvi:
                cell_types = list(map(lambda p: p, latent_adata.obs['cell_type']))
                predictions = list(map(lambda p: p, latent_adata.obs['predicted']))
                for i in range(len(cell_types)):
                    if cell_types[i] == config[parameters.UNLABELED_KEY]:
                        cell_types[i] = predictions[i]
                        predictions[i] = 'yes'
                    else:
                        predictions[i] = 'no'
            final['cell_type'] = cell_types
            final['predicted'] = predictions
        except Exception as e:
            logging.warning(msg = e)

        #Save as .csv
        output_path = config[parameters.OUTPUT_PATH] + ".csv"

        filename = tempfile.mktemp(suffix=".csv")
        final.to_csv(filename)
        utils.store_file_in_s3(filename, output_path)

    def __output_cxg(latent_adata: sc.AnnData, combined_adata: sc.AnnData, config):
        Postprocess.__prepare_output(latent_adata, combined_adata, config)

        #Cellxgene data format requirements
        #1. Expression values in adata.X
        if latent_adata.X is None:
            try:
                latent_adata.X = latent_adata.raw
            except Exception as e:
                logging.warning(msg = e)

        #2. Embedding in adata.obsm (Handled in __prepare_output as needed for .csv and .h5ad)

        #3. Unique var index identifier
        latent_adata.var_names_make_unique()

        #4. Unique obs index identifier
        latent_adata.obs_names_make_unique()

        #Save as .h5ad
        output_path = config[parameters.OUTPUT_PATH] + "_cxg.h5ad"

        filename = tempfile.mktemp(suffix=".h5ad")
        sc.write(filename, latent_adata)
        utils.store_file_in_s3(filename, output_path)

    def output(latent_adata: sc.AnnData, combined_adata: sc.AnnData, configuration, output_types):
        if(output_types.get("csv")):
            #TODO: Change implementation of dropping unnecessary labels?
            obs_to_drop = []

            Postprocess.__output_csv(obs_to_drop, latent_adata, combined_adata, configuration, True)
        if(output_types.get("cxg")):
            Postprocess.__output_cxg(latent_adata, combined_adata, configuration)