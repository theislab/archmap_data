import os
import warnings
from scVI import scVI
import scanpy
import scarches
from scarches.dataset.trvae.data_handling import remove_sparsity
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import utils, parameters
import uncert.uncert_metric as uncert
import logging
import tempfile
import sys
import scvi

import process.processing as processing

import psutil


def setup_modules():
    """
    Set up the warnings filter and the figure parameters
    :return:
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    scanpy.settings.set_figure_params(dpi=200, frameon=False)
    scanpy.set_figure_params(dpi=200)
    scanpy.set_figure_params(figsize=(4, 4))

    torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def setup_anndata_for_scanvi(anndata, configuration):
    """
    Preprocess reference dataset
    :param anndata: reference dataset /source adata
    :param configuration: config
    :return:
    """
    scarches.models.SCANVI.setup_anndata(anndata,
                                         batch_key='dataset',
                                         labels_key='scanvi_label',
                                         unlabeled_category='unlabeled')


def get_scanvi_from_scvi_model(scvi_model, configuration):
    """
    Create the scANVI model instance
    :param scvi_model: the scVI model
    :param configuration: config
    :return: scANVI model
    """
    return scarches.models.SCANVI.from_scvi_model(scvi_model,
                                                  utils.get_from_config(configuration, parameters.UNLABELED_KEY))


def get_latent(model, adata, configuration):
    """
    Create anndata file of latent representation and compute UMAP
    :param model: the created scANVI model
    :param adata: reference dataset / source adata
    :param configuration: config
    :return: latent representation
    """
    # add obs to reference_latent
    reference_latent = scanpy.AnnData(model.get_latent_representation())
    reference_latent.obs["cell_type"] = adata.obs[
        utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)].tolist()
    reference_latent.obs["batch"] = adata.obs[utils.get_from_config(configuration, parameters.CONDITION_KEY)].tolist()

    scanpy.pp.neighbors(reference_latent,
                        n_neighbors=utils.get_from_config(configuration, parameters.NUMBER_OF_NEIGHBORS))
    scanpy.tl.leiden(reference_latent)
    scanpy.tl.umap(reference_latent)

    return reference_latent


def predict(model, latent):
    """
    predict on the latent and compute the accuracy of the predicted value
    :param model: scANVI model
    :param latent: reference latent
    :return: latent with obs "predicted"
    """
    latent.obs['predicted'] = model.predict()
    print("Acc: {}".format(np.mean(latent.obs.predicted == latent.obs.cell_type)))
    return latent


def surgery(reference_latent, source_adata, anndata, configuration):
    """
    Perform surgery on reference model
    :param reference_latent: reference latent
    :param source_adata: reference dataset
    :param anndata: query dataset
    :param configuration: config
    :return: trained model, surgery latent
    """
    model = scarches.models.SCANVI.load_query_data(
        anndata,
        utils.get_from_config(configuration, parameters.PRETRAINED_MODEL_PATH),
        # ist das der richtige Pfad? Ist doch dann schon einmal trainiert?
        freeze_dropout=True,
    )

    model._unlabeled_indices = np.arange(anndata.n_obs)
    model._labeled_indices = []

    print("Labelled Indices: ", len(model._labeled_indices))
    print("Unlabelled Indices: ", len(model._unlabeled_indices))

    model.train(
        max_epochs=utils.get_from_config(configuration, parameters.SCANVI_MAX_EPOCHS),
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,
        use_gpu=utils.get_from_config(configuration, parameters.USE_GPU)
    )

    surgery_latent = get_latent(model, anndata, configuration)

    if utils.get_from_config(configuration, parameters.DEBUG):
        utils.save_umap_as_pdf(surgery_latent, 'figures/surgery.pdf', color=['batch', 'cell_type'])





    ### NEW IMPLEMENTATION ###
    #Check which output types are desired and save latent data
    output_types = utils.get_from_config(configuration, parameters.OUTPUT_TYPE)

    #Combine reference and query data
    combined_adata = source_adata.concatenate(anndata)

    latent_data = model.get_latent_representation(combined_adata)

    for type in output_types:
        if(type == 'csv'):
            #TODO: Change implementation of dropping unnecessary labels?
            obs_to_drop = []

            processing.Postprocess.output_csv(obs_to_drop, latent_data, configuration, True)
        elif(type == 'cxg'):
            processing.Postprocess.output_cxg(latent_data, configuration)
    ### NEW IMPLEMENTATION ###





    # utils.write_combined_csv(reference_latent, surgery_latent, key=utils.get_from_config(configuration, parameters.OUTPUT_PATH))
    # utils.write_full_adata_to_csv(model, source_adata, anndata,
    #                               key=utils.get_from_config(configuration, parameters.OUTPUT_PATH),
    #                               cell_type_key=utils.get_from_config(configuration, parameters.CELL_TYPE_KEY),
    #                               condition_key=utils.get_from_config(configuration, parameters.CONDITION_KEY),
    #                               predictScanvi=True, configuration=configuration)

    model.save('scvi_model', overwrite=True)
    utils.delete_file('scvi_model/model.pt')
    os.rmdir('scvi_model')

    return model, surgery_latent


def query(anndata, source_adata, configuration):
    """
    Perform surgery on reference model and train on query dataset
    :param pretrained_model: pretrained model
    :param reference_latent: reference latent
    :param anndata: target adata / query dataset
    :param source_adata: reference dataset
    :param configuration: config
    :return: trained model, query latent
    """
    print("DEBUGDEBUG QUERY 1")
    print("Load query data to model")
    model = scarches.models.SCANVI.load_query_data(
        anndata,
        'assets/scANVI/' + str(utils.get_from_config(configuration, parameters.ATLAS)) + '/',
        freeze_dropout=True,
    )
    print("DEBUGDEBUG QUERY 2")
    model._unlabeled_indices = np.arange(anndata.n_obs)
    model._labeled_indices = []

    # model._unlabeled_indices = []
    # model._labeled_indices = np.arange(anndata.n_obs)

    if utils.get_from_config(configuration, parameters.DEBUG):
        print("Labelled Indices: ", len(model._labeled_indices))
        print("Unlabelled Indices: ", len(model._unlabeled_indices))

    
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("DEBUGDEBUG QUERY 3")
#TODO: HARDCODING for human lung cell atlas -------------------------------------
    if utils.get_from_config(configuration, parameters.ATLAS) == 'human_lung':
        surgery_epochs = 500
        train_kwargs_surgery = {
            "early_stopping": True,
            "early_stopping_monitor": "elbo_train",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
            "plan_kwargs": {"weight_decay": 0.0},
        }
        print("Train model")
        model.train(
            max_epochs=utils.get_from_config(configuration, parameters.SCANVI_MAX_EPOCHS_QUERY),
            **train_kwargs_surgery,
            use_gpu=utils.get_from_config(configuration, parameters.USE_GPU)
        )
    else:
        print("DEBUGDEBUG QUERY 3.b")
        print("Train model")
        model.train(
            max_epochs=utils.get_from_config(configuration, parameters.SCANVI_MAX_EPOCHS_QUERY),
            plan_kwargs=dict(weight_decay=0.0),
            check_val_every_n_epoch=10,
            use_gpu=utils.get_from_config(configuration, parameters.USE_GPU)
        )
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        print("DEBUGDEBUG QUERY 3.c")
    # print("DEBUGDEBUG QUERY 4")
    # tempdir = tempfile.mkdtemp()
    # model.save(tempdir, overwrite=True)
    # print("DEBUGDEBUG QUERY 5")
    # if utils.get_from_config(configuration, parameters.DEV_DEBUG):
    #     print("DEBUGDEBUG QUERY 5.a")
    #     try:
    #         utils.write_adata_to_csv(model, 'scanvi-query-latent-after-query-training.csv')
    #     except Exception as e:
    #         print(e, file=sys.stderr)
    # if utils.get_from_config(configuration, parameters.DEV_DEBUG):
    #     print("DEBUGDEBUG QUERY 5.b")
    #     try:
    #         utils.store_file_in_s3(tempdir + '/model.pt', 'scanvi-model-after-query-training.pt')
    #     except Exception as e:
    #         print(e, file=sys.stderr)
    # utils.delete_file(tempdir + '/model.pt')
    # os.removedirs(tempdir)

    # # add obs to query_latent
    # print("DEBUGDEBUG QUERY 6")
    # query_latent = get_latent(model, anndata, configuration)
    # print("DEBUGDEBUG QUERY 7")
    # query_latent.obs['cell_type'] = anndata.obs[utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)].tolist()
    # query_latent.obs['dataset'] = anndata.obs[utils.get_from_config(configuration, parameters.CONDITION_KEY)].tolist()
    # print("DEBUGDEBUG QUERY 8")
    # scanpy.pp.neighbors(query_latent, n_neighbors=utils.get_from_config(configuration, parameters.NUMBER_OF_NEIGHBORS))
    # scanpy.tl.leiden(query_latent)
    # scanpy.tl.umap(query_latent)
    # print("DEBUGDEBUG QUERY 9")

    if utils.get_from_config(configuration, parameters.DEBUG):
        utils.save_umap_as_pdf(query_latent, 'figures/query.pdf', color=['batch', 'cell_type'])
        print("DEBUGDEBUG QUERY 3.d")
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    # utils.write_full_adata_to_csv(model, source_adata, anndata,
    #                               key=utils.get_from_config(configuration, parameters.OUTPUT_PATH),
    #                               cell_type_key=utils.get_from_config(configuration, parameters.CELL_TYPE_KEY),
    #                               condition_key=utils.get_from_config(configuration, parameters.CONDITION_KEY),
    #                               predictScanvi=True, configuration=configuration)




    ### NEW IMPLEMENTATION ###
    output_types = utils.get_from_config(configuration, parameters.OUTPUT_TYPE)
    labels_key = utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)
    unlabeled_category = utils.get_from_config(configuration, parameters.UNLABELED_KEY)
    batch_key = utils.get_from_config(configuration, parameters.CONDITION_KEY)

    use_embedding = utils.get_from_config(configuration, parameters.USE_REFERENCE_EMBEDDING)
    
    print("DEBUGDEBUG QUERY 3.e")
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    # print("DEBUGDEBUG QUERY 10.b")
    # anndata.obs["predictions"] = model.predict()
    # anndata.obs[labels_key] = anndata.obs["predictions"]
    # del anndata.obs["predictions"]

    # predict = model.predict(soft=True)

    # #Reset index else max function not working
    # old_index = predict.index
    # predict.reset_index(drop=True, inplace=True)    

    # maxv = predict.max(axis=1)

    # #Set index back to original
    # maxv.set_axis(old_index, inplace=True)

    # #Add uncertainty (1 - probability)
    # anndata.obs["uncertainty"] = 1 - maxv



    #anndata.obsm["latent_rep"] = model.get_latent_representation(anndata)
    # query_latent = scanpy.AnnData(model.get_latent_representation(anndata))
    # try:
    #     #source_adata.obsm["latent_rep"] = model.get_latent_representation(source_adata)
    #     reference_latent = scanpy.AnnData(model.get_latent_representation(source_adata))
    #     reference_latent.obs = source_adata.obs
    #     print("DEBUGDEBUG QUERY 10.C!!!!!!")
    #     uncert.classification_uncert_euclidean(configuration, reference_latent, query_latent, "latent_rep", labels_key, False)
    #     uncert.classification_uncert_mahalanobis(configuration, reference_latent, query_latent, "latent_rep", labels_key, False)
    # except:
    #     print("DEBUGDEBUG QUERY 10.D!!!!!!")
    #     source_adata_sub = source_adata[:,anndata.var.index]
    #     # source_adata_sub.obsm["latent_rep"] = model.get_latent_representation(source_adata_sub)
    #     reference_latent = scanpy.AnnData(model.get_latent_representation(source_adata_sub))
    #     reference_latent.obs = source_adata_sub.obs

    #     uncert.classification_uncert_euclidean(configuration, reference_latent, query_latent, "latent_rep", labels_key, False)
    #     uncert.classification_uncert_mahalanobis(configuration, reference_latent, query_latent, "latent_rep", labels_key, False)

    #Remove later
    # anndata.obs["ann_new"] = False
    # source_adata.obs["scanvi_label"] = utils.get_from_config(configuration, parameters.UNLABELED_KEY)

    # #Get combined and latent data
    # print("Combine reference and query, prepare for export")
    # combined_adata = source_adata.concatenate(anndata, batch_key='bkey', fill_value="None")
    # #combined_adata = source_adata.concatenate(anndata, join="outer", batch_key="bkey")

    ## Alternative approach
    print("DEBUGDEBUG QUERY A")
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
    temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

    print("DEBUGDEBUG QUERY B")
    scanpy.write(temp_reference.name, source_adata)
    scanpy.write(temp_query.name, anndata)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    print("DEBUGDEBUG QUERY C")
    del source_adata
    del anndata
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    import gc
    gc.collect()
    print("DEBUGDEBUG QUERY C after collecting memory")
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    print("DEBUGDEBUG QUERY D")
    from anndata import experimental
    experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    print("DEBUGDEBUG QUERY E")
    combined_adata = scanpy.read_h5ad(temp_combined.name)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    print("DEBUGDEBUG QUERY 11")
    scarches.models.SCANVI.setup_anndata(combined_adata, labels_key=labels_key, unlabeled_category=unlabeled_category, batch_key=batch_key)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("DEBUGDEBUG QUERY 12")
    combined_adata.obsm["latent_rep"] = model.get_latent_representation(combined_adata)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    #Dummy latent adata - Remove line
    print("DEBUGDEBUG QUERY 13")
    latent_adata = None

    #Save output
    print("DEBUGDEBUG QUERY 14")
    processing.Postprocess.output(latent_adata, combined_adata, configuration, output_types)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    
    #Remove created tmp files
    temp_reference.close()
    temp_query.close()
    temp_combined.close()
    ### NEW IMPLEMENTATION ###

    return model


def predict_latent(model, latent):
    """
    Compute Accuracy of model for query dataset
    compare predicted and observed cell types
    :param model: scANVI model
    :param latent: query latent
    :return:
    """
    latent.obs['predicted'] = model.predict()
    print("Acc: {}".format(np.mean(latent.obs.predicted == latent.obs.cell_type)))

    df = latent.obs.groupby(["cell_type", "predicted"]).size().unstack(fill_value=0)
    norm_df = df / df.sum(axis=0)

    figure = plt.figure(figsize=(8, 8), frameon=False)
    _ = plt.grid(False)
    _ = plt.pcolor(norm_df)
    _ = plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
    _ = plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    figure.savefig('predict.png')


# def both_adata(source_adata, target_adata, configuration):
#     adata_full = source_adata.concatenate(target_adata)
#     full_latent = scanpy.AnnData(scarches.models.SCANVI.get_latent_representation(adata=adata_full))
#     full_latent.obs['cell_type'] = adata_full.obs[
#         utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)].tolist()
#     full_latent.obs['batch'] = adata_full.obs[utils.get_from_config(configuration, parameters.CONDITION_KEY)].tolist()

#     scanpy.pp.neighbors(full_latent)
#     scanpy.tl.leiden(full_latent)
#     scanpy.tl.umap(full_latent)

#     full_latent.obs['predicted'] = 'predicted'

#     if utils.get_from_config(configuration, parameters.DEBUG):
#         utils.save_umap_as_pdf(full_latent, 'figures/both.pdf', color=['batch', 'cell_type'])

#     utils.write_latent_csv(full_latent, key=utils.get_from_config(configuration, parameters.OUTPUT_PATH))

#     return full_latent


# def compare_adata(model, source_adata, target_adata, configuration):
#     adata_full = source_adata.concatenate(target_adata)
#     full_latent = scanpy.AnnData(scarches.models.SCANVI.get_latent_representation(adata=adata_full))
#     full_latent.obs['cell_type'] = adata_full.obs[
#         utils.get_from_config(configuration, parameters.CELL_TYPE_KEY)].tolist()
#     full_latent.obs['batch'] = adata_full.obs[utils.get_from_config(configuration, parameters.CONDITION_KEY)].tolist()

#     scanpy.pp.neighbors(full_latent)
#     scanpy.tl.leiden(full_latent)
#     scanpy.tl.umap(full_latent)
#
#     full_latent.obs['predictions'] = 'predicted'
#
#
#     latent.obs['predictions'] = model.predict(adata=adata_full)
#     print("Acc_compare: {}".format(np.mean(latent.obs.predictions == latent.obs.cell_type)))
#     scanpy.pp.neighbors(latent)
#     scanpy.tl.leiden(latent)
#     scanpy.tl.umap(latent)
#
#     if get_from_config(configuration, parameters.DEBUG):
#         utils.save_umap_as_pdf(latent, 'figures/compare.pdf', color=["predictions", "cell_type"])
#
#     utils.write_latent_csv(latent, key=get_from_config(configuration, parameters.OUTPUT_PATH))

#     full_latent.obs['predicted'] = model.predict(adata=adata_full)
#     print("Acc_compare: {}".format(np.mean(full_latent.obs.predicted == full_latent.obs.cell_type)))
#     scanpy.pp.neighbors(full_latent)
#     scanpy.tl.leiden(full_latent)
#     scanpy.tl.umap(full_latent)

#     if utils.get_from_config(configuration, parameters.DEBUG):
#         utils.save_umap_as_pdf(full_latent, 'figures/compare.pdf', color=["predicted", "cell_type"])

#     utils.write_latent_csv(full_latent, key=utils.get_from_config(configuration, parameters.OUTPUT_PATH))


def create_model(source_adata, target_adata, configuration):
    """
    - compute scANVI model and train it on reference dataset
    - compute the accuracy of the learned classifier
    - save the result and write into csv file to the s3

    :param source_adata: reference dataset
    :param target_adata: query dataset
    :param configuration: config
    :return: scANVI model, reference latent
    """
    if utils.get_from_config(configuration, parameters.USE_PRETRAINED_SCANVI_MODEL):
        path = 'assets/scANVI/' + str(utils.get_from_config(configuration, parameters.ATLAS)) + '/'
        return scarches.models.SCANVI.load_query_data(
            target_adata,
            path,
            freeze_dropout=True,
        ), None

    scvi_model, _ = scVI.create_scVI_model(source_adata, target_adata, configuration)
    scanvi = get_scanvi_from_scvi_model(scvi_model, configuration)

    if utils.get_from_config(configuration, parameters.DEBUG):
        print("Labelled Indices: ", len(scanvi._labeled_indices))
        print("Unlabelled Indices: ", len(scanvi._unlabeled_indices))

    scanvi.train(max_epochs=utils.get_from_config(configuration, parameters.SCANVI_MAX_EPOCHS),
                 use_gpu=utils.get_from_config(configuration, parameters.USE_GPU))
    tempdir = tempfile.mkdtemp()
    scanvi.save(tempdir, overwrite=True, save_anndata=True)
    if utils.get_from_config(configuration, parameters.DEV_DEBUG):
        try:
            utils.write_adata_to_csv(scanvi, 'scanvi-reference-latent-after-from-scvi-training.pt')
        except Exception as e:
            print(e, file=sys.stderr)
        try:
            utils.store_file_in_s3(tempdir + '/model.pt', 'scanvi-model-after-first-training.pt')
            utils.store_file_in_s3(tempdir + '/adata.h5ad', 'scanvi-adata-after-first-training.pt')
        except Exception as e:
            print(e, file=sys.stderr)

    utils.delete_file(tempdir + '/model.pt')
    utils.delete_file(tempdir + '/adata.h5ad')
    os.removedirs(tempdir)

    reference_latent = get_latent(scanvi, source_adata, configuration)

    if utils.get_from_config(configuration, parameters.DEBUG):
        utils.save_umap_as_pdf(reference_latent, 'figures/reference.pdf', color=['batch', 'cell_type'])

    reference_latent = predict(scanvi, reference_latent)
    return scanvi, reference_latent


def compute_scANVI(configuration):
    """
    process reference and query dataset with scANVI model
    :param configuration: config
    :return:
    """
    print("DEBUGDEBUG START compute_scANVI")
    if utils.get_from_config(configuration, parameters.DEBUG):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

    print("DEBUGDEBUG  START setup_modules")
    setup_modules()
    print("DEBUGDEBUG END setup_modules")


    #source_adata, target_adata = utils.pre_process_data(configuration)
    print("DEBUGDEBUG  START pre_process_data")
    source_adata, target_adata = processing.Preprocess.pre_process_data(configuration)
    print("DEBUGDEBUG  END pre_process_data")
    #target_adata = processing.Preprocess.scANVI_process_labels(configuration, source_adata, target_adata)
    #scarches.models.SCANVI.setup_anndata(target_adata, labels_key=utils.get_from_config(configuration, parameters.CELL_TYPE_KEY), unlabeled_category=utils.get_from_config(configuration, parameters.UNLABELED_KEY), batch_key=utils.get_from_config(configuration, parameters.CONDITION_KEY))

    print("DEBUGDEBUG  START create_model")
    #scanvi, reference_latent = create_model(source_adata, target_adata, configuration)
    print("DEBUGDEBUG  END create_model")

    print("DEBUGDEBUG  START query")
    model_query, query_latent = query(target_adata, source_adata, configuration)
    print("DEBUGDEBUG  END query")
