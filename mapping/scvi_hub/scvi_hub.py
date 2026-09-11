import scarches
import torch
import scarches_api.utils.utils as utils
import scarches_api.utils.parameters as parameters
from scarches_api.utils import compat  # noqa: F401  (applies the anndata/scarches compat shims)
from huggingface_hub import snapshot_download
from anndata import experimental
from pathlib import Path
import json
import scanpy
import scipy
import gdown
import pandas
import numpy as np
import os
import tempfile
from process.processing import Postprocess
from classifiers.classifiers import Classifiers
from scarches_api.uncert.uncert_metric import classification_uncert_euclidean
from scarches_api.uncert.uncert_metric import classification_uncert_mahalanobis
import ast
import scvi

from scarches_api.utils.metrics import estimate_presence_score, cluster_preservation_score, percent_query_with_anchor, get_wknn
from scarches_api.utils.utils import gene_ensembl_conversion, handle_intersecting_columns, validate_h5ad
from scvi.data._constants import _SETUP_METHOD_NAME

class ScviHub:
    def __init__(self, configuration) -> None:
        self.__configuration = configuration
        self.__training_data_url = None
        self.__model_parent_module = None
        self.__model_cls_name = None
        self._model = None
        self.batch_key_input = "batch"

        #Classifier setup
        self._clf_native = None
        self._clf_xgb = None
        self._clf_knn = None

        self._webhook = utils.get_from_config(configuration, parameters.WEBHOOK_RATIO)
        self._webhook_metrics = utils.get_from_config(configuration, parameters.WEBHOOK_METRICS)
        self._webhook_gene_conversion = utils.get_from_config(configuration, parameters.WEBHOOK_GENE_CONVERSION)
        self._webhook_progress = utils.get_from_config(configuration, parameters.WEBHOOK_PROGRESS)

        self.__download_data()
        utils.notify_backend(self._webhook_progress, {"logs":"Step 1/6: fetching and preprocessing data"})

        

    def map_query(self):

        print("get data")
        self._reference_adata = scanpy.read_h5ad("../scvi_hub/atlas/atlas.h5ad")

        try:
            self._query_adata = scanpy.read_h5ad("../scvi_hub/query/query.h5ad")
            print("Data successfully loaded.")

            valid, message = validate_h5ad(self._query_adata)

            if not valid:
                raise ValueError(message)
            
        except ValueError as e:
            raise RuntimeError(
                f"Error message: {e}. "
                "There is likely an issue with the way your data (anndata object) is formatted upon upload. "
                "Please reach out to ArchMap (archmap.bio@gmail.com) with a screenshot of this error and we can help resolve this."
            )
        

        try:

            temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
            self._query_adata.write_h5ad(temp_query.name)
        except ValueError as e:
            if "is also used by a column whose values are different" in str(e):
                raise ValueError(f"Error message: {e}, Please check your anndata object for columns in .obs and .var that have matching names and delete duplicates") from e
            else:
                raise ValueError(f"Error message: {e}")
            

        if self._query_adata.n_obs>200000:
            raise ValueError(f"The number of cells in the query is over the limit of 200 000 cells. Please divide your data in batches and map the batches separately.")


        # Check if provided query contains batch labels
        if self.batch_key_input not in self._query_adata.obs.columns:
            if self._batch_key in self._query_adata.obs.columns:
                self._query_adata.obs["batch"]=self._query_adata.obs[self._batch_key]
            else:
                self._query_adata.obs["batch"]="mapped_batch"


        # rename duplicate column names
        self._reference_adata.obs = utils.rename_duplicate_columns(self._reference_adata.obs)
        self._reference_adata.var = utils.rename_duplicate_columns(self._reference_adata.var)
        self._query_adata.obs = utils.rename_duplicate_columns(self._query_adata.obs)
        self._query_adata.var = utils.rename_duplicate_columns(self._query_adata.var)

        gene_ensembl_conversion(self._reference_adata, self._query_adata, self._webhook_gene_conversion)

        ref_vars = self._reference_adata.var_names
        query_vars = self._query_adata.var_names

        intersection = ref_vars.intersection(query_vars)
        inter_len = len(intersection)
        ratio = (inter_len / len(ref_vars))*100
        print(ratio)


        if int(ratio)<5:
            raise ValueError(f"Less than 5% of genes (exactly {ratio}%) in your query overlap with the reference data. This will result in a poor mapping quality. Please make sure that either gene symbols or Ensembl IDs are stored in .var_names and you have chosen the correct atlas for your dataset. Currenty, the values in .var_names of your query are: {query_vars.tolist()[:10]}... Check out our FAQs in the docs for more information.")


        utils.notify_backend(self._webhook, {"ratio":ratio})

        del self._reference_adata.uns
        del self._query_adata.uns

        self._reference_adata.obs["type"] = "reference"
        self._query_adata.obs["type"] = "query"


        print("get keys")
        if isinstance(self._cell_type_key,list):
            for key in self._cell_type_key:
                self._query_adata.obs[key] = [self._unlabeled_key]*len(self._query_adata) 
        else:
            self._query_adata.obs[self._cell_type_key] = [self._unlabeled_key]*len(self._query_adata)

        if self._cell_type_key_classifier is None:
            self._cell_type_key_classifier = self._cell_type_key

        if self._cell_type_key_list is None:
            if isinstance(self._cell_type_key_classifier,list):
                self._cell_type_key_list = self._cell_type_key_classifier
            else:
                self._cell_type_key_list = [self._cell_type_key_classifier]

        if self.batch_key_input != self._batch_key:
            self._query_adata.obs[self._batch_key] = self._query_adata.obs[self.batch_key_input].copy()
            del self._query_adata.obs[self.batch_key_input]



        # save counts if counts attr_key is saved in reference model registry
        model_path = "../scvi_hub/model/model.pt"
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if (
            model.get("attr_dict", {})
            .get("registry_", {})
            .get("field_registries", {})
            .get("X", {})
            .get("data_registry", {})
            .get("attr_key") == "counts"
        ):
            self._query_adata.layers["counts"] = self._query_adata.X

        utils.notify_backend(self._webhook_progress, {"logs":"Step 2/6: mapping query"})


        #Conform vars to model for query and reference
        if(self.__model_cls_name == "SCVI"):

            #Align genes and gene order to model 
            self._query_adata.var_names_make_unique()
            
            scvi.model.SCVI.prepare_query_anndata(self._query_adata, "../scvi_hub/model/")

            #Setup adata internals for mapping
            scvi.model.SCVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key)

            #Load scvi model with query
            self._model = scvi.model.SCVI.load_query_data(
                self._query_adata,
                "../scvi_hub/model/",
                freeze_dropout=True,
            )

        
        if(self.__model_cls_name == "SCANVI"):

            self._query_adata.var_names_make_unique()
            scvi.model.SCANVI.prepare_query_anndata(self._query_adata, "../scvi_hub/model/",)

            #Setup adata internals for mapping
            scvi.model.SCANVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key, unlabeled_category=self._unlabeled_key)

            #Load scanvi model with query
            self._model = scvi.model.SCANVI.load_query_data(
                self._query_adata,
                "../scvi_hub/model/",
                freeze_dropout=True,
            )

        print("train")
        try:
            self._model.train(
                max_epochs=50,
                plan_kwargs=dict(weight_decay=0.0),
                check_val_every_n_epoch=10,
            )
        except ValueError as e:
            if "Expected parameter loc" in str(e):
                raise ValueError("Please check that your anndata object has raw counts (not normalized) saved in adata.X. Mapping can only occur with raw count data.") from e
            else:
                raise

        latent_name = f"{self.__model_cls_name.lower()}_latent_qzm"


        qzm = self._reference_adata.obsm[latent_name]
        self._reference_adata.obsm["latent_rep"] = qzm


        #Save out the latent representation for QUERY
        self._query_adata.obsm["latent_rep"] = self._model.get_latent_representation(self._query_adata)

        print("evaluate")
        utils.notify_backend(self._webhook_progress, {"logs":"Step 3/6: calculating mapping uncertainty scores"})
        self._eval_mapping()
        print("classify")
        utils.notify_backend(self._webhook_progress, {"logs":"Step 4/6: transferring labels from reference to query"})
        #Set up classification
        self.classification(ref=self._reference_adata, ref_latent=scanpy.AnnData(self._reference_adata.obsm["latent_rep"], obs=self._reference_adata.obs), query=self._query_adata, query_latent=scanpy.AnnData(self._query_adata.obsm["latent_rep"],obs=self._query_adata.obs))

        print("concat")
        utils.notify_backend(self._webhook_progress, {"logs":"Step 5/6: concatenating reference and query results"})
        self._concat_data()

        print("save")
        utils.notify_backend(self._webhook_progress, {"logs":"Step 6/6: calculating mapping evaluation metrics and saving results"})
        self._save_data()
    


        #Clean up local directories
        print("clean")
        self.__cleanup()

    def _save_data(self):
        # add .X to self._combined_adata

        if self.batch_key_input != self._batch_key:
            self._combined_adata.obs = self._combined_adata.obs.rename(columns={self._batch_key : self.batch_key_input})


        # Calculate presence score

        ref_latent = self._reference_adata.obsm["latent_rep"]
        query_latent = self._query_adata.obsm["latent_rep"]

        self.knn_ref = self.knn_ref_trainer.fit_transform(ref_latent)

        wknn, adjs = get_wknn(
            ref=ref_latent,
            query=query_latent,
            k=15,
            # adj_q2r=self.knn_q2r,
            adj_ref=self.knn_ref,
            return_adjs=True
        )

        presence_score = estimate_presence_score(
            self._reference_adata,
            self._query_adata,
            wknn = wknn)
    
        
        
        self.presence_score = np.concatenate((presence_score["max"],[np.nan]*len(self._query_adata)))

        self._combined_adata.obs["presence_score"] = self.presence_score
        print(f"presence_score: {self.presence_score}")

        self.clust_pres_score=cluster_preservation_score(self._query_adata)
        print(f"clust_pres_score: {self.clust_pres_score}")
        
        self.query_with_anchor=percent_query_with_anchor(adjs["r2q"], adjs["q2r"])
        print(f"query_with_anchor: {self.query_with_anchor}")

        print(f"percent_unknown: {self.percent_unknown}" )

        utils.notify_backend(self._webhook_metrics, {"clust_pres_score":self.clust_pres_score, "query_with_anchor":self.query_with_anchor, "percentage_unknown": self.percent_unknown})
        
        # save adata for user download
        self.temp_output_combined = "finetuned_model/adata.h5ad"
        
        os.makedirs("finetuned_model/", exist_ok=True)
        
        #Save output
        data_cxg = Postprocess.output(None, self._combined_adata, self.__configuration)

        self.data_cxg = data_cxg

    def _concat_data(self):
        self.latent_full_from_mean_var = np.concatenate((self._reference_adata.obsm["latent_rep"], self._query_adata.obsm["latent_rep"]))

        self._query_adata.obs["query"]=["1"]*self._query_adata.n_obs
        self._reference_adata.obs["query"]=["0"]*self._reference_adata.n_obs

        #Added because concat_on_disk only allows csr concat
        if scipy.sparse.issparse(self._query_adata.X) and (self._query_adata.X.format == "csc" or self._reference_adata.X.format == "csc"):

            print("concatenating in memory")
            #self._query_adata.X = csr_matrix(self._query_adata.X.copy())

            self._combined_adata = self._reference_adata.concatenate(self._query_adata, batch_key=self._batch_key,join="outer")

            query_obs=set(self._query_adata.obs.columns)
            ref_obs=set(self._reference_adata.obs.columns)
            inter = ref_obs.intersection(query_obs)
            new_columns = query_obs.union(inter)
            self._combined_adata.obs=self._combined_adata.obs[list(new_columns)]

            self._combined_adata.obsm["latent_rep"] = self.latent_full_from_mean_var

            return
        
        print("concatenating on disk")
        #Added because concat_on_disk only allows inner joins  
        for cell_type_key in self._cell_type_key_list:
            self._reference_adata.obs[cell_type_key + '_uncertainty_euclidean'] = pandas.Series(dtype="float32")
            self._reference_adata.obs[cell_type_key + '_uncertainty_mahalanobis'] = pandas.Series(dtype="float32")
            self._reference_adata.obs[cell_type_key + '_prediction_xgb'] = self._reference_adata.obs[cell_type_key]
            self._reference_adata.obs[cell_type_key + '_prediction_knn'] = self._reference_adata.obs[cell_type_key]
            self._reference_adata.obs[cell_type_key + "_prediction_scanvi"] = self._reference_adata.obs[cell_type_key]
            self._reference_adata.obs[cell_type_key + "_prediction_scanvi_filtered_by_uncert>0.5"] = self._reference_adata.obs[cell_type_key]
            self._reference_adata.obs[cell_type_key + "_prediction_xgb_filtered_by_uncert>0.5"] = self._reference_adata.obs[cell_type_key]
            self._reference_adata.obs[cell_type_key + "_prediction_knn_filtered_by_uncert>0.5"] = self._reference_adata.obs[cell_type_key]


            self._query_adata.obs[cell_type_key] = pandas.Series(dtype="category")
            self._query_adata.obs[cell_type_key] = self._query_adata.obs[cell_type_key].cat.add_categories(["Unknown"])
            self._query_adata.obs[cell_type_key] = self._query_adata.obs[cell_type_key].fillna("Unknown").astype('category')

        
        # intersecting columns with different types cause type errors during concat_on_disk (Can't implicitly convert non-string objects to strings)
        handle_intersecting_columns(self._query_adata, self._reference_adata)

        #Create temp files on disk
        temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
        temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")


        #Write data to temp files
        self._reference_adata.write_h5ad(temp_reference.name)
        self._query_adata.write_h5ad(temp_query.name)

        #Concatenate on disk to save memory
        experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)

        query_obs_columns=set(self._query_adata.obs.columns)
        ref_obs_columns=set(self._reference_adata.obs.columns)
        columns_only_query = list(query_obs_columns.difference(ref_obs_columns))
        query_obs = self._query_adata.obs[columns_only_query].copy()

        print("successfully concatenated")

        #Read concatenated data back in
        self._combined_adata = scanpy.read_h5ad(temp_combined.name)

        # self._combined_adata.obs=self._combined_adata.obs[list(new_columns)]

        print("read concatenated file")

        self._combined_adata.obsm["latent_rep"] = self.latent_full_from_mean_var

        self._combined_adata.obs_names_make_unique()
        
        self._combined_adata.obs=pandas.concat([self._combined_adata.obs,query_obs], axis=1)

        print("added latent rep to adata")

        #convert uncertainty scores to float32 
        for cell_type_key in self._cell_type_key_list:
            self._combined_adata.obs[cell_type_key + '_uncertainty_euclidean'] = self._combined_adata.obs[cell_type_key + '_uncertainty_euclidean'].astype("float32")
            self._combined_adata.obs[cell_type_key + '_uncertainty_mahalanobis'] = self._combined_adata.obs[cell_type_key + '_uncertainty_mahalanobis'].astype("float32")


        return

    def _eval_mapping(self):
        #Create AnnData objects off the latent representation
        query_latent = scanpy.AnnData(self._query_adata.obsm["latent_rep"])
        reference_latent = scanpy.AnnData(self._reference_adata.obsm["latent_rep"])
        reference_latent.obs = self._reference_adata.obs

        #Calculate mapping uncertainty and write into .obs
        self.knn_ref_trainer= classification_uncert_euclidean(self.__configuration, reference_latent, query_latent, self._query_adata, "X", self._cell_type_key_list, False)
        classification_uncert_mahalanobis(self.__configuration, reference_latent, query_latent, self._query_adata, self._cell_type_key_list, False)

         

    def classification(self, ref, ref_latent, query, query_latent):
        #Create directory
        cache_path = Path("../scvi_hub/classifiers/")
        cache_path.mkdir(parents=True, exist_ok=True)

        #Set class for respective cls_name
        if self.__model_cls_name == "SCVI":
            model_class = scarches.models.SCVI.__class__
        if self.__model_cls_name == "SCANVI":
            model_class = scarches.models.SCANVI.__class__

        print(self._clf_native)
        print(self.__model_cls_name)

        #Initialize and create classifier
        if self._clf_native:
            clf = Classifiers(self._clf_xgb, self._clf_knn, self._model, self.__model_cls_name)
        else:
            clf = Classifiers(self._clf_xgb, self._clf_knn, None, model_class)

        clf.create_classifier(adata=ref_latent, adata_X=ref, latent_rep=True, model_path="../scvi_hub/model", label_key=self._cell_type_key, classifier_directory="../scvi_hub/classifiers")
       

        if self._clf_xgb:
            self.percent_unknown=clf.predict_labels(query=query, query_latent=query_latent, classifier_path="../scvi_hub/classifiers/classifier_xgb.ubj", encoding_path="../scvi_hub/classifiers/classifier_encoding.pickle", cell_type_key=self._cell_type_key)
        elif self._clf_knn:
            self.percent_unknown=clf.predict_labels(query=query, query_latent=query_latent, classifier_path="../scvi_hub/classifiers/classifier_knn.pickle", encoding_path="../scvi_hub/classifiers/classifier_encoding.pickle", cell_type_key=self._cell_type_key)
        else:
            self.percent_unknown=clf.predict_labels(query=query, query_latent=query_latent, classifier_path=None, encoding_path=None, cell_type_key=self._cell_type_key)

        self._write_query_prediction_labels()

        
    def _write_query_prediction_labels(self):
        filename = os.path.join(os.getcwd(), "query_prediction_labels.txt")

        pred_cols = [col for col in self._query_adata.obs.columns
                     if (col.endswith('_pred') or '_prediction_' in col)
                     and 'uncertainty' not in col]

        if not pred_cols:
            utils.notify_backend(self._webhook_progress, {"logs": "No prediction columns found in query. Skipping prediction label export."})
            return

        df = self._query_adata.obs[pred_cols].astype(str).copy()
        df.index.name = 'obs_index'
        df.to_csv(filename, sep='\t')

        utils.notify_backend(self._webhook_progress, {"logs": f"Saved query prediction labels to {filename}"})

        output_path = utils.get_from_config(self.__configuration, parameters.OUTPUT_PREDICTION_LABELS_PATH)
        upload_size = utils.store_file_in_s3(filename, output_path)
    
    def __download_data(self):
        scvi_hub_id = utils.get_from_config(self.__configuration, parameters.SCVI_HUB_ID)
        # metadata_path = utils.get_from_config(self.__configuration, parameters.META_DATA_PATH)
        query_download_path = utils.get_from_config(self.__configuration, parameters.QUERY_DATA_PATH)

        #Create directories
        cache_path = Path("../scvi_hub/cache/")
        cache_path.mkdir(parents=True, exist_ok=True)
        download_path = Path("../scvi_hub/download/")
        download_path.mkdir(parents=True, exist_ok=True)
        atlas_path = Path("../scvi_hub/atlas/")
        atlas_path.mkdir(parents=True, exist_ok=True)
        query_path = Path("../scvi_hub/query/")
        query_path.mkdir(parents=True, exist_ok=True)
        metadata_path = Path("../scvi_hub/metadata/")
        metadata_path.mkdir(parents=True, exist_ok=True)
        model_path = Path("../scvi_hub/model/")
        model_path.mkdir(parents=True, exist_ok=True)
        clf_path = Path("../scvi_hub/classifiers/")
        clf_path.mkdir(parents=True, exist_ok=True)

        folder_path = snapshot_download(repo_id=scvi_hub_id, allow_patterns=["*.h5ad","*.pt","*.json","*.md"], cache_dir="../scvi_hub/cache/", local_dir="../scvi_hub/download/")

        self.__read_metadata()

        #Try to download if training_data_url exists in metadata else use the smaller existing one
        if self.__training_data_url is not None:
            gdown.download(self.__training_data_url, "../scvi_hub/atlas/atlas.h5ad")
        else:
            os.replace("../scvi_hub/download/adata.h5ad", "../scvi_hub/atlas/atlas.h5ad")
        utils.fetch_file_from_s3(query_download_path, "../scvi_hub/query/query.h5ad")
        os.replace("../scvi_hub/download/model.pt", "../scvi_hub/model/model.pt")

    def __read_metadata(self):
        f = open("../scvi_hub/download/_scvi_required_metadata.json")
        metadata = json.load(f)
        
        self.__training_data_url = metadata.pop("training_data_url")
        self.__model_parent_module = metadata.pop("model_parent_module")
        self.__model_cls_name = metadata.pop("model_cls_name")

        model_path = "../scvi_hub/download/model.pt"
        
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        attr_dict = model["attr_dict"]

        registry = attr_dict.pop("registry_")
        if _SETUP_METHOD_NAME not in registry.keys():
            registry[_SETUP_METHOD_NAME]="setup_anndata"

        attr_dict["registry_"]=registry
        model["attr_dict"]=attr_dict

        torch.save(model,model_path)

        if "unlabeled_category_" in attr_dict:
            if attr_dict["unlabeled_category_"] is not None:
                self._unlabeled_key = attr_dict["unlabeled_category_"]

        else:
            self._unlabeled_key = "unlabeled"

        self._cell_type_key_list = None
        self._cell_type_key_classifier = None

        model_setup_anndata_args = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.SCVI_HUB_ARGS)
        model_setup_anndata_args = ast.literal_eval(model_setup_anndata_args)

        self._batch_key = model_setup_anndata_args.pop("batch_key")
        self._cell_type_key = model_setup_anndata_args.pop("labels_key")

        classifier_type = utils.get_from_config(configuration=self.__configuration, key=utils.parameters.CLASSIFIER_TYPE)
        classifier_type = ast.literal_eval(classifier_type)

        self._clf_native = classifier_type.pop("Native")
        self._clf_xgb = classifier_type.pop("XGBoost")
        self._clf_knn = classifier_type.pop("kNN")

    def __cleanup(self):
        import shutil

        #Remove directories
        shutil.rmtree("../scvi_hub/cache/")
        shutil.rmtree("../scvi_hub/download/")
        shutil.rmtree("../scvi_hub/atlas/")
        shutil.rmtree("../scvi_hub/query/")
        shutil.rmtree("../scvi_hub/metadata/")
        shutil.rmtree("../scvi_hub/model/")

        shutil.rmtree("../scvi_hub/classifiers/")