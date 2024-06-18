import scarches
import scanpy
import pandas
import numpy as np
import tempfile
import os
import torch
import gc
import numpy as np
import time
import scipy
from scipy.sparse import csr_matrix, csc_matrix
from anndata import experimental
from utils import utils
import scanpy as sc
from scvi.dataloaders import BatchDistributedSampler

from utils import parameters
from utils.metrics import estimate_presence_score, cluster_preservation_score, percent_query_with_anchor, stress_score, get_wknn
from utils.utils import get_from_config
from utils.utils import fetch_file_from_s3
from utils.utils import read_h5ad_file_from_s3, get_file_size_in_gb, replace_X_on_disk 
import pandas as pd

from process.processing import Preprocess
from process.processing import Postprocess

from uncert.uncert_metric import classification_uncert_euclidean
from uncert.uncert_metric import classification_uncert_mahalanobis

from classifiers.classifiers import Classifiers


class ArchmapBaseModel():
    def __init__(self, configuration) -> None:
        self._configuration = configuration

        start_time = time.time() 

        self._atlas = get_from_config(configuration=configuration, key=parameters.ATLAS)
        self._model_type = get_from_config(configuration=configuration, key=parameters.MODEL)
        self._model_path = get_from_config(configuration=configuration, key=parameters.PRETRAINED_MODEL_PATH)
        self._scpoli_attr = get_from_config(configuration=configuration, key=parameters.SCPOLI_ATTR)
        self._scpoli_model_params = get_from_config(configuration=configuration, key=parameters.SCPOLI_MODEL_PARAMS)
        self._scpoli_var_names = get_from_config(configuration=configuration, key=parameters.SCPOLI_VAR_NAMES)
        self._reference_adata_path = get_from_config(configuration=configuration, key=parameters.REFERENCE_DATA_PATH)
        self._query_adata_path = get_from_config(configuration=configuration, key=parameters.QUERY_DATA_PATH)
        self._webhook = utils.get_from_config(configuration, parameters.WEBHOOK_RATIO)
        self._webhook_metrics = utils.get_from_config(configuration, parameters.WEBHOOK_METRICS)
        # self._use_gpu = get_from_config(configuration=configuration, key=parameters.USE_GPU)

        #Has to be empty for the load_query_data function to work properly (looking for "model.pt")
        self._temp_model_path = ""
        self._model = None
        self._temp_clf_model_path = None
        self._temp_clf_encoding_path = None
        self._query_adata = None
        self._reference_adata = None
        self._combined_adata = None

        #Load and process required data
        self._acquire_data()

        #Set respective keys coherent to chosen atlas
        self._cell_type_key = None
        self._batch_key = None
        self._unlabeled_key = None

        # self._cell_type_key, self._batch_key, self._unlabeled_key = Preprocess.get_keys(self._atlas, self._query_adata) 
        self._cell_type_key, self._batch_key, self._unlabeled_key = Preprocess.get_keys(self._atlas, self._query_adata, configuration) 

        self._clf_native = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("Native")
        self._clf_xgb = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("XGBoost")
        self._clf_knn = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_TYPE).pop("kNN")
        self._clf_model_path = get_from_config(configuration=configuration, key=parameters.CLASSIFIER_PATH)
        self._clf_encoding_path = get_from_config(configuration=configuration, key=parameters.ENCODING_PATH)

        end_time = time.time() 
        print(f"time {end_time-start_time}")

    def run(self):
        start_time = time.time()
        self._map_query()
        self._eval_mapping()
        self._transfer_labels()
        self._concat_data()
        self._save_data()
        end_time = time.time() 
        print(f"time {end_time-start_time}")
        self._cleanup()

    def _map_query(self):
        #Map the query onto reference

        # threshold = 10000
        if self._atlas == "fetal_brain":
            lr=0.1
            self._max_epochs = 20
        else:
            lr=0.001

        self._model.train(
            max_epochs=self._max_epochs,
            plan_kwargs=dict(weight_decay=0.0,lr=lr),
            check_val_every_n_epoch=10,
            # datasplitter_kwargs = dict(distributed_sampler = True),
            # strategy='ddp_find_unused_parameters_true',
            # accelerator="cpu", 
            # devices=4
        )

        if "X_latent_qzm" in self._reference_adata.obsm and "X_latent_qzv" in self._reference_adata.obsm:
            print("__________getting X_latent_qzm from minified atlas for scvi-tools models___________")
            qzm = self._reference_adata.obsm["X_latent_qzm"]
            self._reference_adata.obsm["latent_rep"] = qzm

        else:
            self._compute_latent_representation(explicit_representation=self._reference_adata)

        #Save out the latent representation for QUERY
        self._compute_latent_representation(explicit_representation=self._query_adata)



    def _acquire_data(self):
        #Download query and reference from GCP
        self._reference_adata = read_h5ad_file_from_s3(self._reference_adata_path)
        self._reference_adata.obs["type"] = "reference"

        self._query_adata_raw = read_h5ad_file_from_s3(self._query_adata_path) 
        self._query_adata_raw.obs["type"] = "query"

        self._query_adata_raw.obs_names_make_unique()

        #Convert bool to categorical to avoid write error during concatenation
        Preprocess.bool_to_categorical(self._reference_adata)
        Preprocess.bool_to_categorical(self._query_adata_raw)

        ref_vars = self._reference_adata.var_names
        query_vars = self._query_adata_raw.var_names

        intersection = ref_vars.intersection(query_vars)
        inter_len = len(intersection)
        ratio = inter_len / len(ref_vars)
        print(ratio)

        utils.notify_backend(self._webhook, {"ratio":ratio})

        
        # save only necessary data for mapping to new adata
        self._query_adata = self._query_adata_raw.copy()
        del self._query_adata.varm
        del self._query_adata.obsm
        del self._query_adata.layers
        del self._query_adata.uns
        del self._query_adata.obsp
        del self._query_adata.varp

        self._query_adata.layers['counts'] = self._query_adata.X


    def _eval_mapping(self):
        #Create AnnData objects off the latent representation
        query_latent = scanpy.AnnData(self._query_adata.obsm["latent_rep"])
        reference_latent = scanpy.AnnData(self._reference_adata.obsm["latent_rep"])
        reference_latent.obs = self._reference_adata.obs

        #Calculate mapping uncertainty and write into .obs
        self.knn_ref_trainer= classification_uncert_euclidean(self._configuration, reference_latent, query_latent, self._query_adata, "X", self._cell_type_key, False)
        classification_uncert_mahalanobis(self._configuration, reference_latent, query_latent, self._query_adata, "X", self._cell_type_key, False)

        #stress score
        if self._atlas=="hnoca":
            print("calculating stress score")
            stress_score(self._query_adata)
            print(self._query_adata.obs["Hallmark_Glycolysis_Score"])

    def _transfer_labels(self):
        if not self._clf_native and not self._clf_knn and not self._clf_xgb:
            return

        if self._clf_native:
            clf = Classifiers(self._clf_xgb, self._clf_knn, self._model, self._model.__class__)

        #Instantiate xgb or knn classifier if selected
        if self._clf_xgb or self._clf_knn:
            clf = Classifiers(self._clf_xgb, self._clf_knn, None, self._model.__class__)

            #Download classifiers and encoding from GCP if kNN or XGBoost
            if self._clf_xgb:
                self._temp_clf_encoding_path = tempfile.mktemp(suffix=".pickle")
                fetch_file_from_s3(self._clf_encoding_path, self._temp_clf_encoding_path)

                self._temp_clf_model_path = tempfile.mktemp(suffix=".ubj")
                fetch_file_from_s3(self._clf_model_path, self._temp_clf_model_path)
            elif self._clf_knn:
                self._temp_clf_encoding_path = tempfile.mktemp(suffix=".pickle")
                fetch_file_from_s3(self._clf_encoding_path, self._temp_clf_encoding_path)

                self._temp_clf_model_path = tempfile.mktemp(suffix=".pickle")
                fetch_file_from_s3(self._clf_model_path, self._temp_clf_model_path)

        #Compute label transfer and save to respective .obs
        query_latent = scanpy.AnnData(self._query_adata.obsm["latent_rep"])
        
        self.percent_unknown = clf.predict_labels(self._query_adata, query_latent, self._temp_clf_model_path, self._temp_clf_encoding_path)


    def _concat_data(self):

        #save .X and var_names of query in new adata for later concatenation after cellxgene
        self.adata_query_X = scanpy.AnnData(self._query_adata.X.copy())
        self.adata_query_X.var_names = self._query_adata.var_names

        #we can then zero out .X in original query
        if self._query_adata.X.format == "csc":
            all_zeros = csc_matrix(self._query_adata.X.shape)
        else:
            all_zeros = csr_matrix(self._query_adata.X.shape)

        self._query_adata.X = all_zeros.copy()
        
        self.latent_full_from_mean_var = np.concatenate((self._reference_adata.obsm["latent_rep"], self._query_adata.obsm["latent_rep"]))

        self._query_adata.obs["query"]=["1"]*self._query_adata.n_obs
        self._reference_adata.obs["query"]=["0"]*self._reference_adata.n_obs

        #Added because concat_on_disk only allows csr concat
        if scipy.sparse.issparse(self._query_adata.X) and (self._query_adata.X.format == "csc" or self._reference_adata.X.format == "csc"):

            print("concatenating in memory")
            #self._query_adata.X = csr_matrix(self._query_adata.X.copy())

            self._combined_adata = self._reference_adata.concatenate(self._query_adata, batch_key='bkey',join="outer")

            query_obs=set(self._query_adata.obs.columns)
            ref_obs=set(self._reference_adata.obs.columns)
            inter = ref_obs.intersection(query_obs)
            new_columns = query_obs.union(inter)
            self._combined_adata.obs=self._combined_adata.obs[list(new_columns)]

            self._combined_adata.obsm["latent_rep"] = self.latent_full_from_mean_var
            del self._query_adata
            del self._reference_adata
            gc.collect()

            

            return
        
        print("concatenating on disk")
        #Added because concat_on_disk only allows inner joins
        self._reference_adata.obs[self._cell_type_key + '_uncertainty_euclidean'] = pandas.Series(dtype="float32")
        self._reference_adata.obs['uncertainty_mahalanobis'] = pandas.Series(dtype="float32")
        self._reference_adata.obs['prediction_xgb'] = pandas.Series(dtype="category")
        self._reference_adata.obs['prediction_knn'] = pandas.Series(dtype="category")

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
        columns_only_query = query_obs_columns.difference(ref_obs_columns)
        query_obs = self._query_adata.obs[columns_only_query].copy()

        del self._reference_adata
        del self._query_adata
        gc.collect()

        print("successfully concatenated")

        #Read concatenated data back in
        self._combined_adata = scanpy.read_h5ad(temp_combined.name)

        # self._combined_adata.obs=self._combined_adata.obs[list(new_columns)]

        print("read concatenated file")

        self._combined_adata.obsm["latent_rep"] = self.latent_full_from_mean_var

        self._combined_adata.obs_names_make_unique()
        
        self._combined_adata.obs=pd.concat([self._combined_adata.obs,query_obs], axis=1)

        print("added latent rep to adata")

        return


    def _compute_latent_representation(self, explicit_representation):
        #Store latent representation
        explicit_representation.obsm["latent_rep"] = self._model.get_latent_representation(explicit_representation)

    def _save_data(self):
        # add .X to self._combined_adata

        print("adding X from cloud")
        self.add_X_from_cloud()


        combined_downsample = self.downsample_adata()

        # Calculate presence score

        ref_downsample = combined_downsample[combined_downsample.obs["query"]=="0"]
        query_downsample = combined_downsample[combined_downsample.obs["query"]=="1"]

        ref_latent_downsample = combined_downsample[combined_downsample.obs["query"]=="0"].obsm["latent_rep"]
        query_latent_downsample = combined_downsample[combined_downsample.obs["query"]=="1"].obsm["latent_rep"]

        self.knn_ref = self.knn_ref_trainer.fit_transform(ref_latent_downsample)

        wknn, adjs = get_wknn(
            ref=ref_latent_downsample,
            query=query_latent_downsample,
            k=15,
            # adj_q2r=self.knn_q2r,
            adj_ref=self.knn_ref,
            return_adjs=True
        )

        presence_score = estimate_presence_score(
            ref_downsample,
            query_downsample,
            wknn = wknn)
    
        
        self.presence_score = np.concatenate((presence_score["max"],[np.nan]*len(query_downsample)))

        combined_downsample.obs["presence_score"] = self.presence_score

        self.clust_pres_score=cluster_preservation_score(query_downsample)
        print(f"clust_pres_score: {self.clust_pres_score}")
        
        self.query_with_anchor=percent_query_with_anchor(adjs["r2q"], adjs["q2r"])
        print(f"query_with_anchor: {self.query_with_anchor}")

        utils.notify_backend(self._webhook_metrics, {"clust_pres_score":self.clust_pres_score, "query_with_anchor":self.query_with_anchor, "percentage_unknown": self.percent_unknown})
        
        #Save output
        Postprocess.output(None, combined_downsample, self._configuration)

    def add_X_from_cloud(self):
        if True or get_from_config(self._configuration, parameters.WEBHOOK) is not None and len(
                get_from_config(self._configuration, parameters.WEBHOOK)) > 0:
            
            utils.notify_backend(get_from_config(self._configuration, parameters.WEBHOOK), self._configuration)
            if not self._reference_adata_path.endswith("data.h5ad"):
                raise ValueError("The reference data should be named data.h5ad")
            else:
                count_matrix_path = self._reference_adata_path[:-len("data.h5ad")] + "data_only_count.h5ad"

            combined_adata = self._combined_adata
            count_matrix_size_gb = get_file_size_in_gb(count_matrix_path)
            self.temp_output_combined = tempfile.mktemp( suffix=".h5ad")

            if count_matrix_size_gb < 10:
                print("Count matrix size less than 10 gb.")
                count_matrix = read_h5ad_file_from_s3(count_matrix_path)
                #Added because concat_on_disk only allows csr concat
                if count_matrix.X.format == "csc" or self.adata_query_X.X.format == "csc":
                    print("Concatenating query and reference count matrices in memory")
                    combined_data_X = count_matrix.concatenate(self.adata_query_X)

                    del count_matrix
                    del self.adata_query_X
                    gc.collect()

                else:
                    print("Concatenating query and reference count matrices on disk")
                    #Create temp files on disk
                    temp_reference = tempfile.NamedTemporaryFile(suffix=".h5ad")
                    temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
                    temp_combined = tempfile.NamedTemporaryFile(suffix=".h5ad")

                    #Write data to temp files
                    count_matrix.write_h5ad(temp_reference.name)
                    self.adata_query_X.write_h5ad(temp_query.name)

                    del count_matrix
                    del self.adata_query_X
                    gc.collect()
                
                    experimental.concat_on_disk([temp_reference.name, temp_query.name], temp_combined.name)
                    combined_data_X = sc.read_h5ad(temp_combined.name)

                combined_adata.X = combined_data_X.X
                sc.write(self.temp_output_combined, combined_adata)

            else:
                print("Count matrix size larger than 10 gb.")
                temp_query = tempfile.NamedTemporaryFile(suffix=".h5ad")
                self.adata_query_X.write_h5ad(temp_query.name)
                del self.adata_query_X
                gc.collect()
                self.temp_output_combined =replace_X_on_disk(combined_adata,self.temp_output_combined, temp_query.name, count_matrix_path)

            self._combined_adata = combined_adata


    def downsample_adata(self, query_ratio=5):
        """
        Downsamples the reference data to be proportional to the query data.
        
        If 10% of the reference data is less than the query data size times the query_ratio, 
        use a proportional sampling method. Otherwise, sample 10% from each cell type in 
        the reference data.

        Parameters:
        query_ratio (int, optional): The ratio of reference to query data size. Default is 5.

        Returns:
        AnnData: Downsampled AnnData from combined data.
        """
        # Separate reference and query data
        ref_adata = self._combined_adata[self._combined_adata.obs["query"] == "0"]
        query_adata_index = np.where(self._combined_adata.obs["query"] == "1")[0]

        # Check if 10% of reference is less than query size times the ratio
        if len(ref_adata) * 0.1 < len(query_adata_index) * query_ratio:
            # New approach: Proportional sampling based on cell type proportions
            # Calculate total number of cells to sample from reference
            total_ref_cells_to_sample = len(query_adata_index) * query_ratio

            # Get unique cell types
            celltypes = np.unique(self._combined_adata.obs[self._cell_type_key])

            # Calculate the proportion of each cell type in the reference data
            celltype_proportions = {celltype: np.sum(ref_adata.obs[self._cell_type_key] == celltype) / len(ref_adata) for celltype in celltypes}

            # Sample cells from each cell type according to its proportion
            sampled_cell_index = []
            for celltype, proportion in celltype_proportions.items():
                cell_indices = np.where(ref_adata.obs[self._cell_type_key] == celltype)[0]
                sample_size = int(total_ref_cells_to_sample * proportion)
                
                # Adjust sample size if it exceeds the number of available cells
                if sample_size > len(cell_indices):
                    sample_size = len(cell_indices)
                
                sampled_cells = np.random.choice(cell_indices, size=sample_size, replace=False)
                sampled_cell_index.extend(sampled_cells)
        else:
            # Old approach: Sample 10% from each cell type in the reference data
            celltypes = np.unique(self._combined_adata.obs[self._cell_type_key])
            percentage = 0.02 if ref_adata.n_obs> 3000000 else 0.1 # max 1
        
            sampled_cell_index = np.concatenate([np.random.choice(np.where(ref_adata.obs[self._cell_type_key] == celltype)[0], size=int(len(np.where(ref_adata.obs[self._cell_type_key] == celltype)[0]) * percentage), replace=False) for celltype in celltypes])

        # Combine sampled reference cells with query cells
        sampled_cell_index = np.concatenate([sampled_cell_index, query_adata_index])

        # Create downsampled AnnData object
        combined_downsample = self._combined_adata[sampled_cell_index].copy()

        return combined_downsample


    def _cleanup(self):
        #Remove all temp files
        if os.path.exists(os.path.join(self._temp_model_path, "model.pt")):
            os.remove(os.path.join(self._temp_model_path, "model.pt"))
        if os.path.exists(os.path.join(self._temp_model_path, "model_params.pt")):
            os.remove(os.path.join(self._temp_model_path, "model_params.pt"))
        if os.path.exists(os.path.join(self._temp_model_path, "attr.pkl")):
            os.remove(os.path.join(self._temp_model_path, "attr.pkl"))
        if os.path.exists(os.path.join(self._temp_model_path, "var_names.csv")):
            os.remove(os.path.join(self._temp_model_path, "var_names.csv"))

        if self._temp_clf_model_path is not None:
            if os.path.exists(self._temp_clf_model_path):
                os.remove(self._temp_clf_model_path)
        if self._temp_clf_encoding_path is not None:
            if os.path.exists(self._temp_clf_encoding_path):
                os.remove(self._temp_clf_encoding_path)

class ScVI(ArchmapBaseModel):
    def _map_query(self):
        #Align genes and gene order to model      
        scarches.models.SCVI.prepare_query_anndata(self._query_adata, self._temp_model_path)

        #Setup adata internals for mapping
        scarches.models.SCVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key)

        #Load scvi model with query
        model = scarches.models.SCVI.load_query_data(
            self._query_adata,
            self._temp_model_path,
            freeze_dropout=True,
        )

        self._model = model
        self._max_epochs = get_from_config(configuration=self._configuration, key=parameters.SCVI_QUERY_MAX_EPOCHS)

        super()._map_query()

    def _acquire_data(self):
        super()._acquire_data()
        
        #Download model from GCP
        fetch_file_from_s3(self._model_path, "./model.pt")

    def _compute_latent_representation(self, explicit_representation):
        #Setup adata before quering model for latent representation
        scarches.models.SCVI.setup_anndata(explicit_representation, batch_key=self._batch_key)

        super()._compute_latent_representation(explicit_representation=explicit_representation)

class ScANVI(ArchmapBaseModel):
    def _map_query(self, supervised=False):
        #Align genes and gene order to model
        if self._cell_type_key in self._query_adata.obs.columns:
            self._query_adata.obs[f"{self._cell_type_key}_user_input"] = self._query_adata.obs[self._cell_type_key]
        self._query_adata.obs[self._cell_type_key] = [self._unlabeled_key]*len(self._query_adata) 
        scarches.models.SCANVI.prepare_query_anndata(self._query_adata, self._temp_model_path)

        #Setup adata internals for mapping
        scarches.models.SCANVI.setup_anndata(self._query_adata, batch_key=self._batch_key, labels_key=self._cell_type_key, unlabeled_category=self._unlabeled_key)

        #Load scanvi model with query
        model = scarches.models.SCANVI.load_query_data(
            self._query_adata,
            self._temp_model_path,
            freeze_dropout=True,
        )

        #Check if mapping supervised, unsupervised or semi-supervised
        if supervised:
            model._unlabeled_indices = []
            model._labeled_indices = self._query_adata.n_obs
        else:
            model._unlabeled_indices = np.arange(self._query_adata.n_obs)
            model._labeled_indices = []

        self._model = model

        self._max_epochs = get_from_config(configuration=self._configuration, key=parameters.SCANVI_MAX_EPOCHS_QUERY)

        super()._map_query()

    def _acquire_data(self):
        super()._acquire_data()
        
        #Download model from GCP
        fetch_file_from_s3(self._model_path, "./model.pt")

    def _compute_latent_representation(self, explicit_representation):
        #Setup adata before quering model for latent representation
        scarches.models.SCANVI.setup_anndata(explicit_representation, labels_key=self._cell_type_key, unlabeled_category="unlabeled", batch_key=self._batch_key)

        super()._compute_latent_representation(explicit_representation=explicit_representation)

class ScPoli(ArchmapBaseModel):
   
    def _map_query(self):

        model = scarches.models.scPoli.load_query_data(
            adata=self._query_adata,
            reference_model=self._temp_model_path,
            labeled_indices=[],
            map_location=torch.device("cpu")
        )

        self._query_adata = model.adata

        self._model = model
        self._max_epochs = get_from_config(configuration=self._configuration, key=parameters.SCPOLI_MAX_EPOCHS)
        
        self._model.train(
            n_epochs=self._max_epochs,
            pretraining_epochs=40,
            eta=10
        )

        #Compute sample embeddings on query
        self._sample_embeddings()

        #make separate if statements based on the key that is available in atlas. 
        if "X_latent_qzm_scpoli" in self._reference_adata.obsm and "X_latent_qzv_scpoli" in self._reference_adata.obsm:
            print("__________getting X_latent_qzm_scpoli and X_latent_qzv_scpoli from minified atlas___________")
            qzm = self._reference_adata.obsm["X_latent_qzm_scpoli"]
            qzv = self._reference_adata.obsm["X_latent_qzv_scpoli"]
            latent = self._model.model.sampling(torch.tensor(qzm), torch.tensor(qzv)).numpy()
            self._reference_adata.obsm["latent_rep"] = latent

            #Save out the latent representation for QUERY
            self._compute_latent_representation(explicit_representation=self._query_adata)
        
        # in case the atlas provider stored mean for the latent space and want to use that for mapping
        elif "X_latent_qzm_scpoli" in self._reference_adata.obsm:
            print("__________getting X_latent_qzm_scpoli from minified atlas___________")
            qzm = self._reference_adata.obsm["X_latent_qzm_scpoli"]
            self._reference_adata.obsm["latent_rep"] = qzm

            #Save out the latent representation for QUERY
            self._compute_latent_representation(explicit_representation=self._query_adata, mean=True)

        else:
            self._compute_latent_representation(explicit_representation=self._reference_adata)
            self._compute_latent_representation(explicit_representation=self._query_adata)
        


    def _compute_latent_representation(self, explicit_representation, mean=False):
        explicit_representation.obsm["latent_rep"] = self._model.get_latent(explicit_representation, mean=mean)

    def _acquire_data(self):
        super()._acquire_data()
        
        #Download model from GCP
        fetch_file_from_s3(self._scpoli_model_params, "./model_params.pt")
        fetch_file_from_s3(self._scpoli_attr, "./attr.pkl")
        fetch_file_from_s3(self._scpoli_var_names, "./var_names.csv")

    def _sample_embeddings(self):
        from sklearn.decomposition import KernelPCA
        
        sample_embedding = self._model.get_conditional_embeddings()        

        pca = KernelPCA(n_components=2, kernel='linear')
        emb_pca = pca.fit_transform(sample_embedding.X)
        conditions = self._model.conditions_[self._batch_key]

        for i, c in enumerate(conditions):
            # ax.plot([0, emb_pca[i, 0]], [0, emb_pca[i, 1]])
            # ax.text(emb_pca[i, 0], emb_pca[i, 1], c)

            plot_data = [0, emb_pca[i, 0]], [0, emb_pca[i, 1]]
        
        self._query_adata.uns["sample_embeddings"] = emb_pca

        # from sklearn.decomposition import KernelPCA
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # pca = KernelPCA(n_components=2, kernel='linear')
        # emb_pca = pca.fit_transform(sample_embedding.X)
        # conditions = self._model.conditions_['study']
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # sns.scatterplot(x=emb_pca[:, 0], y=emb_pca[:, 1], hue=conditions, ax=ax)
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # for i, c in enumerate(conditions):
        #     ax.plot([0, emb_pca[i, 0]], [0, emb_pca[i, 1]])
        #     ax.text(emb_pca[i, 0], emb_pca[i, 1], c)
        # sns.despine()
