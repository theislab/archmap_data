from ..scarches_api.utils import parameters
import scanpy as sc
import logging

class Process:

    def __init__(self, config):
        self.config = config
        return


#Post-processing
    def compute_umap(self, latent_adata):
        sc.pp.neighbors(latent_adata, n_neighbors=self.config[parameters.NUMBER_OF_NEIGHBORS])
        sc.tl.leiden(latent_adata)
        sc.tl.umap(latent_adata)
        return latent_adata

    def write_csv(self, obs_to_drop: list, latent_adata: sc.AnnData, predict_scanvi):
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
        return

    def make_cxg_complient(self, latent_adata: sc.AnnData):
        if "X_umap" not in latent_adata.obsm:
            self.compute_umap(latent_adata)
        try:
            latent_adata.X = latent_adata.raw
            latent_adata.obs_names_make_unique()
        except Exception as e:
            logging.warning(msg = e)
        latent_adata.var_names_make_unique()
        return latent_adata

    
