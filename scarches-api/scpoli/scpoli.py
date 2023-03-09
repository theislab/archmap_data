import os
import torch
import gdown
import numpy as np
import scanpy as sc
import pandas as pd
import scarches as sca
import matplotlib.pyplot as plt

early_stopping_kwargs = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

condition_key = 'study'
cell_type_key = ['cell_type']

# url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE211799&format=file&file=GSE211799%5Fadata%5Fatlas%2Eh5ad%2Egz'
# output = 'GSE211799_adata_atlas.h5ad.gz'
# gdown.download(url, output)

# import gzip
# import shutil
# with gzip.open(output, 'rb') as f_in:
#     with open('GSE211799_adata_atlas.h5ad', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

adata = sc.read('GSE211799_adata_atlas.h5ad')

adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

adata.obs[condition_key] = adata.obs[condition_key].astype(str)

adata.raw = adata
sc.pp.normalize_total(adata)

scpoli_model = sca.models.scPoli(
    adata=adata.copy(),
    condition_key='study',
    cell_type_keys=['cell_type'],
    embedding_dim=3,
)

scpoli_model.train( 
    n_epochs=50,
    pretraining_epochs=40,
    early_stopping_kwargs=early_stopping_kwargs,
    eta=5,
)

scpoli_model.save('/scpoli_pancreas', overwrite =True)