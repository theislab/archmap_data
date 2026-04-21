import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import scanpy as sc
import scarches as sca
import numpy as np
import datetime
import json

# SET PATHS HERE
query_path = 'queries/hlca_queries/new/hlca_disease_ext.h5ad'
ref_model_path = "models/model_hlca_scanvi"
surgery_model_path = "surgery_models/hlca_disease_ext"
epochNumbers = [1, 5, 10, 50, 100, 250, 500, 750, 1000]

adata = sc.read_h5ad(query_path)
adata.X = adata.raw.X


adata_query = sca.models.SCANVI.prepare_query_anndata(
    adata = adata, reference_model =ref_model_path, inplace=False
)

adata_query.obs["scanvi_label"] = "unlabeled"


surgery_model = sca.models.SCANVI.load_query_data(
    adata_query,
    ref_model_path,
    freeze_dropout=True
)

totalEpochs = 0

for epochNumber in epochNumbers:
    print(f"Training Model up to {epochNumber} epochs".center(120,"*"))
    print(f"Training {epochNumber-totalEpochs} epochs")
    start_time = datetime.datetime.now()
    surgery_model_dir = f"{surgery_model_path}/epoch_{epochNumber}"
    surgery_model.train(max_epochs=epochNumber-totalEpochs)
    surgery_model.save(surgery_model_dir, overwrite=True)
    timespan =  str(datetime.datetime.now()-start_time)
    totalEpochs = epochNumber
    print(f"{epochNumber} epochs took: {timespan}")

print("finished")