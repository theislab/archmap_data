import scib_metrics
import scanpy
import anndata
import numpy
import scib.preprocessing as pp
import scib.integration as ig
import scib.metrics as me

def metrics():
    ### Theislab scib
    adata = scanpy.read_h5ad("atlas_626ea3311d7d1a27de465b64_data.h5ad")

    pp.normalize(adata)
    # pp.scale_batch(adata, "batch")
    # pp.hvg_batch(adata, "batch")
    # pp.hvg_intersect(adata, "batch")
    # pp.reduce_data(adata, "batch")

    scanpy.pp.normalize_total()

    print(adata)


    integration = reference_integration(adata)
    print(integration["combat"])


    # metrics_space = ["feature", "embedding", "kNN_graph"]

    # if(metrics_space == "feature"):
    #     return
    # elif(metrics_space == "embedding"):
    #     return
    #     #embedding_space(unintegraded, integrated);
    # elif(metrics_space == "kNN_graph"):
    #     return

    ### Yoseflab scib
    # '''
    # scib.metrics.metrics_fast include:
    # Biological conservation:
    # 1. HVG overlap
    # 2. Cell type ASW
    # 3. Isolated labels ASW

    # Batch correction:
    # 1. Graph connectivity
    # 2. Batch ASW
    # 3. Principal component regression
    # '''

    # adata = scanpy.read("output_cxg.h5ad")

    # #Biological conservation
    # print(metrics.isolated_labels(adata.X, adata.obs["cell_type"], adata.obs["batch"]))
    # print(metrics.silhouette_label(adata.X, adata.obs["cell_type"]))

    # #Batch correction
    # print(metrics.silhouette_batch(adata.X, adata.obs["cell_type"], adata.obs["batch"]))
    # #metrics.pcr_comparison()
    
    return

def embedding_space(unintegrated = anndata.AnnData(), integrated = anndata.AnnData()):
    #Reduce unintegrated adata to embedding space
    batch_key = "batch"

    pp.reduce_data(unintegrated, batch_key)

    me.metrics_fast(unintegrated, integrated, batch_key)

def reference_integration(adata, combat = True, scanorama = False, scvi = False):
    batch_key = "batch"

    if(combat):
        combat_adata = ig.combat(adata, batch_key)
    else:
        combat_adata = numpy.nan

    if(scanorama):
        scanorama_adata = ig.scanorama(adata, batch_key)
    else:
        scanorama_adata = numpy.nan

    if(scvi):
        scvi_adata = ig.scvi(adata, batch_key)
    else:
        scvi_adata = numpy.nan

    output = {
        "combat": combat_adata,
        "scanorama": scanorama_adata,
        "scvi": scvi_adata
    }

    return output 
    

if __name__ == "__main__":
    metrics()
