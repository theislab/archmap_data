import scib_metrics
import scanpy

def metrics(adata):
    '''
    scib.metrics.metrics_fast include:
    Biological conservation:
    1. HVG overlap
    2. Cell type ASW
    3. Isolated labels ASW

    Batch correction:
    1. Graph connectivity
    2. Batch ASW
    3. Principal component regression
    '''

    adata = scanpy.read("output_cxg.h5ad")

    #Biological conservation
    print(metrics.isolated_labels(adata.X, adata.obs["cell_type"], adata.obs["batch"]))
    print(metrics.silhouette_label(adata.X, adata.obs["cell_type"]))

    #Batch correction
    print(metrics.silhouette_batch(adata.X, adata.obs["cell_type"], adata.obs["batch"]))
    #metrics.pcr_comparison()
    
    return