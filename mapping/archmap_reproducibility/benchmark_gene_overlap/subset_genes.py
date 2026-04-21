
import scanpy as sc
import numpy as np
import scipy as sp

def main():

    atlas = sc.read("adata_core_meyer.h5ad")
    

    sparse_array = sp.sparse.load_npz('adata_core_meyer_rawcounts.npz')

    atlas.X = sparse_array

    query = atlas[atlas.obs["dataset"]=="Meyer_2021_5prime"]

    query.write("mapping/hlca_tutorial/Meyer_2021_5prime.h5ad")

    query_name = "Meyer_2021_5prime"
    ref = sc.read("mapping/hlca_tutorial/model_hlca_scanvi/adata.h5ad")
    query = sc.read(f"mapping/hlca_tutorial/{query_name}.h5ad")

    ref_vars = ref.var_names

    query_vars = query.var_names

    intersection = ref_vars.intersection(query_vars)

    query = query[:,intersection]

    query.write(f"mapping/hlca_tutorial/queries/{query_name}_100.h5ad")

    #subset 
    overlap = [10,50]

    for percentage in overlap:

        num_samples = int(len(intersection) * (percentage / 100))

        # Randomly sample elements
        gene_sample = np.random.choice(intersection, num_samples, replace=False)

        query[:,gene_sample].write(f"mapping/hlca_tutorial/queries/{query_name}_{percentage}.h5ad")


if __name__ == "__main__":
    main()
