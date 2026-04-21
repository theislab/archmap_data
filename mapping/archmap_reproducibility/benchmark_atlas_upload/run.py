import os
from archmap_reproducibility.benchmark_atlas_upload.benchmark_atlas_upload import benchmark, benchmark_plot, minify



def main():

    print(os.environ) # show all environment variables and their values.

    modelPath = os.getenv('modelpath')
    atlasPath = os.getenv('atlaspath')
    modelName = os.getenv('modelname')
    batchkey = os.getenv('batchkey')
    celltypekey = os.getenv('celltypekey')
    atlasName = os.getenv('atlasname')

    print(f"modelpath: {modelPath}")
    print(f"atlaspath: {atlasPath}")


    modelpath_local = "model/"

    # TODO: 
    # Check that data is not minified

    # benchmark integration
    benchmark(modelName, atlasName, modelpath_local, batchkey, celltypekey)
    benchmark_plot(atlasName, batchkey, celltypekey)

    # minify
    minify(modelName, atlasName, modelpath_local)

    # get classifiers and uncert


if __name__ == "__main__":
    main()
