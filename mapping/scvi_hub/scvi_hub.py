import scarches_api.utils.utils as utils
import scarches_api.utils.parameters as parameters
import json
import scanpy
import scvi

class ScviHub:
    def __init__(self, configuration) -> None:
        self.configuration = configuration

        self.__download_data()

    def map_query(self):
        model_parent_module, model_cls_name, batch_key, labels_key = self.__read_metadata()

        reference = scanpy.read_h5ad("atlas/atlas.h5ad")
        query = scanpy.read_h5ad("query/query.h5ad")

        eval(model_parent_module + "." + model_cls_name).prepare_query_anndata(query, "model/")

        model = eval(model_parent_module + "." + model_cls_name).load_query_data(
                    query,
                    "model/",
                    freeze_dropout=True,
                )

        model.train(
            max_epochs=10,
            plan_kwargs=dict(weight_decay=0.0),
            check_val_every_n_epoch=10,
            use_gpu=False
        )

        combined_adata = query.concatenate(reference, batch_key="bkey")
        eval(model_parent_module + "." + model_cls_name).setup_anndata(combined_adata)

        combined_adata.obsm["latent_rep"] = model.get_latent_representation(combined_adata)

        return combined_adata

    def __download_data(self):
        model_path = utils.get_from_config(self.configuration, parameters.MODEL_DATA_PATH)
        atlas_path = utils.get_from_config(self.configuration, parameters.ATLAS_DATA_PATH)
        metadata_path = utils.get_from_config(self.configuration, parameters.META_DATA_PATH)
        query_path = utils.get_from_config(self.configuration, parameters.QUERY_DATA_PATH)

        utils.fetch_file_from_s3(model_path, "model/model.pt")
        utils.fetch_file_from_s3(atlas_path, "atlas/atlas.h5ad")
        utils.fetch_file_from_s3(metadata_path, "metadata/metadata.json")
        utils.fetch_file_from_s3(query_path, "query/query.h5ad")

    def __read_metadata():
        f = open("metadata/metadata.json")
        metadata = json.load(f)
        
        model_parent_module = metadata.pop("model_parent_module")
        model_cls_name = metadata.pop("model_cls_name")
        batch_key = metadata.pop("batch_key")
        labels_key = metadata.pop("labels_key")

        return model_parent_module, model_cls_name, batch_key, labels_key