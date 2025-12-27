import json
from datetime import datetime

ALL_AVAILABLE_EMBEDDERS_MODELS = {}
ALL_AVAILABLE_RERANKERS_MODELS = {}


class EmbeddingModelsConfig:
    def __init__(
        self,
        embedders_config: str = "configs/embedders.json",
        rerankers_config: str = "configs/rerankers.json",
    ):
        self._embedders_config = embedders_config
        self._rerankers_config = rerankers_config

        if len(embedders_config):
            self._load_embedders_cfg()

        if len(rerankers_config):
            self._load_rerankers_cfg()

    @staticmethod
    def get_embedder_path(model_name):
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["path"]

    @staticmethod
    def get_embedder_vector_size(model_name):
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["vector_size"]

    @staticmethod
    def get_embedder_device(model_name):
        return ALL_AVAILABLE_EMBEDDERS_MODELS[model_name]["device"]

    @staticmethod
    def get_reranker_path(model_name):
        return ALL_AVAILABLE_RERANKERS_MODELS[model_name]["path"]

    @staticmethod
    def get_reranker_device(model_name):
        return ALL_AVAILABLE_RERANKERS_MODELS[model_name]["device"]

    @staticmethod
    def embedders():
        return list(ALL_AVAILABLE_EMBEDDERS_MODELS.keys())

    @staticmethod
    def rerankers():
        return list(ALL_AVAILABLE_RERANKERS_MODELS.keys())

    def _load_embedders_cfg(self):
        with open(self._embedders_config, "rt") as models_in:
            whole_emb = json.load(models_in)

        m2config = {}
        for m in whole_emb["models"]:
            m2config[m["name"]] = m

        for a_model in whole_emb["active_models"]:
            ALL_AVAILABLE_EMBEDDERS_MODELS[a_model] = m2config[a_model]

    def _load_rerankers_cfg(self):
        with open(self._rerankers_config, "rt") as models_in:
            whole_emb = json.load(models_in)

        m2config = {}
        for m in whole_emb["models"]:
            m2config[m["name"]] = m

        for a_model in whole_emb["active_models"]:
            if a_model not in ALL_AVAILABLE_RERANKERS_MODELS:
                ALL_AVAILABLE_RERANKERS_MODELS[a_model] = m2config[a_model]
