from typing import List

from engine.controllers.milvus import MilvusHandler
from engine.controllers.embedders_rerankers import EmbeddingModelsConfig
from data.controllers.constants import NORMALIZE_EMBEDDINGS


class SemanticDBController:
    def __init__(self, milvus_config_path: str, prepare_db: bool = False):
        self._milvus_handler = None
        self._milvus_config_path = milvus_config_path
        if prepare_db:
            self._prepare_db()

    def _prepare_db(self) -> None:
        """
        Simple method do use Milvus. Connect to db, create schem, database etc.
        :return: None
        """
        _ = MilvusHandler(
            jsonl_config_path=self._milvus_config_path,
            collection_name="",
            collection_description=None,
            embedding_size=None,
            embedder_model_path=None,
            create_db_if_not_exists=True,
            load_embedder=False,
            load_collection_and_schema=False,
            extended_schema=False,
            vec_1_size=None,
            vec_2_size=None,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )

    def get_add_collection(
        self, collection_name: str, collection_description: str, model_embedder: str
    ):
        embedding_size = EmbeddingModelsConfig.get_embedder_vector_size(
            model_name=model_embedder
        )

        m_handler = self.__prepare_milvus_handler(
            collection_name=collection_name,
            collection_description=collection_description,
            embedding_size=embedding_size,
            load_collection_and_schema=True,
        )
        m_handler.close_connection()

    def collections(self) -> List[str]:
        m_handler = self.__prepare_milvus_handler(
            collection_name=None,
            collection_description=None,
            embedding_size=None,
            load_collection_and_schema=False,
        )
        collections = m_handler.db_collections()
        m_handler.close_connection()
        return collections

    def __prepare_milvus_handler(
        self,
        collection_name: str | None,
        collection_description: str | None,
        embedding_size: int | None,
        load_collection_and_schema: bool,
    ):
        m_handler = MilvusHandler(
            jsonl_config_path=self._milvus_config_path,
            collection_name=collection_name,
            collection_description=collection_description,
            embedding_size=embedding_size,
            embedder_model_path=None,
            create_db_if_not_exists=False,
            load_embedder=False,
            load_collection_and_schema=load_collection_and_schema,
            extended_schema=False,
            vec_1_size=None,
            vec_2_size=None,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )
        return m_handler
