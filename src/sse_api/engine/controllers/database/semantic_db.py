from typing import List

from sse_api.data.controllers.constants import NORMALIZE_EMBEDDINGS

from sse_api.engine.controllers.database.milvus import MilvusHandler
from sse_api.engine.controllers.models_logic.embedders_rerankers import (
    EmbeddingModelsConfig,
)


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

    def drop_collection(self, collection_name: str) -> bool:
        """
        Remove the whole collection from the vector database.
        :param collection_name: Name of the collection to remove
        :return: True when the collection was removed
        """
        m_handler = self.__prepare_milvus_handler(
            collection_name=collection_name,
            collection_description=None,
            embedding_size=None,
            load_collection_and_schema=False,
        )
        dropped = m_handler.drop_collection()
        m_handler.close_connection()
        return dropped

    def delete_texts(self, collection_name: str, external_ids) -> int:
        """
        Remove texts (chunks) of the collection from the vector database.
        :param collection_name: Name of the collection with the texts
        :param external_ids: Relational db ids of the text chunks
        :return: Number of removed vector entries
        """
        return self.__delete_by_external_ids(
            collection_name=collection_name,
            metadata_handler="delete_texts_by_external_ids",
            external_ids=external_ids,
        )

    def delete_documents(self, collection_name: str, external_ids) -> int:
        """
        Remove all texts of the given documents from the vector database.
        :param collection_name: Name of the collection with the documents
        :param external_ids: Relational db ids of the documents
        :return: Number of removed vector entries
        """
        return self.__delete_by_external_ids(
            collection_name=collection_name,
            metadata_handler="delete_documents_by_external_ids",
            external_ids=external_ids,
        )

    def set_documents_active(
        self, collection_name: str, external_ids, is_active: bool
    ) -> int:
        """
        Activate or deactivate the vectors of the given documents, i.e. set
        their "is_active" flag.  The entries are kept in the vector database,
        so a document may be activated again without any re-indexing.
        :param collection_name: Name of the collection with the documents
        :param external_ids: Relational db ids of the documents
        :param is_active: Value to set on the vectors of those documents
        :return: Number of updated vector entries
        """
        external_ids = list(external_ids)
        if not external_ids:
            return 0

        m_handler = self.__prepare_milvus_handler(
            collection_name=collection_name,
            collection_description=None,
            embedding_size=None,
            load_collection_and_schema=False,
        )
        try:
            updated = m_handler.set_active_by_external_ids(
                external_ids=external_ids, is_active=is_active
            )
        finally:
            m_handler.close_connection()
        return updated

    def __delete_by_external_ids(
        self, collection_name: str, metadata_handler: str, external_ids
    ) -> int:
        external_ids = list(external_ids)
        if not external_ids:
            return 0

        m_handler = self.__prepare_milvus_handler(
            collection_name=collection_name,
            collection_description=None,
            embedding_size=None,
            load_collection_and_schema=False,
        )
        try:
            deleted = getattr(m_handler, metadata_handler)(external_ids=external_ids)
        finally:
            m_handler.close_connection()
        return deleted

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
