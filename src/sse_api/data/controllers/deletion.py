"""
deletion.py
-----------

Deletion of the content of a collection in both databases of the engine:
the relational one (documents, pages and text chunks) and the vector one
(embeddings of the removed texts).  The files which the documents were
indexed from are removed too, as the last step of every deletion, so that a
document which is still in the databases never loses its file.

Rows are always removed first and vectors afterwards, so that removed rows
are never indexed again. Every method reports what was removed::

    controller = ContentDeletionController()
    result = controller.delete_documents(
        collection=collection, document_names=["instructor.pdf"]
    )
    print(result.as_dict())
"""

from dataclasses import dataclass, field
from typing import List

from django.db.models.deletion import ProtectedError  # type: ignore[import-untyped]

from sse_api.data.controllers.file_cleanup import (
    FileCleanupResult,
    collection_paths,
    document_paths,
    remove_paths,
)
from sse_api.data.models import CollectionOfDocuments

from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.database.semantic_db import SemanticDBController


class ContentProtected(Exception):
    """
    Raised when other rows are protected against the removal of the content.
    """


@dataclass
class DeletionResult:
    """
    Outcome of a single deletion: what was removed from the relational db,
    from the vector db and what was not found.
    """

    deleted: dict = field(default_factory=dict)
    deleted_vectors: int = 0
    dropped_collection: bool = False
    not_found: List[str] = field(default_factory=list)
    #: Files of the uploads which disappeared from the disk, and the files
    #: which could not be removed (a read-only mount, a file deleted by hand).
    #: A problem with the disk never fails the deletion of the databases.
    deleted_files: int = 0
    file_errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        Deletion outcome prepared for the API response.
        :return: Dictionary with the number of the deleted rows, the number of
        the deleted vectors and the names that were not found
        """
        return {
            "deleted": self.deleted,
            "deleted_vectors": self.deleted_vectors,
            "dropped_collection": self.dropped_collection,
            "not_found": self.not_found,
            "deleted_files": self.deleted_files,
            "file_errors": self.file_errors,
        }


class ContentDeletionController:
    """
    Removes texts, documents and whole collections from the relational and
    the vector database.
    """

    def __init__(self, milvus_config_path: str = "./configs/milvus_config.json"):
        """
        :param milvus_config_path: Milvus connection configuration used to
        remove the vectors
        """
        self._milvus_config_path = milvus_config_path

    def delete_texts(
        self, collection: CollectionOfDocuments, chunk_ids
    ) -> DeletionResult:
        """
        Delete text chunks of the collection.
        :param collection: Collection with the chunks
        :param chunk_ids: Ids of the chunks to delete
        :return: DeletionResult
        """
        wanted_ids = self.__int_ids(chunk_ids)
        chunk_ids = self.__chunks_of_collection(collection, wanted_ids)

        # Chunks referencing given ones through "paged_text" have to be
        # removed together with them.
        chunk_ids = RelationalDBController.chunks_to_delete(chunk_ids=chunk_ids)
        deleted = self.__safe_delete(
            lambda: RelationalDBController.delete_chunks(chunk_ids=chunk_ids)
        )
        deleted_vectors = 0
        if len(chunk_ids):
            deleted_vectors = self.__semantic_db_controller().delete_texts(
                collection_name=collection.name, external_ids=chunk_ids
            )
        return DeletionResult(
            deleted=deleted,
            deleted_vectors=deleted_vectors,
            not_found=sorted(
                str(chunk_id) for chunk_id in set(wanted_ids) - set(chunk_ids)
            ),
        )

    def delete_documents(
        self,
        collection: CollectionOfDocuments,
        document_names=None,
        document_ids=None,
    ) -> DeletionResult:
        """
        Delete documents, together with their pages and texts, from the
        collection.
        :param collection: Collection with the documents
        :param document_names: Names of the documents to delete
        :param document_ids: Ids of the documents to delete
        :return: DeletionResult
        """
        wanted_names = [str(name) for name in (document_names or [])]
        wanted_ids = self.__int_ids(document_ids)

        documents = []
        if len(wanted_ids):
            documents += list(
                RelationalDBController.get_collection_documents_by_ids(
                    collection=collection, document_ids=wanted_ids
                )
            )
        if len(wanted_names):
            documents += list(
                RelationalDBController.get_collection_documents_by_names(
                    collection=collection, document_names=wanted_names
                )
            )
        documents = self.__unique_documents(documents)
        document_ids = [document.pk for document in documents]
        paths = document_paths(documents)

        deleted = self.__safe_delete(
            lambda: RelationalDBController.delete_documents(documents=documents)
        )
        deleted_vectors = 0
        if len(document_ids):
            deleted_vectors = self.__semantic_db_controller().delete_documents(
                collection_name=collection.name, external_ids=document_ids
            )
        # The files of the uploads go last: nothing may lose the file of a
        # document which is still in the databases.
        files = remove_paths(paths) if len(paths) else FileCleanupResult()
        not_found = sorted(
            set(wanted_names) - {document.name for document in documents}
        )
        not_found += sorted(
            str(document_id) for document_id in set(wanted_ids) - set(document_ids)
        )
        return DeletionResult(
            deleted=deleted,
            deleted_vectors=deleted_vectors,
            not_found=not_found,
            deleted_files=files.deleted,
            file_errors=files.errors,
        )

    def delete_collection(self, collection: CollectionOfDocuments) -> DeletionResult:
        """
        Delete the collection with all of its documents, pages and texts.
        :param collection: Collection to delete
        :return: DeletionResult
        """
        collection_name = collection.name
        # The paths of the files have to be read before the rows of the
        # documents are gone, as the databases are where they are kept.
        paths = collection_paths(collection)
        deleted = self.__safe_delete(
            lambda: RelationalDBController.delete_collection(collection=collection)
        )
        semantic_db_controller = self.__semantic_db_controller()
        dropped = semantic_db_controller.drop_collection(
            collection_name=collection_name
        )
        files = remove_paths(paths) if len(paths) else FileCleanupResult()
        return DeletionResult(
            deleted=deleted,
            dropped_collection=dropped,
            deleted_files=files.deleted,
            file_errors=files.errors,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def __chunks_of_collection(
        collection: CollectionOfDocuments, chunk_ids
    ) -> List[int]:
        """
        Ids of the given chunks which belong to the collection.
        :param collection: Collection which chunks are looked for
        :param chunk_ids: Ids of the chunks
        :return: Ids of the chunks of the collection
        """
        return [
            chunk.pk
            for chunk in RelationalDBController.get_collection_chunks_by_ids(
                collection=collection, chunk_ids=chunk_ids
            )
        ]

    @staticmethod
    def __int_ids(ids) -> List[int]:
        """
        Remove empty values and cast the ids to integers.
        :param ids: Ids of any type to cast
        :return: List of integers
        """
        if ids is None:
            return []
        return [int(chunk_id) for chunk_id in ids if str(chunk_id).strip()]

    @staticmethod
    def __unique_documents(documents) -> list:
        """
        Keep only the documents which were not given before.
        :param documents: Documents, possibly repeated
        :return: List of unique documents
        """
        unique_documents = []
        document_ids = set()
        for document in documents:
            if document.pk in document_ids:
                continue
            document_ids.add(document.pk)
            unique_documents.append(document)
        return unique_documents

    @staticmethod
    def __safe_delete(delete_method) -> dict:
        """
        Run the deletion translating protected rows into ContentProtected.
        :param delete_method: Method which removes rows from the relational db
        :return: Number of the deleted rows
        """
        try:
            return delete_method()
        except ProtectedError as error:
            raise ContentProtected(str(error)) from error

    def __semantic_db_controller(self) -> SemanticDBController:
        """
        Vector database controller, without loading any model.
        :return: SemanticDBController
        """
        return SemanticDBController(
            milvus_config_path=self._milvus_config_path, prepare_db=False
        )
