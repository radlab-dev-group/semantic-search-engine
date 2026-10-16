"""
visibility.py
-------------

Enabling and disabling documents of a collection in the search, i.e. setting
``Document.use_in_search``.

Nothing is removed here: the pages, the text chunks and the embeddings of a
disabled document stay in both databases, so it may be enabled again without
any re-indexing.  The embeddings are only deactivated (their ``is_active``
flag), so every consumer of the vector collection — not only the search of
this engine — sees the current state of a document.  Documents are addressed
by their names and/or ids, exactly as in the deletion
(``data/controllers/deletion.py``)::

    controller = ContentVisibilityController()
    result = controller.set_documents_use_in_search(
        collection=collection,
        use_in_search=False,
        document_names=["instructor.pdf"],
    )
    print(result.as_dict())
"""

from dataclasses import dataclass, field
from typing import List

from sse_api.data.models import CollectionOfDocuments

from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.database.semantic_db import SemanticDBController


@dataclass
class VisibilityResult:
    """
    Outcome of a single visibility change: how many documents were updated
    and which names/ids were not found in the collection.
    """

    updated_documents: int = 0
    updated_vectors: int = 0
    use_in_search: bool = False
    not_found: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        Visibility change prepared for the API response.
        :return: Dictionary with the number of the updated documents and
        vectors, the value set on them and the names/ids that were not found
        """
        return {
            "updated_documents": self.updated_documents,
            "updated_vectors": self.updated_vectors,
            "use_in_search": self.use_in_search,
            "not_found": self.not_found,
        }


class ContentVisibilityController:
    """
    Decides which documents of a collection take part in the search.
    """

    def __init__(self, milvus_config_path: str = "./configs/milvus_config.json"):
        """
        :param milvus_config_path: Milvus connection configuration used to
        activate/deactivate the vectors of the documents
        """
        self._milvus_config_path = milvus_config_path

    def set_documents_use_in_search(
        self,
        collection: CollectionOfDocuments,
        use_in_search: bool,
        document_names=None,
        document_ids=None,
    ) -> VisibilityResult:
        """
        Enable or disable the use of documents in the search of the
        collection.
        :param collection: Collection with the documents
        :param use_in_search: Value to set on the found documents
        :param document_names: Names of the documents to change
        :param document_ids: Ids of the documents to change
        :return: VisibilityResult
        """
        wanted_names = [str(name) for name in (document_names or [])]
        wanted_ids = [int(document_id) for document_id in (document_ids or [])]

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

        updated_documents = RelationalDBController.set_documents_use_in_search(
            documents=documents, use_in_search=use_in_search
        )
        document_ids = [document.pk for document in documents]

        # The flag is written to the vector database as well: the embeddings
        # are kept (so enabling a document back does not need any re-indexing)
        # but their "is_active" flag follows the relational one, which makes
        # the state visible to every other consumer of the collection.
        updated_vectors = 0
        if len(document_ids):
            updated_vectors = self.__semantic_db_controller().set_documents_active(
                collection_name=collection.name,
                external_ids=document_ids,
                is_active=use_in_search,
            )

        found_names = {document.name for document in documents}
        found_ids = {document.pk for document in documents}
        not_found = sorted(set(wanted_names) - found_names) + sorted(
            str(document_id) for document_id in set(wanted_ids) - found_ids
        )
        return VisibilityResult(
            updated_documents=updated_documents,
            updated_vectors=updated_vectors,
            use_in_search=use_in_search,
            not_found=not_found,
        )

    def __semantic_db_controller(self) -> SemanticDBController:
        """
        Vector database controller, without loading any model.
        :return: SemanticDBController
        """
        return SemanticDBController(
            milvus_config_path=self._milvus_config_path, prepare_db=False
        )

    @staticmethod
    def __unique_documents(documents) -> list:
        """
        Keep only the documents which were not given before (one document may
        be addressed by its name and its id in the same request).
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
