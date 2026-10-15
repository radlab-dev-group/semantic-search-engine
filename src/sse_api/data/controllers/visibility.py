"""
visibility.py
-------------

Enabling and disabling documents of a collection in the search, i.e. setting
``Document.use_in_search``.

Nothing is removed here: the pages, the text chunks and the embeddings of a
disabled document stay in both databases, so it may be enabled again without
any re-indexing.  Documents are addressed by their names and/or ids, exactly
as in the deletion (``data/controllers/deletion.py``)::

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


@dataclass
class VisibilityResult:
    """
    Outcome of a single visibility change: how many documents were updated
    and which names/ids were not found in the collection.
    """

    updated_documents: int = 0
    use_in_search: bool = False
    not_found: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        Visibility change prepared for the API response.
        :return: Dictionary with the number of the updated documents, the
        value set on them and the names/ids that were not found
        """
        return {
            "updated_documents": self.updated_documents,
            "use_in_search": self.use_in_search,
            "not_found": self.not_found,
        }


class ContentVisibilityController:
    """
    Decides which documents of a collection take part in the search.
    """

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
        found_names = {document.name for document in documents}
        found_ids = {document.pk for document in documents}
        not_found = sorted(set(wanted_names) - found_names) + sorted(
            str(document_id) for document_id in set(wanted_ids) - found_ids
        )
        return VisibilityResult(
            updated_documents=updated_documents,
            use_in_search=use_in_search,
            not_found=not_found,
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
