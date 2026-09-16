"""
data_adapter.py
---------------

Adapter that implements ``RelationalDataProtocol`` by wrapping
``RelationalDBController`` (Django ORM data access) and
``DBTextSearchController`` (PostgreSQL FTS).
"""

from typing import Any, Dict, List, Optional

from django.db.models import QuerySet

from data.models import CollectionOfDocuments, DocumentPageText, Document

# Import original controllers for the adapter layer.
from engine.controllers.database.relational_db import RelationalDBController
from engine.database.milvus_impl import (
    MilvusHandler,
)  # noqa: F401 — exposed for adapter usage
from engine.controllers.search.relational import DBTextSearchController


class RelationalDataAdapter:
    """
    Protocol‑based adapter that provides relational data access to engine_core.

    Internally it uses ``RelationalDBController`` and ``DBTextSearchController``,
    converting protocol calls (which use ``collection_id: int``) into ORM queries.
    """

    @staticmethod
    def _get_collection(collection_id: int) -> CollectionOfDocuments:
        """Resolve a collection from its primary key."""
        return CollectionOfDocuments.objects.get(pk=collection_id)

    # ------------------------------------------------------------------
    # RelationalDataProtocol implementation (direct methods)
    # ------------------------------------------------------------------

    def get_collection_chunks(
        self, collection_id: int, min_char_len: int = 300
    ) -> list[DocumentPageText]:
        """Get all text chunks for a collection (used during indexing)."""
        coll = self._get_collection(collection_id)
        return RelationalDBController.get_collection_chunks(coll, min_char_len)

    def documents_names_from_categories(
        self,
        collection_id: int,
        categories: list[str],
        only_used_to_search: bool = True,
    ) -> list[str]:
        """Get document names matching given categories."""
        coll = self._get_collection(collection_id)
        result = DBTextSearchController.documents_names_from_categories(
            collection=coll,
            categories=categories,
            only_used_to_search=only_used_to_search,
        )
        return (
            list(result)
            if hasattr(result, "__iter__") and not isinstance(result, str)
            else [result]
        )

    def document_names_relative_path_contains(
        self,
        collection_id: int,
        substrings: list[str],
        only_used_to_search: bool = True,
    ) -> list[str]:
        """Find document names whose relative path contains any substring."""
        coll = self._get_collection(collection_id)
        result = DBTextSearchController.document_names_relative_path_contains(
            collection=coll,
            substrings=substrings,
            only_used_to_search=only_used_to_search,
        )
        return (
            list(result)
            if hasattr(result, "__iter__") and not isinstance(result, str)
            else [result]
        )

    def fts_search(
        self,
        query_str: str,
        collection_id: int,
        max_results: int,
        language: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Full‑text search returning ``(text_id, rank)`` tuples."""
        coll = self._get_collection(collection_id)
        return DBTextSearchController.search_texts(
            query_str=query_str,
            collection=coll,
            max_results=max_results,
            filters=None,
            language=language,
        )

    def get_text_for_id(
        self,
        text_ids: list[int],
        surrounding_chunks: int = 0,
    ) -> list[dict]:
        """Get text chunks by ID with surrounding context."""
        return DBTextSearchController.get_texts(
            texts_ids=text_ids,
            texts_scores=[1.0]
            * len(text_ids),  # scores ignored for ordering since we pass ids
            surrounding_chunks=surrounding_chunks,
        )

    def get_all_documents_from_collection(
        self, collection_id: int
    ) -> QuerySet[Document]:
        """Get all document objects for a collection (for metadata filtering)."""
        return Document.objects.filter(collection__pk=collection_id)


# Re-export the concrete implementation class under the protocol name for compatibility.
RelationalDataProtocol = RelationalDataAdapter  # type: ignore[misc]
