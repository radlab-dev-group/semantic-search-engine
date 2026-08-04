"""
Protocol interfaces that define the contract between engine_core (the pure
search engine) and the database / adapter layer.

All classes are ``typing.Protocol`` — they have no runtime cost when used
only for type checking and impose zero overhead on implementations.
"""

from typing import List, Dict, Any, Tuple, Optional, Protocol


class VectorStoreProtocol(Protocol):
    """Interface for vector‑store operations consumed by the search engine."""

    def add_texts(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add texts with metadata to the vector store."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return raw embedding vectors for the given texts."""

    def search(
        self,
        query_text: str,
        max_results: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.  Returns hit dicts with ``score``, ``text_str``, ``metadata``."""

    def rerank(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Rerank results using a cross‑encoder model."""


class RelationalDataProtocol(Protocol):
    """Interface for relational data access consumed by the search engine."""

    def get_collection_chunks(
        self, collection_id: int, min_char_len: int = 300
    ) -> Any:  # QuerySet — avoid importing django.db.models
        """Get all text chunks for a collection (used during indexing)."""

    def documents_names_from_categories(
        self,
        collection_id: int,
        categories: List[str],
        only_used_to_search: bool = True,
    ) -> List[str]:
        """Get document names matching given categories."""

    def document_names_relative_path_contains(
        self,
        collection_id: int,
        substrings: List[str],
        only_used_to_search: bool = True,
    ) -> List[str]:
        """Find document names whose relative path contains any substring."""

    def fts_search(
        self,
        query_str: str,
        collection_id: int,
        max_results: int,
        language: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """Full‑text search returning ``(text_id, rank)`` tuples."""

    def get_text_for_id(
        self,
        text_ids: List[int],
        surrounding_chunks: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get text chunks by ID with surrounding context."""

    def get_all_documents_from_collection(
        self, collection_id: int
    ) -> Any:  # QuerySet — avoid importing django.db.models
        """Get all document objects for a collection (for metadata filtering)."""


class ModelConfigProtocol(Protocol):
    """Interface for embedder/reranker model configuration."""

    @property
    def embedders(self) -> List[str]:
        ...

    @property
    def rerankers(self) -> List[str]:
        ...

    def get_embedder_path(self, model_name: str) -> str:
        ...

    def get_reranker_path(self, model_name: str) -> str:
        ...

    def get_embedder_vector_size(self, model_name: str) -> int:
        ...

    def get_embedder_device(self, model_name: str) -> str:
        ...

    def get_reranker_device(self, model_name: str) -> str:
        ...


class GenerativeServiceProtocol(Protocol):
    """Interface for generative (LLM) answer services."""

    def generative_answer_for_response(
        self,
        user_response: Any,  # DTO / data object — not a Django model
        query_instruction: str,
        query_options: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Any:
        ...


class ExtractiveQAProtocol(Protocol):
    """Interface for extractive question‑answering services."""

    def run_extractive_qa(self, question_str: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        ...
