"""
vector_store_adapter.py
-----------------------

Adapter that implements ``VectorStoreProtocol`` by wrapping ``MilvusHandler``.
Provides a clean protocol boundary so engine_core never needs to know about
pymilvus or Milvus internals.
"""

from typing import Any, Dict, List, Optional

from engine.engine_core.protocols import VectorStoreProtocol
from engine.database.milvus_impl import MilvusHandler


class MilvusVectorStoreAdapter(VectorStoreProtocol):
    """
    Protocol‑based adapter that delegates to ``MilvusHandler``.
    Implements ``VectorStoreProtocol`` by wrapping all low‑level
    Milvus operations.
    """

    def __init__(self, handler: MilvusHandler) -> None:
        self._handler = handler

    # ------------------------------------------------------------------
    # VectorStoreProtocol implementation
    # ------------------------------------------------------------------

    def add_texts(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add texts with metadata to the vector store."""
        if not texts:
            return
        self._handler.add_texts(
            texts=texts,
            metadata=metadata,
            max_text_len=self._handler.max_tokens_in_text,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return raw embedding vectors for the given texts."""
        return self._handler.prepare_embeddings(texts)

    def search(
        self,
        query_text: str,
        max_results: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.  Returns a flat list of hit dicts.

        ``min_similarity`` (optional) is the cosine similarity cutoff.
        """
        # MilvusHandler.search returns list[list[dict]] — flatten to single list
        all_hits = self._handler.search(
            search_text=query_text,
            max_results=max_results,
            additional_output_fields=None,
            post_search_options=None,
            metadata_filter=metadata_filter,
            min_similarity=min_similarity,
        )

        # Flatten: MilvusHandler returns one hit-list per query; we take the first (and usually only) one.
        hits_list = all_hits[0] if all_hits else []

        # Each hit dict has keys: "score", "text", "metadata"
        # Re-key "text" → "text_str" for consistency with downstream code
        results: list[dict[str, Any]] = []
        for hit in hits_list:
            result = {
                "score": hit["score"],
                "text_str": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
            }
            # Pass through any extra fields (e.g., from post_search_options)
            for k, v in hit.items():
                if k not in ("score", "text", "metadata"):
                    result[k] = v
            results.append(result)

        return results

    def rerank(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Rerank results using a cross‑encoder model."""
        if not results:
            return []

        # Format for MilvusHandler._rerank_search_results
        formatted = [
            {self._handler.DB_FIELD_TEXT: r.get("text_str", ""), "original_data": r}
            for r in results
        ]

        try:
            self._handler._load_reranker_model_from_path()
            reranked_all = self._handler._rerank_search_results(
                query_text, [formatted]
            )

            reranked_hits = reranked_all[0]

            final_results: list[dict[str, Any]] = []
            for hit in reranked_hits:
                original_data = hit["original_data"]
                original_data["score"] = float(hit["score"])
                final_results.append(original_data)

            return final_results[:max_results]
        except Exception:
            return results[:max_results]

    @property
    def collection_name(self) -> str:
        """Expose the underlying Milvus collection name."""
        return self._handler.collection_name
