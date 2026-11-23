"""
query_pipeline.py
-----------------

Lightweight query orchestrator that uses engine.core protocols and
pure computation functions.  This class **does not** access Django ORM —
it is the hosting app's responsibility to create ``UserQuery`` /
``UserQueryResponse`` records.
"""

from typing import Any, Dict, Optional

from radlab_data.text.utils import TextUtils

from sse_api.engine.core.protocols import (
    VectorStoreProtocol,
    RelationalDataProtocol,
    ModelConfigProtocol,
)
from sse_api.data.controllers.constants import DEFAULT_MIN_SIMILARITY
from sse_api.engine.core.search_engine import (
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_LEXICAL_WEIGHT,
    SearchEngineCore,
)


def _as_float(value, default: float) -> float:
    """``value`` as a fraction of the fusion, ``default`` when unusable.

    The weights come from a request or from the JSON profile of a
    collection, so they may be anything; a broken value must not break a
    search, it falls back to the default of the engine.
    """
    if isinstance(value, bool) or value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return default
    if not 0.0 <= weight <= 1.0:
        return default
    return weight


class QueryPipeline:
    """
    Orchestrates a single query through the full hybrid search pipeline
    using protocol‑based dependencies.
    """

    @staticmethod
    def execute(
        query_text: str,
        collection_id: int,
        model_config: ModelConfigProtocol,
        vector_store: VectorStoreProtocol,
        data_source: RelationalDataProtocol,
        search_options: Optional[Dict[str, Any]] = None,
        ignore_question_lang_detect: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a complete search pipeline.

        Parameters
        ----------
        query_text : str
            The raw user query.
        collection_id : int
            Primary key of the document collection.
        model_config : ModelConfigProtocol
            Embedder/reranker configuration.
        vector_store : VectorStoreProtocol
            Vector store adapter.
        data_source : RelationalDataProtocol
            Relational data adapter for FTS and text retrieval.
        search_options : dict | None
            Search options (filters, ranking flags, etc.).
        ignore_question_lang_detect : bool, default False
            Skip language detection if ``True``.

        Returns
        -------
        dict
            ``{"results": {"query", "stats", "detailed_results"}, ...}``
        """
        if search_options is None:
            search_options = {}

        max_results = int(search_options.get("max_results", 40))
        hybrid_search = bool(search_options.get("hybrid_search", True))
        rerank_results_flag = bool(search_options.get("rerank_results", False))
        min_similarity = search_options.get("min_similarity", DEFAULT_MIN_SIMILARITY)
        # Share of every leg in the rank fusion; the request (or the profile
        # of the collection) decides, the defaults of the engine otherwise.
        dense_weight = _as_float(
            search_options.get("dense_weight"), DEFAULT_DENSE_WEIGHT
        )
        lexical_weight = _as_float(
            search_options.get("lexical_weight"), DEFAULT_LEXICAL_WEIGHT
        )

        # Detect language if needed
        language: Optional[str] = None
        if not ignore_question_lang_detect and query_text.strip():
            lang = TextUtils.text_language(query_text)
            language = lang or "simple"

        # Resolve document name filters from multiple sources
        filter_doc_names: list[str] = []

        categories = search_options.get("categories", [])
        if categories:
            names = data_source.documents_names_from_categories(
                collection_id, [str(c) for c in categories]
            )
            filter_doc_names.extend(names)

        rel_paths = search_options.get("relative_paths", [])
        if rel_paths:
            names = data_source.document_names_relative_path_contains(
                collection_id, rel_paths
            )
            filter_doc_names.extend(names)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_filter_names: list[str] = []
        for name in filter_doc_names:
            if name not in seen:
                seen.add(name)
                unique_filter_names.append(name)

        # Step 5: Execute search via SearchEngineCore
        engine = SearchEngineCore(
            vector_store=vector_store,
            data_source=data_source,
            model_config=model_config,
        )

        results = engine.hybrid_search(
            query_text=query_text,
            collection_id=collection_id,
            filter_doc_names=unique_filter_names,
            max_results=max_results,
            rerank=rerank_results_flag,
            language=language,
            hybrid_search=hybrid_search,
            min_similarity=min_similarity,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )

        return {
            "results": results,
        }
