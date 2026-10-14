"""
engine.core
-----------

Standalone search‑engine core that can be used without Django or database
dependencies.  Exports protocol interfaces, pure‑computation helper functions,
the ``SearchEngineCore`` orchestrator and the query‑pipeline entry point.
"""

from sse_api.engine.core.protocols import (
    VectorStoreProtocol,
    RelationalDataProtocol,
    ModelConfigProtocol,
    GenerativeServiceProtocol,
    ExtractiveQAProtocol,
)
from sse_api.engine.core.search_engine import (
    rrf_merge,
    rerank_results_with_vector_store,
    prepare_documents_stats,
    filter_stats_to_display_results,
    convert_search_results_to_doc2answer,
    get_accumulated_docs_by_rank_perc,
    reformat_search_results_to_display,
    SearchEngineCore,
)
from sse_api.engine.core.query_pipeline import QueryPipeline
from sse_api.engine.core.config.models_config import (
    EmbeddingModelsConfig,
    ALL_AVAILABLE_EMBEDDERS_MODELS,
    ALL_AVAILABLE_RERANKERS_MODELS,
)
from sse_api.engine.core.generative.extractive import ExtractiveQAController

__all__ = [
    # Protocols
    "VectorStoreProtocol",
    "RelationalDataProtocol",
    "ModelConfigProtocol",
    "GenerativeServiceProtocol",
    "ExtractiveQAProtocol",
    # Pure computation functions
    "rrf_merge",
    "rerank_results_with_vector_store",
    "prepare_documents_stats",
    "filter_stats_to_display_results",
    "convert_search_results_to_doc2answer",
    "get_accumulated_docs_by_rank_perc",
    "reformat_search_results_to_display",
    # Classes
    "SearchEngineCore",
    "QueryPipeline",
    "EmbeddingModelsConfig",
    "ExtractiveQAController",
]

# Also expose the global config dicts as top-level attributes
ALL_AVAILABLE_EMBEDDERS_MODELS = ALL_AVAILABLE_EMBEDDERS_MODELS
ALL_AVAILABLE_RERANKERS_MODELS = ALL_AVAILABLE_RERANKERS_MODELS
