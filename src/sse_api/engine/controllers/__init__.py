"""
engine/controllers
------------------

Backward‑compatibility re‑exports.  All original import paths continue to work.
New consumers should prefer direct imports from ``engine.core`` and
``engine.database``.
"""

from sse_api.engine.controllers.search.semantic import DBSemanticSearchController
from sse_api.engine.controllers.search.query import SearchQueryController
from sse_api.engine.controllers.search.relational import DBTextSearchController
from sse_api.engine.controllers.models_logic.extractive import (
    ExtractiveQAController,
)  # noqa: F401
from sse_api.engine.controllers.database.milvus import MilvusHandler, INDEX_QUERY_PARAMS
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.database.semantic_db import SemanticDBController
from sse_api.engine.controllers.models_logic.embedders_rerankers import (
    EmbeddingModelsConfig,
)
from sse_api.engine.controllers.system_logic.system import EngineSystemController


__all__ = [
    "DBSemanticSearchController",
    "SearchQueryController",
    "DBTextSearchController",
    "MilvusHandler",
    "INDEX_QUERY_PARAMS",
    "RelationalDBController",
    "SemanticDBController",
    "EmbeddingModelsConfig",
    "EngineSystemController",
    "ExtractiveQAController",
]
