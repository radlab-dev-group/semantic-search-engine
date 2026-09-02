"""
engine/controllers
------------------

Backward‑compatibility re‑exports.  All original import paths continue to work.
New consumers should prefer direct imports from ``engine.engine_core`` and
``engine.database``.
"""

from engine.controllers.search.semantic import DBSemanticSearchController
from engine.controllers.search.query import SearchQueryController
from engine.controllers.search.relational import DBTextSearchController
from engine.controllers.models_logic.extractive import (
    ExtractiveQAController,
)  # noqa: F401
from engine.controllers.database.milvus import MilvusHandler, INDEX_QUERY_PARAMS
from engine.controllers.database.relational_db import RelationalDBController
from engine.controllers.database.semantic_db import SemanticDBController
from engine.controllers.models_logic.embedders_rerankers import EmbeddingModelsConfig
from engine.controllers.system_logic.system import EngineSystemController

# GenerativeModelConfig may not be importable if llm_router_lib is missing
try:
    from engine.controllers.models_logic.generative import (
        GenerativeModelConfig,
    )  # noqa: F401
except ImportError:
    pass

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
