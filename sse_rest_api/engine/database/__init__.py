"""
database (adapter layer)
-------------------------

Re-exports the original implementations for backward‑compatibility and
provides protocol‑based adapter classes that bridge engine_core protocols
to concrete database backends.
"""

from engine.database.milvus_impl import MilvusHandler, INDEX_QUERY_PARAMS
from engine.database.semantic_db_impl import SemanticDBController
from engine.database.vector_store_adapter import MilvusVectorStoreAdapter
from engine.database.data_adapter import RelationalDataAdapter

# Re-export original controllers for backward compatibility (existing code still uses these paths).
from engine.controllers.database.relational_db import RelationalDBController
from engine.controllers.search.relational import DBTextSearchController

__all__ = [
    "MilvusHandler",
    "INDEX_QUERY_PARAMS",
    "SemanticDBController",
    "RelationalDBController",
    "DBTextSearchController",
    "MilvusVectorStoreAdapter",
    "RelationalDataAdapter",
]
