"""
database (adapter layer)
-------------------------

Re-exports the original implementations for backward‑compatibility and
provides protocol‑based adapter classes that bridge engine.core protocols
to concrete database backends.
"""

from sse_api.engine.database.milvus_impl import MilvusHandler, INDEX_QUERY_PARAMS
from sse_api.engine.database.semantic_db_impl import SemanticDBController
from sse_api.engine.database.vector_store_adapter import MilvusVectorStoreAdapter
from sse_api.engine.database.data_adapter import RelationalDataAdapter

# Re-export original controllers for backward compatibility (existing code still uses these paths).
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.search.relational import DBTextSearchController

__all__ = [
    "MilvusHandler",
    "INDEX_QUERY_PARAMS",
    "SemanticDBController",
    "RelationalDBController",
    "DBTextSearchController",
    "MilvusVectorStoreAdapter",
    "RelationalDataAdapter",
]
