"""
milvus_impl.py
--------------

Thin re‑export of the original MilvusHandler implementation.
Kept in this file so that engine.core can reference it by name while
the original source file remains unchanged for backward‑compatibility.
"""

# Re-export from the original location — zero code changes needed.
from sse_api.engine.controllers.database.milvus import (
    MilvusHandler,
    INDEX_QUERY_PARAMS,
)

__all__ = ["MilvusHandler", "INDEX_QUERY_PARAMS"]
