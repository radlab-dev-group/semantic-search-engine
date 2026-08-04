"""
milvus_impl.py
--------------

Thin re‑export of the original MilvusHandler implementation.
Kept in this file so that engine_core can reference it by name while
the original source file remains unchanged for backward‑compatibility.
"""

# Re-export from the original location — zero code changes needed.
from engine.controllers.database.milvus import (
    MilvusHandler,
    INDEX_QUERY_PARAMS,
)

__all__ = ["MilvusHandler", "INDEX_QUERY_PARAMS"]
