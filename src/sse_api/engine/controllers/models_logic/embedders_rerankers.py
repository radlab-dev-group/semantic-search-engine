"""
embedders_rerankers.py
---------------------------------
Backward-compatibility shim for the embedder and reranker model configuration.

The implementation lives in :mod:`sse_api.engine.core.config.models_config`;
this module only re-exports it, so that ``path``, ``vector_size``, ``device``
and ``max_model_length`` are read from a single registry whichever import path
a consumer uses.
"""

from sse_api.engine.core.config import models_config as _models_config
from sse_api.engine.core.config.models_config import (  # noqa: F401
    ALL_AVAILABLE_EMBEDDERS_MODELS,
    ALL_AVAILABLE_RERANKERS_MODELS,
    UNKNOWN_MAX_MODEL_LENGTH,
    EmbeddingModelsConfig,
    _max_model_length,
)

__all__ = [
    "EmbeddingModelsConfig",
    "ALL_AVAILABLE_EMBEDDERS_MODELS",
    "ALL_AVAILABLE_RERANKERS_MODELS",
    "UNKNOWN_MAX_MODEL_LENGTH",
]


def _ensure_config_loaded() -> None:
    """Delegate to the single configuration instance of ``engine.core``."""
    _models_config._ensure_config_loaded()
