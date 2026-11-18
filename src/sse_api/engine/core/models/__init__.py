"""
engine/core/models/ — model backend abstraction for the search engine.

``ModelBackend`` hides *where* the embedder and reranker run: in-process
(``LocalBackend``, the default, and what the engine has always done) or in a
standalone model service reached over HTTP (``RemoteBackend``, see the
``sse_models`` package).  ``build_backend()`` picks between them from the
environment, so switching is a config change, not a code change.
"""

from sse_api.engine.core.models.backend import (
    ModelBackend,
    LocalBackend,
    RemoteBackend,
    ModelBackendError,
    build_backend,
    resolve_model_name,
)

__all__ = [
    "ModelBackend",
    "LocalBackend",
    "RemoteBackend",
    "ModelBackendError",
    "build_backend",
    "resolve_model_name",
]
