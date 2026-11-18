"""
sse_models — standalone model service for the Semantic Search Engine.

A small FastAPI service that hosts the embedder and reranker models *outside*
of the main engine process.  The engine talks to it over HTTP (see
``sse_api.engine.core.models.backend.RemoteBackend``), so the heavy
inference can run on a different (GPU) machine, be scaled independently and
be restarted without touching the engine.

The service is self-contained: it does not import Django or the engine.  It
only needs ``fastapi``, ``uvicorn`` and the model stack (``torch``,
``sentence_transformers``).

Run it directly::

    python -m sse_models
    # or
    sse-models

or via uvicorn::

    uvicorn sse_models.app:app --host 0.0.0.0 --port 8280
"""

__version__ = "0.1.0"
