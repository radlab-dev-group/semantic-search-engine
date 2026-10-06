"""
sse_lib
-------

Thin client library for the **Semantic Search Engine** (SSE) REST API.

It wraps the HTTP endpoints of ``sse_rest_api`` (collections, indexing, hybrid
search, RAG answers, chats and model listings) behind a small Python facade;
the server side stays untouched::

    from sse_lib import SSEClient, SearchOptions, GenerativeOptions

    with SSEClient("http://localhost:8271", token="...") as client:
        collection = client.create_collection(
            "my_docs",
            "My Documents",
            "Test collection",
            embedder="radlab/polish-bi-encoder-mean",
            reranker="radlab/polish-cross-encoder",
            index_type="HNSW",
        )
        client.index_documents(collection.name, ["Some text to index."])
        found = client.search(collection.name, "text", max_results=10)
        answer = client.generate_answer(
            found.query_response_id,
            GenerativeOptions(generative_model="radlab/pLLama-3-8B-DPO-L"),
        )
        print(answer.answer)
"""

from __future__ import annotations

from sse_lib import endpoints
from sse_lib.client import SSEClient
from sse_lib.exceptions import (
    SSEAPIError,
    SSEAuthenticationError,
    SSEConfigError,
    SSEError,
    SSEHTTPError,
    SSETransportError,
    SSEValueError,
)
from sse_lib.models import (
    Answer,
    Chat,
    ChatHistory,
    ChatMessage,
    ChatReply,
    Collection,
    IndexingResult,
    SearchHit,
    SearchResponse,
    UploadResult,
)
from sse_lib.options import (
    DEFAULT_INDEX_TYPE,
    INDEX_TYPES,
    GenerativeOptions,
    IndexingOptions,
    SearchOptions,
    build_options,
    text_document,
)
from sse_lib.transport import DEFAULT_TIMEOUT, DEFAULT_TOKEN_TYPE, Transport

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_INDEX_TYPE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TOKEN_TYPE",
    "INDEX_TYPES",
    "Answer",
    "Chat",
    "ChatHistory",
    "ChatMessage",
    "ChatReply",
    "Collection",
    "GenerativeOptions",
    "IndexingOptions",
    "IndexingResult",
    "SSEAPIError",
    "SSEAuthenticationError",
    "SSEClient",
    "SSEConfigError",
    "SSEError",
    "SSEHTTPError",
    "SSETransportError",
    "SSEValueError",
    "SearchHit",
    "SearchOptions",
    "SearchResponse",
    "Transport",
    "UploadResult",
    "build_options",
    "endpoints",
    "text_document",
    "__version__",
]
