"""
token_chunking.py
-----------------

Token chunking of the indexed documents, done by the model service.

A chunk is measured in tokens, and tokens belong to the tokenizer of the
embedder of the collection.  The engine does not carry that tokenizer (it
carries no ``transformers`` at all), so counting them is the model service's
job — ``POST /v1/chunk`` — exactly like counting and truncating for a query.

Without a configured service there is nothing to call and the caller keeps
its historical behaviour: the document reader chunks with its own tokenizer,
which is why a single-host installation still installs the model stack for
the indexing of token-chunked collections.

The shape of the result follows the reader the engine already uses
(``radlab_data``): every chunk is a copy of the page it came from, the
``chunk`` metadata numbers the chunks *inside their page* starting at 1, and
``text_chunk_type`` gains the ``_chunk_max_tokens_<n>`` and ``_overlap_<n>``
parts.  Rows written by the reader and rows written through the service
differ in their boundaries (another tokenizer cut them), never in the way
they are described.
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: ``(texts, max_tokens, overlap) -> chunks of every text``.
Chunker = Callable[[Sequence[str], int, int], List[List[str]]]


def chunk_type_suffix(max_tokens: int, overlap: int = 0) -> str:
    """
    The ``text_chunk_type`` part which describes token chunking.

    Mirrors ``radlab_data.text.loaders.loader_i``: a chunk size of at least
    two tokens is worth naming, an overlap is named when it is positive.
    """
    suffix = ""
    if max_tokens and max_tokens > 1:
        suffix += f"_chunk_max_tokens_{max_tokens}"
        if overlap and overlap > 0:
            suffix += f"_overlap_{overlap}"
    return suffix


def chunk_pages(
    pages: Sequence[Dict[str, Any]],
    chunker: Chunker,
    max_tokens: int,
    overlap: int = 0,
) -> List[Dict[str, Any]]:
    """
    Split the ``page_content`` of every page into token chunks.

    Every page comes back numbered: a page which fits the budget is chunk 1
    of its page, a split one is chunks 2..n, exactly as the reader numbers
    the pages it chunks (the indexing stores ``metadata["chunk"]`` of every
    stored text, so the number must be there in both paths).
    """
    texts = [page.get("page_content") or "" for page in pages]
    if not any(texts):
        return [deepcopy(page) for page in pages]

    window = chunker(texts, max_tokens, overlap)
    if len(window) != len(pages):
        raise ValueError(
            f"chunking returned {len(window)} results for {len(pages)} pages"
        )

    suffix = chunk_type_suffix(max_tokens, overlap)
    chunked_pages: List[Dict[str, Any]] = []
    for page, parts in zip(pages, window):
        parts = [part for part in parts if part]
        metadata = dict(page.get("metadata") or {})
        if suffix:
            metadata["text_chunk_type"] = (
                f"{metadata.get('text_chunk_type', 'raw_text')}{suffix}"
            )
        for number, part in enumerate(
            parts or [page.get("page_content") or ""], start=1
        ):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk"] = number
            chunked_pages.append(
                {**page, "page_content": part, "metadata": chunk_metadata}
            )
    return chunked_pages


def chunk_document_dict(
    doc_as_dict: Dict[str, Any],
    chunker: Chunker,
    max_tokens: int,
    overlap: int = 0,
) -> Dict[str, Any]:
    """
    Token-chunk every page of a document as returned by the reader.

    Only ``pages`` changes; the identity of the document (its paths, category
    and options) is carried over untouched, so the rest of the indexing sees
    the same dictionary it has always seen.
    """
    return {
        **doc_as_dict,
        "pages": chunk_pages(
            doc_as_dict.get("pages") or [], chunker, max_tokens, overlap
        ),
    }


def token_chunker_for(
    embedder: Optional[str],
    max_tokens: Optional[int],
    batch_size: int = 32,
) -> Optional[Chunker]:
    """
    A chunker of ``max_tokens``-sized chunks for a collection's ``embedder``.

    ``None`` when there is nothing to chunk (no chunk size given) or nothing
    to ask (no model service configured, no embedder of the collection); the
    caller then lets the document chunk as it used to.  The embedder of the
    collection is a filesystem path, the service addresses models by their
    catalog name, so the name is resolved first.
    """
    if not max_tokens or max_tokens < 2:
        return None

    from sse_api.engine.core.models.backend import (  # noqa: PLC0415
        build_chunk_client,
        resolve_model_name,
    )

    model_name = resolve_model_name("embedder", embedder)
    chunker = build_chunk_client(model_name)
    if chunker is None:
        return None

    def _chunk_in_batches(
        texts: Sequence[str], tokens: int, overlap: int = 0
    ) -> List[List[str]]:
        chunks: List[List[str]] = []
        for start in range(0, len(texts), batch_size):
            chunks.extend(
                chunker(texts[start : start + batch_size], tokens, overlap)
            )
        return chunks

    return _chunk_in_batches
