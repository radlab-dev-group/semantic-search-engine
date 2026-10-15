"""
chunking.py
-----------

Chunking (splitting texts into chunks) is a property of a collection: mixing
chunk sizes inside one collection makes similarities, thresholds and context
windows incomparable. The values can still be sent with every indexing call,
but they are resolved against the settings of the collection:

* declared on the request  -> used (and, in a strict collection, compared),
* not declared             -> the settings of the collection are used,
* neither of them          -> the global defaults are used and remembered on
  the collection, so the next call does not have to pass anything.

A chunk also has to fit the embedder of the collection: ``sentence-transformers``
truncates a longer text silently, so the end of such a chunk would never reach
the vector.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from sse_api.data.core.constants import (
    DEFAULT_MAX_TOKENS_IN_CHUNK,
    DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
    MAX_TOKENS_IN_CHUNK_KEY,
    MAX_TOKENS_IN_CHUNK_LIMIT,
    NUMBER_OF_OVERLAP_TOKENS_KEY,
)
from sse_api.data.models import CollectionOfDocuments
from sse_api.engine.core.config.models_config import EmbeddingModelsConfig

logger = logging.getLogger(__name__)

#: Chunking resolved from a request, a collection or the global defaults.
Chunking = Dict[str, int]


class ChunkingError(Exception):
    """Base class of the chunking problems reported by this module."""


class InvalidChunking(ChunkingError):
    """The chunk size and its overlap do not describe a usable split."""


class ChunkingTooLong(ChunkingError):
    """
    The chunk does not fit the maximum input length of the embedder.

    Attributes
    ----------
    max_tokens_in_chunk : int
        Chunk size which was asked for.
    max_model_length : int
        Maximum input length of the embedder, in tokens.
    model_name : str
        Name of the embedder declaring that limit.
    """

    def __init__(
        self, max_tokens_in_chunk: int, max_model_length: int, model_name: str
    ):
        self.max_tokens_in_chunk = max_tokens_in_chunk
        self.max_model_length = max_model_length
        self.model_name = model_name
        super().__init__(
            f"max_tokens_in_chunk={max_tokens_in_chunk} exceeds the "
            f"max_model_length={max_model_length} of {model_name}"
        )


class ChunkingMismatch(ChunkingError):
    """
    The request asks for another chunking than the one of the collection.

    Only raised when the collection enforces its settings
    (``metadata_json["strict_chunking"]``).

    Attributes
    ----------
    declared : dict
        Chunking of the collection.
    requested : dict
        Chunking asked for by the request.
    """

    def __init__(self, declared: Chunking, requested: Chunking):
        self.declared = declared
        self.requested = requested
        super().__init__(
            f"collection chunks at {declared}, request asked for {requested}"
        )


def _model_max_length(lookup_name: str, model_name: Optional[str]) -> Optional[int]:
    """
    Read the maximum input length of a model from the configuration.

    A model which is not configured, or a configuration which cannot be read
    at all, means "no known limit" and never interrupts indexing.
    """
    if not model_name:
        return None
    lookup = getattr(EmbeddingModelsConfig, lookup_name, None)
    if lookup is None:
        return None
    try:
        return lookup(model_name)
    except Exception:  # missing config file, unknown model, malformed entry
        logger.warning(
            "Could not read %s of model %r", lookup_name, model_name, exc_info=False
        )
        return None


def embedder_max_length(model_name: Optional[str]) -> Optional[int]:
    """Maximum length of a single text of ``model_name``, in tokens."""
    return _model_max_length("get_embedder_max_model_length", model_name)


def reranker_max_length(model_name: Optional[str]) -> Optional[int]:
    """Maximum length of a ``query + text`` pair of ``model_name``, in tokens."""
    return _model_max_length("get_reranker_max_model_length", model_name)


def declared_chunking(indexing_options: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Read the chunking declared by an ``indexing_options`` dictionary.

    Returns
    -------
    tuple
        ``(max_tokens_in_chunk, number_of_overlap_tokens)``; either element is
        ``None`` when the request did not declare it.
    """
    options = indexing_options if isinstance(indexing_options, dict) else {}
    max_tokens = options.get(MAX_TOKENS_IN_CHUNK_KEY)
    overlap = options.get(NUMBER_OF_OVERLAP_TOKENS_KEY)
    return (
        int(max_tokens) if max_tokens is not None else None,
        int(overlap) if overlap is not None else None,
    )


def check_chunk_size(
    max_tokens_in_chunk: int,
    embedder_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
) -> None:
    """
    Check a chunk size against the models of the collection.

    Parameters
    ----------
    max_tokens_in_chunk : int
        Chunk size in tokens.
    embedder_model : str, optional
        Embedder of the collection; the chunk has to fit its input.
    reranker_model : str, optional
        Reranker of the collection; a chunk which does not fit a
        ``query + text`` pair is only reported, because the query takes part
        of that budget already.

    Raises
    ------
    ChunkingTooLong
        When the chunk is longer than the input of the embedder or above
        :data:`~sse_api.data.core.constants.MAX_TOKENS_IN_CHUNK_LIMIT`.
    """
    if (
        isinstance(max_tokens_in_chunk, bool)
        or not isinstance(max_tokens_in_chunk, int)
        or max_tokens_in_chunk < 1
    ):
        raise InvalidChunking(
            f"max_tokens_in_chunk must be a positive number, got {max_tokens_in_chunk!r}"
        )
    if max_tokens_in_chunk > MAX_TOKENS_IN_CHUNK_LIMIT:
        raise ChunkingTooLong(
            max_tokens_in_chunk, MAX_TOKENS_IN_CHUNK_LIMIT, "sanity limit"
        )

    limit = embedder_max_length(embedder_model)
    if limit is not None and max_tokens_in_chunk > limit:
        raise ChunkingTooLong(max_tokens_in_chunk, limit, embedder_model or "")

    reranker_limit = reranker_max_length(reranker_model)
    if reranker_limit is not None and max_tokens_in_chunk >= reranker_limit:
        logger.warning(
            "Chunks of %d tokens do not fit the reranker %r (max_model_length=%d "
            "for query + text); reranking will see a prefix of every chunk",
            max_tokens_in_chunk,
            reranker_model,
            reranker_limit,
        )


def resolve_chunking(
    collection: CollectionOfDocuments,
    indexing_options: Any,
    bind_defaults: bool = True,
) -> Dict[str, Any]:
    """
    Resolve and validate the chunking of one indexing call.

    Parameters
    ----------
    collection : CollectionOfDocuments
        Collection the texts are indexed into.
    indexing_options : dict
        Options of the request; its chunk keys are optional.
    bind_defaults : bool, default True
        Remember the defaults on a collection which has no chunking yet, so
        that the next call does not have to pass them.

    Returns
    -------
    dict
        ``indexing_options`` with both chunk keys set to the values to use.

    Raises
    ------
    ChunkingMismatch
        A strict collection and another chunking than its own.
    ChunkingTooLong
        The chunk does not fit the embedder of the collection.
    InvalidChunking
        The overlap is not a number of tokens inside ``[0, chunk)``.
    """
    options = dict(indexing_options) if isinstance(indexing_options, dict) else {}
    requested_max, requested_overlap = declared_chunking(options)
    declared = collection.chunking()
    declared_max = _as_optional_int(declared.get(MAX_TOKENS_IN_CHUNK_KEY))
    declared_overlap = _as_optional_int(declared.get(NUMBER_OF_OVERLAP_TOKENS_KEY))

    if requested_max is None and requested_overlap is None:
        # Nothing asked for: the collection decides, or the global defaults
        # which are then remembered on the collection.
        max_tokens, overlap = _declared_or_defaults(
            declared_max, declared_overlap, bind_defaults, collection
        )
    else:
        # Something asked for: the request wins, and what it does not mention
        # comes from the collection (or the defaults for a collection created
        # without chunking).
        max_tokens = (
            requested_max
            if requested_max is not None
            else _as_int(declared_max, DEFAULT_MAX_TOKENS_IN_CHUNK)
        )
        overlap = (
            requested_overlap
            if requested_overlap is not None
            else _as_int(declared_overlap, DEFAULT_NUMBER_OF_OVERLAP_TOKENS)
        )
        if (
            collection.strict_chunking
            and declared_max is not None
            and (
                max_tokens != declared_max or overlap != _as_int(declared_overlap, 0)
            )
        ):
            raise ChunkingMismatch(
                {
                    MAX_TOKENS_IN_CHUNK_KEY: declared_max,
                    NUMBER_OF_OVERLAP_TOKENS_KEY: _as_int(declared_overlap, 0),
                },
                {
                    MAX_TOKENS_IN_CHUNK_KEY: max_tokens,
                    NUMBER_OF_OVERLAP_TOKENS_KEY: overlap,
                },
            )
        if declared_max is None and bind_defaults:
            _bind(collection, max_tokens, overlap)

    if not 0 <= overlap < max_tokens:
        raise InvalidChunking(
            f"number_of_overlap_tokens must be within [0, {max_tokens}), "
            f"got {overlap}"
        )

    check_chunk_size(
        max_tokens,
        embedder_model=collection.model_embedder,
        reranker_model=collection.model_reranker,
    )

    options[MAX_TOKENS_IN_CHUNK_KEY] = max_tokens
    options[NUMBER_OF_OVERLAP_TOKENS_KEY] = overlap
    return options


def _as_int(value: Any, default: int) -> int:
    """Read an integer of a chunking entry, ``default`` when absent."""
    fallback = _as_optional_int(value)
    return default if fallback is None else fallback


def _as_optional_int(value: Any) -> Optional[int]:
    """Read an integer of a chunking entry, ``None`` when it is no number.

    The values come from a JSON column, so they may be anything; a value
    which is not a whole number is treated as "not declared" rather than
    breaking indexing.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _declared_or_defaults(declared_max, declared_overlap, bind_defaults, collection):
    """Chunking of the collection, or the global defaults bound onto it."""
    if declared_max is not None:
        return _as_int(declared_max, DEFAULT_MAX_TOKENS_IN_CHUNK), _as_int(
            declared_overlap, DEFAULT_NUMBER_OF_OVERLAP_TOKENS
        )
    if bind_defaults:
        _bind(
            collection, DEFAULT_MAX_TOKENS_IN_CHUNK, DEFAULT_NUMBER_OF_OVERLAP_TOKENS
        )
    return DEFAULT_MAX_TOKENS_IN_CHUNK, DEFAULT_NUMBER_OF_OVERLAP_TOKENS


def _bind(
    collection: CollectionOfDocuments, max_tokens_in_chunk: int, overlap: int
) -> None:
    """Remember chunking on a collection which does not have it yet."""
    collection.set_chunking(max_tokens_in_chunk, overlap)
    if collection.pk is None:
        return  # not in the database (a stand-in of a test): nothing to update
    try:
        collection.save(update_fields=["metadata_json"])
    except Exception:  # a read-only or not yet migrated database must not stop
        logger.warning(
            "Could not store the chunking defaults on collection %r",
            getattr(collection, "name", None),
            exc_info=False,
        )
