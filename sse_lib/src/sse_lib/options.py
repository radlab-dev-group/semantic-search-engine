"""
options.py
----------

Typed builders for the option dictionaries accepted by the SSE API.

They exist for two reasons:

* ``engine`` and ``data`` read some of the options with ``options["key"]``
  (``IndexingOptions`` below always serialises every required key, so a missing
  key cannot turn into an HTTP 500), while every other option is read with
  ``.get()`` and is therefore omitted when unset;
* the API is not consistent about the wire format - ``options`` of
  ``search_with_options``, ``query_options`` of ``generative_answer`` and
  ``indexing_options`` of ``upload_and_index_files`` are sent as JSON *strings*
  (the views call ``json.loads``), whereas ``add_and_index_texts`` and the
  chat endpoints expect real JSON objects.  The client applies the right
  encoding; callers only ever pass the objects built here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Union

from sse_lib.exceptions import SSEValueError

# Accepted values of ``embedder_index_type`` (``INDEX_QUERY_PARAMS`` in
# ``engine.controllers.database.milvus``).
INDEX_TYPES = ("HNSW", "IVF_FLAT")
DEFAULT_INDEX_TYPE = "HNSW"


@dataclass
class SearchOptions:
    """Options of ``POST search_with_options``.

    Only the keys that are set are sent; the server defaults apply to the rest
    (``max_results``, ``hybrid_search=True``, ``rerank_results=False``, ...).

    Every key here is one the backend actually reads in
    ``engine.controllers.search.semantic``.  ``documents_names`` is the
    exception in the other direction: ``core.validation.search_options`` accepts
    it, but no controller ever looks at it, so it stays out of this class on
    purpose - sending it would only suggest that it filters something.
    """

    categories: Optional[List[str]] = None
    documents: Optional[List[str]] = None
    relative_paths: Optional[List[str]] = None
    relative_path_contains: Optional[List[str]] = None
    metadata_filters: Optional[List[Dict[str, Any]]] = None
    templates: Optional[int] = None
    only_template_documents: Optional[bool] = None
    max_results: Optional[int] = None
    rerank_results: Optional[bool] = None
    rerank_max_results: Optional[int] = None
    return_with_factored_fields: Optional[bool] = None
    hybrid_search: Optional[bool] = None
    min_similarity: Optional[float] = None
    rrf_k: Optional[int] = None
    use_and_operator: Optional[bool] = None
    #: Keys of this dict are merged into the payload; escape hatch for options
    #: that this release of the library does not know about yet.
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the payload understood by the API (``None`` dropped)."""
        payload: Dict[str, Any] = {}
        for meta in fields(self):
            if meta.name == "extra":
                continue
            value = getattr(self, meta.name)
            if value is not None:
                payload[meta.name] = value
        payload.update(self.extra or {})
        return payload

    def to_json(self) -> str:
        """JSON string, the wire format of the ``options`` form field."""
        return json.dumps(self.to_dict())


@dataclass
class IndexingOptions:
    """Options of ``upload_and_index_files`` / ``add_and_index_texts``.

    The five cleaning options are read with ``indexing_options["..."]``, so
    :meth:`to_dict` always emits them.  The two chunking keys belong to the
    collection (see :meth:`SSEClient.create_collection`): left out here, the
    settings of the collection are used, and only :meth:`to_dict` emits a key
    once it has been set -- the same rule as for
    ``delete_files_after_indexing``, which an older engine may not know.
    """

    prepare_proper_pages: bool = True
    merge_document_pages: bool = False
    clear_text: bool = True
    use_text_denoiser: bool = False
    max_tokens_in_chunk: Optional[int] = None
    number_of_overlap_tokens: Optional[int] = None
    check_text_lang: bool = True
    #: Remove the file of a document from the disk of the engine once its
    #: embeddings are stored; ``failed`` documents keep their file.
    delete_files_after_indexing: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Option dictionary; unset chunk keys are omitted on purpose."""
        payload: Dict[str, Any] = {
            "prepare_proper_pages": self.prepare_proper_pages,
            "merge_document_pages": self.merge_document_pages,
            "clear_text": self.clear_text,
            "use_text_denoiser": self.use_text_denoiser,
            "check_text_lang": self.check_text_lang,
        }
        if self.max_tokens_in_chunk is not None:
            payload["max_tokens_in_chunk"] = self.max_tokens_in_chunk
        if self.number_of_overlap_tokens is not None:
            payload["number_of_overlap_tokens"] = self.number_of_overlap_tokens
        if self.delete_files_after_indexing is not None:
            payload["delete_files_after_indexing"] = self.delete_files_after_indexing
        return payload

    def to_json(self) -> str:
        """JSON string, the wire format of the multipart form field."""
        return json.dumps(self.to_dict())


@dataclass
class GenerativeOptions:
    """Options of ``POST generative_answer`` (the ``query_options`` field)."""

    generative_model: Optional[str] = None
    percentage_rank_mass: Optional[float] = None
    answer_language: Optional[str] = None
    translate_answer: Optional[bool] = None
    #: Ask the model with the names of the source documents in the prompt;
    #: ``core.validation.generation_options`` defaults it to ``False``.
    use_doc_names_in_response: Optional[bool] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the payload understood by the API (``None`` dropped)."""
        payload: Dict[str, Any] = {}
        for meta in fields(self):
            if meta.name == "extra":
                continue
            value = getattr(self, meta.name)
            if value is not None:
                payload[meta.name] = value
        payload.update(self.extra or {})
        return payload

    def to_json(self) -> str:
        """JSON string, the wire format of the ``query_options`` field."""
        return json.dumps(self.to_dict())


@dataclass
class ChatOptions:
    """Switches of the chat endpoints, sent in the same ``options`` object.

    ``add_user_message`` and ``new_chat`` read one ``options`` dictionary for
    both the generation settings and the four switches below, which is why
    :meth:`sse_lib.SSEClient.send_chat_message` merges this class into the
    generation options instead of sending a second field.
    """

    query_instruction: Optional[str] = None
    use_intelligent_query_rewrite: Optional[bool] = None
    use_rag_supervisor: Optional[bool] = None
    use_content_supervisor: Optional[bool] = None
    #: Keys of this dict are merged into the payload; escape hatch for switches
    #: that this release of the library does not know about yet.
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the payload understood by the API (``None`` dropped)."""
        payload: Dict[str, Any] = {}
        for meta in fields(self):
            if meta.name == "extra":
                continue
            value = getattr(self, meta.name)
            if value is not None:
                payload[meta.name] = value
        payload.update(self.extra or {})
        return payload

    def to_json(self) -> str:
        """JSON string, for callers which prefer to serialise themselves."""
        return json.dumps(self.to_dict())


OptionsLike = Union[
    SearchOptions,
    IndexingOptions,
    GenerativeOptions,
    ChatOptions,
    Dict[str, Any],
]


def _as_dict(options_cls: Any, options: Optional[OptionsLike]) -> Dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, options_cls):
        data = options.to_dict()
        existing_extra = getattr(options, "extra", None)
        if existing_extra:
            data["extra"] = dict(existing_extra)
        return data
    if isinstance(options, dict):
        return dict(options)
    raise SSEValueError(
        f"{options!r} is neither a {options_cls.__name__} nor a dictionary"
    )


def build_options(
    options_cls: Any,
    options: Optional[OptionsLike] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Normalise ``options`` + ``kwargs`` into an instance of ``options_cls``.

    Keyword arguments win over the entries of a dict/instance passed as
    ``options``.  Names that the dataclass does not declare end up in ``extra``
    (and are still sent), so features added to the backend later stay usable;
    for option classes without ``extra`` an unknown name is an error instead.
    """
    data = _as_dict(options_cls, options)
    known = {meta.name for meta in fields(options_cls)}
    accepts_extra = "extra" in known

    values: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}

    for source in (data, kwargs or {}):
        if accepts_extra:
            extra.update(source.get("extra") or {})
        for key, value in source.items():
            if key == "extra" and accepts_extra:
                continue
            if key in known:
                values[key] = value
                extra.pop(key, None)
            elif accepts_extra:
                extra[key] = value
            else:
                raise SSEValueError(
                    f"{options_cls.__name__} has no '{key}' option; known options "
                    f"are {sorted(known)}"
                )

    if accepts_extra and extra:
        values["extra"] = extra
    return options_cls(**values)


def text_document(
    text: str,
    *,
    name: Optional[str] = None,
    category: str = "",
    relative_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    pages: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build a single element of the ``texts[]`` payload.

    ``add_and_index_texts`` expects a list of document dictionaries::

        {"filepath": ..., "relative_filepath": ..., "category": ...,
         "pages": [{"page_content": ...}, ...], "options": {...}}

    A plain string handed to :meth:`sse_lib.SSEClient.index_documents` is
    converted with this helper.
    """
    if pages is None:
        page_list: List[Any] = [{"page_content": text}]
    else:
        page_list = [
            {"page_content": page} if isinstance(page, str) else page
            for page in pages
        ]
    document_name = name or "text-document"
    document: Dict[str, Any] = {
        "filepath": document_name,
        # The server validates relative_filepath with required_text, so an
        # omitted path falls back to the document name instead of sending "",
        # which the server rejected outright.
        "relative_filepath": relative_path or document_name,
        "category": category,
        "pages": page_list,
    }
    if options:
        document["options"] = dict(options)
    return document


# Fields the backend insists on, per validated endpoint.  ``engine`` runs
# ``query_options`` through generation_options(), which rejects ``None``, so a
# client that leaves them unset used to get an opaque INVALID_PARAMS back.
# The chat endpoints are deliberately *not* covered: they validate ``options``
# with the permissive ``options_object`` and accept anything, so requiring
# these fields when sending a chat message would reject valid requests.
GENERATIVE_REQUIRED_FIELDS = ("generative_model", "percentage_rank_mass")


def _require(options: Any, required: Sequence[str], endpoint: str) -> None:
    missing = [name for name in required if getattr(options, name, None) is None]
    if missing:
        listed = " and ".join(f"'{name}'" for name in missing)
        raise SSEValueError(
            f"{endpoint} requires {listed}; the backend rejects them as null. "
            "Pass them explicitly (or via default_indexing_options on the client)."
        )


def _reject_out_of_range(value: Any, name: str, endpoint: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSEValueError(f"{endpoint} requires {name} to be a number")


def validate_indexing(options: "IndexingOptions", *, endpoint: str) -> None:
    """Check what ``data``'s ``indexing_options`` validator demands.

    Mirrors ``core/validation.py`` so the caller gets the message before a
    round trip.  Chunk size and overlap are optional - they belong to the
    collection - but when they are given the chunk size has to be a positive
    identifier and the overlap has to satisfy ``0 <= overlap < chunk``.
    """
    chunk = options.max_tokens_in_chunk
    overlap = options.number_of_overlap_tokens
    if chunk is not None:
        _reject_out_of_range(chunk, "max_tokens_in_chunk", endpoint)
        if chunk <= 0:
            raise SSEValueError(
                f"{endpoint} requires max_tokens_in_chunk > 0, got {chunk!r}"
            )
    if overlap is not None:
        _reject_out_of_range(overlap, "number_of_overlap_tokens", endpoint)
        if overlap < 0:
            raise SSEValueError(
                f"{endpoint} requires number_of_overlap_tokens >= 0, got {overlap!r}"
            )
        if chunk is not None and not overlap < chunk:
            raise SSEValueError(
                f"{endpoint} requires 0 <= number_of_overlap_tokens < "
                f"max_tokens_in_chunk, got {overlap!r} for {chunk!r}"
            )


def chunking_payload(
    chunking: Any, *, endpoint: str = "new_collection"
) -> Dict[str, int]:
    """
    Normalise the chunking of a collection into the payload of the API.

    Accepts ``{"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64}``,
    an :class:`IndexingOptions` carrying the same two keys, or the short
    ``{"max_tokens_in_chunk": 500}`` form (no overlap).

    Raises
    ------
    SSEValueError
        When the chunk size is missing or the two values do not describe a
        usable split.  The size of a chunk itself is not checked here; the
        server refuses a chunk which does not fit the embedder of the
        collection with ``CHUNKING_TOO_LONG``.
    """
    data = _as_dict(IndexingOptions, chunking)
    max_tokens = data.get("max_tokens_in_chunk")
    overlap = data.get("number_of_overlap_tokens", 0)
    if max_tokens is None:
        raise SSEValueError(
            f"{endpoint} chunking requires max_tokens_in_chunk; pass e.g. "
            '{"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64}'
        )
    options = IndexingOptions(
        max_tokens_in_chunk=max_tokens, number_of_overlap_tokens=overlap
    )
    validate_indexing(options, endpoint=endpoint)
    return {
        "max_tokens_in_chunk": options.max_tokens_in_chunk,
        "number_of_overlap_tokens": options.number_of_overlap_tokens,
    }


def validate_generative(options: "GenerativeOptions", *, endpoint: str) -> None:
    """Check what ``engine``'s ``generation_options`` validator demands.

    The model name and ``percentage_rank_mass`` are both required, the mass is
    a percentage, and ``answer_language`` becomes mandatory as soon as
    ``translate_answer`` is on.
    """
    _require(options, GENERATIVE_REQUIRED_FIELDS, endpoint)
    mass = options.percentage_rank_mass
    _reject_out_of_range(mass, "percentage_rank_mass", endpoint)
    if not 0 <= mass <= 100:
        raise SSEValueError(
            f"{endpoint} requires percentage_rank_mass between 0 and 100, "
            f"got {mass!r}"
        )
    if getattr(options, "translate_answer", False) and not getattr(
        options, "answer_language", None
    ):
        raise SSEValueError(
            f"{endpoint} requires answer_language when translate_answer is set"
        )


__all__ = [
    "DEFAULT_INDEX_TYPE",
    "INDEX_TYPES",
    "ChatOptions",
    "GenerativeOptions",
    "IndexingOptions",
    "SearchOptions",
    "build_options",
    "chunking_payload",
    "text_document",
    "validate_generative",
    "validate_indexing",
]
