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
from typing import Any, Dict, List, Optional, Union

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

    The server reads all seven keys with ``indexing_options["..."]``, so
    :meth:`to_dict` always emits the complete set.
    """

    prepare_proper_pages: bool = True
    merge_document_pages: bool = False
    clear_text: bool = True
    use_text_denoiser: bool = False
    max_tokens_in_chunk: Optional[int] = None
    number_of_overlap_tokens: Optional[int] = None
    check_text_lang: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Full option dictionary, including the keys the server requires."""
        return {
            "prepare_proper_pages": self.prepare_proper_pages,
            "merge_document_pages": self.merge_document_pages,
            "clear_text": self.clear_text,
            "use_text_denoiser": self.use_text_denoiser,
            "max_tokens_in_chunk": self.max_tokens_in_chunk,
            "number_of_overlap_tokens": self.number_of_overlap_tokens,
            "check_text_lang": self.check_text_lang,
        }

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


OptionsLike = Union[
    SearchOptions, IndexingOptions, GenerativeOptions, Dict[str, Any]
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


__all__ = [
    "DEFAULT_INDEX_TYPE",
    "INDEX_TYPES",
    "GenerativeOptions",
    "IndexingOptions",
    "SearchOptions",
    "build_options",
    "text_document",
]
