"""
models.py
---------

Lightweight, read-only views over the JSON bodies returned by the API.

Every dataclass keeps the untouched server payload in ``raw``, so nothing is
lost when the backend adds a field that this release does not model yet.  All
accessors tolerate missing keys: the serializers of the backend
(``data.serializers``, ``chat.serializer``) do not guarantee a stable shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_count(value: Any) -> Optional[int]:
    """A document counter of the server; ``None`` when it was not sent."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Collection:
    """A ``CollectionOfDocuments`` as returned by ``collections``."""

    id: Optional[int] = None
    name: str = ""
    display_name: str = ""
    description: str = ""
    created_on: Optional[str] = None
    document_count: Optional[int] = None
    active_document_count: Optional[int] = None
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any, **known: Any) -> "Collection":
        """Build a collection from a serialized object, plus known local values."""
        payload = _as_dict(data)
        merged = {**payload, **{key: value for key, value in known.items() if value}}
        metadata = merged.get("metadata_json")
        return cls(
            id=merged.get("id"),
            name=merged.get("name", ""),
            display_name=merged.get("display_name", ""),
            description=merged.get("description", ""),
            created_on=merged.get("created_on"),
            document_count=_as_count(merged.get("document_count")),
            active_document_count=_as_count(merged.get("active_document_count")),
            metadata_json=metadata if isinstance(metadata, dict) else {},
            raw=payload,
        )

    @property
    def document_count_or_zero(self) -> int:
        """``document_count`` as a number; ``0`` when the server did not send it."""
        return self.document_count or 0

    @property
    def active_document_count_or_zero(self) -> int:
        """``active_document_count`` as a number; ``0`` when the server did not send it."""
        return self.active_document_count or 0

    @property
    def chunking(self) -> Dict[str, Any]:
        """Chunking declared for this collection, ``{}`` when it has none."""
        chunking = self.metadata_json.get("chunking")
        return chunking if isinstance(chunking, dict) else {}

    @property
    def strict_chunking(self) -> bool:
        """Whether the collection enforces its own chunking on indexing."""
        return bool(self.metadata_json.get("strict_chunking"))


@dataclass
class DocumentRef:
    """A document of a collection, as listed by ``documents?with_disabled``.

    ``id`` and ``use_in_search`` are served only by the ``with_disabled`` view
    of the endpoint (``DocumentVisibilitySerializer``); they are what
    ``delete_documents`` and ``set_documents_use_in_search`` accept in
    ``document_ids[]``.
    """

    id: Optional[int] = None
    name: str = ""
    use_in_search: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "DocumentRef":
        """Build a document reference from one entry of a serialized list."""
        payload = _as_dict(data)
        use_in_search = payload.get("use_in_search")
        return cls(
            id=payload.get("id"),
            name=payload.get("name", "") or "",
            use_in_search=None if use_in_search is None else bool(use_in_search),
            raw=payload,
        )


@dataclass
class DocumentPreviewPage:
    """One page of a document, as served by ``document_content``."""

    page: Optional[int] = None
    content: str = ""
    chunks: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "DocumentPreviewPage":
        payload = _as_dict(data)
        return cls(
            page=payload.get("page"),
            content=payload.get("content") or "",
            chunks=payload.get("chunks") or 0,
            raw=payload,
        )


@dataclass
class DocumentPreview:
    """Whole text of one document (``document_content``).

    ``content`` is every chunk of ``chunk_type`` joined in reading order, which
    is what a preview window shows; ``pages`` keeps the same text page by page
    so the caller can show page numbers.  A document longer than the requested
    window of pages answers ``truncated`` with the ``next_page`` to ask for.
    """

    document_id: Optional[int] = None
    document_name: str = ""
    relative_path: str = ""
    category: str = ""
    use_in_search: Optional[bool] = None
    chunk_type: str = ""
    chunk_types: List[str] = field(default_factory=list)
    pages: List[DocumentPreviewPage] = field(default_factory=list)
    content: str = ""
    characters: int = 0
    chunks: int = 0
    from_page: int = 1
    total_pages: int = 0
    truncated: bool = False
    next_page: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "DocumentPreview":
        payload = _as_dict(data)
        use_in_search = payload.get("use_in_search")
        return cls(
            document_id=payload.get("document_id"),
            document_name=payload.get("document_name") or "",
            relative_path=payload.get("relative_path") or "",
            category=payload.get("category") or "",
            use_in_search=None if use_in_search is None else bool(use_in_search),
            chunk_type=payload.get("chunk_type") or "",
            chunk_types=[
                entry for entry in _as_list(payload.get("chunk_types")) if entry
            ],
            pages=[
                DocumentPreviewPage.from_api(entry)
                for entry in _as_list(payload.get("pages"))
            ],
            content=payload.get("content") or "",
            characters=payload.get("characters") or 0,
            chunks=payload.get("chunks") or 0,
            from_page=payload.get("from_page") or 1,
            total_pages=payload.get("total_pages") or 0,
            truncated=bool(payload.get("truncated")),
            next_page=payload.get("next_page"),
            raw=payload,
        )


@dataclass
class SearchHit:
    """A single chunk returned inside ``results.detailed_results``."""

    score: Optional[float] = None
    score_normalized: Optional[float] = None
    similarity: Optional[float] = None
    document_name: Optional[str] = None
    relative_filepath: Optional[str] = None
    page_number: Optional[int] = None
    text_number: Optional[int] = None
    language: Optional[str] = None
    text: Optional[str] = None
    left_context: Optional[str] = None
    right_context: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "SearchHit":
        """Build a hit from one entry of ``detailed_results``."""
        payload = _as_dict(data)
        return cls(
            score=payload.get("score"),
            score_normalized=payload.get("score_normalized"),
            similarity=payload.get("similarity"),
            document_name=payload.get("document_name"),
            relative_filepath=payload.get("relative_filepath"),
            page_number=payload.get("page_number"),
            text_number=payload.get("text_number"),
            language=payload.get("language"),
            text=payload.get("text_str"),
            left_context=payload.get("left_context"),
            right_context=payload.get("right_context"),
            raw=payload,
        )


@dataclass
class SearchResponse:
    """Body of ``POST search_with_options``."""

    query: str = ""
    query_response_id: Optional[int] = None
    results: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    detailed_results: Any = field(default_factory=dict)
    structured_results: List[Any] = field(default_factory=list)
    template_prompts: List[Any] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "SearchResponse":
        """Build a response from the ``body`` of a search call."""
        payload = _as_dict(body)
        results = _as_dict(payload.get("results"))
        return cls(
            query=str(results.get("query", "")),
            query_response_id=payload.get("query_response_id"),
            results=results,
            stats=_as_dict(results.get("stats")),
            detailed_results=results.get("detailed_results", {}),
            structured_results=_as_list(results.get("structured_results")),
            template_prompts=_as_list(payload.get("template_prompts")),
            raw=payload,
        )

    @property
    def hits(self) -> List[SearchHit]:
        """Reformatted hits, ready to display (``detailed_results``)."""
        entries = self.detailed_results
        if isinstance(entries, dict):
            entries = list(entries.values())
        return [SearchHit.from_api(entry) for entry in _as_list(entries)]


@dataclass
class Answer:
    """Body of ``POST generative_answer``."""

    response_id: Optional[int] = None
    answer: str = ""
    answer_translated: Optional[str] = None
    generation_time: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "Answer":
        """Build an answer from the ``body`` of a generation call."""
        payload = _as_dict(body)
        return cls(
            response_id=payload.get("response_id"),
            answer=payload.get("answer", "") or "",
            answer_translated=payload.get("answer_translated"),
            generation_time=payload.get("generation_time"),
            raw=payload,
        )


@dataclass
class SearchHistoryAnswer:
    """An answer recorded for a historical search."""

    id: Optional[int] = None
    response_id: Optional[int] = None
    is_generative: bool = True
    answer_options: Dict[str, Any] = field(default_factory=dict)
    query_instruction_prompt: Optional[str] = None
    generated_answer: str = ""
    generated_answer_translated: Optional[str] = None
    rate_value: Optional[int] = None
    rate_value_max: Optional[int] = None
    rate_comment: Optional[str] = None
    generation_time: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "SearchHistoryAnswer":
        """Build an answer record from one entry of an answers list."""
        payload = _as_dict(data)
        return cls(
            id=payload.get("id"),
            response_id=payload.get("response_id"),
            is_generative=bool(payload.get("is_generative", True)),
            answer_options=_as_dict(payload.get("answer_options")),
            query_instruction_prompt=payload.get("query_instruction_prompt"),
            generated_answer=payload.get("generated_answer", "") or "",
            generated_answer_translated=payload.get("generated_answer_translated"),
            rate_value=payload.get("rate_value"),
            rate_value_max=payload.get("rate_value_max"),
            rate_comment=payload.get("rate_comment"),
            generation_time=payload.get("generation_time"),
            raw=payload,
        )


@dataclass
class SearchHistory:
    """A historical search entry with complete results and generated answer(s)."""

    query_id: Optional[int] = None
    query_response_id: Optional[int] = None
    query: str = ""
    created_on: Optional[str] = None
    collection_id: Optional[int] = None
    collection_name: Optional[str] = None
    query_options: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    detailed_results: Any = field(default_factory=dict)
    structured_results: List[Any] = field(default_factory=list)
    answers: List[SearchHistoryAnswer] = field(default_factory=list)
    generated_answer: Optional[str] = None
    generated_answer_translated: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "SearchHistory":
        """Build a search history entry from the ``body`` of a search history call."""
        payload = _as_dict(body)
        results = _as_dict(payload.get("results"))
        answers_data = _as_list(payload.get("answers"))
        answers = [SearchHistoryAnswer.from_api(a) for a in answers_data]

        stats = payload.get("stats") or results.get("stats") or {}
        detailed_results = (
            payload.get("detailed_results")
            if "detailed_results" in payload
            else results.get("detailed_results", {})
        )
        structured_results = (
            payload.get("structured_results")
            if "structured_results" in payload
            else results.get("structured_results", [])
        )

        gen_answer = payload.get("generated_answer")
        if gen_answer is None and answers:
            gen_answer = answers[-1].generated_answer

        gen_answer_trans = payload.get("generated_answer_translated")
        if gen_answer_trans is None and answers:
            gen_answer_trans = answers[-1].generated_answer_translated

        return cls(
            query_id=payload.get("query_id"),
            query_response_id=payload.get("query_response_id"),
            query=str(
                payload.get("query")
                or payload.get("query_str")
                or results.get("query", "")
            ),
            created_on=payload.get("created_on"),
            collection_id=payload.get("collection_id"),
            collection_name=payload.get("collection_name"),
            query_options=_as_dict(payload.get("query_options")),
            results=results,
            stats=_as_dict(stats),
            detailed_results=detailed_results,
            structured_results=_as_list(structured_results),
            answers=answers,
            generated_answer=gen_answer,
            generated_answer_translated=gen_answer_trans,
            raw=payload,
        )

    @property
    def hits(self) -> List[SearchHit]:
        """Reformatted hits, ready to display (``detailed_results``)."""
        entries = self.detailed_results
        if isinstance(entries, dict):
            entries = list(entries.values())
        return [SearchHit.from_api(entry) for entry in _as_list(entries)]

    @property
    def answer(self) -> Optional[str]:
        """Shortcut for generated_answer."""
        return self.generated_answer


@dataclass
class UploadResult:
    """Body of ``POST upload_and_index_files`` (``UploadedDocumentsSerializer``)."""

    id: Optional[int] = None
    dir_hash: Optional[str] = None
    dir_path: Optional[str] = None
    is_indexed: Optional[bool] = None
    number_of_uploaded_documents: Optional[int] = None
    number_of_indexed_documents_rel_db: Optional[int] = None
    number_of_indexed_documents_vec_db: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "UploadResult":
        """Build an upload summary from the ``body`` of an upload call."""
        payload = _as_dict(body)
        return cls(
            id=payload.get("id"),
            dir_hash=payload.get("dir_hash"),
            dir_path=payload.get("dir_path"),
            is_indexed=payload.get("is_indexed"),
            number_of_uploaded_documents=payload.get("number_of_uploaded_documents"),
            number_of_indexed_documents_rel_db=payload.get(
                "number_of_indexed_documents_rel_db"
            ),
            number_of_indexed_documents_vec_db=payload.get(
                "number_of_indexed_documents_vec_db"
            ),
            raw=payload,
        )


@dataclass
class IndexingResult:
    """Body of ``POST add_and_index_texts``."""

    indexed_documents: int = 0
    indexed_chunks: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "IndexingResult":
        """Build an indexing summary from the ``body`` of the call."""
        payload = _as_dict(body)
        return cls(
            indexed_documents=int(payload.get("indexed_documents") or 0),
            indexed_chunks=int(payload.get("indexed_chunks") or 0),
            raw=payload,
        )


@dataclass
class IndexingJobStatus:
    """One indexing job as reported by ``GET indexing_status``.

    ``None`` when the collection has no indexing request yet (the server then
    answers with the documents which are still waiting for their embeddings).
    """

    job_id: Optional[int] = None
    status: Optional[str] = None
    total_documents: int = 0
    documents_indexed: int = 0
    documents_failed: int = 0
    documents_pending: int = 0
    current_document: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "IndexingJobStatus":
        payload = _as_dict(data)
        if not payload:
            return cls()
        return cls(
            job_id=payload.get("id"),
            status=payload.get("status"),
            total_documents=payload.get("total_documents") or 0,
            documents_indexed=payload.get("documents_indexed") or 0,
            documents_failed=payload.get("documents_failed") or 0,
            documents_pending=payload.get("documents_pending") or 0,
            current_document=payload.get("current_document"),
            raw=payload,
        )


@dataclass
class IndexingStatus:
    """Progress of an indexing job or of a whole collection.

    The documents of the job are listed with their own ``indexing_status``, so
    a client can show which document is already indexed and how many are left;
    a collection without a job answers with the documents which are still
    waiting for their embeddings.
    """

    job: Optional[IndexingJobStatus] = None
    documents: List[Dict[str, Any]] = field(default_factory=list)
    pending_documents: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "IndexingStatus":
        payload = _as_dict(data)
        return cls(
            job=IndexingJobStatus.from_api(payload.get("job")),
            documents=[
                entry for entry in _as_list(payload.get("documents")) if entry
            ],
            pending_documents=payload.get("pending_documents") or 0,
            raw=payload,
        )


@dataclass
class Chat:
    """Body of ``POST new_chat`` (``ChatSerializer``)."""

    id: Optional[int] = None
    organisation_user: Optional[int] = None
    collection: Optional[int] = None
    created_at: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "Chat":
        """Build a chat from the ``body`` of the ``new_chat`` call."""
        payload = _as_dict(_as_dict(body).get("chat"))
        return cls(
            id=payload.get("id"),
            organisation_user=payload.get("organisation_user"),
            collection=payload.get("collection"),
            created_at=payload.get("created_at"),
            options=_as_dict(payload.get("options")),
            raw=payload,
        )


@dataclass
class ChatMessage:
    """A single chat message (``MessageSerializer``)."""

    id: Optional[int] = None
    chat: Optional[int] = None
    role: Optional[str] = None
    text: str = ""
    text_translated: Optional[str] = None
    number: Optional[int] = None
    date_time: Optional[str] = None
    generation_time: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any) -> "ChatMessage":
        """Build a message from one entry of a serialized history."""
        payload = _as_dict(data)
        return cls(
            id=payload.get("id"),
            chat=payload.get("chat"),
            role=payload.get("role"),
            text=payload.get("text", "") or "",
            text_translated=payload.get("text_translated"),
            number=payload.get("number"),
            date_time=payload.get("date_time"),
            generation_time=payload.get("generation_time"),
            raw=payload,
        )


@dataclass
class ChatReply:
    """Body of ``POST add_user_message``."""

    assistant_message: str = ""
    generation_time: Optional[float] = None
    history: List[ChatMessage] = field(default_factory=list)
    last_state: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "ChatReply":
        """Build a reply from the ``body`` of an ``add_user_message`` call."""
        payload = _as_dict(body)
        return cls(
            assistant_message=payload.get("generated_assistant_message", "") or "",
            generation_time=payload.get("generation_time"),
            history=[
                ChatMessage.from_api(entry)
                for entry in _as_list(payload.get("history"))
            ],
            last_state=payload.get("last_state"),
            raw=payload,
        )


@dataclass
class ChatHistory:
    """Body of ``get_chat_by_hash`` and one entry of ``GET chats``."""

    chat_id: Optional[int] = None
    is_read_only: Optional[bool] = None
    messages: List[ChatMessage] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, body: Any) -> "ChatHistory":
        """Build a history from the ``body`` of a chat retrieval call."""
        payload = _as_dict(body)
        return cls(
            chat_id=payload.get("chat_id"),
            is_read_only=payload.get("is_read_only"),
            messages=[
                ChatMessage.from_api(entry)
                for entry in _as_list(payload.get("chat_history"))
            ],
            raw=payload,
        )

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
