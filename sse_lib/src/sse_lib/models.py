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


@dataclass
class Collection:
    """A ``CollectionOfDocuments`` as returned by ``collections``."""

    id: Optional[int] = None
    name: str = ""
    display_name: str = ""
    description: str = ""
    created_on: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Any, **known: Any) -> "Collection":
        """Build a collection from a serialized object, plus known local values."""
        payload = _as_dict(data)
        merged = {**payload, **{key: value for key, value in known.items() if value}}
        return cls(
            id=merged.get("id"),
            name=merged.get("name", ""),
            display_name=merged.get("display_name", ""),
            description=merged.get("description", ""),
            created_on=merged.get("created_on"),
            raw=payload,
        )


@dataclass
class SearchHit:
    """A single chunk returned inside ``results.detailed_results``."""

    score: Optional[float] = None
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
