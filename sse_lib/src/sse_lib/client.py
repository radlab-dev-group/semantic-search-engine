"""
client.py
---------

:class:`SSEClient` - a thin, dependency-light wrapper over the SSE REST API.

``SSEClient`` knows nothing about Milvus, Django or the models: it turns Python
arguments into the HTTP calls of ``sse_api`` and the JSON answers into
small dataclasses.  The controllers of the backend stay untouched; this is just
the client side of the same contract (including its quirks, e.g. ``options``
sent as a JSON string by ``search_with_options``).
"""

from __future__ import annotations

import mimetypes
import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from sse_lib import endpoints
from sse_lib.exceptions import SSEAPIError, SSEConfigError, SSEValueError
from sse_lib.models import (
    Answer,
    Chat,
    ChatHistory,
    ChatReply,
    Collection,
    IndexingResult,
    SearchResponse,
    UploadResult,
)
from sse_lib.options import (
    DEFAULT_INDEX_TYPE,
    INDEX_TYPES,
    GenerativeOptions,
    IndexingOptions,
    OptionsLike,
    SearchOptions,
    build_options,
    text_document,
)
from sse_lib.transport import DEFAULT_TIMEOUT, Transport

ENV_API_HOST = "SSE_API_HOST"
ENV_API_TOKEN = "SSE_API_TOKEN"
ENV_API_PREFIX = "SSE_API_PREFIX"
ENV_API_LANGUAGE = "SSE_API_LANGUAGE"
ENV_API_TIMEOUT = "SSE_API_TIMEOUT"

#: ``sse_api.core.constants.AVAILABLE_LANGUAGES``.
AVAILABLE_LANGUAGES = ("pl", "en")
DEFAULT_LANGUAGE = "pl"

#: Where the server expects uploaded files to be named.
UPLOAD_FILE_FIELD = "files[]"
#: Where the server expects the input texts of ``add_and_index_texts``.
TEXTS_FIELD = "texts[]"

FileLike = Union[str, os.PathLike, Tuple[str, Any], Tuple[str, Any, str], Any]
CollectionLike = Union[str, Collection, Mapping[str, Any]]


class SSEClient:
    """Client of the Semantic Search Engine REST API.

    ``api_host`` may be given with or without a scheme (``localhost:8271`` is
    treated as ``http``, ``search.example.com`` as ``https``).  Missing
    arguments fall back to ``SSE_API_HOST``, ``SSE_API_PREFIX``,
    ``SSE_API_TOKEN``, ``SSE_API_LANGUAGE`` and ``SSE_API_TIMEOUT``::

        from sse_lib import SSEClient, SearchOptions

        client = SSEClient("http://localhost:8271", token="...")
        response = client.search("my_docs", "policy on data retention",
                                 SearchOptions(max_results=20, rerank_results=True))
        answer = client.generate_answer(
            response.query_response_id,
            GenerativeOptions(generative_model="radlab/pLLama-3-8B-DPO-L"),
        )

    With ``username``/``password`` the client logs in on the first call and
    keeps the returned token; pass ``token`` to reuse one across processes.
    """

    def __init__(
        self,
        api_host: Optional[str] = None,
        *,
        api_prefix: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token_type: Optional[str] = None,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        verify: bool = True,
        session: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        max_retries: int = 2,
        default_search_options: Optional[OptionsLike] = None,
        default_indexing_options: Optional[OptionsLike] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        host = _first_set(api_host, os.environ.get(ENV_API_HOST))
        if transport is None and not host:
            raise SSEConfigError(f"api_host is required (or set {ENV_API_HOST})")
        prefix = api_prefix
        if prefix is None:
            prefix = os.environ.get(ENV_API_PREFIX, endpoints.DEFAULT_API_PREFIX)
        language = (
            _first_set(language, os.environ.get(ENV_API_LANGUAGE))
            or DEFAULT_LANGUAGE
        )
        if language not in AVAILABLE_LANGUAGES:
            raise SSEValueError(
                f"language must be one of {list(AVAILABLE_LANGUAGES)}, got {language!r}"
            )
        timeout = _as_timeout(timeout)

        self.api_prefix = prefix
        self.language = language
        self.username = username
        self.password = password
        self.transport = transport or Transport(
            host or "",
            api_prefix=prefix,
            timeout=timeout,
            verify=verify,
            session=session,
            headers=headers,
            max_retries=max_retries,
        )
        resolved_token = _first_set(token, os.environ.get(ENV_API_TOKEN))
        if resolved_token:
            self.transport.set_token(resolved_token, token_type)
        self._token_type = token_type
        self._refresh_token: Optional[str] = None
        self.default_search_options = default_search_options
        self.default_indexing_options = default_indexing_options

    # --- plumbing ------------------------------------------------------------
    @property
    def base_url(self) -> str:
        """Absolute root of the API, e.g. ``http://localhost:8271/api``."""
        return self.transport.base_url

    @property
    def token(self) -> Optional[str]:
        """Token used in the ``Authorization`` header, if any."""
        return self.transport.token

    def set_token(
        self, token: Optional[str], token_type: Optional[str] = None
    ) -> None:
        """Replace the token used for the subsequent calls."""
        self.transport.set_token(token, token_type or self._token_type)

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self.transport.close()

    def __enter__(self) -> "SSEClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SSEClient(base_url={self.base_url!r}, language={self.language!r})"

    # --- service ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """Dependency status of the server (``GET healthz``).

        Public endpoint, so this never triggers a login and works before any
        credentials are set.  Raises when the server cannot be reached, or
        answers 503 because a dependency is down (the status code is what
        probes act on); use :meth:`is_healthy` for a boolean instead of an
        exception.
        """
        body = self.transport.request("GET", endpoints.HEALTHZ)
        return body if isinstance(body, dict) else {"healthy": False}

    def is_healthy(self) -> bool:
        """Whether the server answers and every dependency is up.

        Returns ``False`` instead of raising, which is what a readiness check
        or a test ``setUpClass`` usually wants.
        """
        try:
            return bool(self.health().get("healthy"))
        except Exception:  # pylint: disable=broad-except
            return False

    def request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        """Escape hatch: call any endpoint and get the unwrapped ``body`` back.

        ``lang`` is added and the login is performed lazily, exactly as in the
        higher-level methods.
        """
        self._ensure_authenticated()
        return self.transport.request(method, endpoint, **kwargs)

    def get(
        self, endpoint: str, params: Optional[Mapping[str, Any]] = None, **kwargs
    ):
        """``GET`` on an endpoint, with ``lang`` added for you."""
        return self.request(
            "GET", endpoint, params=self._with_language(params), **kwargs
        )

    def post(
        self,
        endpoint: str,
        json_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """``POST`` on an endpoint, with ``lang`` added for you."""
        body = dict(json_body or {})
        body.update({"lang": self.language})
        return self.request("POST", endpoint, json_body=body, **kwargs)

    # --- authentication ------------------------------------------------------
    def login(
        self, username: Optional[str] = None, password: Optional[str] = None
    ) -> str:
        """Authenticate with ``POST login`` and remember the returned token."""
        user = username or self.username
        secret = password or self.password
        if not user or not secret:
            raise SSEConfigError(
                "login() needs a username and a password (or pass token= to SSEClient)"
            )
        body = self.transport.post(
            endpoints.LOGIN,
            {"username": user, "password": secret},
            params={"lang": self.language},
        )
        token = body.get("token") if isinstance(body, dict) else None
        if not token:
            raise SSEAPIError(
                [{"message": "Login response does not contain a token"}],
                endpoint=endpoints.LOGIN,
            )
        self.set_token(str(token))
        return str(token)

    def login_url(self) -> str:
        """Keycloak/OAuth: URL that the user has to open (``POST login_url``)."""
        body = self.post(endpoints.LOGIN_URL, {})
        url = body.get("login_url") if isinstance(body, dict) else None
        if not url:
            raise SSEAPIError(
                [{"message": "login_url response has no 'login_url'"}],
                endpoint=endpoints.LOGIN_URL,
            )
        return str(url)

    def login_with_code(
        self, code: str, state: str, session_state: Optional[str] = None
    ) -> str:
        """Keycloak/OAuth: exchange an authorization ``code`` for a token."""
        payload: Dict[str, Any] = {"code": code, "state": state}
        if session_state:
            payload["session_state"] = session_state
        body = self.transport.post(endpoints.LOGIN, payload)
        self._store_oauth_tokens(body)
        return str(self.token)

    def refresh_access_token(self, refresh_token: Optional[str] = None) -> str:
        """Keycloak/OAuth: refresh the access token (``POST refresh_token``)."""
        token = refresh_token or self._refresh_token
        if not token:
            raise SSEConfigError(
                "No refresh token available; pass one or log in first"
            )
        body = self.transport.post(endpoints.REFRESH_TOKEN, {"refresh_token": token})
        self._store_oauth_tokens(body)
        return str(self.token)

    def logout(self) -> Any:
        """Keycloak/OAuth: invalidate the token server side (``POST logout``)."""
        result = self.post(endpoints.LOGOUT, {})
        self.set_token(None)
        return result

    # --- collections ---------------------------------------------------------
    def create_collection(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: str = "",
        *,
        embedder: str,
        reranker: str,
        index_type: str = DEFAULT_INDEX_TYPE,
        group_name: Optional[str] = None,
    ) -> Collection:
        """Create a collection (``POST new_collection``).

        ``embedder`` and ``reranker`` must be names served by the backend, see
        :meth:`list_embedders` / :meth:`list_rerankers`; ``index_type`` is one
        of :data:`sse_lib.options.INDEX_TYPES`.  Spaces in ``name`` are
        replaced by underscores by the server, so the returned collection may
        carry a slightly different name.
        """
        name = _require_text(name, "name")
        if not description or not description.strip():
            raise SSEValueError(
                "description is required by the API (it rejects empty strings)"
            )
        if index_type not in INDEX_TYPES:
            raise SSEValueError(
                f"index_type must be one of {list(INDEX_TYPES)}, got {index_type!r}"
            )
        payload: Dict[str, Any] = {
            "collection_name": name,
            "collection_display_name": display_name or name,
            "collection_description": description,
            "model_embedder": _require_text(embedder, "embedder"),
            "model_reranker": _require_text(reranker, "reranker"),
            "embedder_index_type": index_type,
        }
        if group_name:
            payload["group_name"] = group_name

        body = self.post(endpoints.NEW_COLLECTION, payload)
        created = body if isinstance(body, dict) else {}
        return Collection.from_api(
            None,
            id=created.get("collection_id"),
            name=name.replace(" ", "_"),
            display_name=payload["collection_display_name"],
            description=payload["collection_description"],
        )

    def list_collections(self) -> List[Collection]:
        """Collections visible to the authenticated user (``GET collections``)."""
        body = self.get(endpoints.COLLECTIONS)
        collections = body.get("collections") if isinstance(body, dict) else []
        return [Collection.from_api(entry) for entry in collections or []]

    def get_collection(self, name: str) -> Optional[Collection]:
        """One collection by name, or ``None`` when it does not exist."""
        wanted = _require_text(name, "name")
        for collection in self.list_collections():
            if collection.name == wanted:
                return collection
        return None

    def collection_exists(self, name: str) -> bool:
        """Whether a collection with this name is visible to the user."""
        return self.get_collection(name) is not None

    def list_categories(self, collection: CollectionLike) -> List[str]:
        """Distinct categories of a collection (``GET categories``)."""
        body = self.get(endpoints.CATEGORIES, {"collection_name": _name(collection)})
        categories = body.get("categories") if isinstance(body, dict) else []
        return list(categories or [])

    def list_documents(self, collection: CollectionLike) -> List[str]:
        """Document names of a collection (``GET documents``)."""
        body = self.get(endpoints.DOCUMENTS, {"collection_name": _name(collection)})
        documents = body.get("documents") if isinstance(body, dict) else []
        return [
            entry.get("name") if isinstance(entry, dict) else entry
            for entry in documents or []
        ]

    def list_question_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Query templates grouped by their template collection (``GET``)."""
        body = self.get(endpoints.QUESTION_TEMPLATES)
        templates = body.get("templates") if isinstance(body, dict) else {}
        return dict(templates or {})

    def list_filter_options(self) -> Dict[str, Any]:
        """Metadata used to build filter drop-downs (``GET filter_options``)."""
        body = self.get(endpoints.FILTER_OPTIONS)
        return dict(body or {})

    # --- indexing ------------------------------------------------------------
    def upload_files(
        self,
        collection: CollectionLike,
        files: Union[FileLike, Sequence[FileLike]],
        *,
        indexing_options: Optional[OptionsLike] = None,
        **indexing_kwargs: Any,
    ) -> UploadResult:
        """Upload and index files (``POST upload_and_index_files``).

        ``files`` accepts paths, file objects or ``(name, content)`` /
        ``(name, content, content_type)`` tuples; ZIP archives are extracted by
        the server.  Chunking/cleaning is configured with
        :class:`sse_lib.options.IndexingOptions` (or plain keyword arguments).
        """
        options = self._build_options(
            IndexingOptions,
            self.default_indexing_options,
            indexing_options,
            indexing_kwargs,
        )
        prepared, opened = _prepare_files(files)
        try:
            body = self.request(
                "POST",
                endpoints.UPLOAD_AND_INDEX_FILES,
                data={
                    "collection_name": _name(collection),
                    "indexing_options": options.to_json(),
                    "lang": self.language,
                },
                files=prepared,
            )
        finally:
            for handle in opened:
                handle.close()
        return UploadResult.from_api(body)

    def index_documents(
        self,
        collection: CollectionLike,
        documents: Union[
            str, Mapping[str, Any], Sequence[Union[str, Mapping[str, Any]]]
        ],
        *,
        indexing_options: Optional[OptionsLike] = None,
        name: Optional[str] = None,
        category: str = "",
        options: Optional[Dict[str, Any]] = None,
        **indexing_kwargs: Any,
    ) -> IndexingResult:
        """Index raw texts (``POST add_and_index_texts``).

        Every element is either a plain string - wrapped by
        :func:`sse_lib.options.text_document` with the given ``name``,
        ``category`` and metadata ``options`` - or an already prepared document
        dictionary (``filepath``, ``relative_filepath``, ``category``,
        ``pages``).
        """
        resolved = self._build_options(
            IndexingOptions,
            self.default_indexing_options,
            indexing_options,
            indexing_kwargs,
        )
        payload = {
            "collection_name": _name(collection),
            TEXTS_FIELD: [
                (
                    text_document(
                        entry,
                        name=name,
                        category=category,
                        options=options,
                    )
                    if isinstance(entry, str)
                    else dict(entry)
                )
                for entry in _as_sequence(documents)
            ],
            "indexing_options": resolved.to_dict(),
        }
        body = self.post(endpoints.ADD_AND_INDEX_TEXTS, payload)
        return IndexingResult.from_api(body)

    # --- search & generation -------------------------------------------------
    def search(
        self,
        collection: CollectionLike,
        query: str,
        options: Optional[OptionsLike] = None,
        *,
        ignore_question_lang_detect: bool = False,
        **search_kwargs: Any,
    ) -> SearchResponse:
        """Run a (hybrid) search (``POST search_with_options``).

        ``options`` is a :class:`sse_lib.options.SearchOptions` or a dict; any
        single option can also be passed as a keyword argument, e.g.
        ``client.search("docs", "query", max_results=20, hybrid_search=True)``.
        The returned :class:`sse_lib.models.SearchResponse` carries the
        ``query_response_id`` needed by :meth:`generate_answer`.
        """
        if not query or not str(query).strip():
            raise SSEValueError("query cannot be empty")
        search_options = self._build_options(
            SearchOptions, self.default_search_options, options, search_kwargs
        )
        payload: Dict[str, Any] = {
            "collection_name": _name(collection),
            "query_str": str(query),
            # The view calls json.loads() on this field - it must be a string.
            "options": search_options.to_json(),
        }
        if ignore_question_lang_detect:
            payload["ignore_question_lang_detect"] = True

        body = self.post(endpoints.SEARCH_WITH_OPTIONS, payload)
        return SearchResponse.from_api(body)

    def generate_answer(
        self,
        query_response_id: int,
        options: Optional[OptionsLike] = None,
        *,
        query_instruction: str = "",
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> Answer:
        """Generate a RAG answer for a stored search (``POST generative_answer``).

        ``query_response_id`` comes from :meth:`search` (or from a chat
        message state).  ``options`` is a
        :class:`sse_lib.options.GenerativeOptions` or a dict.
        """
        if query_response_id is None:
            raise SSEValueError("query_response_id is required")
        generation = self._build_options(
            GenerativeOptions, None, options, generation_kwargs
        )
        payload: Dict[str, Any] = {
            "query_response_id": query_response_id,
            "query_options": generation.to_json(),
        }
        if query_instruction:
            payload["query_instruction"] = query_instruction
        if system_prompt:
            payload["system_prompt"] = system_prompt

        body = self.post(endpoints.GENERATIVE_ANSWER, payload)
        return Answer.from_api(body)

    def rate_answer(
        self,
        answer_response_id: int,
        rate_value: int,
        rate_value_max: int,
        comment: Optional[str] = None,
    ) -> None:
        """Rate a generated answer (``POST rate_generative_answer``)."""
        payload: Dict[str, Any] = {
            "answer_response_id": answer_response_id,
            "rate_value": rate_value,
            "rate_value_max": rate_value_max,
        }
        if comment is not None:
            payload["rate_comment"] = comment
        self.post(endpoints.RATE_GENERATIVE_ANSWER, payload)

    def list_generative_models(self) -> Dict[str, Any]:
        """Active generative models and their hosts (``GET generative_models``)."""
        body = self.get(endpoints.GENERATIVE_MODELS)
        return dict(body or {})

    def list_embedders(self) -> List[str]:
        """Names of the embedders configured on the server (``GET embedders``)."""
        body = self.get(endpoints.EMBEDDERS)
        models = body.get("models") if isinstance(body, dict) else []
        return list(models or [])

    def list_rerankers(self) -> List[str]:
        """Names of the rerankers configured on the server (``GET rerankers``)."""
        body = self.get(endpoints.RERANKERS)
        models = body.get("models") if isinstance(body, dict) else []
        return list(models or [])

    # --- chat ----------------------------------------------------------------
    def new_chat(
        self,
        collection: Optional[CollectionLike] = None,
        *,
        options: Optional[Dict[str, Any]] = None,
        search_options: Optional[OptionsLike] = None,
    ) -> Chat:
        """Open a new chat (``POST new_chat``).

        ``options`` configures generation, ``search_options`` the retrieval
        used to answer; both are sent as JSON objects (not strings here).
        """
        payload: Dict[str, Any] = {}
        if collection is not None:
            payload["collection_name"] = _name(collection)
        if options:
            payload["options"] = dict(options)
        if search_options:
            payload["search_options"] = _search_options_dict(
                search_options, self.default_search_options
            )
        body = self.post(endpoints.NEW_CHAT, payload)
        return Chat.from_api(body)

    def send_chat_message(
        self,
        chat_id: int,
        message: str,
        options: Optional[OptionsLike] = None,
        *,
        collection: Optional[CollectionLike] = None,
        search_options: Optional[OptionsLike] = None,
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> ChatReply:
        """Add a user message and get the assistant reply (``add_user_message``).

        ``options`` holds the generation options; single values may be passed
        as keyword arguments (``GenerativeOptions`` semantics).
        ``collection`` is required by the current server, even for a chat
        created with a collection.
        """
        if chat_id is None:
            raise SSEValueError("chat_id is required")
        if not message or not str(message).strip():
            raise SSEValueError("message cannot be empty")
        if collection is None:
            raise SSEValueError("collection is required to send a chat message")
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "user_message": str(message),
            "options": self._build_options(
                GenerativeOptions, None, options, generation_kwargs
            ).to_dict(),
        }
        if collection is not None:
            payload["collection_name"] = _name(collection)
        if search_options:
            payload["search_options"] = _search_options_dict(
                search_options, self.default_search_options
            )
        if system_prompt:
            payload["system_prompt"] = system_prompt

        body = self.post(endpoints.ADD_USER_MESSAGE, payload)
        return ChatReply.from_api(body)

    def save_chat(self, chat_id: int, read_only: bool = True) -> str:
        """Mark a chat as saved (``POST save_chat``) and return its hash."""
        body = self.post(
            endpoints.SAVE_CHAT, {"chat_id": chat_id, "read_only": bool(read_only)}
        )
        chat_hash = body.get("chat_hash") if isinstance(body, dict) else None
        return str(chat_hash) if chat_hash is not None else ""

    def get_chat_by_hash(self, chat_hash: str) -> ChatHistory:
        """Fetch a saved chat (``GET get_chat_by_hash``)."""
        body = self.get(
            endpoints.GET_CHAT_BY_HASH,
            {"chat_hash": _require_text(chat_hash, "chat_hash")},
        )
        return ChatHistory.from_api(body)

    def list_chats(self) -> List[ChatHistory]:
        """All chats of the authenticated user (``GET chats``)."""
        body = self.get(endpoints.CHATS)
        history = body.get("history") if isinstance(body, dict) else []
        return [ChatHistory.from_api(entry) for entry in history or []]

    # --- internals -----------------------------------------------------------
    def _ensure_authenticated(self) -> None:
        if self.transport.token is None and self.username and self.password:
            self.login()

    def _with_language(self, params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = dict(params or {})
        merged.setdefault("lang", self.language)
        return merged

    def _build_options(
        self,
        options_cls: Any,
        client_defaults: Optional[OptionsLike],
        options: Optional[OptionsLike],
        kwargs: Optional[Dict[str, Any]],
    ) -> Any:
        merged: Dict[str, Any] = {}
        if client_defaults is not None:
            merged.update(
                build_options(options_cls, client_defaults, None).to_dict()
            )
        if options is not None:
            if isinstance(options, dict):
                merged.update(options)
            else:
                merged.update(build_options(options_cls, options, None).to_dict())
        return build_options(options_cls, merged, kwargs or {})

    def _store_oauth_tokens(self, body: Any) -> None:
        payload = body if isinstance(body, dict) else {}
        token = payload.get("token")
        if not token:
            raise SSEAPIError(
                [{"message": "Authentication response does not contain a token"}]
            )
        self._refresh_token = payload.get("refresh_token")
        self.set_token(str(token))


def _search_options_dict(
    options: OptionsLike, defaults: Optional[OptionsLike]
) -> Dict[str, Any]:
    base = build_options(SearchOptions, defaults, None).to_dict()
    base.update(build_options(SearchOptions, options, None).to_dict())
    return base


def _as_timeout(value: Any) -> float:
    if value is None:
        value = os.environ.get(ENV_API_TIMEOUT) or DEFAULT_TIMEOUT
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SSEConfigError(f"timeout is not a number: {value!r}") from exc


def _first_set(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SSEValueError(f"{name} cannot be empty")
    return text


def _name(collection: CollectionLike) -> str:
    if isinstance(collection, Collection):
        return collection.name
    if isinstance(collection, Mapping):
        return _require_text(collection.get("name"), "collection name")
    return _require_text(collection, "collection")


def _as_sequence(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _prepare_files(
    files: Union[FileLike, Sequence[FileLike]],
) -> Tuple[List[Tuple[str, Tuple[str, Any, str]]], List[Any]]:
    """Turn the ``files`` argument into requests' multipart tuples.

    Files opened by the client are returned as second element so the caller can
    close them after the request; file objects passed in are left untouched.
    """
    opened: List[Any] = []
    prepared: List[Tuple[str, Tuple[str, Any, str]]] = []
    try:
        for item in [files] if isinstance(files, tuple) else _as_sequence(files):
            prepared.append((UPLOAD_FILE_FIELD, _unpack_file(item, opened)))
        if not prepared:
            raise SSEValueError("upload_files() needs at least one file")
    except Exception:
        for handle in opened:
            handle.close()
        raise
    return prepared, opened


def _unpack_file(item: FileLike, opened: List[Any]) -> Tuple[str, Any, str]:
    if isinstance(item, (str, os.PathLike)):
        path = os.fspath(item)
        handle = open(path, "rb")
        opened.append(handle)
        name = os.path.basename(path)
        return name, handle, _guess_content_type(name)

    if isinstance(item, tuple):
        if len(item) == 2:
            name, content = item
            content_type = _guess_content_type(str(name))
        elif len(item) == 3:
            name, content, content_type = item
        else:
            raise SSEValueError(
                "A file tuple must be (name, content) or (name, content, type)"
            )
        return str(name), content, str(content_type)

    name = getattr(item, "name", None)
    name = os.path.basename(name) if isinstance(name, str) and name else "document"
    if not hasattr(item, "read"):
        raise SSEValueError(f"{item!r} is not a path, file object or a tuple")
    return name, item, _guess_content_type(name)


def _guess_content_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"
