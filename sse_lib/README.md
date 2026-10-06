# sse_lib

Thin Python client for the **Semantic Search Engine** (SSE) REST API
(`sse_rest_api/` in this repository).

`sse_lib` is a client-side layer only: it maps Python calls onto the HTTP
endpoints of a running SSE backend and turns the JSON answers into small
dataclasses. It does **not** import Django, `torch`, `pymilvus` or any other
server dependency — the single runtime requirement is `requests`.

```python
from sse_lib import SSEClient, SearchOptions, GenerativeOptions

client = SSEClient("http://localhost:8271", token="...")

client.create_collection(
    "my_docs",
    "My Documents",
    "Test collection",
    embedder="radlab/polish-bi-encoder-mean",
    reranker="radlab/polish-cross-encoder",
    index_type="HNSW",
)
client.upload_files("my_docs", ["invoice.pdf", "regulations.zip"])
client.index_documents("my_docs", ["Some text to index."], category="Notes")

found = client.search("my_docs", "policy on data retention",
                      SearchOptions(max_results=20, rerank_results=True))
answer = client.generate_answer(
    found.query_response_id,
    GenerativeOptions(generative_model="radlab/pLLama-3-8B-DPO-L"),
)
print(answer.answer)
```

## Installation

From a checkout:

```bash
pip install ./sse_lib            # or: pip install -e ./sse_lib  (development)
```

Once published to a registry, `pip install sse-lib` is all that is needed — the
package is standalone and versioned independently (`sse_lib.__version__`).

Python 3.9+, `requests` as the only runtime dependency.

## Authentication

| Mode | How |
|------|-----|
| Pre-existing token (default DRF token auth) | `SSEClient(host, token="...")` — sent as `Authorization: Token <value>` |
| Login with user and password | `SSEClient(host, username="user", password="...")` — `POST login` runs lazily, before the first request |
| Keycloak / OAuth (backend started with `ENV_USE_KC_AUTH=1`) | `client.login_url()` → redirect the user → `client.login_with_code(code, state)`; later `client.refresh_access_token()` and `client.logout()` |
| Custom scheme | `SSEClient(host, token="...", token_type="Bearer")` |

## Configuration

| Constructor argument | Environment variable | Default | Meaning |
|----------------------|----------------------|---------|---------|
| `api_host` | `SSE_API_HOST` | – | `http://localhost:8271`; the scheme may be omitted (`localhost` → `http`, anything else → `https`) |
| `api_prefix` | `SSE_API_PREFIX` | `api` | Path between the host and the endpoints; set `api/v1` for versioned deployments (`main.api` in `django-config.json`) |
| `token` | `SSE_API_TOKEN` | – | Value of the `Authorization` header |
| `language` | `SSE_API_LANGUAGE` | `pl` | Sent as `lang` on every call (`pl` / `en`) |
| `timeout` | `SSE_API_TIMEOUT` | `60.0` | Per-request timeout in seconds |
| `verify`, `session`, `headers`, `max_retries` | – | `True`, new session, `{}`, `2` | TLS verification, shared `requests.Session`, extra headers, GET retries |
| `default_search_options`, `default_indexing_options` | – | – | Client-wide option defaults merged into every search / indexing call |

`SSEClient` is a context manager; leaving the block closes the HTTP session.

## API coverage

| Client method | REST endpoint |
|---------------|---------------|
| `create_collection(...)` | `POST new_collection` |
| `list_collections()`, `get_collection(name)`, `collection_exists(name)` | `GET collections` |
| `list_categories(collection)` | `GET categories` |
| `list_documents(collection)` | `GET documents` |
| `upload_files(collection, files, ...)` | `POST upload_and_index_files` |
| `index_documents(collection, documents, ...)` | `POST add_and_index_texts` |
| `list_question_templates()` | `GET question_templates` |
| `list_filter_options()` | `GET filter_options` |
| `search(collection, query, options, ...)` | `POST search_with_options` |
| `generate_answer(query_response_id, options, ...)` | `POST generative_answer` |
| `rate_answer(answer_response_id, value, value_max, comment=None)` | `POST rate_generative_answer` |
| `list_generative_models()`, `list_embedders()`, `list_rerankers()` | `GET generative_models`, `GET embedders`, `GET rerankers` |
| `new_chat(...)`, `send_chat_message(...)`, `save_chat(...)`, `get_chat_by_hash(...)`, `list_chats()` | `POST new_chat`, `POST add_user_message`, `POST save_chat`, `GET get_chat_by_hash`, `GET chats` |
| `login()`, `login_with_code()`, `refresh_access_token()`, `logout()`, `login_url()` | `POST login`, `POST refresh_token`, `POST logout`, `POST login_url` |
| `request(method, endpoint, ...)`, `get(...)`, `post(...)` | any endpoint, including ones this release does not wrap |

Collections may be passed as a name, a `Collection` returned by
`create_collection()` / `list_collections()`, or any mapping with a `name` key.

## Options objects

| Class | Used by | Notes |
|-------|---------|-------|
| `SearchOptions` | `search()` | Only the fields that are set are sent; `extra` merges unknown keys |
| `IndexingOptions` | `upload_files()`, `index_documents()` | Always serialises all seven keys the server reads with `options["..."]` |
| `GenerativeOptions` | `generate_answer()`, `send_chat_message()` | `None` fields are dropped |

Single options may also be passed as keyword arguments — the two calls below are
equivalent:

```python
client.search("my_docs", "query", SearchOptions(max_results=20, hybrid_search=True))
client.search("my_docs", "query", max_results=20, hybrid_search=True)
```

### Wire-format quirks handled by the client

The backend is not uniform, and these traps are the reason the wrapper exists:

* `options` of `search_with_options`, `query_options` of `generative_answer` and
  `indexing_options` of `upload_and_index_files` are read with `json.loads()`,
  so they are sent as **JSON strings**;
* `indexing_options` of `add_and_index_texts` and `options`/`search_options` of
  the chat endpoints are read as objects, so they are sent as **JSON objects**;
* uploaded files use the field name `files[]` and input texts `texts[]`;
* endpoint paths have **no trailing slash** (that is how Django registers them);
* business errors come back as HTTP 200 with `{"status": false, "errors": [...]}`.

## Return values

Methods return dataclasses (`sse_lib.models`): `Collection`, `SearchResponse`
(with `.hits` of `SearchHit`), `Answer`, `UploadResult`, `IndexingResult`,
`Chat`, `ChatReply`, `ChatMessage`, `ChatHistory`. Every one of them keeps the
untouched server payload in `.raw`, and
`client.request("GET", "some_endpoint")` returns the unwrapped `body` of any
endpoint for fields that are not modelled yet.

## Errors

| Exception | Raised when |
|-----------|-------------|
| `SSEValueError` | An argument is rejected before the request is sent (empty query, unknown index type, ...) |
| `SSEConfigError` | Invalid configuration (missing host, non-positive timeout, login without credentials) |
| `SSEAPIError` | The API answered `{"status": false, "errors": [...]}` — inspect `.error_names`, `.not_given_params` |
| `SSEAuthenticationError` | HTTP 401/403 |
| `SSEHTTPError` | Any other non-2xx status |
| `SSETransportError` | No response at all (DNS, connection refused, timeout) |

All of them derive from `SSEError`, so a single `except SSEError` is enough for
a coarse handler. Idempotent `GET` calls are retried on connection errors and
on 429/502/503/504; `POST` is never replayed, because indexing twice is a real
side effect.

## Development

```bash
cd sse_lib
PYTHONPATH=src python -m unittest discover -s tests -v   # no installation needed
python -m pytest .                                        # pytest, if available
black --check . && flake8 .
```

The tests never open a socket: `tests/support.py` replaces the HTTP session
with a recording double, which pins the exact payloads sent to the API.
