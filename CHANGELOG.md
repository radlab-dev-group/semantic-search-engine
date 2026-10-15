# Changelog

All notable changes to the Semantic Search Engine are documented in this file.

| Version | Changelog |
|---------|-----------|
| [0.1.1](#0.1.1) | Deletion API (PostgreSQL + Milvus), document deactivation (`use_in_search`), normalized `[0, 1]` document ranking weights, collection-level chunking and `max_model_length` |
| [0.1.0](#010) | Python client library, hybrid search, production deployment tooling, `src/` layout and CI |
| [0.0.1](#001) | Initial version |


## [011]

### Added

#### API — deletion of collection content
- **`POST delete_texts`** — remove text chunks of a collection by their ids
  (`chunk_ids[]`) from PostgreSQL (including the pages left without any
  chunk) and their embeddings from Milvus.
- **`POST delete_documents`** — remove documents of a collection by names
  (`documents[]`) and/or ids (`document_ids[]`) together with their pages
  and text chunks, from both databases.
- **`POST delete_collection`** — remove the whole collection with all of its
  documents, pages, text chunks and the Milvus collection.
- Deletion endpoints accept `DELETE` as well as `POST` and report what was
  removed (`deleted` rows, `deleted_vectors`, `dropped_collection`,
  `not_found`). Deleting is allowed only for the **owner** of a collection.
- `ContentDeletionController` (`data/controllers/deletion.py`) orchestrates
  the removal in both databases: rows first, vectors afterwards, so removed
  rows are never indexed again. Chats and search history referencing a
  deleted collection are removed with it (protected foreign keys).
- `MilvusHandler` deletion API: `delete_texts_by_external_ids`,
  `delete_documents_by_external_ids` (primary-key based delete resolved
  through a metadata query, batched) and `drop_collection`;
  `SemanticDBController` exposes `delete_texts`, `delete_documents` and
  `drop_collection`.
- New errors: `DOCUMENT_NOT_FOUND`, `CHUNK_NOT_FOUND`, `DELETE_PROTECTED`.

#### API — document deactivation in the search
- **`POST set_documents_use_in_search`** — enable or disable documents of a
  collection in the search, i.e. set the `Document.use_in_search` flag.
  Documents are addressed by their names (`documents[]`) and/or by their ids
  (`document_ids[]`); `use_in_search` defaults to `false`, so the call
  deactivates them.
- Nothing is removed: the pages, the text chunks and the embeddings of a
  disabled document stay in both databases, so the document may be enabled
  again without any re-indexing.
- While disabled, its chunks are not returned by `search_with_options`: the
  full-text side (`DBTextSearchController.search_texts`) and the chunk
  resolution (`DBTextSearchController.get_texts`, used by the vector side and
  by the protocol-based engine) skip the chunks of the documents with
  `use_in_search=False` (opt-out via `only_used_to_search=False`).
- The answer reports `updated_documents`, the value which was set
  (`use_in_search`) and the names/ids which were not found (`not_found`);
  identifiers which match nothing are refused with `DOCUMENT_NOT_FOUND`.
  Changing the documents is allowed only for the **owner** of the collection,
  like the deletion endpoints.
- `ContentVisibilityController` (`data/controllers/visibility.py`) flips the
  flag in the relational db, with `VisibilityResult` as the outcome; the
  update itself lives in
  `RelationalDBController.set_documents_use_in_search`.

#### API — chunking of a collection

How the texts of a collection are split is now declared once, on the collection
(`CollectionOfDocuments.metadata_json`), instead of being repeated (or
forgotten) on every indexing call.

- **`POST new_collection`** accepts `chunking`
  (`{"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64}`) and the
  `strict_chunking` flag, and reports both in the answer.
- **`max_tokens_in_chunk`** and **`number_of_overlap_tokens`** of
  `indexing_options` are **optional**: `resolve_chunking()`
  (`data/controllers/chunking.py`) takes the values of the request, then the
  ones of the collection, then the global defaults
  (`DEFAULT_MAX_TOKENS_IN_CHUNK` = 500, no overlap), which are remembered on a
  collection that has no settings yet, so the next call may pass nothing.
  Mixed chunk lengths inside one collection remain possible — the invariant of
  a collection is its embedder, vector size and metric, not one chunk size.
- A strict collection refuses indexing options other than its own
  (`CHUNKING_MISMATCH`, `000006_DATA`), and a chunk longer than the input of
  the embedder of the collection is refused instead of being truncated
  silently by the model (`CHUNKING_TOO_LONG`, `000005_DATA`).
- **`GET collections`** exposes `metadata_json`, so a UI can read the settings
  of a collection; **`GET filter_options`** reports `chunking_defaults` to
  prefill a form with (`data/core/constants.py` holds the defaults and keys).
- Management command **`check_collection_chunking`** reports what every
  collection declares and, with `--verify`, measures the stored chunks with the
  tokenizer of the embedder and points at the chunks longer than declared
  (`--collection NAME`, `--limit`, `--json`).

#### Model input length (`max_model_length`)

- `configs/embedders.json` and `configs/rerankers.json` may declare
  `max_model_length`, the largest number of tokens a model takes in: one text
  of an embedder, the whole `query + text` pair of a reranker. An entry
  without the key declares no limit, so existing configurations keep working.
- New accessors `EmbeddingModelsConfig.get_embedder_max_model_length()` /
  `get_reranker_max_model_length()` and the descriptions
  `embedders_info()` / `rerankers_info()`; `GET embedders` and `GET rerankers`
  return them as `models_info`. A chunk which does not fit the pair budget of
  the reranker is only reported in the server log, because the query already
  occupies part of it.
- `engine.controllers.models_logic.embedders_rerankers` is now a shim of
  `engine.core.config.models_config` — one registry, `path`, `vector_size`,
  `device` and `max_model_length` read from a single place.

#### Client library
- **`SSEClient.delete_texts`, `SSEClient.delete_documents` and
  `SSEClient.delete_collection`** with client side validation of the given
  ids/names, plus `DELETE_TEXTS`, `DELETE_DOCUMENTS`, `DELETE_COLLECTION`
  endpoint constants.
- **`SSEClient.set_documents_use_in_search`** with the
  **`SSEClient.deactivate_documents`** / **`SSEClient.activate_documents`**
  shortcuts and the `SET_DOCUMENTS_USE_IN_SEARCH` endpoint constant.
- **`SSEClient.create_collection(..., chunking=..., strict_chunking=...)`**
  declares the chunking of the new collection
  (`sse_lib.options.chunking_payload` normalises and checks it), and
  `IndexingOptions` no longer sends `max_tokens_in_chunk` /
  `number_of_overlap_tokens` when they are unset, so the settings of the
  collection apply. `Collection.chunking` / `.strict_chunking` /
  `.metadata_json` read them back.
- **`SSEClient.embedders_info()`** and **`SSEClient.rerankers_info()`** return
  `models_info`: `vector_size`, `device` and `max_model_length` per model.

#### Document ranking — normalized weights
- Chunk scores are now normalized onto `[0, 1]` before document statistics
  are computed (`normalize_relevance`), matching the scale each pipeline
  stage actually reports: cosine similarity (`(1 + s) / 2`), RRF values
  (`s * (k + 1) / 2`) or cross-encoder logits (logistic squash). The scale
  is derived from the search options by `retrieval_score_scale`.
- `prepare_documents_stats` gains `score_scale` / `rrf_k` parameters and
  returns, in addition to `score` and the hit/page counters:
  `score_normalized` (mean relevance in `[0, 1]`), `score_weighted`
  (non-negative document mass), `score_weighted_scaled` (share of the total
  mass; always sums up to `1.0`) and `weight_normalized` (mass relative to
  the best document).
- Fixes the old scaling, which clamped negative reranker logits to the same
  value, shifted the weights with `abs(min())` even when every weight was
  positive, and did not sum up to `1.0` for result sets of unrelated
  documents — all of which distorted `percentage_rank_mass`.
- The protocol-based engine no longer scores every chunk with a constant
  `1.0`: retrieval scores (cosine or fused RRF) are propagated through
  `RelationalDataAdapter.get_text_for_id(text_scores=...)`.
- `DBSemanticSearchController.prepare_documents_stats` now delegates to the
  pure `engine.core.search_engine` implementation (the duplicated copy was
  removed).
- Search responses now make the score scale explicit:
  - `results.score_scale` says which metric ordered the results: `"cosine"`
    (plain vector search), `"rrf"` (hybrid search) or `"reranker"` (cross
    encoder was applied).
  - every chunk carries `score_normalized`, the same relevance mapped onto
    `[0, 1]`, so numbers from different pipelines are comparable.
  - every chunk carries `similarity`, the cosine similarity of the vector
    search — reported **only** when the vector search is the single ranking
    criterion (`hybrid_search` off). In hybrid search the ranking is decided
    by the fused ranks and full-text-only chunks have no similarity, so a
    cosine next to such a score would be misleading. The same rule applies
    to the new `stats[document].similarity` (mean of the hits, `null` in
    hybrid mode).

## [0.1.0]

### Added

#### Client library
- **`sse_lib` — Python client for the SSE REST API**: `SSEClient` with an HTTP transport layer (URL building, token authentication, retry logic, error mapping), typed response models (collections, search results, uploads, indexing, chats, messages), typed option builders, and a custom exception hierarchy.
- **`SSEClient.health()` / `is_healthy()`** and a `HEALTHZ` endpoint constant for readiness checks; `is_healthy()` returns `False` instead of raising.
- **Client-side option validation** in `sse_lib`: indexing and generative options the backend refuses as `null` (`max_tokens_in_chunk`, `number_of_overlap_tokens`, `generative_model`, `percentage_rank_mass`) now fail fast with an actionable message instead of an opaque `INVALID_PARAMS` round trip.

#### Search & retrieval
- **Hybrid search** combining Milvus vector results with PostgreSQL full-text search through Reciprocal Rank Fusion (RRF), with LLM-powered query reformulation and improved reranking.
- **`min_similarity` parameter** for cosine similarity filtering across the hybrid search pipeline, vector store adapter, Milvus controller, and search controllers, backed by unit tests for `COSINE` metric alignment.
- **`milvus_index_to_cosine.py`** migration script that converts existing Milvus collections to COSINE metric indexes without re-embedding data, including a `chat` namespace.
- **Protocol-based adapters** for vector store and relational data, establishing clean protocol boundaries with backward-compatible re-exports of legacy controllers.

#### API & validation
- **Upload API** expanded with safety checks, detailed validation, and configurable limits.
- **Chat API** hardened with input validation, improved collection handling, and detailed error responses; `collection` is now required in `send_chat_message`.
- **`INVALID_PARAMS` error** and stricter parameter and language validation in request decorators; tightened token validation in `authorization`.
- **`/healthz` probe** reporting database and Milvus dependency status, registered both under the configured API root and as a bare `/healthz` so the container `HEALTHCHECK` and nginx keep working when the API version prefix changes.

#### Deployment & tooling
- **Production setup**: gunicorn configuration, `.env` template, Docker Compose, and supporting scripts.
- **nginx reverse proxy** with TLS configuration and a README covering certificate setup.
- **RAG evaluation tooling**: evaluator script for LLM-based RAG quality assessment plus `EVALUATE.md` documentation.
- **`SETUP.md`**, English startup script `run-en.sh`, and `DEPENDENCIES.md` listing Python packages, infrastructure, ML models, and optional services.

#### Packaging, CI, and project layout (`feat/modularity`)
- **`src/` layout with a single installable package**: `sse_rest_api` and `sse_tools` moved under `src/`, `engine_core` flattened into `engine.core`, and `pyproject.toml` extended with build metadata, package discovery, and pytest configuration.
- **CI pipeline** (`.github/workflows/ci.yml`) running the test suite on Python 3.10/3.11/3.12 plus a lint gate (black, flake8, bandit, mypy); `requirements-dev.txt` keeps the dev toolchain out of the production image.
- **Config templates** shipped under `configs/` (`django-config`, `auth-config`, `milvus_config`, `user-group-organisation`, `secret-key`) so a fresh checkout provides every expected file.
- **Initial migration** for the `authorization` app, which was missing from the repository.
- **Test infrastructure**: Django test settings wired to the real URLconf, route-table coverage for each authentication mode, `healthz` tests, input-validation tests, and `sse_lib` tests for required options and the health endpoint.

### Fixed
- **Auth configuration no longer read at import time**: `RdlAuthStateHandler`, `RdlAuthGrantAccTokenHandler`, and `AuthAuthenticationMiddleware` opened `configs/auth-config.json` from class definitions, so loading the URLconf raised `FileNotFoundError` on any machine without a deployment config.
- **`sse_lib` stopped sending an empty `relative_filepath`**: an omitted path now falls back to the document name, because the server validates the field with `required_text` and rejected `""`.
- **Two latent `F821` bugs** and the remaining lint errors resolved; the whole tree formatted with black.

### Changed
- **Package rename and restructure**: `semantic_search_engine` → `sse_rest_api` → `sse_apps`, and finally to the `src/` layout, with all imports, scripts, docs, and path references updated.
- **Batch text indexing** with pre-fetched context maps to cut redundant queries; lazy initialization of embedder/reranker configs and controller dependencies.
- **Embedder and reranker device** switched from CUDA to CPU.
- **Lint configuration consolidated** into `pyproject.toml` and `setup.cfg`; redundant `tox.ini` and `mypy.ini` removed.
- **Documentation refreshed**: English README with architecture diagram and usage guide, `README-pl.md`, module docstrings across `chat`, `data`, `engine`, `system`, and `sse_lib`, and README coverage of upload policies, access rules, and `min_similarity`.
- `requirements.txt` introduced for the engine (pymilvus, `langchain_community`, python-docx).

### Removed
- `tmp/` scratch data and the unused `llm-router-config.json`, superseded by the shipped `configs/` templates.

## [0.0.1]

### Added
- Initial release of the Semantic Search Engine: Django REST API server with the `main`, `engine`, `data`, `chat`, `authorization`, and `system` apps.
- Semantic search on Milvus (handler, index creation, query controllers) with PostgreSQL-backed relational data and text indexing.
- Query denoising (`ContentSupervisor`, `DenoiserInference`) and query-template filtering.
- Generative-model, embedder, and reranker configuration loaders with pluggable local and remote model hosts.
- Token-based authorization with user/group organisation mapping and login endpoint.
- Chat module (models, serializers, API views, URLs) for multi-turn retrieval-augmented conversations.
- Installation and initialization scripts, local development tooling configs (black, flake8, pylint, mypy, bandit, tox).
- Root and per-package READMEs covering overview, architecture, quick-start, configuration, and typical workflows.
