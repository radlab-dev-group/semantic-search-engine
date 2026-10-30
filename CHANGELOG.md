# Changelog

All notable changes to the Semantic Search Engine are documented in this file.

| Version | Changelog |
|---------|-----------|
| [0.1.4](#014) | Preview of a whole document: `GET document_content` (PostgreSQL only) and `sse_lib.SSEClient.document_content` / `indexing_status`. Counters of the documents in `GET collections` (`document_count`, `active_document_count`), one query for the whole answer; `sse_lib.Collection` exposes both. Date of a search in the history (`UserQuery.created_on`, `created_on` in `search_history` / `get_search_history`); `sse_lib.SearchHistory.created_on`. Files of the uploads on the disk: `delete_documents` / `delete_collection` remove them with the rows (`deleted_files`, `file_errors`), and the new `delete_files_after_indexing` option of an indexing call removes the file of every document once it is indexed. Documents of an upload are deduplicated by name **and** the path they were read from, so a zip with files of the same name in different subdirectories indexes every one of them. `./run.sh down` stops every SSE process (worker, API), `./run.sh status` lists them |
| [0.1.3](#013) | Indexing in a Celery + RabbitMQ queue, per-document indexing progress, `indexing_status` endpoint. Deactivation propagates to Milvus (`is_active` of the vectors); Milvus 3.0 |
| [0.1.2](#012) | Deletion of the search history (`delete_search_history`), form-encoded `[]` list params |
| [0.1.1](#0.1.1) | Deletion API (PostgreSQL + Milvus), document deactivation (`use_in_search`), normalized `[0, 1]` document ranking weights, collection-level chunking and `max_model_length` |
| [0.1.0](#010) | Python client library, hybrid search, production deployment tooling, `src/` layout and CI |
| [0.0.1](#001) | Initial version |


## [0.1.4]

### Changed

#### Default chunking of a collection
- A collection created without chunking settings now uses **400 tokens** per
  chunk with an **overlap of 60 tokens** instead of 500 tokens with no overlap
  (`DEFAULT_MAX_TOKENS_IN_CHUNK`, `DEFAULT_NUMBER_OF_OVERLAP_TOKENS` in
  `src/sse_api/data/core/constants.py`).  The values are still remembered on
  the collection, so they can be overridden per request or per collection.

### Added

#### API server under gunicorn
- A new launcher, `src/sse_api/run-rest-api-gunicorn.sh`, starts the REST API on
  gunicorn (the same environment and the same port `8271` as `run-api.sh`, but
  without the development server). `./run.sh start` now uses it; `./run.sh down`
  stops the gunicorn processes too. Flags: `--port`, `--bind`, `--workers`.

### Fixed

#### Indexing of files with the same name in different subdirectories
- A zip (or a directory upload) may hold several files of the same name in
  different subdirectories.  The name of a document is not its identity —
  the same name in two different directories is two different documents —
  and `exists_document_in_db` used the name alone when deciding that a
  document was already in the database, so only the first of the files was
  indexed and the rest were silently skipped.  The check now compares the
  name **and** the path the file was read from, which is also the identity
  the storage step (`add_document_from_dict`) already uses, so a retried
  job of one upload still skips what it already stored.

### Added

#### Files of the uploads
- **`delete_documents` and `delete_collection` remove the files of the uploads
  from the disk** as the last step of the deletion, after the rows and the
  vectors, so a document which is still in the databases never loses its
  file.  The answer reports how many of them went away (`deleted_files`) and
  which ones could not be removed (`file_errors`); a problem with the disk
  never fails a deletion.  `delete_texts` touches no file — the rest of the
  document is still in the databases.
- A new module, `data/controllers/file_cleanup.py`, owns that removal: only
  the paths inside the upload directory may be removed, and a directory
  disappears only when its last file is gone, so a directory of another
  collection is never touched.
- **`delete_files_after_indexing`**, a new option of `indexing_options`
  (`upload_and_index_files`, `add_and_index_texts`): the file of every
  document disappears from the disk once its embeddings are stored.  It is
  saved on `IndexingJob.indexing_options`, which is how a worker of the queue
  learns about it.  `indexed` and `skipped` remove the file, `failed` keeps it
  — the file of a document whose embeddings are missing is the only thing a
  retry can still work on.  The option is optional, unlike the five cleaning
  switches, so a client which does not know it keeps working.
- `sse_lib.IndexingOptions.delete_files_after_indexing`.

#### API — the date of a search
- `UserQuery` remembers **when the search was made** (`created_on`, migration
  `0002_userquery_created_on`), so the history of the searches can show the
  date next to the question instead of only the order of the entries.
- `search_history` and `get_search_history` add `created_on` to every entry,
  as an ISO 8601 string in UTC (the same shape as the other timestamps of the
  API); the ordering of the history does not change.
- `sse_lib.SearchHistory.created_on` exposes it, `None` for an answer of an
  engine which does not know the field yet.

#### API — counters of the documents of a collection
- **`GET collections`** answers with two numbers for every collection, so a
  client listing them does not have to call `GET documents` once per
  collection: `document_count` (every document of the collection, the ones
  deactivated in the search included) and `active_document_count` (the
  documents with `use_in_search=True`, i.e. what `GET documents` lists by
  default).
- `RelationalDBController.with_document_counts` annotates a queryset of
  collections with both counters in **one query**, and `ListCollections` uses
  it for the collections of the user and for the organisational ones before
  merging them — the number of queries does not grow with the number of
  collections.
- `CollectionOfDocumentsSerializer` reads the annotations and counts a
  collection on its own when they are missing, so any other place which
  serializes a collection gets correct numbers instead of zeros.

#### API — preview of a whole document
- **`GET document_content`** — the whole text of one document of a collection,
  which is what the preview opened from a row of the documents tab shows.
  `collection_name` with `document_name` or `document_id`, optionally
  `chunk_type` (which stored form of the text to read, e.g. `clear_texts` or
  `raw_text`), `from_page` and `max_pages` (default `PREVIEW_MAX_PAGES` = 200,
  at most `PREVIEW_MAX_PAGES_LIMIT` = 1000).  The answer carries the stitched
  `content`, the same text page by page (`pages[]` with `page`, `content` and
  `chunks`), the `chunk_type` which was read together with all the
  `chunk_types` of the document, and the window of pages (`from_page`,
  `total_pages`, `truncated`, `next_page`) so a client can walk through a
  document longer than the window.
- `ContentPreviewController` (`data/controllers/preview.py`) reads the preview
  from PostgreSQL only — the pages of the document and the chunks of one
  `text_chunk_type`, joined in reading order — so neither the source file nor
  Milvus is needed.  A name which stands for more than one row is previewed by
  its newest document, and an unknown `chunk_type` falls back to the newest
  form the document was indexed with.  A document of another organisation is
  `COLLECTION_NOT_FOUND`, a missing one `DOCUMENT_NOT_FOUND`.
- `sse_lib.SSEClient.document_content(collection, document,`
  `from_page=None, max_pages=None, chunk_type=None)` wraps the endpoint and
  answers with `DocumentPreview` (one `DocumentPreviewPage` per page of the
  document); `document` is a name or an identifier.
- `sse_lib.SSEClient.indexing_status(collection=None, job_id=None)` wraps
  `GET indexing_status` with `IndexingStatus` / `IndexingJobStatus`: the
  documents of a job with their own `indexing_status`, the documents still
  waiting for their embeddings and the counter of the job.  The route has been
  registered by the backend since 0.1.3 and was the one the client library did
  not know.

### Changed

#### Client library
- `sse_lib.Collection` carries `document_count` and `active_document_count`
  (`None` when the server did not send them, e.g. an engine older than this
  release), with `document_count_or_zero` / `active_document_count_or_zero`
  for callers which prefer a plain number.

- README: the REST endpoints table and a new "Counters of a collection"
  section describe the two fields (`README-pl.md` in Polish).

#### Stopping the processes of the SSE by `run.sh`
- `./run.sh down` (and `clean`) stop **every** process of the SSE — the worker
  of the indexing queue and the API server — and no longer only the PID written
  to `.celery-worker.pid`.  Several checkouts of the project work on one broker
  and one database whenever their `.env` points at the same hosts, and a worker
  left behind by an earlier start keeps consuming the tasks of the `indexing`
  queue with the code it was started with: an indexing job then behaves like an
  older build — documents silently skipped — however many times the current
  checkout is restarted.  A process which ignores `SIGTERM` is sent `SIGKILL`
  after 10 seconds.
- `./run.sh status` lists the SSE processes it finds, so a second worker, or a
  server of another checkout sharing the broker, is visible before it can
  mislead anyone.

## [0.1.3]

### Added

#### Indexing queue (Celery + RabbitMQ)
- Indexing can be handed over to a **Celery** queue with **RabbitMQ**:
  `ENV_USE_CELERY=1` (or `"use_celery": true` in `django-config.json`) makes
  `upload_and_index_files` and `add_and_index_texts` store the content, write an
  `IndexingJob` and answer immediately; a worker does the parsing and the
  embeddings.  With the flag off, both endpoints index in the request exactly as
  before, and both modes report the same progress.
- The **document** is the unit of work: `prepare_indexing_job` stores the
  documents of one upload and queues one `index_document` per document
  (`sse_api/data/tasks.py`).  A document is retried three times with an
  exponential backoff and then marked `failed` without stopping the rest of the
  job; both tasks acknowledge late, so a killed worker loses no message.
- `IndexingJob` (kind, status, `total_documents`, `documents_indexed`,
  `documents_failed`, `documents_skipped`, `current_document`, `error_message`,
  timestamps) and `Document.indexing_status` (`pending` / `indexing` / `indexed`
  / `failed` / `skipped`, `NULL` for documents indexed before this change) keep
  the progress in the relational database, so it survives a restart of a worker.
  New migration `data/0003_indexing_jobs`.
- **`GET indexing_status`** — the progress of one job (`job_id`) or of the
  newest job of a collection (`collection_name`): the job with its counters and
  `percent`, its documents with their own status, and
  `pending_documents` of the collection.  `upload_and_index_files` and
  `add_and_index_texts` answer with the same job under `indexing_job`, plus
  `"async": true` when the work was only queued.  `documents` reports
  `indexing_status` per document.  New error: `INDEXING_JOB_NOT_FOUND`.
- `python -m sse_api.manage reset_stale_indexing_jobs` closes jobs and
  documents which a killed worker left running (`--minutes`, `--dry-run`).
- `docker compose --profile indexing up -d` starts `rabbitmq` and `worker`;
  without the profile the compose stack is unchanged.  New variables in
  `.env.example`: `ENV_CELERY_RESULT_BACKEND` (empty by default: the progress is
  kept in PostgreSQL), `ENV_CELERY_TASK_TIME_LIMIT`, `RABBITMQ_USER`,
  `RABBITMQ_PASSWORD`, `RABBITMQ_BIND_HOST`, `RABBITMQ_HOST_PORT`,
  `CELERY_CONCURRENCY`.

### Changed
- `sse_api/config/__init__.py` creates the Celery application (imported only
  when Celery is installed), `config/settings.py` configures it through the
  `CELERY_*` namespace instead of adding Celery to `INSTALLED_APPS`.
- `celery` is a dependency now; the embedder of a collection is loaded once per
  worker process.
- `POST set_documents_use_in_search` now propagates the visibility change to
  the vector database as well.  The embeddings of the changed documents are
  **not removed** — their `is_active` flag in Milvus follows the relational
  `Document.use_in_search` one, so the state is consistent for **every**
  consumer of the vector collection, not only for the search of this engine.
  Re-activating a document still needs no re-indexing.  `MilvusHandler` gains
  `set_active_by_external_ids`, which queries only the matching primary keys
  and writes back `pk` and `is_active` with a Milvus **partial update**, so
  the embeddings are never read nor rewritten.  Updating an entry of an
  `auto_id` collection needs a server which supports it (Milvus >= 2.4.15);
  against an older one (such as the shipped `milvusdb/milvus:v2.3.0`) the
  relational flag is still stored and the search still respects it, and a
  warning is logged instead of failing the request.
- The answer of `POST set_documents_use_in_search` reports how many vectors
  were changed as well (`updated_vectors`, next to `updated_documents`).
- The bundled vector database is now **Milvus 3.0**
  (`milvusdb/milvus:v3.0.3`) with a matching `pymilvus` 3.x client
  (`requirements.txt` pins `pymilvus>=3.0`).  The `DEPLOY_MODE=STANDALONE`
  environment variable is passed to the standalone container — Milvus >= 3.0
  refuses to start with an embedded etcd without it — and the compose stack
  sets `ETCD_AUTH_ENABLED=false` for its unauthenticated sidecar etcd.
- Upgrading from Milvus 2.x needs a re-index: the old collections do not
  have the `is_active` field, so they are dropped and re-created (PostgreSQL
  holds the documents and the embeddings are rebuilt).

## [0.1.2]

### Added

#### API — deletion of the search history
- **`POST delete_search_history`** (also accepted as `DELETE`) — remove the
  stored queries of the authenticated user by their ids (`query_ids[]` and/or
  `query_response_ids[]`, the same identifiers the rest of the history API
  serves) together with their responses, the generated answers and the RAG
  states of the chats which reference them.  The `state` of a chat message is
  detached instead of removed, so the message keeps its life and only loses the
  reference.  The answer reports what was removed (`deleted`) and which of
  the requested ids were not found (`not_found`, per identifier type).
  New errors: `QUERY_NOT_FOUND`, `DELETE_PROTECTED` (engine).
- `SearchHistoryDeletionController`
  (`engine/controllers/history/deletion.py`) removes the rows in the order
  which respects the `PROTECT` foreign keys, inside one transaction, so a
  failure leaves the history untouched.
- `sse_lib.SSEClient.delete_search_history(query_ids=None,`
  `query_response_ids=None)` wraps the endpoint.
- `request_params` reads the `[]` list parameters of a form-encoded body with
  `getlist`, so a form-encoded request behaves like the JSON one (previously
  a repeated key kept only its last value).

## [0.1.1]

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
- **`--fix`** repairs the chunking *declared* on the collections
  (`plan_chunking_fix()`): a collection without settings gets the defaults, a
  chunk above `max_model_length` of its embedder is clamped to it and an
  overlap outside the chunk is corrected; `--dry-run` shows the plan without
  writing it. Metadata only — no text is re-split, chunks already in the
  database require re-indexing.

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
- **`SETUP.md`**, English startup script `run.sh`, and `DEPENDENCIES.md` listing Python packages, infrastructure, ML models, and optional services.

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
