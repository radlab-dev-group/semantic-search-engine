# Changelog

All notable changes to the Semantic Search Engine are documented in this file.

| Version | Changelog |
|---------|-----------|
| [0.1.9](#019) | Search quality: the engine core no longer ships the ML stack (text cleaning, token counting and truncation move to the model service — `POST /v1/tokenize`, `/v1/truncate`, `/v1/denoise`), indexing embeds chunks with a document context header (document, category, path, section), the Milvus `filename` / `document_name` filter bug is fixed, short chunks (40+ characters) reach the index, texts are split on paragraph, code-fence and table boundaries, every collection stores its own search profile and RAG model (`POST /api/v1/set_collection_options`; the search and generation endpoints fall back to the profile when the request names no value), the hybrid search is rebalanced: weighted RRF (`dense_weight` / `lexical_weight`), equal candidate pools for both legs, the document filter on the full-text leg, the cosine similarity surviving the fusion, per-model `query_instruction` for asymmetric embedders and HNSW tuned for recall, and the post-retrieval stage: the reranker sees the name of the document it scores, a bigger rerank pool, per-document chunk caps (MMR-style diversification), a logarithmic document mass and the stitching of neighbouring chunks into one passage for the RAG context; the evaluation gains the standard IR metrics (NDCG, MRR, Precision/Recall, AP/MAP) scored on every question and a grid-search tuner of the fusion weights, the RRF constant and the rerank switch, and extractive question answering moving to the model service (`POST /v1/extractive_qa`), with the engine verified to import without `torch`, `transformers` or `sentence-transformers` |
| [0.1.8](#018) | Standalone model service (`src/sse_models`) with `POST /v1/embed` / `POST /v1/rerank`, run with `./run.sh models` or `./run-models.sh`: the engine can delegate embedding and reranking to it (`SSE_MODEL_SERVICE_ENABLED`) with a circuit breaker and in-process fallback, or run inference on another machine entirely. Keyword filters in search (`options.keywords` with `keywords_match: "and"` / `"any"`), lexical-only search (`"lexical_only": true` answers from the full-text leg alone, `score_scale: "fts"`), a `POST update_knowledge_quality` endpoint with monotonic per-document upvote/downvote counters, and `model_embedder` / `model_reranker` in `GET collections` |
| [0.1.7](#017) | Deletion of user chats: `POST delete_chat` (also accepted as `DELETE`, `chat_id` / `chat_ids[]`), `ChatDeletionController`, and `sse_lib.SSEClient.delete_chat` (`endpoints.DELETE_CHAT`). Deleting a collection also removes messages and states of its chats cleanly. Chat history (`GET chats` and `GET get_chat_by_hash`) now serializes messages with `DeepMessageSerializer` (including `state` with RAG responses, answers, and citations) and `collection_name` |
| [0.1.6](#016) | Conversational RAG with memory: the generator sees the previous questions **and** answers, follow-up questions are rewritten into standalone retrieval queries (new `POST rewrite_query`), answers carry numbered source citations (`citations`, per-chat numbering, `answer_source`/`rewritten_query` in the reply) and short commands ("make it a table", "shorten", "translate") reuse the previous sources without a new search. Every search hit and citation carries the chunk's identity and position (`text_id`, `document_id`, `char_start`, `char_end`); `GET document_content?text_id=` opens exactly the chunk's page with a `highlight`. `sse_lib`: `rewrite_query()`, `generative_answer_with_citations()`, `document_content(text_id=...)` and the new reply/citation fields |
| [0.1.5](#015) | `./run.sh start` now runs the REST API on **gunicorn** (`src/sse_api/run-rest-api-gunicorn.sh`, the same environment and port `8271` as `run-api.sh`; flags `--port`, `--bind`, `--workers`). `./run.sh down` stops **every** SSE process (worker, API) and `./run.sh status` lists them; the bundled Milvus moves to `v3.0.5`. Default chunking of a collection: **400 tokens** per chunk with a **60-token overlap** (was 500 with no overlap) |
| [0.1.4](#014) | Preview of a whole document: `GET document_content` (PostgreSQL only) and `sse_lib.SSEClient.document_content` / `indexing_status`. Counters of the documents in `GET collections` (`document_count`, `active_document_count`), one query for the whole answer; `sse_lib.Collection` exposes both. Date of a search in the history (`UserQuery.created_on`, `created_on` in `search_history` / `get_search_history`); `sse_lib.SearchHistory.created_on`. Files of the uploads on the disk: `delete_documents` / `delete_collection` remove them with the rows (`deleted_files`, `file_errors`), and the new `delete_files_after_indexing` option of an indexing call removes the file of every document once it is indexed. Documents of an upload are deduplicated by name **and** the path they were read from, so a zip with files of the same name in different subdirectories indexes every one of them |
| [0.1.3](#013) | Indexing in a Celery + RabbitMQ queue, per-document indexing progress, `indexing_status` endpoint. Deactivation propagates to Milvus (`is_active` of the vectors); Milvus 3.0 |
| [0.1.2](#012) | Deletion of the search history (`delete_search_history`), form-encoded `[]` list params |
| [0.1.1](#0.1.1) | Deletion API (PostgreSQL + Milvus), document deactivation (`use_in_search`), normalized `[0, 1]` document ranking weights, collection-level chunking and `max_model_length` |
| [0.1.0](#010) | Python client library, hybrid search, production deployment tooling, `src/` layout and CI |
| [0.0.1](#001) | Initial version |


## [0.1.9]

### Added

#### Context enrichment of the indexed vectors
- A chunk on its own is context-blind: *"Wnioski należy składać do 30
  kwietnia"* does not say which regulation it belongs to, and a query naming
  the regulation never reaches it.  Before a chunk is embedded, a small
  bounded header is prepended —
  `Dokument: <nazwa> | Kategoria: <kategoria> | Ścieżka: <ścieżka> |
  Sekcja: <nagłówek>` — so that the identity of the document and the
  section of the chunk move into every token of the vector.  The enrichment
  is for the embedder only: the raw text is what is stored in the vector
  database and PostgreSQL, what the API displays, and what a re-indexing
  reads (an already enriched text is never enriched twice).
- `src/sse_api/data/core/context_enrichment.py` builds the header
  (`build_enriched_chunk_text`, `build_context_header`, `extract_section_title`,
  `is_enriched`, `raw_chunk_text`); the ingestion pipeline
  (`DBSemanticSearchController.index_texts_from_list`) and
  `SearchEngineCore.index_collection` both enrich before embedding, with the
  section title read from a Markdown heading or a short title line of the
  chunk itself.
- `sse_models` gains `POST /v1/tokenize` (the tokenizers of the catalog,
  cached, without the model weights), `POST /v1/truncate` (token-exact
  budget) and `POST /v1/denoise` (the T5 denoiser ported from the engine,
  scored against the original with a Levenshtein similarity).  New role
  `denoise` and setting `SSE_MODELS_DENOISE_MAX_MODEL_INPUT` (default 256).
  `ModelBackend` gains `count_tokens` / `truncate_texts` / `denoise_text`:
  `RemoteBackend` calls the service, `LocalBackend` the in-process models,
  both loaded lazily.  `DenoiserController` routes through the service and
  degrades to the original text with a warning when it cannot be cleaned, so
  indexing never fails on a missing T5; the historical path moves to
  `denoiser_legacy.py`.

#### Structure-aware chunking
- `split_text_respecting_structure()` in `src/sse_api/data/controllers/chunking.py`
  splits a text without cutting a sentence, a Markdown table or a fenced
  code block: the text is first divided into structure-aware blocks
  (paragraphs, code fences, tables) which are packed into chunks, and a
  block longer than a chunk is cut on a paragraph, a line or a sentence
  boundary.  It is a pure helper, available to the parsers and to the
  `check_collection_chunking` audit.

#### Search and RAG profile of a collection
- Every collection now stores its own search hyperparameters
  (`metadata_json["default_search_options"]`: `max_results`, `hybrid_search`,
  `rerank_results`, `rerank_max_results`, `dense_weight`, `lexical_weight`,
  `rrf_k`, `min_similarity`, `max_chunks_per_document`) and its RAG profile
  (`metadata_json["default_generation_options"]`: `generative_model`,
  `temperature`, `repetition_penalty`, `top_p`, `top_k`, `typical_p`,
  `max_new_tokens`, `percentage_rank_mass`, `system_prompt`,
  `answer_language`, `translate_answer`).
- `POST /api/v1/set_collection_options` saves or updates the two profiles
  (owner of the collection only; a call changes exactly the keys it sends).
  `GET /api/v1/collections` answers with both profiles, completed with the
  defaults of the engine, so a client always reads a complete set.
- The endpoints fall back to the profile of the collection:
  `search_with_options` fills every search option a request does not declare
  from the collection (request value → collection → engine default), and
  `generative_answer` and the conversational RAG answer with the model and
  the sampling parameters the collection saved when the request does not name
  its own.  The generation fallback fills only what the collection really
  saved, so a deployment which never pinned a model keeps the answers it had.
- `src/sse_api/core/validation.py`: `collection_search_options` and
  `collection_generation_options` validate the profiles; the sampling
  parameters get the same ranges and aliases in every place
  (`rep_penalty` → `repetition_penalty`, `topp` → `top_p`, `topk` → `top_k`,
  `typicalp` → `typical_p`, `max_tokens` → `max_new_tokens`).
- `sse_lib`: `SSEClient.set_collection_options(...)` with explicit
  keyword arguments for the sampling parameters, `SSEClient.
  update_knowledge_quality(...)` (the `update_knowledge_quality` route was
  missing from the endpoint catalog), `Collection.default_search_options` /
  `default_generation_options` / `default_generative_model`, and
  `SearchOptions.metadata_filter_groups` which the server reads.

#### Weighted fusion of the hybrid search
- The rank fusion of a hybrid search is now **weighted Reciprocal Rank
  Fusion**: each leg contributes `weight / (k + rank)`.  `dense_weight`
  (default 0.6) and `lexical_weight` (default 0.4) are options of a search
  request and of the profile of a collection; only their ratio matters, a
  zero weight drops the leg, and a fusion without any weight is refused with
  a `ValueError` instead of silently ranking nothing.  Both implementations
  (the protocol pipeline and the historical controller) call the same
  `rrf_merge`, so a weight means the same thing everywhere.
- The two legs of a hybrid search are **balanced**: each contributes
  `max_results * 10` candidates (the constant
  `HYBRID_CANDIDATE_POOL_FACTOR`), and the document filter of the request is
  forwarded to the full-text leg (`fts_search(..., filter_doc_names=...)`),
  so a search restricted to one document does not spend the candidate pool
  of its lexical leg on the rest of the collection.
- The **cosine similarity of a chunk survives the fusion**: it is a fact
  about the chunk and the query, not about the leg which found it, so the
  document statistics, the reranker and the RAG context keep a comparable
  score in a hybrid search; a chunk which only the text index found carries
  `similarity: null`.
- **Query instruction of asymmetric embedders**: an entry of
  `configs/embedders.json` may declare `query_instruction`, the prefix the
  model wants in front of a query (`google/embeddinggemma-2` ships
  `task: search result | query: `).  `MilvusHandler.search` applies it to
  every query it embeds; an entry without the key means "no prefix", so a
  plain sentence-transformer is not instructed.
- The HNSW index of Milvus is tuned for recall: `M = 16`,
  `efConstruction = 128` at the build, `ef = 200` at the query (was
  `8 / 64 / 100`).
- `sse_lib.SearchOptions` gains `dense_weight` and `lexical_weight`.

#### Contextual reranking and result diversity
- The cross-encoder reranker is no longer **context-blind**: a candidate is
  sent to it prefixed with the name of the document it comes from
  (`Dokument: {name} | Treść: {text}`, the shape of the indexing
  enrichment), so a query naming a document, a regulation or a product
  matches a chunk which does not repeat the name inside itself
  (`MilvusHandler._rerank_candidate`).
- The **rerank pool grows**: the default number of candidates handed to the
  reranker is now `max(50, max_results * 3)` (was `max_results * 2`), so a
  relevant chunk which ranks 22nd after the fusion is still scored when the
  answer shows ten results; the request option `rerank_max_results` and the
  profile of a collection override it.
- **Diversification of the results** (MMR in the spirit of, without the
  pairwise similarity): a chunk of a document which is already represented
  pays a fraction of the relevance range of the answer (the
  `diversity_decay` of the new `engine.core.diversity`), and at most
  `max_chunks_per_document` (default 2, a search option and a profile key)
  chunks of one document are shown in front of the chunks of the others.
  A skipped chunk is never dropped: the list is topped up with it, so a
  search with exactly `max_results` results only reorders them.  The
  document statistics are computed from the undiversified hits on purpose:
  the mass of a document is its evidence, not a presentation rule.
- The **document mass is saturated, not a product of the bulk**:
  `score_weighted` is now `score_normalized * max(1, log2(1 + hits) *
  log2(1 + pages_count))` (was `score_normalized * hits * pages_count`), so
  a hundred-page manual with marginal hits no longer drowns a one-page
  note which answers the question exactly in the selection of the
  documents for the RAG context.
- **Stitching of neighbouring chunks** before they are read by a model:
  two hits of one document which follow each other in the text (the same
  page with the next `text_number`, or the context attached to the later
  hit being exactly the text of the earlier one) are joined into one
  continuous passage in the RAG context and in the enumerated citations of
  a conversational answer (`stitch_adjacent_chunks` in
  `engine.core.search_engine`), so a sentence cut by the chunking window is
  not cut twice.  A merged passage keeps the identity of its first part
  (`text_id`, `char_start`) and the widest span it can prove.
- `sse_lib.SearchOptions` gains `max_chunks_per_document`.

#### Evaluation: the standard IR metrics and the tuner of the search
- The evaluation now speaks the language of Information Retrieval:
  `src/sse_tools/evaluation/evaluator/ir_metrics.py` provides `dcg_at_k`,
  `ndcg_at_k`, `mrr_at_k`, `precision_recall_at_k`,
  `average_precision_at_k` and `map_at_k` as pure functions -- no PyTorch,
  no engine, a ranked list and a ground truth are all they need.  Every
  question of an evaluation run is now scored with them (the ranking of the
  documents by their mass against the files of the human answer, at the
  depth of the test) and the scores land in the XLSX report next to the
  old percentage of the found files.
- `tune_search_parameters.py`: the grid search over the hyperparameters of
  the hybrid search.  A golden dataset (questions with their relevant
  documents) is run through the grid of the fusion weights, the constant
  of the RRF and the rerank switch; the combinations are ordered by
  NDCG@k, the report is written as JSON (and XLSX), and the winner is
  printed as a search profile ready to be stored on a collection with
  `POST /api/v1/set_collection_options`.  A combination whose fusion
  weights share a ratio with an earlier one is skipped: only the ratio of
  `dense_weight` to `lexical_weight` decides a fusion, and the default grid
  of 200 cells is therefore measured in 160 runs.
- The tools of the repository are repaired: the import of
  `PublicRelationDBController` from `sse_api.engine.controllers.database`
  now works (the package exports its controllers lazily), the typo of
  `eavaluate_embedder_search.py` is corrected to
  `evaluate_embedder_search.py`, and the private IPs of the Llama services
  in `article_question_generator.py` come from the environment
  (`LLAMA_SERVICES_HOSTS`, whitespace or comma separated) instead of the
  source.

#### Extractive question answering in the model service
- Extracting an answer from the retrieved passages was the last operation
  which loaded a HuggingFace `pipeline` inside the engine process.  It moves
  to the model service as **`POST /v1/extractive_qa`**: the engine posts the
  question together with the passages it has just retrieved and gets back one
  `{answer, score}` per passage, in input order, so the caller keeps its own
  ranking of which passage answered best.
- `sse_models`: `src/sse_models/extractive_qa.py` (one pass per passage, a
  blank passage answered without the model), the `qa_model()` loader of
  `ModelRegistry` (lazy and cached, like the denoiser), schemas
  `ExtractiveQARequest` / `ExtractiveQAResponse`, a per-model inference lock,
  and the new **`extractive_qa` role** with setting
  `SSE_MODELS_EXTRACTIVE_QA_MAX_CONTEXTS` (default 64, above it a request is
  answered `413`).  The model comes from the `extractive_qa` section of
  `configs/models.json` (`model_name`, `model_path`, `device`).
- The forward pass is written against the model rather than requested as a
  pipeline: **transformers v5 removed the `question-answering` pipeline**
  (the `*ForQuestionAnswering` classes stay), so the service tokenises
  `[CLS] question [SEP] context [SEP]`, takes the best `start <= end` pair of
  the start/end logits and slices the span back out of the passage by its
  character offsets, closing a cut which landed mid-word.  Same answers, no
  dependency on a removed API and no pin to an old release.
- Engine: `ModelBackend` gains `extractive_qa(question, contexts)` —
  `RemoteBackend` calls the service (circuit breaker and local fallback like
  every other operation), `LocalBackend` the in-process pipeline of the new
  `sse_api/data/controllers/extractive_qa_local.py`.  `build_extractive_qa_client()`
  is the standalone client, the mirror of `build_denoise_client()`.
- `ExtractiveQAController` keeps its public shape (`run_extractive_qa` answers
  a `search_results` dict the same way as before) but no longer loads a model:
  it prefers the model service, falls back to the in-process pipeline when the
  service is down, and answers with an empty answer and a score of `0.0` when
  neither is available, so a search never fails on a missing QA model.  The
  class had two near-identical copies; `controllers/models_logic/extractive.py`
  now re-exports the one of `core/generative/extractive.py` under the
  import path it has always had.
- The `RemoteBackend` fallback covers the text operations it was missing: a
  failed `count_tokens` / `truncate_texts` / `denoise_text` / `extractive_qa`
  is answered by the local backend instead of falling into the rerank branch
  (which also removes a block of unreachable code).

### Changed

#### Dependencies of the engine
- `requirements.txt` of the API image no longer installs
  `torch`, `transformers`, `sentence-transformers`, `sentencepiece` or
  `spacy`; the engine core imports none of them at module level (an
  in-process QA pipeline or T5 is loaded only by the explicit local fallbacks
  of `sse_api/data/controllers/`).  Inference and all heavy text operations
  live in the model service (`requirements-models.txt` is the only model
  stack), including extractive question answering
  (`POST /v1/extractive_qa`).
- `Dockerfile.models` builds the inference image standalone.
- The import boundary is now enforced, not intended: importing the engine —
  the API, the search, a worker which only answers queries — no longer
  reaches `transformers`.  The reader of the uploaded files
  (`radlab_data`, whose loaders import `transformers` on import) is imported
  when a file is really read (`directory_file_reader()` /
  `input_text_document()` in `sse_api/engine/controllers/database/relational_db.py`),
  the tokenizer of the embedder is gone from `DBSemanticSearchController`
  (truncation goes to `POST /v1/truncate` through the new
  `build_truncate_client()`, then to the in-process backend, then leaves the
  text to the embedder), and
  `src/sse_api/engine/tests/test_import_boundaries.py` imports the engine in
  a process where `torch`, `transformers` and `sentence-transformers` raise
  `ImportError`.  File indexing stays the documented exception: reading and
  tokenizing a document is `radlab_data`'s work and it needs the model stack.

### Fixed

#### Indexing with the model service
- `MilvusHandler.add_texts` refused to embed with
  `Embedder model is not initialized!` whenever no model was loaded **in the
  engine process** — which is exactly the state of a service-backed
  deployment, where indexing therefore never started.  The guard now asks the
  backend which is going to answer and names both ways of satisfying it.
- `MilvusVectorStoreAdapter` could not be imported on its own
  (`cannot import name 'MilvusHandler' from a partially initialized module`):
  the adapter imported the handler from the re-export module which imports
  the controllers, which import the adapter.  The type of the wrapped handler
  is a type-checking import now, so the cycle is gone.

#### Metadata of the indexed vectors
- `SearchEngineCore.index_collection` wrote the name of the document under
  `document_name` only, while `MilvusHandler` filtered on `filename`:
  a search restricted to document names returned no hit for the collections
  indexed by the new path.  Both paths now write `filename` **and**
  `document_name`, and the Milvus filter accepts either key
  (`metadata["filename"] in [...] or metadata["document_name"] in [...]`).
- The minimum length of an indexed chunk drops from 300 to **40 characters**
  (`MIN_CHUNK_CHAR_LEN` in `sse_api/engine/core/constants.py`), so short but
  essential entries — definitions, single legal articles, price-list
  positions — reach the index instead of being dropped.

## [0.1.8]

### Added

#### API — model service (embedding / reranking)
- `src/sse_models`: embedding and reranking as a separate process, so
  inference can run on another machine (a GPU box), scale per operation,
  and reload a model without restarting the engine. One service, two
  endpoints — `POST /v1/embed` (texts to vectors) and `POST /v1/rerank`
  (query plus candidates to relevance scores, answered best-first with the
  `index` of each document in the request restored) — plus `GET /v1/models`
  (the served catalog with load state), `GET /healthz` and `GET /readyz`.
  Models are addressed by name, never by path: a request sends
  `google/embeddinggemma-300m` and the service resolves it against
  `configs/embedders.json` / `configs/rerankers.json`; nothing can ask it
  to load an arbitrary directory. Roles (`all` / `embed` / `rerank`) let
  one host serve both operations or split them across hosts, and a
  per-model inference lock keeps a shared model from being asked twice at
  once.
- Run it with `./run.sh models` (venv by default, `--no-venv` to use the
  current interpreter; fails fast when a dependency of the service is
  missing) or `./run-models.sh` (tries the interpreters in turn until one
  can import the service, applies its flags after `.env`, so a flag always
  wins; `--host`, `--port`, `--role`, `--warm`, `--dry-run`, `--config-dir`,
  `--token`). Dependencies live in `requirements-models.txt` — no Django,
  no pymilvus.
- `ModelBackend` (`engine/core/models/backend.py`): the engine's embedding
  and reranking go through a small interface with two implementations —
  `LocalBackend`, the in-process models (the historical behaviour), and
  `RemoteBackend`, HTTP to the model service with a connect/read timeout
  split, a circuit breaker (`SSE_MODEL_SERVICE_FAILURE_THRESHOLD` +
  `SSE_MODEL_SERVICE_COOLDOWN_SECONDS`) and degrade-to-fallback: a dead
  service costs one timeout, not one per query, and a rejected model or
  token opens the circuit immediately. `build_backend()` is feature-flagged:
  `SSE_MODEL_SERVICE_ENABLED` off (the default) returns `LocalBackend`, and
  with it on the model name is resolved from the collection's path via the
  already-loaded config catalog, so there is no second mapping to maintain.
- `MilvusHandler` delegates `prepare_embeddings` and
  `_rerank_search_results` to the backend; with the service on, the eager
  model loads are skipped (the models live in the other process) and the
  `torch` / `sentence_transformers` imports move out of module level, so an
  engine that delegates inference never loads the model stack at import
  time.
- Configuration (`.env`, all optional, off by default): for the engine
  `SSE_MODEL_SERVICE_ENABLED`, `SSE_MODEL_SERVICE_URL`,
  `SSE_MODEL_SERVICE_TOKEN`, `SSE_MODEL_SERVICE_TIMEOUT_SECONDS`,
  `SSE_MODEL_SERVICE_CONNECT_TIMEOUT_SECONDS`,
  `SSE_MODEL_SERVICE_FALLBACK_LOCAL`, `SSE_MODEL_SERVICE_FAILURE_THRESHOLD`,
  `SSE_MODEL_SERVICE_COOLDOWN_SECONDS`; for the service process
  `SSE_MODELS_HOST`, `SSE_MODELS_PORT`, `SSE_MODELS_ROLE`,
  `SSE_MODELS_API_TOKEN`, `SSE_MODELS_WARM_MODELS`,
  `SSE_MODELS_RERANK_MAX_DOCUMENTS`, `SSE_MODELS_LOG_LEVEL`. Per-model
  placement is the `device` key of the config catalog
  (`cpu` / `cuda` / `cuda:1` ...; `CUDA_VISIBLE_DEVICES` remaps the
  indices, so `cuda:1` is the Nth GPU in that list, not GPU number N).

#### API — keyword filters
- `options.keywords`: a list of nonempty strings that replaces `query_str`
  **only in the PostgreSQL full-text leg**. Each entry is parsed as a plain
  `SearchQuery`, so `["pool timeout", "TypeError"]` means
  `(pool AND timeout) OR TypeError` after language-dependent tokenization —
  keywords are lexical terms, not exact-string or path filters. The vector
  branch, the reranker question and the RRF merge are unchanged: the
  keywords steer which candidates the FTS leg contributes, nothing else.
- `options.keywords_match`: `"and"` (omitted, the default) requires every
  keyword entry to match; `"any"` requires at least one. Any other value —
  `"all"`, `"or"`, `"AND"`, an empty string, `null` — is rejected by
  validation, even when hybrid search is off. The boolean match is enforced
  with the FTS `@@` operator before `SearchRank` scores the candidates,
  because a rank is a score, not a match constraint.
- `keywords` and `keywords_match` are persisted in
  `UserQuery.query_options` alongside `query_str`; no schema migration.

#### API — knowledge quality update
- **`POST update_knowledge_quality`** (`collection_name`, `document_name`,
  `upvotes`, `downvotes`) — owner-only update of the monotonic quality
  counters in a document's `metadata_json`: every request adds to the
  counters, never replaces them, and no arbitrary metadata is writable
  through the endpoint. The write is row-locked and transactional, so
  concurrent updates of the same document cannot lose each other, and the
  counters persist across re-indexing without any re-embedding.

#### API — lexical-only search
- A `search_with_options` request may now answer from the full-text leg
  alone: `"lexical_only": true` in `options` skips the embedding of the
  query and the vector store entirely and ranks the chunks with the
  Postgres full-text rank, which answers a literal question better than a
  cosine when it is literal — an exception class, an identifier, a config
  key. The answer reports `score_scale: "fts"`, the one scale the rest of
  the pipeline did not already use, and a reranker over a lexical search is
  a reranker: it overwrites the scale it is given.
- The option is a boolean of the search options like every other switch,
  so `"true"` that arrives as a string is not a string the controller has
  to guess at. It takes precedence over `hybrid_search`, because asking
  for both is asking to skip the vectors.

#### API — collections
- `GET collections` reports `model_embedder` and `model_reranker` of each
  collection — the models its vectors were built with, which is what a
  client needs to know before it moves the collection to another index.

#### `sse_lib`
- `SearchOptions.keywords` and `SearchOptions.keywords_match`: the client
  sends keyword filters the same way it sends any other search option, and
  says nothing when it did not set them.
- `SearchOptions.lexical_only`: the client asks for it the same way it
  asks for `hybrid_search`, and says nothing when it did not set it.

### Changed
- `retrieval_score_scale` knows the leg that ran: a lexical-only search
  reports `"fts"`, a hybrid one `"rrf"`, a plain vector search a cosine,
  and a reranker over any of them `"reranker"`.  The new scale is clamped
  to `[0, 1]` like a raw rank, because a Postgres `SearchRank` is already
  on that scale and there is nothing to squash.


## [0.1.7]

### Added

#### API — deletion of user chats
- **`POST delete_chat`** (also accepted as `DELETE`) — remove one or more chats
  of the authenticated user by their ids (`chat_id` and/or `chat_ids[]`)
  together with their messages and message states (`MessageState`,
  `RAGMessageState`, `ContentSupervisorState`).  The underlying search queries,
  responses and answers referenced by the RAG states are kept intact.  The
  answer reports how many rows were removed (`deleted`) and which of the
  requested ids were not found (`not_found`).  Foreign or missing chats return
  `CHAT_ID_NOT_FOUND` without leaking ownership.
- `ChatDeletionController` (`chat/controllers/deletion.py`) removes the chat
  rows and their states in the order which respects the `PROTECT` foreign keys,
  inside a single transaction.  Collection deletion (`_delete_collection_chats`)
  now uses the same helper so collections with non-empty chats are removed
  cleanly.

#### `sse_lib`
- `SSEClient.delete_chat(chat_ids=None, *, chat_id=None)` (alias
  `delete_chats`) wraps `POST delete_chat` (`endpoints.DELETE_CHAT`), validating
  identifiers client-side before sending the request.
- `ChatHistory` and `ChatMessage` expose `collection_name` and message `state`
  (`prev_message`, `next_message`) returned by `GET chats` and
  `GET get_chat_by_hash` (which now use `DeepMessageSerializer` so historical
  chat messages carry their associated RAG search results, answers and
  citations).


## [0.1.6]

### Added

#### RAG memory of the conversation
- The RAG answer now receives the previous questions **and** answers as
  `historical_messages` of the router call, bounded by
  `rag_history_max_turns` (default `8`, maximum `50`, whole turns only) and
  `rag_history_max_chars` (default `24000`, maximum `100000`).  Older turns stay
  in the database, but they are not automatically summarised.
- Follow-up questions are rewritten into standalone retrieval queries through
  the router instead of concatenating the old questions onto the current one.
  The rewriting model comes from `query_rewrite_model`, then
  `LLM_ROUTER_QUERY_REWRITE_MODEL`, and finally `generative_model`; the first
  question is not rewritten, and on any router failure the original question is
  searched.  Rewriting is on by default; an explicit `use_intelligent_query_rewrite: false`
  keeps the previous behaviour (the current question searched as-is).
- Sources are numbered per chat (`system_prompt_with_enumerated_documents`, on
  by default in the chat): every fragment gets its own number `[n]`, independent
  of the document name, and the number-to-fragment map — `n`, `text_id`,
  `document_id`, `page_number`, `char_start`, `char_end`, text — is returned in
  `citations`, stored with the answer in
  `sse_answer.generative_qa_result_json` and rehydrated from the database (not
  the client session) for the later turns.  The generator's reply is checked to
  reference only the known numbers.  A standalone `generative_answer` keeps the
  numbering off unless the option is set, and then numbers per answer.
- Short commands such as “zrób z tego tabelę”, “skróć” or “przetłumacz” (PL/EN)
  — or an explicit `history_only: true` — skip the search and reuse the
  previous query, response and sources; the reply reports
  `answer_source: "history"` with an empty `rewritten_query`.  Ambiguous
  commands, or a missing source context, fall back to the regular search.
  `answer_source` (`"rag"`, `"history"`, `"llm_fallback"`) is returned in
  `add_user_message`, stored with the answer, and exposed in
  `last_state.rag_message_state` (`answer_source`, `rewritten_query`,
  `citations`) together with the citations.
- A change of collection or search scope stops reusing turns from another
  scope; deleted, disabled or changed sources are not used again.  A template
  prompt now prefixes the user's `system_prompt` instead of replacing it.
- `add_user_message` runs in a single database transaction with the chat row
  locked, so concurrent messages of one chat do not interleave.

#### `POST rewrite_query`
- Rewrites a follow-up question into a standalone retrieval query through the
  router: `query_str`, `generative_model` and optional `historical_messages`
  (chronological `{user, assistant}` string turns, validated before the router
  is called).  Returns `rewritten_query`; a failure returns the standard
  generation error.  The endpoint does not search and does not store anything.

#### Chunk coordinates in the results and the preview
- Every search hit and citation carries the identity and the position of its
  chunk: `text_id`, `document_id` and `char_start`/`char_end` — zero-based
  Unicode code points with an exclusive end, relative to `pages[].content` of
  the relational preview (not byte offsets, not JavaScript UTF-16 indices).
  Missing locations are `null`.  The coordinates come from one query per batch
  (`data/controllers/chunk_offsets.py`); a stale, changed, empty or unmappable
  chunk gets `null` instead of a guessed position, and nothing is normalised to
  force a match.
- `GET document_content` accepts `text_id` (with the usual `collection_name`,
  optionally `document_id`/`document_name`/`chunk_type` which must match the
  chunk).  It resolves the chunk only inside the authorized collection, returns
  exactly the chunk's page and its stored chunk type — overriding
  `from_page`/`max_pages` — and adds
  `highlight: {text_id, page_number, char_start, char_end}`.  Without
  `text_id` the page window, the response shape and the pagination are
  unchanged; an unknown, deleted or out-of-scope chunk returns the existing
  not-found envelope.

### Added

#### `sse_lib`
- `SSEClient.rewrite_query(query, generative_model=..., historical_messages=...)`
  returns a standalone retrieval query; history turns are validated as
  `{user, assistant}` strings and the client language is forwarded.
- `SSEClient.generative_answer_with_citations(query_response_id, options, ...)`
  always enables the numbered sources (without mutating the passed options) and
  returns an `Answer` with `.citations`.
- `SSEClient.document_content` accepts `text_id` (the document argument becomes
  optional) and `DocumentPreview` exposes `.highlight`; `SearchHit` carries
  `text_id`, `document_id`, `char_start`/`char_end`.
- `ChatOptions` gains `user_msg_is_rag_question`,
  `system_prompt_with_enumerated_documents`, `query_rewrite_model`,
  `rag_history_max_turns`, `rag_history_max_chars` and `history_only`;
  `GenerativeOptions` gains `system_prompt_with_enumerated_documents`;
  `ChatReply` exposes `answer_source`, `rewritten_query` and `citations`.

### Changed

#### Dependencies
- The published `radlab-llm-router` is no longer in `requirements.txt`, and
  `initialize.sh` installs the router from the local `llm-router` checkout.
  The deployment needs a router (and `llm_router_lib`) build that provides
  `rewrite_query`, `generative_answer_with_citations` and
  `enumerated_documents` with `historical_messages` in the RAG payload; the
  quality of the attribution of a claim to a fragment still depends on the LLM.

## [0.1.5]

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

### Changed

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

#### Bundled Milvus
- The Docker image of the bundled Milvus moves from `v3.0.3` to `v3.0.5`
  (`MILVUS_IMAGE` in `run.sh`); the `milvus_data` volume is kept as before.

## [0.1.4]

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
