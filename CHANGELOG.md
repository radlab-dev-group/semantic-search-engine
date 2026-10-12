# Changelog

All notable changes to the Semantic Search Engine are documented in this file.

| Version | Changelog |
|---------|-----------|
| [0.1.0](#010) | Python client library, hybrid search, production deployment tooling, `src/` layout and CI |
| [0.0.1](#001) | Initial version |

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
