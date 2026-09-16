# Project Dependencies

This document lists all dependencies for the **Semantic Search Engine (SSE)** — a Django-based platform combining semantic vector search, generative AI (RAG pipelines), and content supervision for context-aware information retrieval.

---

## Python Package Dependencies

### Core Framework

| Package | Purpose |
|---------|---------|
| `django` | Web framework |
| `djangorestframework` | REST API layer |
| `django-rest-swagger` | API documentation |
| `django-filter` | Query filtering |

### Authentication

| Package | Purpose |
|---------|---------|
| `pyjwt` | JWT token handling |

Local module `sse_rest_api/authorization/` provides Keycloak and OAuth v1/v2 authentication support.

### Machine Learning / NLP

| Package | Purpose |
|---------|---------|
| `transformers` | Hugging Face model pipelines (tokenizers, T5, cross-encoders) |
| `torch` | PyTorch deep learning framework |
| `sentence_transformers` | Sentence embeddings and cross-encoding |
| `sentencepiece` | Tokenizer for Google models |
| `spacy` | NLP processing |
| `fasttext-langdetect` | Language detection |
| `openai` | OpenAI API client (cloud LLMs) |
| `huggingface_hub` | Hugging Face model downloads |

### Database

| Package | Purpose |
|---------|---------|
| `pymilvus` | Milvus vector database client |
| `psycopg2-binary` | PostgreSQL adapter |

### Document Processing

| Package | Purpose |
|---------|---------|
| `markdown` | Markdown parsing |
| `python-docx` | Word document handling |
| `pypdf` | PDF parsing |
| `xlsxwriter` | Excel export |

### Utilities

| Package | Purpose |
|---------|---------|
| `gunicorn` | WSGI production server |
| `deepl` | DeepL translation API client |
| `Levenshtein` | String edit distance |
| `pybind11` | C++ binding (build dependency) |

### Dynamically Installed Packages

Installed at deployment via `sse_rest_api/initialize.sh`:

| Package | Source | Purpose |
|---------|--------|---------|
| `radlab-data` | GitHub (`radlab-dev-group/radlab-data`) | Internal preprocessing pipeline and text utilities |
| `llm-router` | GitHub (`radlab-dev-group/llm-router`) | LLM routing client for local model API |

---

## Infrastructure Dependencies (Required)

### PostgreSQL — Relational Database

- **Host:** `192.168.100.67`
- **Port:** `5457`
- **Database:** `sse_backend`
- **Config:** `sse_rest_api/configs/django-config.json` (`database` section)

Used for relational data storage and **Full-Text Search (FTS)**. The system utilizes `django.contrib.postgres` to perform weighted searches (`SearchVector`, `SearchRank`) over document text, which are then merged with vector results via RRF.

### Milvus — Vector Database

- **Host:** `192.168.100.67`
- **Port:** `19530`
- **Database:** `sse_backend`
- **Config:** `sse_rest_api/configs/milvus_config.json`

Used for semantic vector search over indexed document collections. Supports IVF_FLAT and HNSW indexing.

### LLM Router API — Generative Model Endpoint

- **Host:** `192.168.100.65:8080`
- **Config:** `sse_rest_api/configs/generative-models.json`

Provides 7 generative models via HTTP POST:

| Model | Type |
|-------|------|
| `gpt-oss:120b` | Large language model |
| `google/gemma-3-12b-it` | Instruction-tuned LLM |
| `google/gemini-2.5-flash-lite` | Fast inference model |
| `qwen3-coder:30b` | Code-focused LLM |
| `speakleash/Bielik-11B-v2.3-Instruct` | Polish instruction-tuned LLM |
| `dolphin-mistral` | Mistral-based model |
| `dolphin3:8b` | Lightweight Dolphin model |

---

## Infrastructure Dependencies (Optional)

### Keycloak — Authentication / SSO

- **Host:** `https://login.radlab.dev`
- **Config:** `sse_rest_api/configs/auth-config.json`
- **Enable via:** `ENV_USE_KC_AUTH=1`
- **Grant type:** `authorization_code`

When enabled, replaces default token-based authentication with Keycloak SSO.

### AWS S3 — Object Storage

- **Region:** `eu-central-1`
- **Config file:** `sse_rest_api/configs/aws_config.json` (must be created at deployment)
- **Enable via:** `ENV_USE_AWS=true`
- **Library:** `boto3`

Provides document object storage. Disabled by default.

### DeepL API — Translation Service

- **Endpoint:** Cloud DeepL service
- **Config:** Environment variable `DEEPL_AUTH_KEY`
- **Library:** `deepl`

Translates generated answers to Polish. Enabled when `DEEPL_AUTH_KEY` is set and the query parameter `translate_answer=true`.

### Celery / RabbitMQ — Task Queue

- **Config:** `sse_rest_api/configs/django-config.json` (`broker_url`)
- **Enable via:** `ENV_USE_CELERY=1`

Handles async background tasks (document ingestion, indexing). Disabled by default.

---

## ML Models (Local Files)

Models are loaded from `/mnt/data2/llms/models/radlab-open/` and defined in config files:

### Embedders (`sse_rest_api/configs/embedders.json`)

| Model | Dimensions | Target |
|-------|-----------|--------|
| `radlab/article-bi-encoder-20240901` | 1024 | cuda:0 |
| `radlab/polish-bi-encoder-mean` | 1024 | cuda:0 |
| `ipipan/silver-retriever-base-v1.1` | 768 | cuda:0 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | CPU |
| `google/embeddinggemma-300m` | 768 | cuda:0 |

### Rerankers (`sse_rest_api/configs/rerankers.json`)

| Model | Dimensions | Target |
|-------|-----------|--------|
| `radlab/polish-cross-encoder` | 1024 | cuda:0 |

### Denoiser (`sse_rest_api/configs/models.json`)

| Model | Target |
|-------|--------|
| `radlab/polish-denoiser-t5-base` | cuda:0 |

---

## Docker Infrastructure

Used for local development of PostgreSQL and Milvus services:

- **PostgreSQL:** `scripts/admin/postgres.sh` — `docker run` with credentials from config
- **Milvus:** `milvusdb/milvus:2.3.0` — standalone container, ports 19530/19121

---

## Additional Directories

| Directory | Purpose |
|-----------|---------|
| `mcp/` | Model Context Protocol server implementations (FastMCP servers and Ollama client) |
| `sse_apps/` | Standalone admin utilities, evaluation scripts (BLEU, ROUGE scoring), and document converters |
| `scripts/` | Shell utility scripts for infrastructure management |
