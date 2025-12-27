# Semantic Search Engine

*A lightweight, extensible toolkit for preparing, indexing, and querying textual data for semantic search and downstream
NLP tasks.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Directory Layout](#directory-layout)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
    - [Django Settings](#django-settings)
    - [AWS S3 Integration](#aws-s3-integration)
    - [Model Registry](#model-registry)
    - [Environment Variables](#environment-variables-summary)
7. [Data Ingestion & Preparation](#data-ingestion--preparation)
    - [Any‑Text‑to‑JSON Converter](#anytexttojson-converter)
    - [Doccano Export Converter](#doccano-export-converter)
    - [Upload & Index Pipeline](#upload--index-pipeline)
8. [Semantic Indexing (Milvus)](#semantic-indexing-milvus)
    - [Embedding & Reranking Models](#embedding--reranking-models)
    - [Indexing Scripts](#indexing-scripts)
9. [Search API](#search-api)
    - [REST Endpoints](#rest-endpoints)
    - [Search Options & Filters](#search-options--filters)
    - [Query Templates & Grammar](#query-templates--grammar)
10. [Conversational RAG (Chat) Service](#conversational-rag-chat-service)
    - [Chat Lifecycle](#chat-lifecycle)
    - [Content Supervisor](#content-supervisor)
    - [Generative Model Integration](#generative-model-integration)
11. [Management & Administrative Tools](#management--administrative-tools)
    - [Django Management Commands](#django-management-commands)
    - [Shell Scripts](#shell-scripts)
12. [Testing & Evaluation](#testing--evaluation)
13. [Extending the Engine](#extending-the-engine)
14. [Contribution Guidelines](#contribution-guidelines)
15. [License](#license)

---  

## Project Overview

The **Semantic Search Engine** provides a full‑stack solution for building semantic search applications:

| Layer                                    | Purpose                                                                                                                | Main Modules                                                            |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Data Ingestion**                       | Convert raw text or annotation exports into a uniform JSON/JSONL format, optionally chunk, clean, and detect language. | `any_text_to_json.py`, `doccano_converter.py`                           |
| **Storage**                              | Persist raw documents and metadata in PostgreSQL (relational) and Milvus (vector) databases.                           | `data.models`, `engine.models`, `semantic_search_engine/aws_handler.py` |
| **Embedding & Reranking**                | Compute dense vector representations and optionally re‑rank using cross‑encoders.                                      | `embedders_rerankers.py`, `engine.controllers.embedders_rerankers`      |
| **Search**                               | Retrieve nearest‑neighbor chunks with optional metadata filters, template‑based constraints, and pagination.           | `engine.controllers.search`, `data.controllers.relational_db`           |
| **RAG (Retrieval‑Augmented Generation)** | Combine retrieved snippets with LLMs (local or OpenAI) to generate context‑aware answers.                              | `chat.controllers`, `engine.controllers.models`                         |
| **API Layer**                            | Expose all functionality via a clean, versioned REST API built on Django Rest Framework.                               | `chat.api`, `data.api`, `engine.api`, `system.api`                      |
| **Administration**                       | User, organisation, and group management, plus collection lifecycle utilities.                                         | `system.controllers`, `system.models`                                   |

The repository is deliberately modular: you can use the data preparation scripts standalone, or run the full
Django‑powered service for end‑to‑end pipelines.

---  

## Architecture Diagram

```
+----------------------+      +---------------------+      +-------------------+
|  Raw Text / Doccano  | ---> |  Ingestion Scripts  | ---> |  PostgreSQL (RDB) |
+----------------------+      +---------------------+      +-------------------+
                                   |                               |
                                   v                               v
                           +-------------------+          +-------------------+
                           |  Chunking /       |          |  Metadata &      |
                           |  Language Detect  |          |  Relations        |
                           +-------------------+          +-------------------+
                                   |                               |
                                   v                               v
                           +-------------------+          +-------------------+
                           |  JSON/JSONL Files |          |  Document Models  |
                           +-------------------+          +-------------------+
                                   |                               |
                                   +---------------+---------------+
                                                   |
                                                   v
                                         +-------------------+
                                         |   Milvus (Vector) |
                                         +-------------------+
                                                   |
                                                   v
                                         +-------------------+
                                         |   Search Service  |
                                         +-------------------+
                                                   |
                                                   v
                                         +-------------------+
                                         |   RAG (LLM)       |
                                         +-------------------+
                                                   |
                                                   v
                                         +-------------------+
                                         |   REST API (Django)|
                                         +-------------------+
```

---  

## Directory Layout

```
semantic-search-engine/
├── apps_sse/                     # Core apps for SSE (semantic search engine)
│   ├── admin/                    # Management scripts (add org, users, etc.)
│   ├── dataset/                  # Generators for synthetic QA datasets
│   ├── evaluator/                # Test harnesses and evaluation utilities
│   ├── installed/                # Bundled third‑party utilities (denoiser, converters)
│   ├── add_files_from_dir.py      # CLI for uploading & indexing a directory
│   ├── index_collection_to_milvus.py
│   ├── index_to_milvus.sh
│   ├── index_to_postgresql.sh
│   └── semantic_search_app.py     # Interactive REPL for ad‑hoc queries
├── chat/                         # Conversational (RAG) service
│   ├── models.py
│   ├── controllers.py
│   ├── api.py
│   └── urls.py
├── data/                         # Document, collection, and query‑template models
│   ├── models.py
│   ├── controllers.py
│   ├── api.py
│   └── urls.py
├── engine/                       # Search & generative model orchestration
│   ├── models.py
│   ├── controllers/
│   │   ├── search.py
│   │   └── models.py
│   ├── api.py
│   └── urls.py
├── system/                       # Organisation & authentication
│   ├── models.py
│   ├── controllers.py
│   ├── api.py
│   └── urls.py
├── main/                         # Django project entry point
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── semantic_search_engine/        # Shared utilities (AWS handler, constants)
│   ├── __init__.py
│   ├── aws_handler.py
│   ├── constants.py
│   └── models.json                # Model metadata registry
├── doccano_converter.py          # Doccano → dataset conversion
├── any_text_to_json.py           # Directory → JSON/JSONL conversion
├── manage.py                     # Django management script
├── requirements.txt
├── README.md                     # <-- you are reading this file!
└── scripts/                      # Optional helper scripts for CI / deployment
```

---  

## Prerequisites

| Component                     | Minimum Version       | Why it matters                                                                                   |
|-------------------------------|-----------------------|--------------------------------------------------------------------------------------------------|
| **Python**                    | 3.11 (rc1 acceptable) | All scripts and Django run on 3.11+.                                                             |
| **pip**                       | latest                | To install dependencies from `requirements.txt`.                                                 |
| **PostgreSQL**                | 12+                   | Relational store for documents, users, collections.                                              |
| **Milvus**                    | 2.3+                  | Vector database for semantic embeddings.                                                         |
| **Django**                    | 4.x                   | Core web framework for the API layer.                                                            |
| **boto3**                     | any                   | Required only if you enable AWS S3 storage.                                                      |
| **radlab_data** package       | ≥ 0.2.0               | Provides `DirectoryFileReader`, `TextUtils`, and other utilities used by the conversion scripts. |
| **CUDA (optional)**           | 11+                   | If you want to run embeddings / LLMs on GPU.                                                     |
| **OpenAI API key** (optional) | –                     | Enables the OpenAI generative backend.                                                           |
| **Deepl API key** (optional)  | –                     | Enables answer translation.                                                                      |

All third‑party Python dependencies are listed in `requirements.txt`.

---  

## Installation

### 1. Clone the repository

```shell script
git clone https://github.com/radlab-dev-group/semantic-search-engine.git
cd semantic-search-engine
```

### 2. Create a virtual environment (highly recommended)

```shell script
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```shell script
pip install -r requirements.txt
```

> **Tip:** If you encounter a `pip` version conflict with system packages, set `export PIP_BREAK_SYSTEM_PACKAGES=1`
> before installing (as done in `full-install.sh`).

### 4. Set up PostgreSQL

Create a database and a user (example for a local dev setup):

```sql
CREATE
DATABASE semantic_search;
CREATE
USER sse_user WITH PASSWORD 'sse_pass';
GRANT ALL PRIVILEGES ON DATABASE
semantic_search TO sse_user;
```

### 5. Run Django migrations

```shell script
python manage.py migrate
```

### 6. (Optional) Install Milvus locally

If you don’t have a remote Milvus cluster, you can run a Docker container:

```shell script
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 19121:19121 \
  milvusdb/milvus:2.3.0
```

Make sure the Milvus connection details match the values in `configs/milvus_config.json`.

---  

## Configuration

Configuration files live under the `configs/` directory. The most important ones are:

| File                             | Description                                                              |
|----------------------------------|--------------------------------------------------------------------------|
| `configs/milvus_config.json`     | Milvus connection, index settings, and embedding model references.       |
| `configs/aws_config.json`        | S3 credentials and bucket name (used by `AwsHandler`).                   |
| `configs/embedders.json`         | Registry of available embedding models (paths, vector size, device).     |
| `configs/rerankers.json`         | Registry of cross‑encoder reranker models.                               |
| `configs/generative-models.json` | Mapping of generative model names to API endpoints (local or OpenAI).    |
| `configs/query-templates.json`   | Definition of query‑template grammars and per‑template metadata filters. |

### Django Settings

`main/settings.py` loads configuration through the `SystemSettingsHandler` (see `main/src/settings.py`). The handler
reads values from:

1. **JSON config files** (`django-config.json` by default).
2. **Environment variables** (prefixed with `ENV_`).

You can override any setting by exporting the corresponding environment variable. For example:

```shell script
export ENV_DEBUG=true
export ENV_DB_HOST=localhost
export ENV_USE_AWS=1
export ENV_USE_KC_AUTH=0
```

#### Key Django Settings

| Setting         | Source                                | Default                 |
|-----------------|---------------------------------------|-------------------------|
| `DEBUG`         | `ENV_DEBUG` or `django-config.json`   | `false`                 |
| `DATABASES`     | `ENV_DB_*` or `django-config.json`    | PostgreSQL config       |
| `MAIN_API_URL`  | Computed from `api` section in config | `/api/`                 |
| `ALLOWED_HOSTS` | `ENV_ALLOWED_HOSTS` or config         | `[]`                    |
| `MAIN_LOGGER`   | Configured via `logger` section       | Python `logging` module |

### AWS S3 Integration

The `AwsHandler` expects a JSON file (default: `configs/aws_config.json`). Example:

```json
{
  "aws": {
    "REGION_NAME": "us-east-1",
    "ENDPOINT_URL": "https://s3.amazonaws.com",
    "ACCESS_KEY_ID": "<your-access-key>",
    "SECRET_ACCESS_KEY": "<your-secret>",
    "STORAGE_BUCKET_NAME": "my-semsearch-bucket"
  }
}
```

All required fields are validated on `AwsHandler` initialization; missing values raise an `AssertionError`.

### Model Registry (`models.json`)

`semantic_search_engine/models.json` holds model metadata used by the embedding and reranker loaders. A minimal entry
looks like:

```json
{
  "denoiser": {
    "model_name": "radlab/polish-denoiser-t5-base",
    "model_path": "/opt/models/denoiser/t5-base",
    "device": "cuda:0"
  }
}
```

You can add new models by extending the JSON and updating `embedders.json` / `rerankers.json` accordingly.

### Environment Variables (summary)

| Variable                                                                                                | Meaning                                                     |
|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `ENV_DEBUG`                                                                                             | Turn Django debug mode on/off.                              |
| `ENV_DB_NAME` / `ENV_DB_HOST` / `ENV_DB_PORT` / `ENV_DB_ENGINE` / `ENV_DB_USERNAME` / `ENV_DB_PASSWORD` | Override PostgreSQL connection.                             |
| `ENV_USE_AWS`                                                                                           | Enable the `AwsHandler` S3 wrapper.                         |
| `ENV_USE_KC_AUTH`                                                                                       | Use Keycloak authentication (requires `rdl_authorization`). |
| `ENV_USE_OAUTH_V1_AUTH` / `ENV_USE_OAUTH_V2_AUTH`                                                       | Use OAuth v1/v2 authentication.                             |
| `ENV_USE_INTROSPECT`                                                                                    | Enable token introspection for OAuth.                       |
| `ENV_PUBLIC_API_AVAILABLE`                                                                              | Expose public API endpoints without authentication.         |
| `ENV_CELERY_BROKER_URL`                                                                                 | URL of the Celery broker (RabbitMQ, Redis, etc.).           |
| `ENV_DEFAULT_LANGUAGE`                                                                                  | Default language for API responses (`pl` or `en`).          |
| `OPENAI_API_KEY`                                                                                        | API key for OpenAI model calls.                             |
| `DEEPL_AUTH_KEY`                                                                                        | API key for DeepL translation service.                      |

---  

## Data Ingestion & Preparation

The engine expects **documents** to be stored in a normalized JSON/JSONL format. Two primary CLI utilities help you get
there.

### Any‑Text‑to‑JSON Converter

`any_text_to_json.py` walks a directory tree, reads plain‑text files, and emits a single JSON document (or JSONL)
containing:

- `filepath`, `relative_filepath`
- `category` (derived from folder hierarchy)
- `pages` → list of `{page_number, page_content, metadata}`
- Optional language detection, chunking, overlapping tokens, and denoising.

#### Basic usage

```shell script
python any_text_to_json.py \
  -d /path/to/raw_texts \
  -o /tmp/dataset.json
```

#### Advanced options

| Flag                                | Description                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------|
| `--proper-pages`                    | Merge all texts belonging to the same logical page before chunking.                             |
| `--merge-document-pages`            | Collapse an entire document into a single page (useful for short docs).                         |
| `--clear-texts`                     | Run the radlab‑data denoiser on each chunk.                                                     |
| `--split-to-max-tokens-in-chunk N`  | Limit each chunk to *N* tokens (uses the tokenizer of the chosen embedder).                     |
| `--overlap-tokens-between-chunks M` | Overlap *M* tokens between successive chunks (must accompany `--split-to-max-tokens-in-chunk`). |
| `--check-text-language`             | Detect language with `fasttext-langdetect` and store it in metadata.                            |
| `--processes K`                     | Parallelise the pipeline with *K* worker processes.                                             |

Output format is automatically chosen: if the output file ends with `.json`, a single JSON object containing a
`"documents"` list is written; otherwise a JSONL stream is produced (one document per line).

### Doccano Export Converter

`doccano_converter.py` transforms Doccano annotation exports (JSONL) into datasets ready for **sequence classification**
or **token classification** training.

#### Sequence Classification Example

```shell script
python doccano_converter.py \
  -I /data/doccano/seq_export \
  -e .jsonl \
  --show-class-labels-histogram \
  -O ./prepared_datasets/seq/20231208 \
  --sequence-classification
```

#### Token Classification Example

```shell script
python doccano_converter.py \
  -I /data/doccano/token_export \
  -e .jsonl \
  --show-class-labels-histogram \
  -O ./prepared_datasets/token/20231208 \
  --token-classification
```

Key flags:

| Flag                            | Meaning                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| `-I / --input-dir`              | Directory with Doccano export files.                                     |
| `-e / --dataset-extension`      | File extension of source files (`.jsonl` default).                       |
| `-O / --output-dir`             | Destination directory for the prepared dataset.                          |
| `--sequence-classification`     | Produce a dataset for text‑level classification.                         |
| `--token-classification`        | Produce a dataset for token‑level labeling.                              |
| `--show-class-labels-histogram` | Print a histogram of class label frequencies (useful for sanity checks). |
| `--save-excel-annotations`      | Export label statistics to an Excel file.                                |
| `--save-iob-standard`           | Save the token‑level dataset in IOB format.                              |
| `--mapping-file`                | Optional CSV mapping from source label names to canonical names.         |

### Upload & Index Pipeline

The high‑level CLI `add_files_from_dir.py` bundles the previous steps: it uploads a directory to the **relational DB**,
runs optional preprocessing, and then triggers **semantic indexing** into Milvus.

```shell script
python add_files_from_dir.py \
  -d /path/to/texts \
  -c MyCollection \
  --clear-texts \
  --merge-document-pages \
  --split-to-max-tokens-in-chunk=200 \
  --overlap-tokens-between-chunks=20 \
  --processes=10
```

Internally the command:

1. Creates a temporary upload folder (or uses an existing `UploadedDocuments` record).
2. Calls `DirectoryFileReader` → `RelationalDBController.add_uploaded_documents_to_db`.
3. Instantiates `DBSemanticSearchController` for the target collection and indexes all resulting `DocumentPageText`
   objects.

---  

## Semantic Indexing (Milvus)

### Embedding & Reranking Models

The system supports **dynamic model loading** via the JSON registries in `configs/embedders.json` and
`configs/rerankers.json`. Each entry contains:

```json
{
  "name": "sentence-transformers/all-MiniLM-L6-v2",
  "path": "/opt/models/all-MiniLM-L6-v2",
  "vector_size": 384,
  "device": "cuda:0"
}
```

- **Embedders** produce dense vectors for each text chunk.
- **Rerankers** (cross‑encoders) optionally re‑score the top‑K retrieved vectors using a second model, improving
  relevance.

### Indexing Scripts

| Script                          | Description                                                                                     |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| `index_collection_to_milvus.py` | Indexes all `DocumentPageText` objects from a relational collection into a Milvus collection.   |
| `index_to_milvus.sh`            | Convenience wrapper that sets the index type (`HNSW` or `IVF_FLAT`) and runs the Python script. |
| `index_to_postgresql.sh`        | Uses `add_files_from_dir.py` to ingest raw files and then indexes them.                         |

All scripts accept the following arguments (common to both):

- `--from-collection` – name of the **relational** collection containing the raw chunks.
- `--to-collection` – name of the **Milvus** collection that will be created.
- `--chunk-type` – the `text_chunk_type` field used to differentiate between plain, cleaned, or overlapped chunks.
- `--index-name` – one of the supported Milvus index types (`HNSW`, `IVF_FLAT`).

> **Note:** The Milvus handler automatically creates the collection if it does not exist, using the embedding dimensions
> defined in the model registry.

---  

## Search API

The REST API follows a **versioned** pattern (`/api/vX.Y/…`). All endpoints are defined in the Django `urls.py` files of
each app.

### REST Endpoints

| Method | URL (example)                  | Description                                                                     |
|--------|--------------------------------|---------------------------------------------------------------------------------|
| `POST` | `/api/new_collection/`         | Create a new document collection (specify embedder, reranker, index type).      |
| `GET`  | `/api/collections/`            | List all collections visible to the authenticated user.                         |
| `POST` | `/api/upload_and_index_files/` | Upload files (multipart) and trigger indexing.                                  |
| `POST` | `/api/search_with_options/`    | Perform a semantic search with rich filtering options.                          |
| `POST` | `/api/generative_answer/`      | Generate a RAG answer for a previously stored query response.                   |
| `GET`  | `/api/question_templates/`     | List available query‑template collections.                                      |
| `GET`  | `/api/filter_options/`         | Retrieve mock filter metadata (URLs, categories) – useful for UI drop‑downs.    |
| `GET`  | `/api/categories/`             | List distinct categories present in a collection.                               |
| `GET`  | `/api/documents/`              | List document names within a collection (used for UI selectors).                |
| `GET`  | `/api/chats/`                  | Retrieve all chats for the current user.                                        |
| `POST` | `/api/add_user_message/`       | Append a user message to a chat, optionally run RAG and return assistant reply. |
| `POST` | `/api/save_chat/`              | Mark a chat as saved and optionally set it to read‑only.                        |
| `GET`  | `/api/get_chat_by_hash/`       | Retrieve a saved chat by its hash (read‑only).                                  |

All endpoints require an **Authorization** header (`Token <jwt>`), unless `ENV_PUBLIC_API_AVAILABLE` is set.

### Search Options & Filters

The `search_with_options` endpoint expects a JSON payload with:

```json
{
  "collection_name": "MyCollection",
  "query_str": "Jakie są najnowsze regulacje podatkowe?",
  "options": {
    "categories": [
      "Prawo",
      "Finanse"
    ],
    "documents": [
      "regulacje_2023.pdf"
    ],
    "relative_paths": [
      "/2023/"
    ],
    "templates": [
      1,
      5
    ],
    "only_template_documents": false,
    "max_results": 40,
    "rerank_results": true,
    "return_with_factored_fields": false,
    "relative_path_contains": [
      "https://gov.pl"
    ],
    "metadata_filters": [
      {
        "operator": "eq",
        "field": {
          "metadata_json.type": "regulation"
        }
      }
    ],
    "use_and_operator": true
  }
}
```

Key concepts:

- **Categories** – map to the `category` field of `Document`.
- **Templates** – IDs of `QueryTemplate` objects; the system will filter documents that satisfy the template’s
  `data_filter_expressions`.
- **Metadata filters** – low‑level JSON‑logic style filters (`in`, `eq`, `gt`, `hse`, …) applied directly on the
  `metadata_json` column.
- **`use_and_operator`** – determines whether multiple filter lists are intersected (`true`) or unioned (`false`).

The response contains:

- `stats` – aggregated per‑document hit counts, scores, and page statistics.
- `detailed_results` – list of matching chunks with left/right context.
- `structured_results` – optional structured output generated by a template that requests it.
- `template_prompts` – any system prompts extracted from the used templates (useful for LLM instruction).

### Query Templates & Grammar

Query templates are defined in `configs/query-templates.json`. They consist of:

- **Grammar** – token set and alphabet used to generate natural‑language queries.
- **Data connector** – static key‑value pairs that must match `Document.metadata_json`.
- **Data filter expressions** – dynamic Python expressions evaluated against document metadata (e.g., date range
  checks).
- **Structured response flag** – when `true`, the engine extracts specified fields from the document metadata and
  returns them alongside the text snippets.

The `QueryTemplateController` automatically validates a document against its grammar (`use_document_in_sse`) and filter
expressions (`use_document_in_sse`), ensuring only relevant documents participate in the search.

---  

## Conversational RAG (Chat) Service

The **Chat** app adds a conversational layer on top of the semantic search engine. It stores chat history, supports
content supervision, and integrates with generative models.

### Chat Lifecycle

1. **Create a new chat** – `POST /api/new_chat/` (optional collection and search options).
2. **Send a user message** – `POST /api/add_user_message/` with `chat_id`, `user_message`, and generation options.
3. **System processes the message**:
    - Detects whether the message is a RAG query (`user_msg_is_rag_question` flag).
    - Optionally extracts an explicit question (`pytanie:`) and/or instruction.
    - Runs the **semantic search** using the collection linked to the chat (or the global default).
    - If `use_rag_supervisor` is enabled, the system creates a `RAGMessageState` linking the query, response, and
      generated answer.
    - If `use_content_supervisor` is enabled, URLs in the user text are fetched and their content added to the prompt.
4. **Generate assistant response** – either via a local LLM endpoint or OpenAI, with optional translation via DeepL.
5. **Persist messages** – `Message` objects store role (`user`, `assistant`, `system`), text, timestamps, and optional
   `MessageState`.

All chats are tied to an `OrganisationUser` for permission checks. The `read_only` flag prevents further message
addition.

### Content Supervisor

Implemented in `ChatController._prepare_content_supervisor_state`. It uses **radlab_content_supervisor** to:

- Detect URLs (`URLRegexProcessor`).
- Fetch remote content (HTML) and store it in `ContentSupervisorState.www_content`.
- Append fetched snippets to the user message before sending it to the LLM.

This feature is useful for **dynamic knowledge retrieval** (e.g., fetching the latest news article).

### Generative Model Integration

The `GenerativeModelController` orchestrates calls to:

- **Local model APIs** – defined in `configs/generative-models.json` with custom endpoints (`/generate`).
- **OpenAI** – via `OpenAIGenerativeController`.

Both paths support:

- **Generation options** (`top_k`, `temperature`, etc.).
- **System prompt** – static instruction that can be injected from a query template.
- **Answer translation** – using Deepl if `translate_answer` is true and a key is present.

Generated answers are stored in `UserQueryResponseAnswer` and can be rated via the `SetRateForQueryResponseAnswer`
endpoint.

---  

## Management & Administrative Tools

### Django Management Commands

All standard Django commands are available (`runserver`, `migrate`, `createsuperuser`).  
Additional project‑specific commands are provided by the `apps_sse` scripts:

| Command                                   | Description                                                                                    |
|-------------------------------------------|------------------------------------------------------------------------------------------------|
| `python manage.py runserver 0.0.0.0:8271` | Starts the development server (default port 8271 in `run-api.sh`).                             |
| `python manage.py migrate`                | Applies database migrations (including the generated migration files under `chat/migrations`). |
| `python manage.py createsuperuser`        | Creates a Django superuser for the admin UI (if `ENV_SHOW_ADMIN_WINDOW` is true).              |

### Shell Scripts

- **`full-install.sh`** – End‑to‑end installer that sets up the virtual environment, installs required packages, creates
  the database, runs migrations, prepares the Milvus collection, adds a default organisation/user, and loads query
  templates.
- **`run-api.sh`** – Convenience wrapper that sets common environment variables (CUDA device, data upload limits, API
  keys) and starts the Django dev server.
- **`index_to_milvus.sh`** & **`index_to_postgresql.sh`** – Quick wrappers around the Python indexing scripts for ad‑hoc
  runs.

All scripts are idempotent where possible; they log progress to stdout and rely on the underlying Python modules for
error handling.

---  

## Testing & Evaluation

The **evaluator** package (`apps_sse/evaluator`) provides utilities for:

- **Loading test configurations** (JSON files defining test cases).
- **Running semantic search** with various parameters (max results, rerank, etc.).
- **Generating answers** using configured generative models.
- **Computing metrics** (BLEU, ROUGE) via `torchmetrics`.

The main entry point is `tests_loader.py`. It writes results to Excel (`.xlsx`) files for downstream analysis. Example
usage:

```shell script
python apps_sse/evaluator/src/tests_loader.py \
  -u alice \
  -c ./tests/config.json \
  -o results.xlsx \
  -p results_chunks.xlsx
```

The output contains:

- **Aggregated statistics** per collection, test case, and model.
- **Human‑annotated answer comparisons** (precision/recall style).
- **Per‑chunk detailed results** for error analysis.

---  

## Extending the Engine

The codebase is deliberately **plug‑and‑play**:

1. **Add a new embedding model** – Append an entry to `configs/embedders.json` and place the model files where
   `model_path` points. No code changes required.
2. **Add a new reranker** – Same procedure with `configs/rerankers.json`.
3. **Create a custom query template** – Edit `configs/query-templates.json` (or add a new file and load it via
   `QueryTemplatesLoaderController`). Define `data_filter_expressions` using Python syntax; the engine evaluates them
   safely with `eval`.
4. **Expose a new API endpoint** – Create a view inheriting from `APIView`, add the route to the appropriate `urls.py`,
   and use existing decorators (`required_params_exists`, `get_organisation_user`).
5. **Swap the LLM backend** – Implement a new subclass of `GenerativeModelControllerApi.LocalModelAPI` with custom
   request payload/response handling, then reference the new model name in `configs/generative-models.json`.

All extensions respect the existing permission model (token‑based authentication) and will automatically appear in the
OpenAPI/Swagger UI if `django-rest-swagger` is enabled.

---  

## Contribution Guidelines

1. **Fork the repository** and create a feature branch (`git checkout -b feature/your‑idea`).
2. **Write tests** for any new functionality (use the existing `apps_sse/evaluator` test harness as a reference).
3. **Run the full test suite**: `pytest` (install via `pip install pytest`).
4. **Format code** with `black` and lint with `flake8`.
5. **Update documentation** – add a new section to this README, or improve docstrings.
6. **Submit a Pull Request** targeting the `main` branch. Ensure CI passes (if configured).

---  

## License

This project is licensed under the **Apache 2.0 License** – see the `LICENSE` file for full details.

---  

*Happy coding! 🚀*  