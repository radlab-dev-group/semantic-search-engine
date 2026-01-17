# Semantic Search Engine (SSE)

## Overview

The **Semantic Search Engine** (SSE) is a modular, Django‑based platform that combines **semantic vector search**, *
*generative AI**, and **content supervision** to provide powerful, context‑aware information retrieval.  
It supports:

- Indexing documents (PDF, text, etc.) into a Milvus vector store.
- Rich metadata‑driven filtering and query‑template matching.
- RAG (Retrieval‑Augmented Generation) pipelines with locally‑hosted LLMs or OpenAI models.
- Multi‑user, organisation‑aware access control.

## Architecture & Packages

| Package    | Purpose                                                                                                            | Key Modules                                                             |
|------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **chat**   | Conversational interface on top of the search engine. Handles chat sessions, message state, and RAG orchestration. | `controllers.py`, `api.py`, `models.py`, `serializer.py`                |
| **data**   | Ingestion pipeline, relational DB models, and utilities for preparing documents before semantic indexing.          | `relational_db.py`, `semantic_db.py`, `query_templates.py`, `upload.py` |
| **engine** | Core search, embedding, reranking, and generative answer logic.                                                    | `search.py`, `models.py`, `embedders_rerankers.py`, `system.py`         |
| **system** | Organisation, group, and user management with token‑based authentication.                                          | `models.py`, `controllers.py`, `api.py`                                 |
| **main**   | Global configuration, settings handling, error utilities, and Django entry points.                                 | `settings.py`, `constants.py`, `decorators.py`, `response.py`           |

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd semantic-search-engine
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (see *Configuration* below)
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export DEEPL_AUTH_KEY=your_deepl_key
   export ENV_USE_KC_AUTH=0   # set to 1 to enable Keycloak auth
   ```

4. **Run migrations**
   ```bash
   ./initialize.sh migrate
   ```

5. **Start the server**
   ```bash
   ./run-api.sh
   ```

   The API will be available at `http://localhost:8271/api/...`.

## Configuration

All runtime settings are defined in `configs/django-config.json` (or overridden via environment variables).  
Key sections:

- **Database** – PostgreSQL connection details.
- **Milvus** – Vector store connection (`milvus_config.json`).
- **Models** – Embedders, rerankers, and generative models (`embedders.json`, `rerankers.json`,
  `generative-models.json`).
- **Authentication** – Enable Keycloak/OAuth via `ENV_USE_KC_AUTH`, `ENV_USE_OAUTH_V1_AUTH`, `ENV_USE_OAUTH_V2_AUTH`.
- **Logging** – Choose between `dev`, `demo`, `prod` loggers in `django-config.json`.

Refer to each package’s README (e.g., `chat/README.md`, `data/README.md`) for detailed settings.

## Typical Workflows

### 1. Create a Collection & Index Documents

``` bash
# Create a new collection (POST /api/collections/)
curl -X POST http://localhost:8271/api/collections/ \
  -H "Authorization: Token <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "my_docs",
        "collection_display_name": "My Documents",
        "collection_description": "Test collection",
        "model_embedder": "radlab/polish-bi-encoder-mean",
        "model_reranker": "radlab/polish-cross-encoder",
        "embedder_index_type": "HNSW"
      }'
```

Upload files (supports ZIP extraction, optional denoising, chunking):

``` bash
curl -X POST http://localhost:8271/api/upload_and_index_files/ \
  -H "Authorization: Token <your_token>" \
  -F "files[]=@/path/to/file.pdf" \
  -F "collection_name=my_docs" \
  -F "indexing_options={\"prepare_proper_pages\":true,\"clear_text\":true}"
```

### 2. Search with Options

``` bash
curl -X POST http://localhost:8271/api/search_with_options/ \
  -H "Authorization: Token <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "my_docs",
        "query_str": "What is the policy on data retention?",
        "options": {
          "categories": ["Law"],
          "max_results": 20,
          "rerank_results": true
        }
      }'
```

### 3. Generate an Answer (RAG)

``` bash
curl -X POST http://localhost:8271/api/generative_answer/ \
  -H "Authorization: Token <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "query_response_id": 123,
        "query_options": {
          "generative_model": "radlab/pLLama-3-8B-DPO-L",
          "percentage_rank_mass": 0.3,
          "answer_language": "pl",
          "translate_answer": false
        }
      }'
```

### 4. Chat Interaction

Start a new chat session:

``` bash
curl -X POST http://localhost:8271/api/new_chat/ \
  -H "Authorization: Token <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"options": {}, "collection_name": "my_docs"}'
```

Add a user message and receive an assistant reply:

``` bash
curl -X POST http://localhost:8271/api/add_user_message/ \
  -H "Authorization: Token <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "chat_id": 1,
        "user_message": "Explain the GDPR article 15.",
        "options": {"generative_model":"radlab/pLLama-3-8B-DPO-L"},
        "collection_name":"my_docs"
      }'
```

## Extending the Project

- **Add a new embedder or reranker**: Update `configs/embedders.json` or `configs/rerankers.json` and restart the
  server.
- **Custom query templates**: Edit `configs/query-templates.json` to define new grammar and data‑connector filters.
- **New generative backend**: Implement a wrapper similar to `OpenAIGenerativeController` or extend
  `GenerativeModelControllerApi`.
- **Authentication**: Switch to Keycloak or OAuth by setting the appropriate `ENV_USE_*` variables and providing the
  relevant config (`configs/auth-config.json`).

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.

--- 

*Happy searching!* 🚀

