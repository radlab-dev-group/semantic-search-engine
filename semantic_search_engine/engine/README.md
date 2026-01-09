### Overview

The **engine** package implements the core search‑and‑generation logic of the Semantic Search Engine.  
It glues together vector‑based semantic search (Milvus), embedding/reranking models, and generative language models (
OpenAI or locally hosted).

### Package Structure

```
engine/
├─ controllers/
│   ├─ __init__.py
│   ├─ embedders_rerankers.py      # Loads embedder & reranker model configs
│   ├─ milvus.py                   # Milvus handler (connection, queries, etc.)
│   ├─ models.py                   # High‑level models for queries, responses & answers
│   ├─ question_answer.py.depr     # Deprecated QA helper (kept for reference)
│   ├─ search.py                   # Semantic search, indexing, filtering & result processing
│   └─ system.py                   # System‑level actions (e.g. rating answers)
├─ core/
│   ├─ __init__.py
│   └─ middleware.py               # Authentication‑related middleware helpers
├─ migrations/
│   └─ __init__.py
├─ __init__.py
├─ api.py                          # Public API endpoints (search, generative answer, rating, model listing)
├─ apps.py                         # Django AppConfig
├─ models.py                       # Django ORM models for queries, responses and answers
└─ urls.py                         # URL routing for engine‑specific endpoints
```

### Key Components

| Component                  | Purpose                                                                                                                                                                                                                                                                                                                                            |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **embedders_rerankers.py** | Reads `configs/embedders.json` and `configs/rerankers.json`, builds global registries (`ALL_AVAILABLE_EMBEDDERS_MODELS`, `ALL_AVAILABLE_RERANKERS_MODELS`) and provides helper methods (`get_embedder_path`, `get_reranker_path`, etc.).                                                                                                           |
| **search.py**              | - `SearchQueryController` creates a `UserQuery`, runs the semantic search and stores the `UserQueryResponse`. <br> - `DBSemanticSearchController` handles Milvus indexing/search, metadata filtering, reranking and result post‑processing. <br> - `DBTextSearchController` fetches raw text fragments from the relational DB for display.         |
| **models.py**              | Django ORM models: `UserQuery`, `UserQueryResponse`, `UserQueryResponseAnswer`. They store the query, its detailed results, and generated answers (including rating fields).                                                                                                                                                                       |
| **system.py**              | Service class (`EngineSystemController`) for system‑level actions such as persisting user ratings on generated answers.                                                                                                                                                                                                                            |
| **api.py**                 | REST API views: <br> • `SearchWithOptions` – run a search with complex options. <br> • `GenerativeAnswerForQuestion` – generate an answer from a stored query response. <br> • `ListGenerativeModels`, `ListEmbeddersModels`, `ListRerankersModels` – expose available model names. <br> • `SetRateForQueryResponseAnswer` – record a user rating. |
| **middleware.py**          | Helper that registers a callback (when external auth is enabled) to auto‑create a default organisation hierarchy for newly authenticated Django users.                                                                                                                                                                                             |

### Typical Workflow

1. **Indexing** – Use `DBSemanticSearchController` (via the data layer) to push document text chunks into Milvus.
2. **Search** – Client POSTs to `/api/<version>/search_with_options/` with a collection name, query string and search
   options. `SearchWithOptions` invokes `SearchQueryController.new_query()`, which: <br>a) stores the `UserQuery`. <br>
   b) Calls `DBSemanticSearchController.search_with_options()` to retrieve ranked text fragments. <br>c) Persists a
   `UserQueryResponse`.
3. **Generative Answer** – Client POSTs to `/api/<version>/generative_answer/` with the `query_response_id` and
   generation options. The view uses `GenerativeModelController` to call either an OpenAI model or a local model via
   HTTP, optionally translating the answer. The generated answer is saved as a `UserQueryResponseAnswer`.
4. **Rating** – Users can rate the generated answer via `/api/<version>/rate_generative_answer/`;
   `EngineSystemController.set_rating()` updates the rating fields.

### Extending the Engine

- **Add a new embedder or reranker**: Extend the JSON config files under `configs/`, then re‑run the server – the
  registries will pick up the new entries automatically.
- **Custom post‑processing**: Subclass `DBSemanticSearchController` or add new helper methods in `search.py` and expose
  them via additional API endpoints.
- **New generative back‑ends**: Implement a wrapper similar to `OpenAIGenerativeController` or extend
  `GenerativeModelControllerApi.LocalModelAPI` to communicate with another service.