### Overview

The **data** package manages all document‑related entities, ingestion pipelines, and auxiliary utilities needed before
the data reaches the semantic search engine. It provides models for collections, uploaded files, and query‑template
structures, plus controllers for denoising, uploading, relational DB interactions, and query‑template handling.

### Package Structure

```
data/
├─ controllers/
│   ├─ __init__.py
│   ├─ constants.py               # Simple feature flags
│   ├─ denoiser.py                # Text denoising using a T5‑based model
│   ├─ query_templates.py         # Grammar & filtering for query templates
│   ├─ relational_db.py           # CRUD + indexing helpers for the relational DB
│   ├─ semantic_db.py             # Thin wrapper around Milvus for semantic indexing
│   └─ upload.py                  # Handles file upload, extraction, and optional denoising
├─ migrations/
│   └─ __init__.py
├─ __init__.py
├─ api.py                          # Public endpoints for collections, uploads, etc.
├─ apps.py                         # Django AppConfig
├─ models.py                       # Django ORM models (collections, documents, uploads)
├─ serializers.py                  # DRF serializers for the models
└─ urls.py                         # URL routing for data‑specific endpoints
```

### Core Models (`models.py`)

| Model                                                                       | Description                                                                                                                         |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **CollectionOfDocuments**                                                   | Represents a named collection of documents. Stores display name, description, embedding model configuration, and visibility groups. |
| **UploadedDocuments**                                                       | Tracks a physical directory where raw files were uploaded, plus indexing statistics.                                                |
| **Document**                                                                | Individual file metadata (name, path, hash, category, optional JSON metadata). Linked to a collection and an uploading user.        |
| **DocumentPage**                                                            | A page of a multi‑page document (e.g., PDF).                                                                                        |
| **DocumentPageText**                                                        | Text chunk extracted from a page; contains raw and optionally denoised text, language, and metadata.                                |
| **CollectionOfQueryTemplates**, **QueryTemplateGrammar**, **QueryTemplate** | Structures for defining reusable query templates, their grammar, and per‑template data‑connector constraints.                       |

### Important Controllers

| Controller                                         | Role                                                                                                                                                                                                                                                                                             |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **DenoiserController** (`denoiser.py`)             | Loads a T5‑based denoising model (cached) and provides `denoise_text()` for cleaning raw OCR output.                                                                                                                                                                                             |
| **RelationalDBController** (`relational_db.py`)    | - Retrieves user collections. <br> - Adds uploaded documents to the relational DB, handling page creation, optional denoising, and metadata extraction. <br> - Provides helper methods for fetching documents, pages, and categories.                                                            |
| **SemanticDBController** (`semantic_db.py`)        | Thin wrapper around Milvus; prepares a collection, creates schemas, and offers a `collections()` helper.                                                                                                                                                                                         |
| **UploadDocumentsController** (`upload.py`)        | Orchestrates the full upload‑index pipeline: <br>  1️⃣ Stores files in a structured directory (`<upload_dir>/<org>/<user>/<collection>/<timestamp>/`). <br>  2️⃣ Calls `RelationalDBController.add_uploaded_documents_to_db()`. <br>  3️⃣ Triggers semantic indexing via `SemanticDBController`. |
| **QueryTemplateController** (`query_templates.py`) | Parses a JSON configuration of query templates, validates grammar, and filters documents based on template‑defined data‑connector constraints.                                                                                                                                                   |

### Public API (`api.py`)

| Endpoint                                   | Description                                                                                                        |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **new_collection**                         | Create a new `CollectionOfDocuments` with embedding & reranker settings; optionally link to an organisation group. |
| **collections**                            | List all collections accessible to the authenticated user (personal + organisation‑wide).                          |
| **upload_and_index_files**                 | Accept multipart file upload, store files, run relational indexing, and optionally trigger Milvus indexing.        |
| **add_and_index_texts**                    | Accept raw JSON‑encoded texts (already split into pages) and index them directly.                                  |
| **categories**, **documents**              | Retrieve distinct categories or document names for a given collection.                                             |
| **question_templates**, **filter_options** | Expose the available query templates and mock filter data (useful for UI prototypes).                              |

### Typical Ingestion Flow

1. **Create a collection** – POST to `/api/<version>/new_collection/` with embedding model names, index type, etc.
2. **Upload files** – POST multipart to `/api/<version>/upload_and_index_files/` with `files[]`, `collection_name`, and
   `indexing_options` (e.g., `use_text_denoiser`, `max_tokens_in_chunk`). The controller stores files, extracts pages,
   optionally denoises, and finally indexes the cleaned text into Milvus.
3. **Optional text‑only indexing** – Use `/api/<version>/add_and_index_texts/` when the client already has pre‑processed
   JSON text chunks.

### Extending the Data Layer

- **New metadata fields** – Extend `Document.metadata_json` and update
  `RelationalDBController._filter_documents_based_on_metadata` to support additional operators.
- **Alternative storage** – Swap `UploadedDocuments` handling with a cloud bucket (e.g., S3) by customizing `AwsHandler`
  in the `main` package and adjusting `UploadDocumentsController._store_single_file_to_upload_dir`.
- **Custom denoiser** – Provide a different model path in `configs/models.json`; the `DenoiserController` will pick it
  up automatically.
