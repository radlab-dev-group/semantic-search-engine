### Overview

The **data** package manages all document‑related entities, ingestion pipelines, and auxiliary utilities needed before
the data reaches the semantic search engine. It provides models for collections, uploaded files, and query‑template
structures, plus controllers for denoising, uploading, relational DB interactions, and query‑template handling.

### Package Structure

```
data/
├─ controllers/
│   ├─ common/
│   │   └─ constants.py           # Simple feature flags
│   ├─ denoising/
│   │   └─ denoiser_logic.py      # Text denoising using a T5‑based model
│   ├─ templates/
│   │   └─ template_logic.py      # Grammar & filtering for query templates
│   ├─ upload/
│   │   └─ upload_logic.py        # Handles file upload, extraction, and optional denoising
│   └─ __init__.py                # Facade for backward compatibility
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
| **DocumentPageText**                                                        | Text chunk extracted from a page; contains raw and optionally denoised text, language, and metadata. Used for both semantic search and PostgreSQL Full-Text Search. |
| **CollectionOfQueryTemplates**, **QueryTemplateGrammar**, **QueryTemplate** | Structures for defining reusable query templates, their grammar, and per‑template data‑connector constraints.                       |

### Important Controllers

| Controller                                         | Role                                                                                                                                                                                                                                                                                             |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **DenoiserController** (`denoiser_logic.py`)       | Loads a T5‑based denoising model (cached) and provides `denoise_text()` for cleaning raw OCR output.                                                                                                                                                                                             |
| **RelationalDBController** (`engine/controllers/database/relational_db.py`) | - Retrieves user collections. <br> - Adds uploaded documents to the relational DB, handling page creation, optional denoising, and metadata extraction. <br> - Provides helper methods for fetching documents, pages, and categories.                                                            |
| **SemanticDBController**   (`engine/controllers/database/semantic_db.py`)   | Thin wrapper around Milvus; prepares a collection, creates schemas, and offers a `collections()` helper.                                                                                                                                                          |
| **UploadDocumentsController** (`upload_logic.py`)  | Orchestrates the full upload‑index pipeline: <br>  1️⃣ Stores files in a structured directory (`<upload_dir>/<org>/<user>/<collection>/<timestamp>/`). <br>  2️⃣ Calls `RelationalDBController.add_uploaded_documents_to_db()`. <br>  3️⃣ Triggers high-performance batch semantic indexing via `SemanticDBController`. |
| **Chunking** (`controllers/chunking.py`) | `resolve_chunking()` resolves the chunking of one indexing call against the collection (the request wins → the collection → the defaults, bound onto the collection), `check_chunk_size()` refuses a chunk which does not fit `max_model_length` of the embedder of the collection, and a strict collection refuses values other than its own (`ChunkingMismatch`). |
| **QueryTemplateController** (`template_logic.py`) | Parses a JSON configuration of query templates, validates grammar, and filters documents based on template‑defined data‑connector constraints.                                                                                                                                                   |

### Public API (`api.py`)

| Endpoint                                   | Description                                                                                                        |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **new_collection**                         | Create a new `CollectionOfDocuments` with embedding & reranker settings, optional `chunking` (`max_tokens_in_chunk`, `number_of_overlap_tokens`) and `strict_chunking`; optionally link to an organisation group. |
| **collections**                            | List all collections accessible to the authenticated user (personal + organisation‑wide).                          |
| **upload_and_index_files**                 | Accept multipart file upload, store files, run relational indexing, and optionally trigger Milvus indexing.        |
| **add_and_index_texts**                    | Accept raw JSON‑encoded texts (already split into pages) and index them directly.                                  |
| **categories**, **documents**              | Retrieve distinct categories or document names for a given collection.                                             |
| **question_templates**, **filter_options** | Expose the available query templates and mock filter data (useful for UI prototypes).                              |
| **delete_texts**                           | Delete text chunks of a collection (`chunk_ids[]`), from the relational db and Milvus.                             |
| **delete_documents**                       | Delete documents (by `documents[]` names and/or `document_ids[]`) with their pages and chunks, from both databases.|
| **delete_collection**                      | Delete the whole collection with all of its content from the relational db and Milvus.                            |
| **set_documents_use_in_search**            | Enable/disable documents of a collection in the search (`use_in_search` flag; nothing is removed).                |

The deletion is performed by `data/controllers/deletion.py`
(`ContentDeletionController`): rows are removed first and the embeddings of
the removed texts afterwards, and the outcome (deleted rows, deleted vectors,
not‑found names/ids) is reported in the response. Only the **owner** of a
collection may delete it; a collection which is merely shared with the groups
of the user is reported as not found. Deleting a collection also removes the
chats and the search history which reference it (protected foreign keys).

Document visibility is performed by `data/controllers/visibility.py`
(`ContentVisibilityController`): it only flips the `Document.use_in_search`
flag in the relational db and reports the updated documents and the
not‑found names/ids. Nothing is removed, so a deactivated document may be
enabled again without any re‑indexing. While disabled, its chunks are not
returned by the search (vector and full‑text sides alike).

### Chunking of a collection

`CollectionOfDocuments.metadata_json` keeps the settings which belong to the collection as a
whole: `chunking` (`max_tokens_in_chunk`, `number_of_overlap_tokens`) and the `strict_chunking`
flag. They are declared with `new_collection`, read back with `collections` (the serializer
exposes `metadata_json`), and every indexing call is resolved against them by
`data/controllers/chunking.py`: the values of the request win, then the ones of the collection,
then `DEFAULT_MAX_TOKENS_IN_CHUNK` / `DEFAULT_NUMBER_OF_OVERLAP_TOKENS`
(`data/core/constants.py`), which are bound onto the collection so that the next call does not
have to pass them. A strict collection raises `ChunkingMismatch` instead of accepting other
values, and a chunk which does not fit `max_model_length` of the embedder of the collection
raises `ChunkingTooLong`. Mixed chunk lengths are allowed — the invariant of a collection is its
embedder and vector size, not a single chunk size.

`python -m sse_api.manage check_collection_chunking` reports what every collection declares;
`--verify` measures the stored chunks with the tokenizer of the embedder and points at the chunks
which are longer than declared (`--collection NAME`, `--limit`, `--json` shape the report).

### Typical Ingestion Flow

1. **Create a collection** – POST to `/api/<version>/new_collection/` with embedding model names, index type and,
   optionally, the `chunking` every later indexing call falls back to.
2. **Upload files** – POST multipart to `/api/<version>/upload_and_index_files/` with `files[]`, `collection_name`, and
   `indexing_options` (e.g., `use_text_denoiser`; `max_tokens_in_chunk` may be left to the collection). The controller
   stores files, extracts pages, optionally denoises, and finally indexes the cleaned text into Milvus.
3. **Optional text‑only indexing** – Use `/api/<version>/add_and_index_texts/` when the client already has pre‑processed
   JSON text chunks.

### Upload safety and limits

`data/controllers/upload.py` copies uploads in 64 KiB chunks. ZIP archives are
spooled to a temporary file outside the recursively scanned upload directory and
deleted on completion or rejection. Only extracted regular files are returned.
Absolute paths, traversal, symlinks, special files, duplicate names, case-only
collisions and file/directory collisions are rejected; existing files are never
overwritten. A rejected request removes its entire unique staging directory,
creates no upload record and never starts relational or semantic indexing.

Configure these Django settings in `main/settings.py` (defaults apply when absent):

| Setting | Default | Scope |
|---|---:|---|
| `UPLOAD_MAX_FILES` | `1000` | Regular files across the whole multipart request |
| `UPLOAD_MAX_UNCOMPRESSED_BYTES` | `104857600` | Total plain/extracted bytes across the request |
| `UPLOAD_MAX_COMPRESSION_RATIO` | `100` | Per ZIP member: uncompressed/compressed bytes |
| `UPLOAD_MAX_INPUT_BYTES` | `104857600` | Total incoming ZIP and plain file bytes |

Use positive finite limits. ZIP declared sizes/counts/ratios are checked before
extraction, and actual bytes, ratios and file counts are checked during copying.
Each ZIP also permits at most `2 * UPLOAD_MAX_FILES` entries, including directories.
Set Django multipart upload handlers and reverse-proxy request limits separately:
these controller limits apply after Django receives the multipart request.

`number_of_uploaded_documents` counts stored regular files (including nested ZIP
members), not ZIP containers or directories. Existing relational/page and vector
index counters retain their prior semantics. The response envelope and business
error HTTP status remain unchanged; rejected uploads return `000001_DATA`.

### Extending the Data Layer

- **New metadata fields** – Extend `Document.metadata_json` and update
  `RelationalDBController._filter_documents_based_on_metadata` to support additional operators.
- **Alternative storage** – Swap `UploadedDocuments` handling with a cloud bucket (e.g., S3) by customizing `AwsHandler`
  in the `sse_api.config` package and adjusting `UploadDocumentsController._store_single_file_to_upload_dir`.
- **Custom denoiser** – Provide a different model path in `configs/models.json`; the `DenoiserController` will pick it
  up automatically.
