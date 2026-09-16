# Semantic Search Engine (SSE) — Setup & Run Instructions

> English counterpart of `URUCHOMIENIE.md` (script: `run-en.sh`). Polish originals: `URUCHOMIENIE.md` + `run.sh`.
>
> In sync with the current code (service in `sse_rest_api/`). The top-level `README.md` describes the **old**
> file layout — skip it.
> Most faithful source: `sse_rest_api/README.md` + the code itself (`*/urls.py`, `configs/`, `initialize.sh`, `run-api.sh`).

## ⚡ Quick start (TL;DR)

```bash
cd semantic-search-engine
./run-en.sh               # == all: venv + dependencies, PG + Milvus, initialization, start API
./run-en.sh cosine        # (one-off) existing collections → COSINE metric, no re-embedding
# API: http://localhost:8271/api/   (login: default_admin / password)
```

Step-by-step details below. Search computes **cosine similarity** (see §11).

## 0. What it is and where the code lives

- **Repo:** `semantic-search-engine/`
- **Service (backend API): `sse_rest_api/`** — `manage.py`, `initialize.sh`, `run-api.sh`, `configs/`,
  `requirements.txt`.
- `sse_apps/admin/` + `scripts/admin/` — admin scripts (user, templates, Milvus, Postgres).
- Stack: **Python 3.11, Django + DRF**. Hybrid search = **Milvus** (vectors) + **PostgreSQL** (full-text) + RRF.
  Optionally an **LLM router** (RAG/chat).

## 1. Requirements

- **Python 3.11**, **Docker**.
- **PostgreSQL** — db `sse_backend`, user `admin`, password `SuperSecretPassword123`, host `localhost`,
  **port 5471**.
- **Milvus 2.3+** — host `localhost`, **port 19530** (+19121), db `sse_backend_engine`, `sse_user`/`sse_password`.
- **ML models** (paths from `configs/`) — embedders (`embedders.json`), reranker `radlab/polish-cross-encoder`
  (`rerankers.json`), denoiser `radlab/polish-denoiser-t5-base` (`models.json`). Most of them live locally under
  **`/mnt/data2/llms/models/...`** (they must exist; the denoiser uses `cuda:0`).
- **Optional** **LLM router** (RAG/chat only) — `LLM_ROUTER_API` + a model (`generative-models.json`).

> **Search-only works without the LLM router/DeepL/OpenAI.** RAG/chat require the router + generative models.

## 2. Virtual environment + dependencies

```bash
cd semantic-search-engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r sse_rest_api/requirements.txt
cd sse_rest_api
./initialize.sh dep     # pip install radlab-data + llm-router (from git)
```

Key packages: `django`, `djangorestframework`, `transformers`, `torch`, `sentence_transformers`, `spacy`,
`pymilvus`, `psycopg2-binary`, `openai`, `deepl`, `huggingface_hub`.

## 3. Start PostgreSQL

```bash
cd semantic-search-engine
bash scripts/admin/postgres.sh
```

Container `pg_sse_backend_engine`, `localhost:5471 → 5432`, user `admin`, db `sse_backend`, password
`SuperSecretPassword123` (1:1 with `configs/django-config.json`).
> Alternative: your own PG with the same parameters, or `ENV_DB_NAME/HOST/PORT/USERNAME/PASSWORD`.

## 4. Start Milvus

```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 19121:19121 milvusdb/milvus:2.3.0
```

Target: `localhost:19530`, db `sse_backend_engine`, `sse_user`/`sse_password` (`configs/milvus_config.json`).
> Overrides: `ENV_MILVUS_HOST/PORT/DBNAME/USER/PASSWORD`.

## 5. (Optional) LLM router — RAG/chat only

- Set `LLM_ROUTER_API` (in `run-api.sh`: `http://192.168.100.65:8080` — change it to yours).
- Model as defined in `configs/generative-models.json` (`google/gemma-4-12b-it` → `localhost:8080`).
- `OPENAI_API_KEY`/`DEEPL_AUTH_KEY` (placeholders in `run-api.sh`) — only when using those backends.

## 6. Database initialization + account + templates

Everything runs from `sse_rest_api/` (the scripts use relative paths):

```bash
cd semantic-search-engine/sse_rest_api
./initialize.sh migrate               # manage.py migrate (PostgreSQL)
./initialize.sh semantic              # creates the DB in Milvus (sse_backend_engine)
./initialize.sh add_user              # org/group/user from user-group-organisation.json
./initialize.sh add_query_templates   # templates from query-templates.json
```

or everything at once: `./initialize.sh all`.

- **Default user** (`configs/user-group-organisation.json`): `default_admin` / `default_admin@email.me` /
  `password`; org `sse_default`.

## 7. Start the API server

```bash
cd semantic-search-engine/sse_rest_api
./run-api.sh     # python3 manage.py runserver 0.0.0.0:8271
```

**API: `http://localhost:8271/api/...`**

## 8. Verification (smoke test)

By default **DRF Token** auth (`ENV_USE_*_AUTH=0`):

```bash
curl -X POST http://localhost:8271/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"default_admin","password":"password"}'
# => {"token":"..."}   # then: Authorization: Token <token>
```

Flow (endpoints verified in `*/urls.py`; `MAIN_API_URL = "api"`, no version prefix):

```bash
# 1) Create a collection
curl -X POST http://localhost:8271/api/new_collection \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"collection_name":"my_docs","collection_display_name":"My Docs","collection_description":"test",
       "model_embedder":"radlab/polish-bi-encoder-mean","model_reranker":"radlab/polish-cross-encoder",
       "embedder_index_type":"HNSW"}'

# 2) Upload + index (multipart, files[])
curl -X POST http://localhost:8271/api/upload_and_index_files \
  -H "Authorization: Token $TOKEN" -F "files[]=@/path/doc.pdf" -F "collection_name=my_docs" \
  -F 'indexing_options={"prepare_proper_pages":true,"clear_text":true}'

# 3) Search (hybrid)
curl -X POST http://localhost:8271/api/search_with_options \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"collection_name":"my_docs","query_str":"data retention policy",
       "options":{"max_results":20,"rerank_results":true,"hybrid_search":true}}'
```

RAG/chat (requires the router): `POST /api/generative_answer`, `POST /api/new_chat`, `POST /api/add_user_message`.

> ⚠️ `sse_rest_api/README.md` shows collection creation under `POST /api/collections/`, but in the code **creation** is
> `api/new_collection`, while `api/collections` is **listing** (GET). When calling the API, rely on `*/urls.py`.

## 9. What can be overridden (env / config)

- `django-config.json` (+ `ENV_DB_*`, `ENV_DEBUG`, `ENV_SECRET_KEY`, `ENV_ALLOWED_HOSTS`, `ENV_LOGGER`, `ENV_USE_AWS`,
  `ENV_USE_CELERY`).
- `milvus_config.json` (+ `ENV_MILVUS_*`).
- Models: `embedders.json`, `rerankers.json`, `models.json`, `generative-models.json` — after changing them,
  **restart the server**.
- Auth: `ENV_USE_KC_AUTH`/`ENV_USE_OAUTH_V1_AUTH`/`ENV_USE_OAUTH_V2_AUTH` (=1) + `configs/auth-config.json`
  (token auth by default).

## 10. Common "gotchas"

- Run `initialize.sh` / `run-api.sh` **from the `sse_rest_api/` directory**.
- The models under `/mnt/data2/llms/models/...` must exist; the denoiser runs on `cuda:0` (check the GPU,
  `CUDA_VISIBLE_DEVICES=0`).
- Ports: **PG 5471**, **Milvus 19530/19121**, **API 8271** — avoid collisions.
- `prepare_semantic_db.py` reads `./configs/milvus_config.json` by default (which is why `initialize.sh semantic`
  copies the script into `sse_rest_api/`).
- **Collections created before COSINE was enabled** (before 2026-09-02) may have an `IP` index — their results will
  not be cosine; migrate them: §11.

## 11. Search — COSINE metric

The code **already computes cosine similarity by default** (commit `d9831f9`, 2026-09-02):

- new collections: `metric_type=COSINE` (`MilvusHandler.DEFAULT_COLLECTION_METRIC_TYPE`, `__add_milvus_collection`),
- indexes and queries: `INDEX_QUERY_PARAMS` (HNSW and IVF_FLAT) → `metric_type: COSINE`,
- the score from Milvus = cosine similarity; the `min_similarity` threshold (default **0.5**, range [-1, 1]) drops
  results weaker than the threshold,
- compatibility tests: `sse_rest_api/engine/tests/test_similarity.py`.

**New collections** (created through the API `POST /api/new_collection`) automatically get COSINE — nothing to do.

**Existing collections** (`IP`/legacy index) — migrate them **without re-embedding** (the vectors are already in the
DB — the script releases the index and creates a new one with COSINE):

```bash
cd semantic-search-engine
./run-en.sh cosine                          # all collections (index IVF_FLAT by default)
./run-en.sh cosine --dry-run                # preview the operations (no changes)
./run-en.sh cosine --collection my_coll     # only the given collection (repeatable)
./run-en.sh cosine --index HNSW             # index type: HNSW | IVF_FLAT
```

Equivalent direct invocation (requires a running Milvus):

```bash
cd semantic-search-engine/sse_rest_api
python ../scripts/admin/milvus_index_to_cosine.py --all
python ../scripts/admin/milvus_index_to_cosine.py --collection my_coll --dry-run
```

Verification: `describe_index` for the collection must return `metric_type: COSINE` (e.g. via `pymilvus`).

> Note: `cosine` does not re-embed and does not touch the data — indexes only. Collections must exist; `--all`
> requires Milvus to be reachable (port 19530).

---
*Alternatively: `./run-en.sh all` performs steps 2–7 automatically (flags: `--no-venv/--no-infra/--no-init/--skip-server`);
`./run-en.sh cosine` — COSINE metric for existing data.*
