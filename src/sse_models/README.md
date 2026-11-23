# Model service (`sse_models`)

Embedding and reranking as a separate process, so the Semantic Search Engine can
run its models on another machine, scale the two operations independently, and
reload a model without restarting the engine.

The engine keeps working exactly as before when the feature is off — turning it
on is a configuration change, not a code change.

---

## 1. Why

The engine used to load `SentenceTransformer` (and, on the first reranked query,
`CrossEncoder`) into its own process. On CPU that costs:

| operation | measured on this machine (CPU) |
|---|---|
| embed, one query, model warm | ~0.4 s |
| rerank, 4 pairs, model warm | tens of seconds |
| loading a model | ~30 s (paid inside a request) |

Three consequences, and the service addresses each:

* **The model is loaded inside a search request.** `MilvusHandler` loaded the
  reranker lazily, on the first query that asked for reranking, so whoever got
  that query waited ~30 s. A service loads once, at its own startup.
* **The engine and the models compete for the same CPU.** The engine is I/O
  bound (HTTP, Milvus, Postgres); a reranker wants every core. Separate
  processes means separate machines, and the reranker can go where the GPU is.
* **Reranking scales differently from embedding.** Embedding is cheap and
  needed by every write and every read; reranking is expensive and only needed
  by some searches. Two roles on one image let you scale them separately.

---

## 2. Shape of the API

**One service, two endpoints** — not two services. Embedding and reranking share
the model catalog, the auth, the batching and the health checks; what differs is
how much hardware they want, and that is expressed by a **role**, not by a second
codebase.

| Endpoint | Purpose | Payload |
|---|---|---|
| `POST /v1/embed` | texts → vectors (indexing and query embedding) | `model`, `texts[]`, `normalize?`, `batch_size?` |
| `POST /v1/rerank` | query + candidates → relevance scores | `model`, `query`, `documents[]`, `top_n?`, `return_documents?`, `batch_size?` |
| `POST /v1/tokenize` | count tokens with a catalog model's tokenizer (no weights loaded) | `model`, `texts[]` |
| `POST /v1/truncate` | cut texts to a token budget, token-exact | `model`, `texts[]`, `max_tokens` |
| `POST /v1/denoise` | clean a noisy text (OCR artefacts) with the T5 denoiser | `text`, `max_length?`, `batch_size?` |
| `GET /v1/models` | catalog this instance serves, with load state | — |
| `GET /healthz` | liveness; no model, no credentials | — |
| `GET /readyz` | readiness; reports per-model load errors | — |

Two decisions worth knowing:

* **Models are addressed by name, never by path.** A client sends
  `google/embeddinggemma-300m`; the service resolves that against
  `configs/embedders.json`. Nothing can ask the service to load an arbitrary
  directory.
* **Rerank answers with the input order restored.** The service returns hits
  sorted best-first, each carrying the `index` of the document in the request,
  and the engine maps them back. The engine decides the final ordering; the
  service never drops a candidate.

### Examples

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"google/embeddinggemma-300m","texts":["kot na dywanie"]}' \
  http://127.0.0.1:8280/v1/embed
# {"model":"...","dim":768,"embeddings":[[...]],"count":1,"took_ms":404.855}

curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"radlab/polish-cross-encoder","query":"Jaka jest stolica Polski?",
       "documents":["Polska ma wiele miast.","Warszawa jest stolicą Polski."]}' \
  http://127.0.0.1:8280/v1/rerank
# {"results":[{"index":1,"score":0.61},{"index":0,"score":0.12}],
#  "count":2,"scored":2,"took_ms":...}
```

Interactive docs: `http://<host>:8280/docs`.

---

## 3. Running it

```bash
pip install -r requirements-models.txt      # no Django, no pymilvus
python -m sse_models                        # or ./run.sh models
```

`./run-models.sh` is a launcher for people who do not want to remember which
interpreter has the stack: it tries the project venv and the system python
until one imports `fastapi`/`uvicorn`, and it applies its flags *after* `.env`,
so they win over the file.

```bash
./run-models.sh                             # interpreter and port from .env
./run-models.sh --port 8281 --role rerank   # flags override .env
./run-models.sh --warm                      # load the models before binding
./run-models.sh --dry-run                   # show what would run
./run-models.sh --help                      # every flag and its variable
```

> **`.env` wins over the command line.** `run.sh` sources `.env` with `set -a`,
> so a value already in `.env` (for example `SSE_MODELS_PORT=8280`, which the
> template ships) overrides the same variable you pass in front of the command:
> `SSE_MODELS_PORT=8331 ./run.sh models` still binds 8280. Edit the `.env`
> line, or run `python -m sse_models` directly, when you want a one-off value.


Everything is configured by environment variables (`sse_models/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `SSE_MODELS_HOST` | `0.0.0.0` | bind address |
| `SSE_MODELS_PORT` | `8280` | bind port |
| `SSE_MODELS_ROLE` | `all` | `all` \| `embed` \| `rerank` \| `denoise`; endpoints outside the role answer 503 |
| `SSE_MODELS_API_TOKEN` | *(empty)* | when set, `/v1/*` requires `Authorization: Bearer`; health stays open |
| `SSE_CONFIG_DIR` | `./configs` | where `embedders.json` / `rerankers.json` live |
| `SSE_MODELS_WARM_MODELS` | `0` | preload every active model before binding the port |
| `SSE_MODELS_RERANK_MAX_DOCUMENTS` | `256` | per-request candidate cap (413 above it) |
| `SSE_MODELS_BATCH_SIZE` | *(unset)* | forward a batch size to the model; unset keeps the engine's own call shape |
| `SSE_MODELS_DENOISE_MAX_MODEL_INPUT` | `256` | token window the denoiser feeds to T5 |
| `SSE_MODELS_LOG_LEVEL` | `info` | logging level |

The model catalog is the **same** `configs/embedders.json` and
`configs/rerankers.json` the engine reads, including their `active_models` list
and per-model `device`. A model the engine knows is a model the service can
serve; a model deactivated there is not served here either.

### Model placement (device)

The `device` key is a **per-model** torch device string, passed to the loader
untouched. Any string torch accepts works: `cpu`, `cuda`, `cuda:1`, `cuda:2`.

```json
{ "name": "google/embeddinggemma-300m", "path": "/models/embeddinggemma-300m",
  "device": "cuda:1" }
{ "name": "qwen/Qwen3-Reranker-0.6", "path": "/models/Qwen3-Reranker-0.6B",
  "device": "cuda:2" }
```

Two models pinned to different GPUs are independent: inference on one is locked
**per model, not per process**, so an embed batch on `cuda:1` and a rerank batch
on `cuda:2` run at the same time. (Before that, a single process-wide lock
serialized every model against every other, which would have made the whole
point of splitting them across GPUs meaningless.)

Notes:

* A `device` of `cuda:N` is only as good as the GPUs actually present and
  visible to the process. `CUDA_VISIBLE_DEVICES` **remaps** the indices — with
  `CUDA_VISIBLE_DEVICES=1,2`, `cuda:1` is the *second* GPU in that list, not
  GPU number 1.
* There is no validation: a typo (`cuda:7` on a two-GPU box) fails at load with
  the torch error, surfaced as `500` on that model's endpoint while the rest
  keep serving.
* Pinning is per **catalog entry**, i.e. per model, and is honored by the
  engine's in-process path too — same `device` key, same value, either side.

---

## 4. Pointing the engine at it

The engine side lives in `src/sse_api/engine/core/models/backend.py`. Default
off; when on, `MilvusHandler` delegates both operations.

| Variable | Default | Meaning |
|---|---|---|
| `SSE_MODEL_SERVICE_ENABLED` | `0` | master switch |
| `SSE_MODEL_SERVICE_URL` | — | base url, e.g. `http://192.168.100.70:8280` |
| `SSE_MODEL_SERVICE_TOKEN` | *(empty)* | shared secret |
| `SSE_MODEL_SERVICE_TIMEOUT_SECONDS` | `30` | read timeout of one call |
| `SSE_MODEL_SERVICE_CONNECT_TIMEOUT_SECONDS` | `3` | connect timeout |
| `SSE_MODEL_SERVICE_FALLBACK_LOCAL` | `1` | on failure use in-process models |
| `SSE_MODEL_SERVICE_FAILURE_THRESHOLD` | `3` | consecutive failures that open the circuit |
| `SSE_MODEL_SERVICE_COOLDOWN_SECONDS` | `30` | how long the circuit stays open |
| `SSE_MODEL_SERVICE_EMBED_MODEL` / `..._RERANK_MODEL` | resolved | override the name sent to the service |

**Model names are resolved, not duplicated.** Collections are configured by
filesystem path, the service speaks names. `resolve_model_name()` looks the path
up in the already-loaded catalog, so there is no second mapping to keep in sync;
a path that is not in the catalog falls back to its directory name.

**What turning it on actually changes in the engine:**

* `MilvusHandler.__init__` no longer loads the embedder, and
  `_load_reranker_model_from_path()` returns without loading. The engine starts
  without touching the model stack.
* `prepare_embeddings()` and `_rerank_search_results()` call the backend
  instead of the in-process model. Everything around them — the Milvus query,
  the sort, the score bookkeeping — is untouched.
* The `torch` / `sentence_transformers` imports moved out of module level for
  the same reason: an engine that delegates inference should not pay to import
  them.

### Failure behaviour

Reranking is an enhancement: a search without it still returns results, one
that raises returns nothing. So:

1. A failed call falls back to the in-process model when
   `SSE_MODEL_SERVICE_FALLBACK_LOCAL=1`. The local model is loaded lazily at
   that moment, never eagerly.
2. After `FAILURE_THRESHOLD` consecutive failures the circuit opens for
   `COOLDOWN_SECONDS` and the engine goes straight to local, instead of paying a
   timeout on every query.
3. A rejected model name or token (404/401/403) opens the circuit immediately —
   retrying cannot make a misconfiguration succeed.
4. Failures are logged once per state change, not once per query; a dead service
   under traffic must not flood the log. Recovery is logged too.

---

## 5. Deployment

**Same host, one process** (default role `all`): simplest, and already removes
the load-inside-a-request problem.

**Split by role** — the arrangement the design is for:

```
engine  ──HTTP──▶  service role=embed    (CPU host, replicas scale with indexing)
       └─HTTP──▶  service role=rerank    (GPU host, where the cost actually is)
```

```bash
# GPU host
SSE_MODELS_ROLE=rerank SSE_MODELS_WARM_MODELS=1 SSE_MODELS_PORT=8281 ./run.sh models
# engine
SSE_MODEL_SERVICE_ENABLED=1 \
SSE_MODEL_SERVICE_URL=http://cpu-host:8280 \
SSE_MODEL_SERVICE_RERANK_MODEL=qwen/Qwen3-Reranker-0.6 ...
```

The role only gates what is *served*; a model is loaded on first use unless
`SSE_MODELS_WARM_MODELS=1`, which loads before the port is bound so a load
balancer never sees a ready-but-cold instance.

**One batch per model.** Each model's inference is guarded by its own lock:
two concurrent batches on the *same* model mostly queue on the device and
inflate latency for both, so one model serves one batch at a time. Different
models — and especially different GPUs — do not wait on each other. Scale a
single model with replicas, not with concurrency.

**Security.** `SSE_MODELS_API_TOKEN` is a shared secret, not identity: it stops
casual use, not an attacker. The service returns model output, never file paths
(`/v1/models` deliberately omits them). Keep it on an internal interface.

---

## 6. Operations

```bash
curl -s http://127.0.0.1:8280/healthz   # {"status":"ok","role":"all","loaded":[...]}
curl -s http://127.0.0.1:8280/readyz    # "degraded" + per-model errors when load failed
```

`loaded` is the list of models resident in memory. Empty right after a start
with warm-up off is normal.

Every inference call logs `model`, batch size, result count and server-side
`took_ms`, so engine-vs-service latency splits cleanly.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot handle pad token` / `Cannot handle batch sizes > 1 if no padding token is defined` | rerankers derived from causal LLMs (Qwen3) ship a tokenizer with no pad symbol and no `pad_token_id` in `config.json` | handled at load by `_ensure_pad_token` (substitutes `eos`, sets the id on **both** the sentence-transformers config and the wrapped model config, which are separate objects) |
| `Some weights ... 'score.weight' were not initialized` | model saved by a newer sentence-transformers than installed — the scoring head is rebuilt **with random weights** and scores are meaningless | upgrade `sentence-transformers` (Qwen3-Reranker needs >= 5.x). Take the warning seriously: it fails silently, not loudly |
| `model service rejected the model` | name not in the catalog / not in `active_models` | add it to `configs/*.json`, or point `SSE_MODEL_SERVICE_*_MODEL` at one that is |
| Every query pays a timeout | service unreachable, fallback off | `SSE_MODEL_SERVICE_FALLBACK_LOCAL=1`, and check `SSE_MODEL_SERVICE_URL` reachability from the engine host |
| First query after start is slow | lazy load | `SSE_MODELS_WARM_MODELS=1` |
| `413` from `/v1/rerank` | more candidates than `SSE_MODELS_RERANK_MAX_DOCUMENTS` | narrow the candidate set — reranking hundreds of pairs is the cost you are trying to avoid |

---

## 7. Tests

```bash
pytest src/sse_models/tests/test_api.py -q          # 21 tests, no ML stack needed
```

They run against fake embedder/reranker objects, so they cover the contract —
status codes, auth, role gating, ordering, index mapping, `top_n`, limits —
without loading a model. `test_registry.py` covers catalog handling, per-model
`device` pass-through and the per-model inference lock. The engine side is covered by
`src/sse_api/engine/tests/test_model_backend.py` (17 tests: the factory, name
resolution, payload shapes, the circuit breaker and the fallback).
