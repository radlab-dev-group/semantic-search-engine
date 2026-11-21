"""
app.py
------

FastAPI application exposing the embedder and reranker models over HTTP.

Endpoints
---------
``POST /v1/embed``    embed a batch of texts
``POST /v1/rerank``   score ``query`` against ``documents``
``GET  /v1/models``   list the models this instance can serve
``GET  /healthz``     liveness (no model, no credentials)
``GET  /readyz``      readiness (reports load state per model)

Design notes
------------
* The handlers are plain (non-``async``) functions, so FastAPI runs them in
  its threadpool and a long forward pass never blocks the event loop.
* One instance serves one *role* (``all`` / ``embed`` / ``rerank``).  Deploying
  the embedder and the reranker as separate deployments of the same image is
  how they get scaled independently — the reranker wants the GPU, the embedder
  usually does not.
* The service holds no request state and no engine state: it is a pure
  function of (model name, texts).
"""

import logging
import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status

from sse_models import __version__
from sse_models.registry import ModelConfigError, ModelNotFoundError, ModelRegistry
from sse_models.schemas import (
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    RerankHit,
    RerankRequest,
    RerankResponse,
    DenoiseRequest,
    DenoiseResponse,
    TokenizeRequest,
    TokenizeResponse,
    TruncateRequest,
    TruncateResponse,
)
from sse_models.settings import ModelServiceSettings

logger = logging.getLogger("sse_models")

DESCRIPTION = """\
Standalone embedding / reranking service for the Semantic Search Engine.

Hosts the models defined in ``configs/embedders.json`` and
``configs/rerankers.json`` outside of the engine process, so inference can be
placed on another (GPU) machine, scaled per operation and restarted without
restarting the engine.
"""


def _require_token(settings: ModelServiceSettings):
    """
    Build a dependency that enforces the shared secret on ``/v1/*``.

    Health endpoints do not use it: a probe must work without credentials and
    exposes no text and no model paths.  When ``SSE_MODELS_API_TOKEN`` is empty
    the dependency accepts everything, which is the right default for a service
    reachable only on an internal network.
    """

    def dependency(request: Request) -> None:
        if not settings.token_required():
            return
        provided = request.headers.get("Authorization", "")
        if provided != f"Bearer {settings.api_token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid model service token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency


def create_app(
    settings: Optional[ModelServiceSettings] = None,
    registry: Optional[ModelRegistry] = None,
) -> FastAPI:
    """Build the service app; ``settings``/``registry`` are injectable for tests."""
    settings = settings or ModelServiceSettings()
    registry = registry or ModelRegistry(settings)

    app = FastAPI(
        title="SSE model service",
        description=DESCRIPTION,
        version=__version__,
    )
    app.state.settings = settings
    app.state.registry = registry

    auth = [Depends(_require_token(settings))]

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------
    @app.get("/healthz", response_model=HealthResponse, tags=["health"])
    def healthz() -> HealthResponse:
        """Liveness: the process is up. Never touches a model."""
        return HealthResponse(
            status="ok", role=settings.role, loaded=registry.loaded()
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["health"])
    def readyz() -> HealthResponse:
        """Readiness: reports which models are loaded and which failed."""
        errors = registry.errors()
        return HealthResponse(
            status="degraded" if errors else "ok",
            role=settings.role,
            loaded=registry.loaded(),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    @app.get(
        "/v1/models",
        response_model=ModelsResponse,
        response_model_exclude_none=True,
        tags=["models"],
        dependencies=auth,
    )
    def list_models() -> ModelsResponse:
        """List the catalog this instance serves, with load state."""
        return ModelsResponse(
            role=settings.role,
            models=[ModelInfo(**item) for item in registry.list_models()],
        )

    # ------------------------------------------------------------------
    # /v1/embed
    # ------------------------------------------------------------------
    @app.post(
        "/v1/embed",
        response_model=EmbedResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        dependencies=auth,
        responses={
            404: {"model": ErrorResponse, "description": "Unknown model name"},
            503: {"model": ErrorResponse, "description": "Role disabled here"},
        },
    )
    def embed(payload: EmbedRequest) -> EmbedResponse:
        """Embed texts with a named embedder model."""
        if not settings.embed_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"this instance serves role '{settings.role}', not embeddings",
            )

        normalize = True if payload.normalize is None else payload.normalize
        batch_size = payload.batch_size or settings.batch_size

        model = _load_model(registry, "embedder", payload.model)

        encode_kwargs = {"convert_to_numpy": True}
        if batch_size:
            encode_kwargs["batch_size"] = batch_size

        start = time.perf_counter()
        try:
            with registry.inference_lock("embedder", payload.model):
                vectors = model.encode(
                    sentences=payload.texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                    **encode_kwargs,
                )
        except Exception as exc:  # noqa: BLE001 - report, keep the server alive
            logger.exception("embedding failed for model=%s", payload.model)
            raise HTTPException(
                status_code=500, detail=f"embedding failed: {exc}"
            ) from exc
        took_ms = (time.perf_counter() - start) * 1000.0

        embeddings = [[float(value) for value in row] for row in vectors]
        dim = len(embeddings[0]) if embeddings else 0

        logger.info(
            "embed model=%s texts=%d dim=%d took=%.1fms",
            payload.model,
            len(payload.texts),
            dim,
            took_ms,
        )
        return EmbedResponse(
            model=payload.model,
            dim=dim,
            embeddings=embeddings,
            count=len(embeddings),
            took_ms=round(took_ms, 3),
        )

    # ------------------------------------------------------------------
    # /v1/rerank
    # ------------------------------------------------------------------
    @app.post(
        "/v1/rerank",
        response_model=RerankResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        dependencies=auth,
        responses={
            404: {"model": ErrorResponse, "description": "Unknown model name"},
            413: {"model": ErrorResponse, "description": "Too many documents"},
            503: {"model": ErrorResponse, "description": "Role disabled here"},
        },
    )
    def rerank(payload: RerankRequest) -> RerankResponse:
        """Score documents against a query with a named reranker model."""
        if not settings.rerank_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"this instance serves role '{settings.role}', not reranking",
            )

        if len(payload.documents) > settings.rerank_max_documents:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"{len(payload.documents)} documents exceeds the limit of "
                    f"{settings.rerank_max_documents}; narrow the candidate set "
                    "before reranking"
                ),
            )

        if not payload.query.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="query must not be empty",
            )

        batch_size = payload.batch_size or settings.batch_size

        model = _load_model(registry, "reranker", payload.model)

        pairs = [[payload.query, text] for text in payload.documents]

        # The engine calls predict(pairs) with no batch size; forwarding one
        # only when it was explicitly asked for keeps scores identical to the
        # in-process path (see the note in settings.py about pad tokens).
        predict_kwargs = {}
        if batch_size:
            predict_kwargs["batch_size"] = batch_size

        start = time.perf_counter()
        try:
            with registry.inference_lock("reranker", payload.model):
                scores = model.predict(pairs, **predict_kwargs)
        except Exception as exc:  # noqa: BLE001 - report, keep the server alive
            logger.exception("reranking failed for model=%s", payload.model)
            raise HTTPException(
                status_code=500, detail=f"reranking failed: {exc}"
            ) from exc
        took_ms = (time.perf_counter() - start) * 1000.0

        ordered = sorted(
            range(len(payload.documents)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        if payload.top_n is not None:
            ordered = ordered[: payload.top_n]

        hits = [
            RerankHit(
                index=index,
                score=float(scores[index]),
                document=(
                    payload.documents[index] if payload.return_documents else None
                ),
            )
            for index in ordered
        ]

        logger.info(
            "rerank model=%s documents=%d returned=%d took=%.1fms",
            payload.model,
            len(payload.documents),
            len(hits),
            took_ms,
        )
        return RerankResponse(
            model=payload.model,
            results=hits,
            count=len(hits),
            scored=len(payload.documents),
            took_ms=round(took_ms, 3),
        )

    # ------------------------------------------------------------------
    # /v1/tokenize and /v1/truncate
    # ------------------------------------------------------------------
    @app.post(
        "/v1/tokenize",
        response_model=TokenizeResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        dependencies=auth,
        responses={
            404: {"model": ErrorResponse, "description": "Unknown model name"},
            500: {"model": ErrorResponse, "description": "Bad model config"},
        },
    )
    def tokenize(payload: TokenizeRequest) -> TokenizeResponse:
        """Count the tokens of every text, using the tokenizer of a model.

        Only the tokenizer is loaded — counting tokens never pulls the model
        weights into memory.
        """
        kind = _resolve_kind(registry, payload.model)
        start = time.perf_counter()
        tokenizer = _load_tokenizer(registry, kind, payload.model)
        counts = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in payload.texts
        ]
        took_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "tokenize model=%s texts=%d took=%.1fms",
            payload.model,
            len(counts),
            took_ms,
        )
        return TokenizeResponse(
            model=payload.model,
            token_counts=counts,
            max_length=registry.max_length_of(kind, payload.model),
            took_ms=round(took_ms, 3),
        )

    @app.post(
        "/v1/truncate",
        response_model=TruncateResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        dependencies=auth,
        responses={
            404: {"model": ErrorResponse, "description": "Unknown model name"},
            500: {"model": ErrorResponse, "description": "Bad model config"},
        },
    )
    def truncate(payload: TruncateRequest) -> TruncateResponse:
        """Truncate texts to ``max_tokens`` (token-exact, model tokenizer)."""
        kind = _resolve_kind(registry, payload.model)
        start = time.perf_counter()
        tokenizer = _load_tokenizer(registry, kind, payload.model)

        truncated: List[str] = []
        was_truncated: List[bool] = []
        for text in payload.texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= payload.max_tokens:
                truncated.append(text)
                was_truncated.append(False)
                continue
            truncated.append(
                tokenizer.decode(
                    token_ids[: payload.max_tokens], skip_special_tokens=True
                )
            )
            was_truncated.append(True)

        took_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "truncate model=%s texts=%d cut=%d took=%.1fms",
            payload.model,
            len(truncated),
            sum(1 for flag in was_truncated if flag),
            took_ms,
        )
        return TruncateResponse(
            model=payload.model,
            truncated_texts=truncated,
            was_truncated=was_truncated,
            took_ms=round(took_ms, 3),
        )

    # ------------------------------------------------------------------
    # /v1/denoise
    # ------------------------------------------------------------------
    @app.post(
        "/v1/denoise",
        response_model=DenoiseResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        dependencies=auth,
        responses={
            500: {"model": ErrorResponse, "description": "Denoiser unavailable"},
            503: {"model": ErrorResponse, "description": "Role disabled here"},
        },
    )
    def denoise(payload: DenoiseRequest) -> DenoiseResponse:
        """Clean a noisy text (OCR artefacts, broken sentences) with the T5 denoiser."""
        if not settings.denoise_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"this instance serves role '{settings.role}', not denoising",
            )
        if not payload.text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="text must not be empty",
            )

        bundle = _load_denoiser(registry)
        max_model_input = payload.max_length or settings.denoise_max_model_input

        from sse_models.denoising import denoise_text

        start = time.perf_counter()
        try:
            with registry.inference_lock("denoiser", "denoiser"):
                cleaned, score = denoise_text(
                    bundle,
                    payload.text,
                    max_model_input=max_model_input,
                    batch_size=payload.batch_size or settings.batch_size or 50,
                )
        except Exception as exc:  # noqa: BLE001 - report, keep the server alive
            logger.exception("denoising failed")
            raise HTTPException(
                status_code=500, detail=f"denoising failed: {exc}"
            ) from exc
        took_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "denoise chars=%d score=%.3f took=%.1fms",
            len(payload.text),
            score,
            took_ms,
        )
        return DenoiseResponse(
            cleaned_text=cleaned,
            correctness_score=score,
            took_ms=round(took_ms, 3),
        )

    return app


def _load_model(registry: ModelRegistry, kind: str, name: str):
    """Map registry errors onto HTTP codes (404 unknown name, 500 bad config)."""
    try:
        return registry.get(kind, name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _resolve_kind(registry: ModelRegistry, name: str) -> str:
    """
    Which catalog a model name lives in ("embedder" or "reranker").

    A tokenizer request names a model, not a kind; the embedder catalog is
    checked first because that is the model whose tokenizer matters for
    chunking.
    """
    if registry.has_model("embedder", name):
        return "embedder"
    if registry.has_model("reranker", name):
        return "reranker"
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"no model named '{name}' in the catalog",
    )


def _load_tokenizer(registry: ModelRegistry, kind: str, name: str):
    try:
        return registry.tokenizer_for(kind, name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _load_denoiser(registry: ModelRegistry):
    try:
        return registry.denoiser()
    except ModelConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
