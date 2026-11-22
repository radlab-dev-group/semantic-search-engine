"""
backend.py
----------

Backend abstraction for the two model operations the search engine performs:
embedding texts and reranking ``query``/``passage`` pairs.

Two implementations
-------------------
``LocalBackend``
    Loads the models in the engine process with ``sentence_transformers``.
    This is the historical behaviour and stays the default: nothing changes
    when the feature is off.

``RemoteBackend``
    Calls the standalone model service (``sse_models``) over HTTP.  The models
    then live outside the engine, so they can sit on a GPU box, scale per
    operation and reload without restarting the engine.

Both return the same shapes, so ``MilvusHandler`` does not care which one it
has.  ``build_backend()`` decides from the environment.

Why a fallback rather than a hard failure
-----------------------------------------
Reranking is an enhancement: a search that skips it still returns results, one
that raises returns nothing.  When the service is unreachable the
``RemoteBackend`` therefore degrades to the local backend (when a fallback is
configured) and, failing that, reports the failure to the caller, which is
already built to treat a missing rerank as non-fatal.  A short circuit-breaker
cooldown keeps a dead service from adding a timeout to every single query.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from sse_api.core.constants import get_logger

# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
#: Master switch for the whole feature.  Off by default: the engine keeps
#: loading its models in-process exactly as before.
ENV_MODEL_SERVICE_ENABLED = "SSE_MODEL_SERVICE_ENABLED"
#: Base url of the model service, e.g. ``http://192.168.100.70:8280``.
ENV_MODEL_SERVICE_URL = "SSE_MODEL_SERVICE_URL"
#: Shared secret sent as ``Authorization: Bearer`` (optional).
ENV_MODEL_SERVICE_TOKEN = "SSE_MODEL_SERVICE_TOKEN"
#: Overall request timeout.  A remote embed/rerank must never outlive the
#: engine's own search budget by more than this.
ENV_MODEL_SERVICE_TIMEOUT = "SSE_MODEL_SERVICE_TIMEOUT_SECONDS"
#: Separate connect timeout: a service that is simply not there should be
#: recognised in milliseconds, not after the full read timeout.
ENV_MODEL_SERVICE_CONNECT_TIMEOUT = "SSE_MODEL_SERVICE_CONNECT_TIMEOUT_SECONDS"
#: Explicit model names to send to the service.  When unset, the engine's
#: configured model path is resolved against configs/embedders.json and
#: configs/rerankers.json to find the right name.
ENV_MODEL_SERVICE_EMBED_MODEL = "SSE_MODEL_SERVICE_EMBED_MODEL"
ENV_MODEL_SERVICE_RERANK_MODEL = "SSE_MODEL_SERVICE_RERANK_MODEL"
#: Fall back to in-process models when the service fails.
ENV_MODEL_SERVICE_FALLBACK_LOCAL = "SSE_MODEL_SERVICE_FALLBACK_LOCAL"
#: Seconds to stop calling the service after a failure (circuit breaker).
ENV_MODEL_SERVICE_COOLDOWN = "SSE_MODEL_SERVICE_COOLDOWN_SECONDS"
#: How many consecutive failures trip the breaker.
ENV_MODEL_SERVICE_FAILURE_THRESHOLD = "SSE_MODEL_SERVICE_FAILURE_THRESHOLD"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 3.0
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_FAILURE_THRESHOLD = 3


class ModelBackendError(Exception):
    """A model operation could not be completed by any available backend."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# ----------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------
class ModelBackend:
    """
    Operations the engine needs from an embedder and a reranker.

    ``embed`` returns one vector per input text, in input order.
    ``rerank_scores`` returns one score per input text, in input order — the
    caller sorts, so the backend never reorders or drops a candidate.
    """

    def embed(
        self, texts: Sequence[str], normalize: Optional[bool] = None
    ) -> List[List[float]]:
        raise NotImplementedError

    def rerank_scores(self, query: str, texts: Sequence[str]) -> List[float]:
        raise NotImplementedError

    # -- text operations (tokenization / truncation / denoising) ---------

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """Token count of every text, per the embedder's tokenizer."""
        raise NotImplementedError

    def truncate_texts(
        self, texts: Sequence[str], max_tokens: int
    ) -> Tuple[List[str], List[bool]]:
        """
        Truncate texts to ``max_tokens`` tokens.

        Returns ``(texts, was_truncated)`` — one entry per input text, in
        input order; ``was_truncated[i]`` is True when the i-th text was cut.
        """
        raise NotImplementedError

    def chunk_texts(
        self, texts: Sequence[str], max_tokens: int, overlap: int = 0
    ) -> List[List[str]]:
        """
        Split texts into chunks of at most ``max_tokens`` tokens.

        Returns one list of chunks per input text, in input order; a text
        which fits the budget comes back as a single chunk.  Chunking is
        measured in the tokens of the embedder, which is why it is a model
        operation and not a string operation.
        """
        raise NotImplementedError

    def denoise_text(self, text: str) -> str:
        """Clean ``text`` (OCR artefacts, broken sentences) via the denoiser."""
        raise NotImplementedError

    def extractive_qa(
        self, question: str, contexts: Sequence[str]
    ) -> List[Tuple[str, float]]:
        """
        Answer ``question`` with a span of every context.

        Returns one ``(answer, score)`` pair per input context, in input
        order; the caller keeps its own ranking of which passage answered.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Short human-readable summary for the logs."""
        return self.__class__.__name__


# ----------------------------------------------------------------------
# local
# ----------------------------------------------------------------------
class LocalBackend(ModelBackend):
    """
    In-process backend wrapping already-loaded model objects.

    It takes zero-argument callables that return the model rather than the
    models themselves: ``MilvusHandler`` loads an embedder eagerly and a
    reranker lazily (only when a query asks for reranking), and the backend
    must not disturb that ordering.
    """

    def __init__(
        self,
        embedder_provider: Optional[Callable[[], Any]] = None,
        reranker_provider: Optional[Callable[[], Any]] = None,
        normalize_embeddings: bool = True,
        label: str = "local",
    ) -> None:
        self._embedder_provider = embedder_provider
        self._reranker_provider = reranker_provider
        self._normalize_embeddings = normalize_embeddings
        self._label = label
        self._logger = get_logger()

    def embed(
        self, texts: Sequence[str], normalize: Optional[bool] = None
    ) -> List[List[float]]:
        import torch

        if self._embedder_provider is None:
            raise ModelBackendError("no embedder model configured")

        model = self._embedder_provider()
        if model is None:
            raise ModelBackendError("embedder model is not loaded")

        do_normalize = (
            self._normalize_embeddings if normalize is None else bool(normalize)
        )
        with torch.no_grad():
            vectors = model.encode(
                sentences=list(texts),
                normalize_embeddings=do_normalize,
                show_progress_bar=False,
            )
        return [[float(value) for value in row] for row in vectors]

    def rerank_scores(self, query: str, texts: Sequence[str]) -> List[float]:
        if self._reranker_provider is None:
            raise ModelBackendError("no reranker model configured")

        model = self._reranker_provider()
        if model is None:
            raise ModelBackendError("reranker model is not loaded")

        pairs = [[query, text] for text in texts]
        scores = model.predict(pairs)
        return [float(score) for score in scores]

    # -- text operations -------------------------------------------------
    def _tokenizer(self):
        """Tokenizer of the in-process embedder, loaded through its provider."""
        if self._embedder_provider is None:
            raise ModelBackendError("no embedder model configured")
        model = self._embedder_provider()
        if model is None:
            raise ModelBackendError("embedder model is not loaded")
        tokenizer = getattr(model, "tokenizer", None) or getattr(
            model, "tokenizer_model", None
        )
        if tokenizer is None:
            raise ModelBackendError("embedder exposes no tokenizer")
        return tokenizer

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        tokenizer = self._tokenizer()
        return [len(tokenizer.encode(text)) for text in texts]

    def truncate_texts(
        self, texts: Sequence[str], max_tokens: int
    ) -> Tuple[List[str], List[bool]]:
        tokenizer = self._tokenizer()
        truncated: List[str] = []
        was_truncated: List[bool] = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            if len(token_ids) <= max_tokens:
                truncated.append(text)
                was_truncated.append(False)
                continue
            truncated.append(
                tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
            )
            was_truncated.append(True)
        return truncated, was_truncated

    def chunk_texts(
        self, texts: Sequence[str], max_tokens: int, overlap: int = 0
    ) -> List[List[str]]:
        tokenizer = self._tokenizer()
        step = max_tokens - max(0, min(overlap, max_tokens - 1))
        chunked: List[List[str]] = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_tokens:
                chunked.append([text])
                continue
            windows = [
                tokenizer.decode(
                    token_ids[position : position + max_tokens],
                    skip_special_tokens=True,
                )
                for position in range(0, len(token_ids), step)
            ]
            chunked.append([window for window in windows if window] or [text])
        return chunked

    def denoise_text(self, text: str) -> str:
        # The local denoiser is the engine's historical T5 path; it is kept
        # importable but torch arrives only when a text is really cleaned.
        from sse_api.data.controllers.denoiser import denoise_text_locally

        return denoise_text_locally(text)

    def extractive_qa(
        self, question: str, contexts: Sequence[str]
    ) -> List[Tuple[str, float]]:
        # The local QA pipeline is the engine's historical path: transformers
        # arrives only when an answer is really extracted in this process.
        from sse_api.data.controllers.extractive_qa_local import (
            answer_locally,
        )

        return answer_locally(question, list(contexts))

    def describe(self) -> str:
        return f"local({self._label})"


# ----------------------------------------------------------------------
# remote
# ----------------------------------------------------------------------
class RemoteBackend(ModelBackend):
    """
    Backend that calls the standalone model service over HTTP.

    Failures are counted; once ``failure_threshold`` consecutive failures have
    happened the backend opens its circuit for ``cooldown_seconds`` and goes
    straight to the fallback instead of paying a timeout on every query.  With
    no fallback configured the error is raised to the caller.
    """

    def __init__(
        self,
        base_url: str,
        embed_model: str,
        rerank_model: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        fallback: Optional[ModelBackend] = None,
        session: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._api_token = api_token or ""
        self._timeout = (connect_timeout, timeout)
        self._cooldown = cooldown_seconds
        self._failure_threshold = failure_threshold
        self._fallback = fallback

        self._logger = get_logger()
        self._session = session
        self._lock = threading.Lock()
        self._failures = 0
        self._circuit_open_until = 0.0
        # Failures are logged once per state change, not once per query: a dead
        # service under live traffic would otherwise flood the log.
        self._last_error_logged = False

    # ------------------------------------------------------------------
    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _record_success(self) -> None:
        with self._lock:
            if self._failures:
                self._logger.info("model service recovered: %s", self._base_url)
            self._failures = 0
            self._circuit_open_until = 0.0
            self._last_error_logged = False

    def _record_failure(self, message: str) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._circuit_open_until = time.monotonic() + self._cooldown
                self._failures = 0
            if not self._last_error_logged:
                self._last_error_logged = True
                self._logger.warning(
                    "model service call failed (%s); using %s",
                    message,
                    "local fallback" if self._fallback else "no fallback",
                )

    def _call(self, path: str, payload: dict) -> dict:
        """POST to the service, translating transport and HTTP errors."""
        url = f"{self._base_url}{path}"
        try:
            response = self._http().post(
                url, json=payload, headers=self._headers(), timeout=self._timeout
            )
        except Exception as exc:  # requests.* — connection/timeout problems
            raise ModelBackendError(f"{url} unreachable: {exc}") from exc

        if response.status_code == 404:
            # Unknown model name is a configuration error, not a service
            # outage.  It still opens the circuit (see _is_config_error): the
            # answer cannot change between two queries, so repeating the call
            # only adds latency to every search.
            raise ModelBackendError(
                f"model service rejected the model: {self._detail(response)}"
            )
        if response.status_code in (401, 403):
            raise ModelBackendError(
                f"model service refused the token: {self._detail(response)}"
            )
        if response.status_code >= 400:
            raise ModelBackendError(
                f"{url} responded {response.status_code}: {self._detail(response)}"
            )

        try:
            return response.json()
        except Exception as exc:  # malformed body
            raise ModelBackendError(f"{url} returned a non-JSON body") from exc

    @staticmethod
    def _detail(response: Any) -> str:
        try:
            body = response.json()
        except Exception:
            return (response.text or "")[:200]
        if isinstance(body, dict):
            return str(body.get("detail") or body)[:200]
        return str(body)[:200]

    def _is_config_error(self, error: ModelBackendError) -> bool:
        """
        True when retrying cannot possibly help.

        A rejected model name or token is a misconfiguration: the service will
        answer the same way to the next call, so the circuit opens immediately
        instead of after ``failure_threshold`` identical round-trips.
        """
        text = str(error)
        return "rejected the model" in text or "refused the token" in text

    # ------------------------------------------------------------------
    def embed(
        self, texts: Sequence[str], normalize: Optional[bool] = None
    ) -> List[List[float]]:
        payload = {
            "model": self._embed_model,
            "texts": list(texts),
        }
        if normalize is not None:
            payload["normalize"] = bool(normalize)
        data = self._run("/v1/embed", payload)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ModelBackendError(
                "model service returned an unexpected embedding payload"
            )
        return [list(map(float, row)) for row in vectors]

    def rerank_scores(self, query: str, texts: Sequence[str]) -> List[float]:
        if not self._rerank_model:
            raise ModelBackendError(
                "no reranker model name configured for the service"
            )

        payload = {
            "model": self._rerank_model,
            "query": query,
            "documents": list(texts),
        }
        data = self._run("/v1/rerank", payload)
        results = data.get("results")
        if not isinstance(results, list):
            raise ModelBackendError(
                "model service returned an unexpected rerank payload"
            )

        # The service returns hits sorted best-first with their original index;
        # rebuild the input order so the caller's own sort stays authoritative.
        scores: List[Optional[float]] = [None] * len(texts)
        for hit in results:
            index = hit.get("index")
            if isinstance(index, int) and 0 <= index < len(texts):
                scores[index] = float(hit.get("score", 0.0))
        if any(score is None for score in scores):
            raise ModelBackendError(
                "model service returned scores for only part of the candidates"
            )
        return [float(score) for score in scores]  # type: ignore[arg-type]

    # -- text operations -------------------------------------------------
    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        data = self._run(
            "/v1/tokenize",
            {"model": self._embed_model, "texts": list(texts)},
        )
        counts = data.get("token_counts")
        if not isinstance(counts, list) or len(counts) != len(texts):
            raise ModelBackendError(
                "model service returned an unexpected tokenize payload"
            )
        return [int(count) for count in counts]

    def truncate_texts(
        self, texts: Sequence[str], max_tokens: int
    ) -> Tuple[List[str], List[bool]]:
        data = self._run(
            "/v1/truncate",
            {
                "model": self._embed_model,
                "texts": list(texts),
                "max_tokens": int(max_tokens),
            },
        )
        truncated = data.get("truncated_texts")
        flags = data.get("was_truncated")
        if (
            not isinstance(truncated, list)
            or len(truncated) != len(texts)
            or not isinstance(flags, list)
            or len(flags) != len(texts)
        ):
            raise ModelBackendError(
                "model service returned an unexpected truncate payload"
            )
        return [str(text) for text in truncated], [bool(flag) for flag in flags]

    def chunk_texts(
        self, texts: Sequence[str], max_tokens: int, overlap: int = 0
    ) -> List[List[str]]:
        data = self._run(
            "/v1/chunk",
            {
                "model": self._embed_model,
                "texts": list(texts),
                "max_tokens": int(max_tokens),
                "overlap": int(overlap),
            },
        )
        chunks = data.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(texts):
            raise ModelBackendError(
                "model service returned an unexpected chunk payload"
            )
        document_chunks: List[List[str]] = []
        for entry in chunks:
            if not isinstance(entry, list) or not all(
                isinstance(part, str) for part in entry
            ):
                raise ModelBackendError(
                    "model service returned an unexpected chunk of a text"
                )
            document_chunks.append([str(part) for part in entry])
        return document_chunks

    def denoise_text(self, text: str) -> str:
        data = self._run("/v1/denoise", {"text": text})
        cleaned = data.get("cleaned_text")
        if not isinstance(cleaned, str):
            raise ModelBackendError(
                "model service returned an unexpected denoise payload"
            )
        return cleaned

    def extractive_qa(
        self, question: str, contexts: Sequence[str]
    ) -> List[Tuple[str, float]]:
        data = self._run(
            "/v1/extractive_qa",
            {"question": question, "contexts": list(contexts)},
        )
        answers = data.get("answers")
        if not isinstance(answers, list) or len(answers) != len(contexts):
            raise ModelBackendError(
                "model service returned an unexpected extractive QA payload"
            )
        pairs: List[Tuple[str, float]] = []
        for entry in answers:
            if not isinstance(entry, dict):
                raise ModelBackendError(
                    "model service returned an unexpected extractive QA answer"
                )
            pairs.append(
                (str(entry.get("answer") or ""), float(entry.get("score") or 0.0))
            )
        return pairs

    def _run(self, path: str, payload: dict) -> dict:
        if self.circuit_open:
            return self._degrade(
                ModelBackendError("model service circuit is open"), path, payload
            )
        try:
            data = self._call(path, payload)
        except ModelBackendError as exc:
            if self._is_config_error(exc):
                self._open_circuit()
            else:
                self._record_failure(str(exc))
            return self._degrade(exc, path, payload)
        self._record_success()
        return data

    def _open_circuit(self) -> None:
        with self._lock:
            self._circuit_open_until = time.monotonic() + self._cooldown

    def _degrade(self, error: ModelBackendError, path: str, payload: dict) -> dict:
        if self._fallback is None:
            raise error
        # The fallback answers locally, so nothing more is sent to the service.
        if path == "/v1/embed":
            return {"embeddings": self._fallback.embed(payload["texts"])}
        if path == "/v1/extractive_qa":
            return {
                "answers": [
                    {"answer": answer, "score": score}
                    for answer, score in self._fallback.extractive_qa(
                        payload["question"], payload["contexts"]
                    )
                ]
            }
        if path == "/v1/tokenize":
            return {"token_counts": self._fallback.count_tokens(payload["texts"])}
        if path == "/v1/truncate":
            truncated, was_truncated = self._fallback.truncate_texts(
                payload["texts"], payload["max_tokens"]
            )
            return {
                "truncated_texts": truncated,
                "was_truncated": was_truncated,
            }
        if path == "/v1/chunk":
            return {
                "chunks": self._fallback.chunk_texts(
                    payload["texts"], payload["max_tokens"], payload["overlap"]
                )
            }
        if path == "/v1/denoise":
            return {"cleaned_text": self._fallback.denoise_text(payload["text"])}
        return {
            "results": [
                {"index": index, "score": score}
                for index, score in enumerate(
                    self._fallback.rerank_scores(
                        payload["query"], payload["documents"]
                    )
                )
            ]
        }

    def describe(self) -> str:
        return f"remote({self._base_url}, embed={self._embed_model})"


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------
def service_enabled() -> bool:
    """True when the engine should reach models through the model service."""
    return _env_bool(ENV_MODEL_SERVICE_ENABLED, False) and bool(
        os.environ.get(ENV_MODEL_SERVICE_URL, "").strip()
    )


def service_backend_for(embed_model: str) -> Optional[RemoteBackend]:
    """
    A :class:`RemoteBackend` bound to ``embed_model``, or ``None``.

    For callers which need one text operation of the service (counting or
    truncating tokens) without an embedder model of their own — a management
    command auditing chunk sizes, for instance.  ``None`` when the service is
    not configured, which tells the caller to use its local fallback.
    """
    if not service_enabled() or not embed_model:
        return None
    return RemoteBackend(
        base_url=os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/"),
        embed_model=embed_model,
        api_token=os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip(),
        timeout=_env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS),
        connect_timeout=_env_float(
            ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
        cooldown_seconds=_env_float(
            ENV_MODEL_SERVICE_COOLDOWN, DEFAULT_COOLDOWN_SECONDS
        ),
        failure_threshold=_env_int(
            ENV_MODEL_SERVICE_FAILURE_THRESHOLD, DEFAULT_FAILURE_THRESHOLD
        ),
        fallback=None,
    )


def build_denoise_client() -> Optional[Callable[[str], str]]:
    """
    Build a ``text -> cleaned text`` client for the model service, or ``None``.

    Denoising is a model-service operation and the engine must not need torch
    for it; this is the one place the HTTP shape of ``POST /v1/denoise`` is
    known on the engine side.  ``None`` means "no service configured", and
    the caller decides what cleaning means then (usually: nothing).
    """
    if not service_enabled():
        return None

    base_url = os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/")
    api_token = os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip()
    timeout = _env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS)
    connect_timeout = _env_float(
        ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
    )

    def _denoise(text: str) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.post(
            f"{base_url}/v1/denoise",
            json={"text": text},
            headers=headers,
            timeout=(connect_timeout, timeout),
        )
        if response.status_code >= 400:
            raise ModelBackendError(
                f"{base_url}/v1/denoise responded {response.status_code}"
            )
        body = response.json()
        cleaned = body.get("cleaned_text")
        if not isinstance(cleaned, str):
            raise ModelBackendError(
                "model service returned an unexpected denoise payload"
            )
        return cleaned

    return _denoise


def build_extractive_qa_client() -> (
    Optional[Callable[[str, Sequence[str]], List[Tuple[str, float]]]]
):
    """
    Build a ``(question, contexts) -> answers`` client for the model service.

    Extractive QA is a model-service operation and the engine must not need
    torch for it; this is the one place the HTTP shape of
    ``POST /v1/extractive_qa`` is known on the engine side.  ``None`` means
    "no service configured", and the caller decides what answering means then
    (the in-process pipeline, or nothing).
    """
    if not service_enabled():
        return None

    base_url = os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/")
    api_token = os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip()
    timeout = _env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS)
    connect_timeout = _env_float(
        ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
    )

    def _answer(question: str, contexts: Sequence[str]) -> List[Tuple[str, float]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.post(
            f"{base_url}/v1/extractive_qa",
            json={"question": question, "contexts": list(contexts)},
            headers=headers,
            timeout=(connect_timeout, timeout),
        )
        if response.status_code >= 400:
            raise ModelBackendError(
                f"{base_url}/v1/extractive_qa responded {response.status_code}"
            )
        body = response.json()
        answers = body.get("answers")
        if not isinstance(answers, list) or len(answers) != len(contexts):
            raise ModelBackendError(
                "model service returned an unexpected extractive QA payload"
            )
        return [
            (
                str(entry.get("answer") or ""),
                float(entry.get("score") or 0.0),
            )
            for entry in answers
        ]

    return _answer


def build_truncate_client(
    embed_model: Optional[str],
) -> Optional[Callable[[Sequence[str], int], Tuple[List[str], List[bool]]]]:
    """
    Build a ``(texts, max_tokens) -> (truncated, was_truncated)`` client.

    Truncation is a model-service operation (the embedder's own tokenizer on
    the service side, ``POST /v1/truncate``) and the engine must not need
    ``transformers`` for it.  ``None`` means "no service configured or no
    model to address", and the caller then decides what cutting means —
    usually: nothing.
    """
    if not service_enabled() or not embed_model:
        return None

    base_url = os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/")
    api_token = os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip()
    timeout = _env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS)
    connect_timeout = _env_float(
        ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
    )

    def _truncate(
        texts: Sequence[str], max_tokens: int
    ) -> Tuple[List[str], List[bool]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.post(
            f"{base_url}/v1/truncate",
            json={
                "model": embed_model,
                "texts": list(texts),
                "max_tokens": int(max_tokens),
            },
            headers=headers,
            timeout=(connect_timeout, timeout),
        )
        if response.status_code >= 400:
            raise ModelBackendError(
                f"{base_url}/v1/truncate responded {response.status_code}"
            )
        body = response.json()
        truncated = body.get("truncated_texts")
        flags = body.get("was_truncated")
        if (
            not isinstance(truncated, list)
            or len(truncated) != len(texts)
            or not isinstance(flags, list)
            or len(flags) != len(texts)
        ):
            raise ModelBackendError(
                "model service returned an unexpected truncate payload"
            )
        return [str(text) for text in truncated], [bool(flag) for flag in flags]

    return _truncate


def build_chunk_client(
    embed_model: Optional[str],
) -> Optional[Callable[[Sequence[str], int, int], List[List[str]]]]:
    """
    Build a ``(texts, max_tokens, overlap) -> chunks`` client.

    Token chunking is a model-service operation: the chunk size is counted in
    the tokens of the embedder of the collection, and that tokenizer lives on
    the service (``POST /v1/chunk``).  ``None`` means "no service configured
    or no model to address" and the caller decides what chunking means then —
    usually: the historical in-process chunker of the document reader.
    """
    if not service_enabled() or not embed_model:
        return None

    base_url = os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/")
    api_token = os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip()
    timeout = _env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS)
    connect_timeout = _env_float(
        ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
    )

    def _chunk(
        texts: Sequence[str], max_tokens: int, overlap: int = 0
    ) -> List[List[str]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.post(
            f"{base_url}/v1/chunk",
            json={
                "model": embed_model,
                "texts": list(texts),
                "max_tokens": int(max_tokens),
                "overlap": int(overlap),
            },
            headers=headers,
            timeout=(connect_timeout, timeout),
        )
        if response.status_code >= 400:
            raise ModelBackendError(
                f"{base_url}/v1/chunk responded {response.status_code}"
            )
        body = response.json()
        chunks = body.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(texts):
            raise ModelBackendError(
                "model service returned an unexpected chunk payload"
            )
        return [
            (
                [str(part) for part in entry]
                if isinstance(entry, list)
                else [str(entry)]
            )
            for entry in chunks
        ]

    return _chunk


def resolve_model_name(kind: str, model_path: Optional[str]) -> Optional[str]:
    """
    Map a model *path* back to its catalog *name*.

    The engine configures collections by filesystem path while the model
    service addresses models by the name in ``configs/embedders.json`` /
    ``configs/rerankers.json``.  Looking the path up in the already-loaded
    config gives the right name without a second mapping to maintain.  The
    directory name is the fallback when a path is not in the catalog.
    """
    if not model_path:
        return None

    try:  # imported lazily: pulls Django settings, which must stay optional here
        from sse_api.engine.core.config.models_config import (
            ALL_AVAILABLE_EMBEDDERS_MODELS,
            ALL_AVAILABLE_RERANKERS_MODELS,
        )
    except Exception:  # pragma: no cover - config layer unavailable
        catalog = {}
    else:
        catalog = (
            ALL_AVAILABLE_EMBEDDERS_MODELS
            if kind == "embedder"
            else ALL_AVAILABLE_RERANKERS_MODELS
        )

    normalized = os.path.normpath(model_path)
    for name, entry in catalog.items():
        entry_path = entry.get("path")
        if entry_path and os.path.normpath(entry_path) == normalized:
            return name

    return os.path.basename(normalized) or None


def build_backend(
    embedder_provider: Optional[Callable[[], Any]] = None,
    reranker_provider: Optional[Callable[[], Any]] = None,
    embedder_model_path: Optional[str] = None,
    reranker_model_path: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> ModelBackend:
    """
    Build the backend the environment asks for.

    With the feature off (the default) this returns a :class:`LocalBackend`
    wrapping the given providers, which is exactly the pre-existing behaviour.
    With it on, a :class:`RemoteBackend` is returned; it keeps the local
    providers as an optional fallback so a service outage degrades latency and
    quality rather than breaking search.
    """
    local = LocalBackend(
        embedder_provider=embedder_provider,
        reranker_provider=reranker_provider,
        normalize_embeddings=normalize_embeddings,
    )

    if not service_enabled():
        return local

    base_url = os.environ.get(ENV_MODEL_SERVICE_URL, "").strip().rstrip("/")
    embed_model = os.environ.get(
        ENV_MODEL_SERVICE_EMBED_MODEL
    ) or resolve_model_name("embedder", embedder_model_path)
    rerank_model = os.environ.get(
        ENV_MODEL_SERVICE_RERANK_MODEL
    ) or resolve_model_name("reranker", reranker_model_path)

    if not embed_model:
        # Nothing to address on the service side; better to stay local and say
        # so than to send a request the service cannot serve.
        get_logger().warning(
            "%s is on but no embedder model name could be resolved for %s; "
            "continuing with in-process models",
            ENV_MODEL_SERVICE_ENABLED,
            embedder_model_path,
        )
        return local

    return RemoteBackend(
        base_url=base_url,
        embed_model=embed_model,
        rerank_model=rerank_model,
        api_token=os.environ.get(ENV_MODEL_SERVICE_TOKEN, "").strip(),
        timeout=_env_float(ENV_MODEL_SERVICE_TIMEOUT, DEFAULT_TIMEOUT_SECONDS),
        connect_timeout=_env_float(
            ENV_MODEL_SERVICE_CONNECT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
        cooldown_seconds=_env_float(
            ENV_MODEL_SERVICE_COOLDOWN, DEFAULT_COOLDOWN_SECONDS
        ),
        failure_threshold=_env_int(
            ENV_MODEL_SERVICE_FAILURE_THRESHOLD, DEFAULT_FAILURE_THRESHOLD
        ),
        fallback=(
            local if _env_bool(ENV_MODEL_SERVICE_FALLBACK_LOCAL, True) else None
        ),
    )
