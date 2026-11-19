"""
Tests for the engine-side model backend abstraction.

Covers the factory (feature flag, name resolution, fallback wiring) and the
remote transport (payload shapes, index mapping, the circuit breaker and the
degrade-to-fallback behaviour).  No models are loaded: the providers and the
HTTP session are fakes.
"""

import pytest

from sse_api.engine.core.models import backend as b


class FakeSession:
    """Minimal stand-in for requests.Session that records and replays posts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def ok_embed(dim=2, count=2):
    embeddings = [
        [1.0 if i == row else 0.0 for i in range(dim)] for row in range(count)
    ]
    return FakeResponse(
        200, {"model": "m", "dim": dim, "embeddings": embeddings, "count": count}
    )


def ok_rerank():
    # best first; index 1 is the top hit
    return FakeResponse(
        200,
        {
            "model": "m",
            "results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}],
        },
    )


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------
def test_feature_off_by_default(monkeypatch):
    monkeypatch.delenv(b.ENV_MODEL_SERVICE_ENABLED, raising=False)
    backend = b.build_backend()
    assert isinstance(backend, b.LocalBackend)


def test_feature_on_without_url_stays_local(monkeypatch):
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_ENABLED, "1")
    monkeypatch.delenv(b.ENV_MODEL_SERVICE_URL, raising=False)
    assert isinstance(b.build_backend(), b.LocalBackend)


def test_feature_on_builds_remote(monkeypatch, caplog):
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_ENABLED, "1")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_URL, "http://models:8280/")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_EMBED_MODEL, "google/embeddinggemma-300m")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_RERANK_MODEL, "qwen/Qwen3-Reranker-0.6")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_TOKEN, "sekret")

    backend = b.build_backend()
    assert isinstance(backend, b.RemoteBackend)
    assert backend._base_url == "http://models:8280"  # trailing slash stripped
    assert backend._embed_model == "google/embeddinggemma-300m"
    assert backend._rerank_model == "qwen/Qwen3-Reranker-0.6"
    assert backend._api_token == "sekret"
    # a local backend is kept as the fallback by default
    assert isinstance(backend._fallback, b.LocalBackend)


def test_feature_on_without_fallback(monkeypatch):
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_ENABLED, "1")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_URL, "http://models:8280")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_EMBED_MODEL, "m")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_FALLBACK_LOCAL, "0")
    assert b.build_backend()._fallback is None


def test_unresolvable_embedder_model_stays_local(monkeypatch, caplog):
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_ENABLED, "1")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_URL, "http://models:8280")
    monkeypatch.setenv(b.ENV_MODEL_SERVICE_EMBED_MODEL, "")
    # no embedder path -> no name can be resolved
    backend = b.build_backend(embedder_model_path=None)
    assert isinstance(backend, b.LocalBackend)


# ----------------------------------------------------------------------
# name resolution
# ----------------------------------------------------------------------
def test_resolve_model_name_uses_catalog_when_loaded(monkeypatch):
    # The catalog is populated by the config loader; simulate it being loaded.
    from sse_api.engine.core.config import models_config as mc

    monkeypatch.setitem(
        mc.ALL_AVAILABLE_EMBEDDERS_MODELS,
        "google/embeddinggemma-300m",
        {"path": "/mnt/data2/llms/models/community/google/embeddinggemma-300m"},
    )
    assert (
        b.resolve_model_name(
            "embedder", "/mnt/data2/llms/models/community/google/embeddinggemma-300m"
        )
        == "google/embeddinggemma-300m"
    )


def test_resolve_model_name_falls_back_to_basename():
    assert (
        b.resolve_model_name("embedder", "/opt/models/some-model-9")
        == "some-model-9"
    )


def test_resolve_model_name_none_for_blank():
    assert b.resolve_model_name("embedder", "") is None


# ----------------------------------------------------------------------
# remote transport
# ----------------------------------------------------------------------
def _remote(session, **kwargs):
    return b.RemoteBackend(
        base_url="http://models:8280",
        embed_model="emb",
        rerank_model="rrk",
        fallback=None,
        session=session,
        **kwargs,
    )


def test_remote_embed_posts_expected_payload():
    session = FakeSession([ok_embed()])
    backend = _remote(session)
    vectors = backend.embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    call = session.calls[0]
    assert call["url"] == "http://models:8280/v1/embed"
    assert call["json"] == {"model": "emb", "texts": ["a", "b"]}


def test_remote_embed_sends_token():
    session = FakeSession([ok_embed()])
    backend = _remote(session)
    backend._api_token = "sekret"
    backend.embed(["a", "b"])
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sekret"


def test_remote_rerank_reorders_scores_back_to_input_order():
    session = FakeSession([ok_rerank()])
    backend = _remote(session)
    scores = backend.rerank_scores("q", ["doc0", "doc1"])
    # input order: doc0 scored 0.2 (rank 2), doc1 scored 0.9 (rank 1)
    assert scores == [0.2, 0.9]
    call = session.calls[0]
    assert call["json"] == {
        "model": "rrk",
        "query": "q",
        "documents": ["doc0", "doc1"],
    }


def test_remote_rerank_missing_score_raises():
    session = FakeSession(
        [FakeResponse(200, {"results": [{"index": 0, "score": 0.5}]})]
    )
    backend = _remote(session)
    with pytest.raises(b.ModelBackendError):
        backend.rerank_scores("q", ["doc0", "doc1"])


def test_remote_unknown_model_opens_circuit_immediately():
    """A rejected model will not fix itself, so stop calling for a while."""
    session = FakeSession([FakeResponse(404, {"detail": "unknown model"})])
    backend = _remote(session)
    with pytest.raises(b.ModelBackendError):
        backend.embed(["a", "b"])
    assert backend.circuit_open is True


def test_remote_http_error_without_fallback_raises():
    session = FakeSession([FakeResponse(500, {"detail": "boom"})])
    backend = _remote(session)
    with pytest.raises(b.ModelBackendError):
        backend.embed(["a"])


def test_remote_falls_back_to_local_on_outage():
    class Local:
        def embed(self, texts, normalize=None):
            return [[0.0] for _ in texts]

        def rerank_scores(self, query, texts):
            return [0.5] * len(texts)

        def describe(self):
            return "local-fallback"

    session = FakeSession([FakeResponse(503, {"detail": "service down"})])
    backend = b.RemoteBackend(
        base_url="http://models:8280",
        embed_model="emb",
        rerank_model="rrk",
        fallback=Local(),
        session=session,
    )
    assert backend.embed(["a"]) == [[0.0]]
    assert backend.rerank_scores("q", ["a", "b"]) == [0.5, 0.5]


def test_circuit_opens_after_threshold_then_degrades_without_calling():
    class Local:
        def embed(self, texts, normalize=None):
            return [[9.9] for _ in texts]

        def rerank_scores(self, query, texts):
            return [9.9] * len(texts)

    # one outage per call; breaker trips after 2 failures (default threshold 3)
    session = FakeSession(
        [
            FakeResponse(500, {"detail": "down"}),
            FakeResponse(500, {"detail": "down"}),
        ]
    )
    backend = b.RemoteBackend(
        base_url="http://models:8280",
        embed_model="emb",
        rerank_model="rrk",
        fallback=Local(),
        session=session,
        failure_threshold=2,
    )
    backend.embed(["a"])  # failure 1
    backend.embed(["a"])  # failure 2 -> circuit opens
    assert backend.circuit_open is True
    calls_before = len(session.calls)
    backend.embed(["a"])  # must NOT hit the service again
    assert len(session.calls) == calls_before
    assert backend.embed(["a"]) == [[9.9]]


def test_success_resets_failures():
    session = FakeSession([ok_embed(count=1), ok_embed(count=1)])
    backend = b.RemoteBackend(
        base_url="http://models:8280",
        embed_model="emb",
        rerank_model="rrk",
        session=session,
    )
    backend._record_failure("x")
    backend._failures = 1
    backend.embed(["a"])
    assert backend._failures == 0
