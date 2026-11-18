"""
Tests for the model service HTTP API.

No machine-learning stack is needed: a fake registry hands out stub embedder /
reranker objects, so the tests cover the contract (status codes, ordering,
index mapping, auth, limits) rather than the numerics.
"""

import pytest
from fastapi.testclient import TestClient

from sse_models.app import create_app
from sse_models.registry import ModelNotFoundError
from sse_models.settings import ModelServiceSettings

EMBED_MODEL = "fake/embedder"
RERANK_MODEL = "fake/reranker"
TOKEN = "secret-token"


class FakeEmbedder:
    """Returns a deterministic vector per text, and records the call."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim
        self.calls = []

    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        self.calls.append(
            {"sentences": list(sentences), "normalize": normalize_embeddings}
        )
        # A one-hot-ish vector whose magnitude is 2, so normalisation is visible.
        return [
            [2.0 if i % self.dim == 0 else 0.0 for i in range(self.dim)]
            for _ in sentences
        ]


class FakeReranker:
    """Scores by text length, so the expected order is known up front."""

    def __init__(self) -> None:
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": pairs, "kwargs": kwargs})
        return [float(len(pair[1])) for pair in pairs]


class FakeRegistry:
    def __init__(self, embedder=None, reranker=None) -> None:
        self.embedder = embedder or FakeEmbedder()
        self.reranker = reranker or FakeReranker()

    def get(self, kind, name):
        if kind == "embedder" and name == EMBED_MODEL:
            return self.embedder
        if kind == "reranker" and name == RERANK_MODEL:
            return self.reranker
        raise ModelNotFoundError(f"no {kind} named {name}")

    def list_models(self, kind=None):
        return [
            {
                "name": EMBED_MODEL,
                "kind": "embedder",
                "device": "cpu",
                "loaded": True,
            },
            {
                "name": RERANK_MODEL,
                "kind": "reranker",
                "device": "cpu",
                "loaded": True,
            },
        ]

    def loaded(self):
        return ["embedder:fake/embedder"]

    def errors(self):
        return {}


def _client(**kwargs) -> TestClient:
    settings = ModelServiceSettings(
        env={
            "SSE_MODELS_API_TOKEN": kwargs.pop("token", TOKEN),
            **kwargs.pop("env", {}),
        }
    )
    return TestClient(
        create_app(
            settings=settings, registry=kwargs.pop("registry", FakeRegistry())
        )
    )


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# ----------------------------------------------------------------------
# health / discovery
# ----------------------------------------------------------------------
def test_healthz_needs_no_token():
    client = _client()
    assert client.get("/healthz").status_code == 200


def test_models_requires_token():
    client = _client()
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=_auth()).status_code == 200


def test_wrong_token_rejected():
    client = _client()
    response = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_no_token_configured_means_open():
    client = _client(token="")
    assert client.get("/v1/models").status_code == 200


def test_models_lists_catalog():
    body = _client().get("/v1/models", headers=_auth()).json()
    assert body["role"] == "all"
    assert {m["name"] for m in body["models"]} == {EMBED_MODEL, RERANK_MODEL}


# ----------------------------------------------------------------------
# /v1/embed
# ----------------------------------------------------------------------
def test_embed_returns_vectors_in_order():
    registry = FakeRegistry()
    client = _client(registry=registry)

    response = client.post(
        "/v1/embed",
        headers=_auth(),
        json={"model": EMBED_MODEL, "texts": ["first", "second", "third"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["dim"] == 3
    assert len(body["embeddings"]) == 3
    assert registry.embedder.calls[0]["sentences"] == ["first", "second", "third"]


def test_embed_normalizes_by_default():
    registry = FakeRegistry()
    client = _client(registry=registry)
    client.post(
        "/v1/embed", headers=_auth(), json={"model": EMBED_MODEL, "texts": ["a"]}
    )
    assert registry.embedder.calls[0]["normalize"] is True


def test_embed_normalize_can_be_turned_off():
    registry = FakeRegistry()
    client = _client(registry=registry)
    client.post(
        "/v1/embed",
        headers=_auth(),
        json={"model": EMBED_MODEL, "texts": ["a"], "normalize": False},
    )
    assert registry.embedder.calls[0]["normalize"] is False


def test_embed_rejects_empty_batch():
    assert (
        _client()
        .post("/v1/embed", headers=_auth(), json={"model": EMBED_MODEL, "texts": []})
        .status_code
        == 422
    )


def test_embed_unknown_model_is_404():
    response = _client().post(
        "/v1/embed", headers=_auth(), json={"model": "nope", "texts": ["a"]}
    )
    assert response.status_code == 404


def test_embed_reports_model_failure_as_500():
    class Broken:
        def encode(self, **kwargs):
            raise RuntimeError("cuda exploded")

    client = _client(registry=FakeRegistry(embedder=Broken()))
    response = client.post(
        "/v1/embed", headers=_auth(), json={"model": EMBED_MODEL, "texts": ["a"]}
    )
    assert response.status_code == 500
    assert "cuda exploded" in response.json()["detail"]


# ----------------------------------------------------------------------
# /v1/rerank
# ----------------------------------------------------------------------
def test_rerank_sorts_best_first_with_original_indices():
    documents = ["short", "the longest document of them all", "medium length"]
    response = _client().post(
        "/v1/rerank",
        headers=_auth(),
        json={"model": RERANK_MODEL, "query": "q", "documents": documents},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scored"] == 3
    assert [hit["index"] for hit in body["results"]] == [1, 2, 0]
    assert [hit["score"] for hit in body["results"]] == [32.0, 13.0, 5.0]


def test_rerank_omits_documents_unless_asked():
    body = (
        _client()
        .post(
            "/v1/rerank",
            headers=_auth(),
            json={"model": RERANK_MODEL, "query": "q", "documents": ["a", "bb"]},
        )
        .json()
    )
    assert "document" not in body["results"][0]


def test_rerank_returns_documents_when_asked():
    body = (
        _client()
        .post(
            "/v1/rerank",
            headers=_auth(),
            json={
                "model": RERANK_MODEL,
                "query": "q",
                "documents": ["a", "bb"],
                "return_documents": True,
            },
        )
        .json()
    )
    assert body["results"][0]["document"] == "bb"


def test_rerank_top_n_truncates_after_sorting():
    body = (
        _client()
        .post(
            "/v1/rerank",
            headers=_auth(),
            json={
                "model": RERANK_MODEL,
                "query": "q",
                "documents": ["a", "bbb", "cc"],
                "top_n": 2,
            },
        )
        .json()
    )
    assert [hit["index"] for hit in body["results"]] == [1, 2]
    assert body["count"] == 2 and body["scored"] == 3


def test_rerank_rejects_too_many_documents():
    client = _client(env={"SSE_MODELS_RERANK_MAX_DOCUMENTS": "2"})
    response = client.post(
        "/v1/rerank",
        headers=_auth(),
        json={"model": RERANK_MODEL, "query": "q", "documents": ["a", "b", "c"]},
    )
    assert response.status_code == 413


def test_rerank_rejects_blank_query():
    response = _client().post(
        "/v1/rerank",
        headers=_auth(),
        json={"model": RERANK_MODEL, "query": "   ", "documents": ["a"]},
    )
    assert response.status_code == 422


def test_rerank_does_not_send_batch_size_by_default():
    """Parity with the engine: it calls predict(pairs) with no batch size."""
    registry = FakeRegistry()
    _client(registry=registry).post(
        "/v1/rerank",
        headers=_auth(),
        json={"model": RERANK_MODEL, "query": "q", "documents": ["a", "b"]},
    )
    assert registry.reranker.calls[0]["kwargs"] == {}


def test_rerank_forwards_batch_size_when_given():
    registry = FakeRegistry()
    _client(registry=registry).post(
        "/v1/rerank",
        headers=_auth(),
        json={
            "model": RERANK_MODEL,
            "query": "q",
            "documents": ["a", "b"],
            "batch_size": 4,
        },
    )
    assert registry.reranker.calls[0]["kwargs"] == {"batch_size": 4}


# ----------------------------------------------------------------------
# roles
# ----------------------------------------------------------------------
def test_embed_only_role_refuses_rerank():
    client = _client(env={"SSE_MODELS_ROLE": "embed"})
    assert (
        client.post(
            "/v1/embed", headers=_auth(), json={"model": EMBED_MODEL, "texts": ["a"]}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/rerank",
            headers=_auth(),
            json={"model": RERANK_MODEL, "query": "q", "documents": ["a"]},
        ).status_code
        == 503
    )


def test_rerank_only_role_refuses_embed():
    client = _client(env={"SSE_MODELS_ROLE": "rerank"})
    assert (
        client.post(
            "/v1/embed", headers=_auth(), json={"model": EMBED_MODEL, "texts": ["a"]}
        ).status_code
        == 503
    )
    assert (
        client.post(
            "/v1/rerank",
            headers=_auth(),
            json={"model": RERANK_MODEL, "query": "q", "documents": ["a"]},
        ).status_code
        == 200
    )
