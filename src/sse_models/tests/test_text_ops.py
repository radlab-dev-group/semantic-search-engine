"""
Tests for the text-operation endpoints of the model service:
``/v1/tokenize``, ``/v1/truncate`` and ``/v1/denoise``.

As in ``test_api.py`` no real model is involved: a fake registry hands out a
stub tokenizer and a stub denoiser bundle, so the contract (status codes,
ordering, shape of the answers) is what gets tested.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from sse_models.app import create_app
from sse_models.registry import ModelNotFoundError
from sse_models.settings import ModelServiceSettings

TOKEN = "secret-token"
EMBED_MODEL = "fake/embedder"


class FakeTokenizer:
    """Splits on whitespace; a token is one word."""

    def encode(self, text, add_special_tokens=True):
        return [i for i, word in enumerate(text.split())] + [0]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(word) for word in token_ids)


class FakeDenoiserBundle:
    """A stand-in for the ``{"model", "tokenizer"}`` bundle."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, key):
        if key == "model":
            return self

        class _Tokenizer:
            def encode(text, add_special_tokens=True):
                return list(range(len(text.split())))

            def decode(token_ids, skip_special_tokens=True):
                return " ".join(str(i) for i in token_ids)

        return _Tokenizer()

    def parameters(self):
        class _Device:
            pass

        return iter([_Device()])


class FakeRegistry:
    def __init__(self):
        self._locks = {}
        self._locks_guard = threading.Lock()
        self.max_lengths = {("embedder", EMBED_MODEL): 1024}

    def has_model(self, kind, name):
        return kind == "embedder" and name == EMBED_MODEL

    def tokenizer_for(self, kind, name):
        if not self.has_model(kind, name):
            raise ModelNotFoundError(f"no {kind} named {name}")
        return FakeTokenizer()

    def max_length_of(self, kind, name):
        return self.max_lengths.get((kind, name), 0)

    def denoiser(self):
        return FakeDenoiserBundle()

    def inference_lock(self, kind, name):
        key = f"{kind}:{name}"
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]


def _client(**kwargs) -> TestClient:
    settings = ModelServiceSettings(
        env={
            "SSE_MODELS_API_TOKEN": TOKEN,
            **kwargs.pop("env", {}),
        }
    )
    return TestClient(create_app(settings=settings, registry=kwargs.pop("registry", FakeRegistry())))


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# ----------------------------------------------------------------------
# /v1/tokenize
# ----------------------------------------------------------------------
def test_tokenize_counts_words():
    client = _client()
    res = client.post(
        "/v1/tokenize",
        json={"model": EMBED_MODEL, "texts": ["one two three", "x"]},
        headers=_auth(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == EMBED_MODEL
    assert body["token_counts"] == [4, 2]  # 3 words + 1 fake special token
    assert body["max_length"] == 1024


def test_tokenize_unknown_model_is_404():
    client = _client()
    res = client.post(
        "/v1/tokenize",
        json={"model": "nope/none", "texts": ["x"]},
        headers=_auth(),
    )
    assert res.status_code == 404


def test_tokenize_requires_auth():
    client = _client()
    res = client.post(
        "/v1/tokenize", json={"model": EMBED_MODEL, "texts": ["x"]}
    )
    assert res.status_code == 401


# ----------------------------------------------------------------------
# /v1/truncate
# ----------------------------------------------------------------------
def test_truncate_cuts_to_token_budget():
    client = _client()
    res = client.post(
        "/v1/truncate",
        json={"model": EMBED_MODEL, "texts": ["a b c d e", "short"], "max_tokens": 3},
        headers=_auth(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["was_truncated"] == [True, False]
    assert len(body["truncated_texts"]) == 2
    # the kept text decodes back to at most max_tokens tokens
    assert len(body["truncated_texts"][0].split()) <= 3


def test_truncate_rejects_bad_budget():
    client = _client()
    res = client.post(
        "/v1/truncate",
        json={"model": EMBED_MODEL, "texts": ["a"], "max_tokens": 0},
        headers=_auth(),
    )
    assert res.status_code == 422


# ----------------------------------------------------------------------
# /v1/denoise
# ----------------------------------------------------------------------
def test_denoise_disabled_outside_roles():
    client = _client(env={"SSE_MODELS_ROLE": "embed"})
    res = client.post(
        "/v1/denoise", json={"text": "some noisy text"}, headers=_auth()
    )
    assert res.status_code == 503


def test_denoise_rejects_blank_text():
    client = _client()
    res = client.post(
        "/v1/denoise", json={"text": "   "}, headers=_auth()
    )
    assert res.status_code == 422
