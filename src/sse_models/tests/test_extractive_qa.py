"""
Tests for the extractive QA endpoint of the model service.

As elsewhere in this package no real model is involved: a fake registry hands
out a fake ``question-answering`` pipeline which answers with the first words
of its context, so the contract (status codes, one answer per passage in
input order, the role gate and the request limits) is what gets tested.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from sse_models.app import create_app
from sse_models.extractive_qa import answer
from sse_models.registry import ModelConfigError
from sse_models.settings import ModelServiceSettings

TOKEN = "secret-token"


class FakeQAPipeline:
    """Answers with the first two words of the context; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, question, context):
        self.calls.append({"question": question, "context": context})
        words = context.split()
        return {"answer": " ".join(words[:2]), "score": 0.75}


class FakeRegistry:
    def __init__(self, qa_model=None):
        self._locks = {}
        self._locks_guard = threading.Lock()
        self.qa_calls = 0
        self._qa_model = qa_model if qa_model is not None else FakeQAPipeline()

    def qa_model(self):
        self.qa_calls += 1
        return self._qa_model

    def inference_lock(self, kind, name):
        key = f"{kind}:{name}"
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]


def _client(**kwargs) -> TestClient:
    settings = ModelServiceSettings(
        env={"SSE_MODELS_API_TOKEN": TOKEN, **kwargs.pop("env", {})}
    )
    return TestClient(
        create_app(
            settings=settings, registry=kwargs.pop("registry", FakeRegistry())
        )
    )


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# ----------------------------------------------------------------------
# POST /v1/extractive_qa
# ----------------------------------------------------------------------
def test_extractive_qa_answers_every_passage_in_order():
    registry = FakeRegistry()
    client = _client(registry=registry)
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "Ile to kosztuje?", "contexts": ["a b c", "x y z"]},
        headers=_auth(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2
    assert [item["answer"] for item in body["answers"]] == ["a b", "x y"]
    assert [item["score"] for item in body["answers"]] == [0.75, 0.75]
    # one pipeline call per passage, question always the same
    pipeline = registry._qa_model
    assert [call["context"] for call in pipeline.calls] == ["a b c", "x y z"]
    assert {call["question"] for call in pipeline.calls} == {"Ile to kosztuje?"}


def test_extractive_qa_requires_auth():
    client = _client()
    res = client.post("/v1/extractive_qa", json={"question": "q", "contexts": ["a"]})
    assert res.status_code == 401


def test_extractive_qa_rejects_blank_question():
    client = _client()
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "   ", "contexts": ["a"]},
        headers=_auth(),
    )
    assert res.status_code == 422


def test_extractive_qa_rejects_an_empty_context_list():
    client = _client()
    res = client.post(
        "/v1/extractive_qa", json={"question": "q", "contexts": []}, headers=_auth()
    )
    assert res.status_code == 422


def test_extractive_qa_skips_the_model_for_a_blank_passage():
    registry = FakeRegistry()
    client = _client(registry=registry)
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "q", "contexts": ["   ", "a b c"]},
        headers=_auth(),
    )
    assert res.status_code == 200
    answers = res.json()["answers"]
    assert answers[0] == {"answer": "", "score": 0.0}
    assert answers[1]["answer"] == "a b"
    pipeline = registry._qa_model
    assert [call["context"] for call in pipeline.calls] == ["a b c"]


def test_extractive_qa_disabled_outside_its_roles():
    client = _client(env={"SSE_MODELS_ROLE": "embed"})
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "q", "contexts": ["a"]},
        headers=_auth(),
    )
    assert res.status_code == 503


def test_extractive_qa_is_the_only_endpoint_of_its_role():
    client = _client(env={"SSE_MODELS_ROLE": "extractive_qa"})
    assert (
        client.post(
            "/v1/extractive_qa",
            json={"question": "q", "contexts": ["a"]},
            headers=_auth(),
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/embed",
            json={"model": "fake/embedder", "texts": ["a"]},
            headers=_auth(),
        ).status_code
        == 503
    )


def test_extractive_qa_caps_the_number_of_passages():
    client = _client(env={"SSE_MODELS_EXTRACTIVE_QA_MAX_CONTEXTS": "2"})
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "q", "contexts": ["a", "b", "c"]},
        headers=_auth(),
    )
    assert res.status_code == 413


def test_extractive_qa_reports_a_model_failure_as_500():
    class BrokenPipeline:
        def __call__(self, question, context):
            raise RuntimeError("CUDA out of memory")

    registry = FakeRegistry(qa_model=BrokenPipeline())
    client = _client(registry=registry)
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "q", "contexts": ["a"]},
        headers=_auth(),
    )
    assert res.status_code == 500
    assert "extractive QA failed" in res.json()["detail"]


def test_extractive_qa_reports_a_bad_config_as_500():
    class UnconfiguredRegistry(FakeRegistry):
        def qa_model(self):
            raise ModelConfigError("extractive_qa section has no model_path")

    client = _client(registry=UnconfiguredRegistry())
    res = client.post(
        "/v1/extractive_qa",
        json={"question": "q", "contexts": ["a"]},
        headers=_auth(),
    )
    assert res.status_code == 500


def test_extractive_qa_model_is_loaded_once():
    registry = FakeRegistry()
    client = _client(registry=registry)
    for _ in range(3):
        assert (
            client.post(
                "/v1/extractive_qa",
                json={"question": "q", "contexts": ["a"]},
                headers=_auth(),
            ).status_code
            == 200
        )
    assert registry.qa_calls == 3  # asked per request, cached inside the registry


# ----------------------------------------------------------------------
# sse_models.extractive_qa.answer
# ----------------------------------------------------------------------
def test_answer_helper_returns_pairs_in_input_order():
    pairs = answer(FakeQAPipeline(), "q", ["one two three", "four five"])
    assert pairs == [("one two", 0.75), ("four five", 0.75)]


def test_answer_helper_without_any_context():
    assert answer(FakeQAPipeline(), "q", []) == []
