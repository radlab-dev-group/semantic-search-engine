"""
Tests for :class:`ExtractiveQAController`.

The controller answers through the model service; these tests cover the three
ways it can answer (an injected pipeline, the service, the in-process
fallback) and what happens when a path fails.  No model is loaded and no
socket is opened: the service client and the local pipeline are fakes.
"""

import pytest

from sse_api.engine.core.generative.extractive import ExtractiveQAController


class FakePipeline:
    """A stand-in for a HuggingFace question-answering pipeline."""

    def __init__(self, answer="Warszawa", score=0.8):
        self.answer = answer
        self.score = score
        self.calls = []

    def __call__(self, question, context):
        self.calls.append({"question": question, "context": context})
        return {"answer": self.answer, "score": self.score}


def _search_results(*texts):
    return {
        "results": [
            {
                "result": {
                    "text": {
                        "text_str": text,
                        "document_name": f"doc-{index}.pdf",
                        "page_number": index + 1,
                        "text_number": index,
                    }
                }
            }
            for index, text in enumerate(texts)
        ]
    }


# ----------------------------------------------------------------------
# an injected pipeline (the historical contract)
# ----------------------------------------------------------------------
def test_controller_answers_with_the_injected_pipeline():
    pipeline = FakePipeline()
    controller = ExtractiveQAController(model_path=None, qa_pipeline=pipeline)

    answers = controller.run_extractive_qa(
        "Jaka jest stolica Polski?", _search_results("tekst pierwszy", "tekst drugi")
    )

    assert answers == {
        "doc-0.pdf": {
            "page_number": 1,
            "text_number": 0,
            "answer": "Warszawa",
            "score": 0.8,
        },
        "doc-1.pdf": {
            "page_number": 2,
            "text_number": 1,
            "answer": "Warszawa",
            "score": 0.8,
        },
    }
    assert [call["context"] for call in pipeline.calls] == [
        "tekst pierwszy",
        "tekst drugi",
    ]


def test_controller_sends_a_blank_passage_to_the_model():
    pipeline = FakePipeline()
    controller = ExtractiveQAController(model_path=None, qa_pipeline=pipeline)
    answers = controller.run_extractive_qa("q", _search_results("   ", "a"))
    assert pipeline.calls == [{"question": "q", "context": "a"}]
    assert answers["doc-0.pdf"]["answer"] == ""
    assert answers["doc-0.pdf"]["score"] == 0.0


def test_controller_without_results_answers_nothing():
    controller = ExtractiveQAController(model_path=None, qa_pipeline=FakePipeline())
    assert controller.run_extractive_qa("q", {"results": []}) == {}


def test_controller_answers_an_empty_question_without_calling_the_model():
    pipeline = FakePipeline()
    controller = ExtractiveQAController(model_path=None, qa_pipeline=pipeline)
    answers = controller.run_extractive_qa("   ", _search_results("tekst"))
    assert answers["doc-0.pdf"]["answer"] == ""
    assert pipeline.calls == []


# ----------------------------------------------------------------------
# the model service
# ----------------------------------------------------------------------
def test_controller_prefers_the_model_service(monkeypatch):
    from sse_api.engine.core.models import backend

    calls = {}

    def fake_client():
        def answer(question, contexts):
            calls["question"] = question
            calls["contexts"] = list(contexts)
            return [(f"a-{index}", 0.5) for index, _ in enumerate(contexts)]

        return answer

    monkeypatch.setattr(backend, "build_extractive_qa_client", fake_client)

    controller = ExtractiveQAController(model_path=None)
    answers = controller.run_extractive_qa("q", _search_results("pierwszy", "drugi"))

    assert calls["contexts"] == ["pierwszy", "drugi"]
    assert answers["doc-0.pdf"]["answer"] == "a-0"
    assert answers["doc-1.pdf"]["answer"] == "a-1"
    assert answers["doc-1.pdf"]["score"] == 0.5


def test_controller_falls_back_to_local_when_the_service_fails(monkeypatch):
    from sse_api.engine.core.models import backend

    def broken_client():
        def answer(question, contexts):
            raise backend.ModelBackendError("service is down")

        return answer

    monkeypatch.setattr(backend, "build_extractive_qa_client", broken_client)
    _patch_local_answer(
        monkeypatch, lambda question, contexts: [("lokalna", 0.4) for _ in contexts]
    )

    controller = ExtractiveQAController(model_path=None)
    answers = controller.run_extractive_qa("q", _search_results("tekst"))
    assert answers["doc-0.pdf"] == {
        "page_number": 1,
        "text_number": 0,
        "answer": "lokalna",
        "score": 0.4,
    }


def test_controller_degrades_to_no_answer_when_nothing_works(monkeypatch):
    from sse_api.engine.core.models import backend

    monkeypatch.setattr(
        backend, "build_extractive_qa_client", lambda: None, raising=False
    )

    def explode(question, contexts):
        raise RuntimeError("no torch on this host")

    _patch_local_answer(monkeypatch, explode)

    controller = ExtractiveQAController(model_path=None)
    answers = controller.run_extractive_qa("q", _search_results("tekst"))
    assert answers["doc-0.pdf"]["answer"] == ""
    assert answers["doc-0.pdf"]["score"] == 0.0


def _patch_local_answer(monkeypatch, answer_func):
    """The in-process fallback of the controller, with a fake answer."""
    from sse_api.data.controllers import extractive_qa_local

    monkeypatch.setattr(
        extractive_qa_local,
        "answer_locally",
        lambda question, contexts, bundle=None: answer_func(question, contexts),
    )


def test_controller_answerer_override_wins():
    controller = ExtractiveQAController(
        model_path=None,
        answerer=lambda question, contexts: [("nadpisane", 1.0) for _ in contexts],
    )
    answers = controller.run_extractive_qa("q", _search_results("tekst"))
    assert answers["doc-0.pdf"]["answer"] == "nadpisane"


def test_controller_needs_no_model_path_anymore():
    """No pipeline and no path: the service or the local config answer."""
    controller = ExtractiveQAController(model_path=None)
    assert controller.model_path == ExtractiveQAController.model_path


def test_controller_exports_are_stable():
    from sse_api.engine.controllers.models_logic.extractive import (
        ExtractiveQAController as FromControllers,
    )

    assert FromControllers is ExtractiveQAController
