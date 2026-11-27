"""Completion tests of the multi-collection search plan.

The endpoint tests use mocks for Milvus/PostgreSQL; the aggregation and RAG
helpers are exercised on plain dictionaries.  Together they cover the parts
which the first implementation left unfinished: single-collection provenance
in the database, ``collection_names`` without the legacy field, global score
cutoff, rank mass of the merged stats and collection-aware RAG prompts.
"""

from unittest.mock import MagicMock, patch

import pytest
from django.test import TestCase

from sse_api.core.validation import collection_names
from sse_api.engine.api import SearchWithOptions
from sse_api.engine.controllers.models_logic import generative
from sse_api.engine.core.multi_collection import (
    aggregate_results,
    annotate_results,
    dedupe_collection_names,
    document_label,
    shared_search_options,
)
from sse_api.engine.core.search_engine import (
    convert_search_results_to_doc2answer,
    convert_search_results_to_enumerated_documents,
    get_accumulated_docs_by_rank_perc,
)
from sse_api.tests.input_support import EndpointInputMixin


def make_collection(name: str, display: str) -> MagicMock:
    coll = MagicMock()
    coll.name = name
    coll.display_name = display
    coll.apply_search_defaults.side_effect = lambda options: dict(options)
    return coll


def one_collection_results(score: float = 0.8, doc: str = "doc.pdf") -> dict:
    return {
        "query": "pytanie",
        "score_scale": "cosine",
        "stats": {
            doc: {"score_weighted": score, "score_weighted_scaled": 1.0, "hits": 2}
        },
        "detailed_results": {
            "0": {
                "score": score,
                "document_name": doc,
                "text_str": "fragment",
                "page_number": 1,
                "text_number": 1,
            }
        },
    }


# --------------------------------------------------------------------------- #
# Pure aggregation helpers
# --------------------------------------------------------------------------- #


def test_collection_names_validator_accepts_json_list_and_single_name():
    assert collection_names('["a", "b"]') == ["a", "b"]
    assert collection_names(["a", " a "]) == ["a", "a"]
    assert collection_names("a") == ["a"]
    assert collection_names(None) is None
    with pytest.raises(ValueError):
        collection_names('{"name": "a"}')
    with pytest.raises(ValueError):
        collection_names(["a", 7])


def test_dedupe_preserves_order():
    assert dedupe_collection_names([" b ", "a", "b", "", None, "a"]) == [
        "b",
        "a",
    ]


def test_annotation_keeps_single_collection_document_names():
    coll = make_collection("col_a", "Kolekcja A")
    annotated = annotate_results(one_collection_results(), coll)
    hit = annotated["detailed_results"]["0"]
    assert hit["document_name"] == "doc.pdf"
    assert hit["original_document_name"] == "doc.pdf"
    assert hit["collection_name"] == "col_a"
    assert hit["collection_display_name"] == "Kolekcja A"
    assert list(annotated["stats"]) == ["doc.pdf"]


def test_annotation_disambiguates_the_same_document_in_two_collections():
    coll = make_collection("col_b", "Kolekcja B")
    annotated = annotate_results(
        one_collection_results(), coll, rename_document=True
    )
    assert annotated["detailed_results"]["0"]["document_name"] == (
        "[Kolekcja B] doc.pdf"
    )
    assert "[Kolekcja B] doc.pdf" in annotated["stats"]
    assert annotated["stats"]["[Kolekcja B] doc.pdf"][
        "original_document_name"
    ] == "doc.pdf"


def test_shared_options_drop_filters_of_one_collection():
    options = {
        "max_results": 4,
        "hybrid_search": True,
        "categories": ["A"],
        "documents": ["doc.pdf"],
        "relative_paths": ["x"],
        "templates": [1],
        "metadata_filters": [{"operator": "eq", "field": {"x": 1}}],
        "keywords": ["timeout"],
    }
    shared = shared_search_options(options)
    assert shared == {
        "max_results": 4,
        "hybrid_search": True,
        "keywords": ["timeout"],
    }


def test_global_limit_cuts_and_rescales_even_with_zero_scores():
    detailed = [
        {"document_name": "[A] a.pdf", "score": "0.90", "collection_name": "a"},
        {"document_name": "[B] b.pdf", "score": 0.75, "collection_name": "b"},
        {"document_name": "[A] c.pdf", "score": None, "collection_name": "a"},
    ]
    stats = {
        "[A] a.pdf": {"score_weighted": 0.0},
        "[B] b.pdf": {"score_weighted": 0.0},
        "[A] c.pdf": {"score_weighted": 0.0},
    }
    items, merged_stats = aggregate_results(
        detailed, stats, mode="global_limit", max_results=2
    )
    assert [item["document_name"] for item in items] == [
        "[A] a.pdf",
        "[B] b.pdf",
    ]
    assert set(merged_stats) == {"[A] a.pdf", "[B] b.pdf"}
    assert all("score_weighted_scaled" in row for row in merged_stats.values())
    assert sum(row["score_weighted_scaled"] for row in merged_stats.values()) == (
        pytest.approx(1.0)
    )
    assert set(get_accumulated_docs_by_rank_perc({"stats": merged_stats}, 100)) == {
        "[A] a.pdf",
        "[B] b.pdf",
    }


def test_global_limit_can_rank_by_a_separate_rerank_score():
    detailed = [
        {
            "document_name": "[A] a.pdf",
            "rerank_score": 0.22,
            "collection_name": "a",
        },
        {
            "document_name": "[B] b.pdf",
            "rerank_score": "0.91",
            "collection_name": "b",
        },
    ]
    items, stats = aggregate_results(
        detailed, {}, mode="global_limit", max_results=1
    )
    assert [item["document_name"] for item in items] == ["[B] b.pdf"]
    assert stats["[B] b.pdf"]["score_weighted"] == pytest.approx(0.91)


def test_aggregate_adds_stats_for_chunks_without_document_statistics():
    detailed = [
        {
            "document_name": "[A] orphan.pdf",
            "original_document_name": "orphan.pdf",
            "collection_name": "a",
            "collection_display_name": "Kolekcja A",
            "score": 0.4,
        }
    ]
    items, stats = aggregate_results(
        detailed, {}, mode="per_collection", max_results=5
    )
    assert len(items) == 1
    assert stats["[A] orphan.pdf"]["score_weighted_scaled"] == pytest.approx(1.0)
    assert stats["[A] orphan.pdf"]["collection_name"] == "a"


def test_document_label():
    assert document_label("HR", "regulamin.pdf") == "[HR] regulamin.pdf"


# --------------------------------------------------------------------------- #
# RAG context / system prompt
# --------------------------------------------------------------------------- #


def test_enumerated_documents_keep_collection_provenance():
    coll = make_collection("col_b", "Kolekcja B")
    annotated = annotate_results(
        one_collection_results(), coll, rename_document=True
    )
    documents = convert_search_results_to_enumerated_documents(
        annotated["detailed_results"]
    )
    assert documents[0]["document_name"] == "[Kolekcja B] doc.pdf"
    assert documents[0]["collection_name"] == "col_b"
    assert documents[0]["collection_display_name"] == "Kolekcja B"
    assert documents[0]["original_document_name"] == "doc.pdf"


def test_doc2answer_keeps_plan_citation_format_in_the_prompt():
    coll = make_collection("col_b", "Kolekcja B")
    fragments = annotate_results(
        one_collection_results(), coll, rename_document=True
    )["detailed_results"]
    answers = convert_search_results_to_doc2answer(fragments, None)
    assert list(answers) == ["[Kolekcja B] doc.pdf"]
    assert answers["[Kolekcja B] doc.pdf"] == [
        "[[Kolekcja: Kolekcja B] doc.pdf]: fragment"
    ]
    # A traditional single-collection result keeps the old text format.
    single = convert_search_results_to_doc2answer(
        one_collection_results()["detailed_results"], None
    )
    assert single == {"doc.pdf": ["fragment"]}


def test_legacy_multi_collection_answer_gets_citation_instruction(monkeypatch):
    from llm_router_lib.data_models.response import GenerativeAnswerResponse

    monkeypatch.setenv("LLM_ROUTER_API", "http://router.example.com")
    coll = make_collection("col_b", "Kolekcja B")
    fragments = annotate_results(
        one_collection_results(), coll, rename_document=True
    )["detailed_results"]
    client = MagicMock()
    client.return_value.generative_answer.return_value = GenerativeAnswerResponse(
        response="Odpowiedź [[Kolekcja B] doc.pdf]", generation_time=0.2
    )
    controller = object.__new__(generative.GenerativeModelControllerApi)
    controller.models_config = MagicMock()
    metadata = {}
    with patch.dict(generative.__dict__, {"LLMRouterClient": client}):
        answer, _ = controller.generative_answer_local_api_model(
            question_str="pytanie",
            search_results=fragments,
            qa_gen_model="test-model",
            system_prompt="Bądź zwięzły",
            response_metadata=metadata,
        )
    payload = client.return_value.generative_answer.call_args.kwargs["payload"]
    assert "Bądź zwięzły" in payload.system_prompt
    assert "[[Kolekcja: HR] regulamin.pdf]" in payload.system_prompt
    assert answer.startswith("Odpowiedź")
    assert metadata["citations"][0]["collection_name"] == "col_b"
    assert metadata["citations"][0]["document_name"] == "[Kolekcja B] doc.pdf"


def test_single_collection_prompt_is_unchanged(monkeypatch):
    from llm_router_lib.data_models.response import GenerativeAnswerResponse

    monkeypatch.setenv("LLM_ROUTER_API", "http://router.example.com")
    client = MagicMock()
    client.return_value.generative_answer.return_value = GenerativeAnswerResponse(
        response="Odpowiedź", generation_time=0.2
    )
    controller = object.__new__(generative.GenerativeModelControllerApi)
    controller.models_config = MagicMock()
    with patch.dict(generative.__dict__, {"LLMRouterClient": client}):
        controller.generative_answer_local_api_model(
            question_str="pytanie",
            search_results=one_collection_results()["detailed_results"],
            qa_gen_model="test-model",
            system_prompt="Bądź zwięzły",
        )
    payload = client.return_value.generative_answer.call_args.kwargs["payload"]
    assert payload.system_prompt == "Bądź zwięzły"


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


class MultiCollectionCompletionTests(EndpointInputMixin, TestCase):
    def test_collection_names_alone_is_enough(self):
        coll = make_collection("col_a", "Kolekcja A")
        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                return_value=coll,
            ) as lookup,
            patch(
                "sse_api.engine.api.SearchQueryController.new_query",
                return_value={"results": {}, "query_response_id": 1},
            ) as new_query,
        ):
            response = self.invoke(
                SearchWithOptions,
                {
                    "collection_names": '["col_a"]',
                    "query_str": "pytanie",
                    "options": {},
                },
            )
        self.assertTrue(response.data["status"])
        lookup.assert_called_once()
        self.assertTrue(new_query.call_args.kwargs["annotate_collection"])
        self.assertEqual(
            new_query.call_args.kwargs["search_options_dict"]["collection_names"],
            ["col_a"],
        )

    def test_invalid_multi_collection_mode_is_rejected(self):
        with patch(
            "sse_api.engine.api.SearchQueryController.new_query"
        ) as new_query:
            for payload in (
                {"options": {"multi_collection_mode": "top_k"}},
                {"multi_collection_mode": "top_k", "options": {}},
            ):
                with self.subTest(payload=payload):
                    self.assert_denial(
                        self.invoke(
                            SearchWithOptions,
                            {
                                "collection_name": "col_a",
                                "query_str": "pytanie",
                                **payload,
                            },
                        )
                    )
            new_query.assert_not_called()

    def test_duplicate_names_do_not_run_the_same_search_twice(self):
        coll = make_collection("col_a", "Kolekcja A")
        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                return_value=coll,
            ) as lookup,
            patch(
                "sse_api.engine.api.SearchQueryController.new_query",
                return_value={"results": {}, "query_response_id": 1},
            ) as new_query,
        ):
            response = self.invoke(
                SearchWithOptions,
                {
                    "collection_name": "col_a",
                    "collection_names": '["col_a", "col_a"]',
                    "query_str": "pytanie",
                    "options": {},
                },
            )
        self.assertTrue(response.data["status"])
        lookup.assert_called_once()
        new_query.assert_called_once()

    def test_single_collection_provenance_is_persisted(self):
        coll = make_collection("col_a", "Kolekcja A")
        sem = MagicMock()
        sem.search_with_options.return_value = (
            one_collection_results(),
            [],
            [],
        )
        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                return_value=coll,
            ),
            patch(
                "sse_api.engine.api.DBSemanticSearchController.prepare_controller_for_collection",
                return_value=sem,
            ),
            patch("sse_api.engine.models.UserQuery.objects.create") as uq,
            patch(
                "sse_api.engine.models.UserQueryResponse.objects.create"
            ) as uqr,
        ):
            uq.return_value = MagicMock()
            uqr.return_value = MagicMock(pk=11)
            response = self.invoke(
                SearchWithOptions,
                {
                    "collection_name": "col_a",
                    "query_str": "pytanie",
                    "options": {},
                },
            )
        self.assertTrue(response.data["status"])
        persisted = uqr.call_args.kwargs["detailed_results_json"]["0"]
        self.assertEqual(persisted["document_name"], "doc.pdf")
        self.assertEqual(persisted["collection_name"], "col_a")
        self.assertEqual(persisted["collection_display_name"], "Kolekcja A")
        self.assertEqual(
            uqr.call_args.kwargs["general_stats_json"]["doc.pdf"][
                "collection_display_name"
            ],
            "Kolekcja A",
        )

    def test_multi_collection_drops_collection_scoped_filters(self):
        col_a = make_collection("col_a", "Kolekcja A")
        col_b = make_collection("col_b", "Kolekcja B")
        calls = []

        def search_with_options(**kwargs):
            calls.append(kwargs)
            coll = kwargs["collection"]
            return (
                one_collection_results(doc=f"{coll.name}.pdf"),
                [],
                [],
            )

        sem = MagicMock()
        sem.search_with_options.side_effect = search_with_options
        lookup = lambda collection_name, created_by: {  # noqa: E731
            "col_a": col_a,
            "col_b": col_b,
        }.get(collection_name)
        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                side_effect=lookup,
            ),
            patch(
                "sse_api.engine.api.DBSemanticSearchController.prepare_controller_for_collection",
                return_value=sem,
            ),
            patch("sse_api.engine.models.UserQuery.objects.create") as uq,
            patch(
                "sse_api.engine.models.UserQueryResponse.objects.create"
            ) as uqr,
        ):
            uq.return_value = MagicMock()
            uqr.return_value = MagicMock(pk=12)
            response = self.invoke(
                SearchWithOptions,
                {
                    "collection_name": "col_a",
                    "collection_names": '["col_a", "col_b"]',
                    "query_str": "pytanie",
                    "options": {
                        "max_results": 2,
                        "categories": ["A"],
                        "documents": ["col_a.pdf"],
                        "templates": [3],
                        "multi_collection_mode": "per_collection",
                    },
                },
            )
        self.assertTrue(response.data["status"])
        self.assertEqual(len(calls), 2)
        for call in calls:
            params = call["search_params"]
            self.assertNotIn("categories", params)
            self.assertNotIn("documents", params)
            self.assertNotIn("templates", params)
            self.assertEqual(params["max_results"], 2)
        body = response.data["body"]
        self.assertEqual(len(body["results"]["detailed_results"]), 2)
        self.assertEqual(
            uq.call_args.kwargs["query_options"]["collection_names"],
            ["col_a", "col_b"],
        )
        self.assertEqual(
            uq.call_args.kwargs["query_options"]["multi_collection_mode"],
            "per_collection",
        )

    def test_global_limit_uses_numeric_string_scores_and_persists_stats(self):
        col_a = make_collection("col_a", "Kolekcja A")
        col_b = make_collection("col_b", "Kolekcja B")

        def search_with_options(**kwargs):
            coll = kwargs["collection"]
            results = one_collection_results(
                score=0.51 if coll.name == "col_a" else 0.99,
                doc=f"{coll.name}.pdf",
            )
            results["detailed_results"]["0"]["score"] = (
                "0.51" if coll.name == "col_a" else "0.99"
            )
            return results, [], []

        sem = MagicMock()
        sem.search_with_options.side_effect = search_with_options
        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                side_effect=lambda collection_name, created_by: {
                    "col_a": col_a,
                    "col_b": col_b,
                }.get(collection_name),
            ),
            patch(
                "sse_api.engine.api.DBSemanticSearchController.prepare_controller_for_collection",
                return_value=sem,
            ),
            patch("sse_api.engine.models.UserQuery.objects.create") as uq,
            patch(
                "sse_api.engine.models.UserQueryResponse.objects.create"
            ) as uqr,
        ):
            uq.return_value = MagicMock()
            uqr.return_value = MagicMock(pk=13)
            response = self.invoke(
                SearchWithOptions,
                {
                    "collection_names": '["col_a", "col_b"]',
                    "query_str": "pytanie",
                    "options": {
                        "max_results": 1,
                        "multi_collection_mode": "global_limit",
                    },
                },
            )
        self.assertTrue(response.data["status"])
        detailed = response.data["body"]["results"]["detailed_results"]
        self.assertEqual(len(detailed), 1)
        self.assertEqual(detailed["0"]["collection_name"], "col_b")
        self.assertEqual(detailed["0"]["document_name"], "[Kolekcja B] col_b.pdf")
        stats = uqr.call_args.kwargs["general_stats_json"]
        self.assertEqual(len(stats), 1)
        self.assertIn("score_weighted_scaled", next(iter(stats.values())))
