import json
import unittest

from sse_lib import GenerativeOptions, SSEValueError
from support import build_client, ok, GENERATIVE


class RewriteAndCitationsTests(unittest.TestCase):
    def test_citations_force_numbered_context_without_mutating_options(self):
        options = GenerativeOptions(
            **GENERATIVE, system_prompt_with_enumerated_documents=False
        )
        client = build_client([ok({"answer": "Fact [7].", "citations": [{"n": 7}]})])
        answer = client.generative_answer_with_citations(
            42, options, system_prompt="Be brief.", query_instruction="Use a table."
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/generative_answer")
        self.assertTrue(
            json.loads(call.json["query_options"])[
                "system_prompt_with_enumerated_documents"
            ]
        )
        self.assertEqual(call.json["query_response_id"], 42)
        self.assertEqual(call.json["system_prompt"], "Be brief.")
        self.assertEqual(call.json["query_instruction"], "Use a table.")
        self.assertFalse(options.system_prompt_with_enumerated_documents)
        self.assertEqual(answer.citations, [{"n": 7}])

    def test_citations_require_generation_options(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.generative_answer_with_citations(1)
        self.assertEqual(client.fake_session.calls, [])

    def test_rewrite_sends_history_and_language(self):
        history = [{"user": "Who founded Acme?", "assistant": "Ada."}]
        client = build_client(
            [ok({"rewritten_query": "When did Ada found Acme?"})], language="en"
        )
        result = client.rewrite_query(
            "When?", generative_model="model", historical_messages=history
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/rewrite_query")
        self.assertEqual(
            call.json,
            {
                "query_str": "When?",
                "generative_model": "model",
                "historical_messages": history,
                "lang": "en",
            },
        )
        self.assertEqual(result, "When did Ada found Acme?")

    def test_rewrite_without_history(self):
        client = build_client([ok({"rewritten_query": "Question"})])
        self.assertEqual(
            client.rewrite_query("Question", generative_model="model"), "Question"
        )
        self.assertEqual(
            client.fake_session.calls[0].json["historical_messages"], []
        )

    def test_rewrite_rejects_bad_inputs_before_http(self):
        client = build_client([])
        for kwargs in (
            {"query": " ", "generative_model": "model"},
            {"query": "q", "generative_model": " "},
            {"query": "q", "generative_model": "model", "historical_messages": {}},
            {
                "query": "q",
                "generative_model": "model",
                "historical_messages": [{"user": "q"}],
            },
            {
                "query": "q",
                "generative_model": "model",
                "historical_messages": [
                    {"user": "q", "assistant": "a", "system": "instruction"}
                ],
            },
            {
                "query": "q",
                "generative_model": "model",
                "historical_messages": [{"user": "q", "assistant": 1}],
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(SSEValueError):
                client.rewrite_query(**kwargs)
        self.assertEqual(client.fake_session.calls, [])

    def test_rewrite_rejects_invalid_server_response(self):
        client = build_client([ok({"rewritten_query": " "})])
        from sse_lib import SSEAPIError

        with self.assertRaises(SSEAPIError):
            client.rewrite_query("q", generative_model="model")
