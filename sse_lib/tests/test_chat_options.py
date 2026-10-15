"""
Wire-format tests for the chat switches of :class:`sse_lib.SSEClient`.

The backend reads generation settings and the pipeline switches from one
``options`` dictionary; these tests pin that both land in the same JSON
object of ``new_chat`` and ``add_user_message``.
"""

from __future__ import annotations

import unittest

from sse_lib import ChatOptions, GenerativeOptions, SSEValueError

from support import GENERATIVE, build_client, ok


class NewChatOptionsTest(unittest.TestCase):
    def test_chat_options_merge_into_the_generation_options(self):
        client = build_client([ok({"chat": {"id": 3}})])
        client.new_chat(
            "my_docs",
            options={"generative_model": "m"},
            chat_options=ChatOptions(use_rag_supervisor=True),
        )

        payload = client.fake_session.calls[0].json
        self.assertEqual(
            payload["options"],
            {"generative_model": "m", "use_rag_supervisor": True},
        )

    def test_chat_options_alone_are_sent_as_an_object(self):
        client = build_client([ok({"chat": {"id": 3}})])
        client.new_chat(
            chat_options=ChatOptions(
                use_content_supervisor=True, query_instruction="Summarize"
            )
        )

        payload = client.fake_session.calls[0].json
        self.assertEqual(
            payload["options"],
            {"use_content_supervisor": True, "query_instruction": "Summarize"},
        )
        self.assertNotIn("collection_name", payload)

    def test_no_options_means_no_options_field(self):
        client = build_client([ok({"chat": {"id": 3}})])
        client.new_chat("my_docs")
        self.assertNotIn("options", client.fake_session.calls[0].json)

    def test_plain_mappings_pass_through(self):
        client = build_client([ok({"chat": {"id": 3}})])
        client.new_chat(
            options=GENERATIVE,
            chat_options={"use_intelligent_query_rewrite": True},
        )
        self.assertEqual(
            client.fake_session.calls[0].json["options"],
            {**GENERATIVE, "use_intelligent_query_rewrite": True},
        )


class SendChatMessageOptionsTest(unittest.TestCase):
    def test_chat_options_merge_into_the_generation_options(self):
        client = build_client([ok({"generated_assistant_message": "a"})])
        client.send_chat_message(
            3,
            "q",
            GenerativeOptions(generative_model="m", percentage_rank_mass=40),
            chat_options=ChatOptions(use_rag_supervisor=True),
            collection="docs",
        )

        payload = client.fake_session.calls[0].json
        self.assertEqual(
            payload["options"],
            {
                "generative_model": "m",
                "percentage_rank_mass": 40,
                "use_rag_supervisor": True,
            },
        )

    def test_chat_keywords_win_over_the_chat_options(self):
        client = build_client([ok({"generated_assistant_message": "a"})])
        client.send_chat_message(
            3,
            "q",
            collection="docs",
            use_rag_supervisor=False,
            chat_options=ChatOptions(use_rag_supervisor=True),
        )
        self.assertTrue(
            client.fake_session.calls[0].json["options"]["use_rag_supervisor"]
        )

    def test_unknown_chat_keys_are_an_error(self):
        client = build_client([ok({"generated_assistant_message": "a"})])
        with self.assertRaises(SSEValueError):
            client.send_chat_message(
                3, "q", collection="docs", chat_options="not-an-options"
            )


class ListSearchHistoryTest(unittest.TestCase):
    BODY = {"history": []}

    def setUp(self):
        self.client = build_client([ok(self.BODY), ok(self.BODY)])
        self.session = self.client.fake_session

    def test_query_response_id_is_sent_as_a_parameter(self):
        self.client.list_search_history(query_response_id=42)
        self.assertEqual(
            self.session.calls[0].params,
            {"query_response_id": 42, "lang": "pl"},
        )

    def test_both_filters_may_be_combined(self):
        self.client.list_search_history("docs", query_response_id=42)
        self.assertEqual(
            self.session.calls[0].params,
            {"collection_name": "docs", "query_response_id": 42, "lang": "pl"},
        )

    def test_without_filters_no_params_are_sent(self):
        self.client.list_search_history()
        self.assertEqual(self.session.calls[0].params, {"lang": "pl"})


if __name__ == "__main__":
    unittest.main()
