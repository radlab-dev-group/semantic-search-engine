from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from sse_lib import (
    GenerativeOptions,
    IndexingOptions,
    SearchOptions,
    SSEClient,
    SSEValueError,
    build_options,
)
from support import HOST, FakeSession, build_client, ok


class OptionsRegressionTest(unittest.TestCase):
    def test_unknown_dictionary_options_are_preserved(self):
        for cls in (SearchOptions, GenerativeOptions):
            with self.subTest(cls=cls):
                self.assertEqual(
                    build_options(cls, {"custom_option": 1}).to_dict(),
                    {"custom_option": 1},
                )

    def test_unknown_indexing_dictionary_option_is_rejected(self):
        with self.assertRaises(SSEValueError):
            build_options(IndexingOptions, {"custom_option": 1})

    def test_keywords_override_extra_and_dictionary_values(self):
        options = build_options(
            SearchOptions,
            {"extra": {"custom_option": 1}, "custom_option": 2},
            {"custom_option": 3},
        )
        self.assertEqual(options.to_dict(), {"custom_option": 3})


class ClientRegressionTest(unittest.TestCase):
    def test_indexing_dictionary_does_not_reset_client_defaults(self):
        client = build_client(
            [ok({})], default_indexing_options={"clear_text": False}
        )
        client.index_documents(
            "docs", ["text"], indexing_options={"max_tokens_in_chunk": 200}
        )
        payload = client.fake_session.calls[0].json
        self.assertFalse(payload["indexing_options"]["clear_text"])
        self.assertEqual(payload["indexing_options"]["max_tokens_in_chunk"], 200)

    def test_get_forwards_request_options(self):
        client = build_client([ok({"collections": []})])
        result = client.get("collections", unwrap=False, headers={"X-Test": "yes"})
        self.assertEqual(result, {"status": True, "body": {"collections": []}})
        self.assertEqual(client.fake_session.calls[0].headers["X-Test"], "yes")

    def test_tuple_documents_are_a_sequence(self):
        client = build_client([ok({})])
        client.index_documents("docs", ("text A", "text B"))
        documents = client.fake_session.calls[0].json["texts[]"]
        self.assertEqual(
            [doc["pages"][0]["page_content"] for doc in documents],
            ["text A", "text B"],
        )

    def test_single_file_tuple_is_still_supported(self):
        client = build_client([ok({})])
        client.upload_files("docs", ("a.txt", b"text"))
        self.assertEqual(len(client.fake_session.calls[0].files), 1)
        self.assertEqual(
            client.fake_session.calls[0].files[0][1][:2], ("a.txt", b"text")
        )

    def test_explicit_empty_prefix_overrides_environment(self):
        with patch.dict("os.environ", {"SSE_API_PREFIX": "custom"}):
            client = SSEClient(HOST, api_prefix="", session=FakeSession())
        self.assertEqual(client.base_url, HOST)

    def test_chat_requires_collection_before_sending_request(self):
        client = build_client([])
        with self.assertRaisesRegex(SSEValueError, "collection"):
            client.send_chat_message(1, "question")
        self.assertEqual(client.fake_session.calls, [])

    def test_search_preserves_custom_options_through_client_defaults(self):
        client = build_client([ok({})], default_search_options={"custom": 1})
        client.search("docs", "q", {"custom": 2})
        sent = json.loads(client.fake_session.calls[0].json["options"])
        self.assertEqual(sent["custom"], 2)
