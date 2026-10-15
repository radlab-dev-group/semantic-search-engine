"""
test_required_options.py
------------------------

What the backend insists on, checked before the request leaves the client.

``generative_model`` / ``percentage_rank_mass`` of ``generative_answer`` are
required and rejected as ``null``, so the client refuses to send them unset.
The chunk keys of indexing are the opposite: they belong to the collection and
are optional, but when they are sent they still have to describe a usable
split (``0 <= overlap < chunk``), and a client-wide default is applied.

The chat endpoints are covered from the other side: they validate ``options``
permissively, so the client must NOT demand those fields there.
"""

from __future__ import annotations

import json
import unittest

from support import INDEXING, build_client, ok

from sse_lib import SSEValueError


class IndexingChunkingTest(unittest.TestCase):
    """Chunk size and overlap are the settings of a collection, not of a call."""

    def test_chunk_keys_may_be_left_to_the_collection(self):
        client = build_client([ok({})])
        client.index_documents("docs", ["text"])
        sent = client.fake_session.calls[0].json["indexing_options"]
        self.assertNotIn("max_tokens_in_chunk", sent)
        self.assertNotIn("number_of_overlap_tokens", sent)

    def test_upload_without_chunk_keys_is_sent(self):
        client = build_client([ok({})])
        client.upload_files("docs", [("a.txt", b"text")])
        call = client.fake_session.calls[0]
        sent = json.loads(call.data["indexing_options"])
        self.assertNotIn("max_tokens_in_chunk", sent)
        self.assertEqual(call.url, "http://sse.test:8271/api/upload_and_index_files")

    def test_complete_options_are_sent(self):
        client = build_client([ok({})])
        client.index_documents("docs", ["text"], **INDEXING)
        sent = client.fake_session.calls[0].json["indexing_options"]
        self.assertEqual(sent["max_tokens_in_chunk"], 128)
        self.assertEqual(sent["number_of_overlap_tokens"], 0)

    def test_client_defaults_are_still_applied(self):
        client = build_client([ok({})], default_indexing_options=INDEXING)
        client.index_documents("docs", ["text"])
        sent = client.fake_session.calls[0].json["indexing_options"]
        self.assertEqual(sent["max_tokens_in_chunk"], 128)
        self.assertEqual(sent["number_of_overlap_tokens"], 0)

    def test_overlap_alone_is_sent_and_validated(self):
        client = build_client([ok({})])
        client.index_documents("docs", ["text"], number_of_overlap_tokens=20)
        sent = client.fake_session.calls[0].json["indexing_options"]
        self.assertEqual(sent["number_of_overlap_tokens"], 20)
        self.assertNotIn("max_tokens_in_chunk", sent)

    def test_overlap_must_be_smaller_than_the_chunk(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError) as caught:
            client.index_documents(
                "docs",
                ["text"],
                max_tokens_in_chunk=128,
                number_of_overlap_tokens=128,
            )
        self.assertIn("number_of_overlap_tokens", str(caught.exception))

    def test_chunk_size_must_be_positive(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError):
            client.index_documents(
                "docs", ["text"], max_tokens_in_chunk=0, number_of_overlap_tokens=0
            )


class GenerativeRequiredFieldsTest(unittest.TestCase):
    def test_missing_percentage_rank_mass_is_rejected_before_sending(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError) as caught:
            client.generate_answer(1, {"generative_model": "m"})
        self.assertIn("percentage_rank_mass", str(caught.exception))
        self.assertEqual(client.fake_session.calls, [])

    def test_complete_options_are_sent(self):
        client = build_client([ok({"answer": "x"})])
        client.generate_answer(
            1, {"generative_model": "m", "percentage_rank_mass": 40}
        )
        sent = client.fake_session.calls[0].json
        self.assertEqual(
            sent["query_options"],
            '{"generative_model": "m", "percentage_rank_mass": 40}',
        )

    def test_percentage_must_be_a_percentage(self):
        client = build_client([ok({})])
        for mass in (-1, 101):
            with self.subTest(mass=mass):
                with self.assertRaises(SSEValueError):
                    client.generate_answer(
                        1, {"generative_model": "m", "percentage_rank_mass": mass}
                    )

    def test_translate_answer_needs_a_language(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError) as caught:
            client.generate_answer(
                1,
                {
                    "generative_model": "m",
                    "percentage_rank_mass": 40,
                    "translate_answer": True,
                },
            )
        self.assertIn("answer_language", str(caught.exception))


class ChatIsNotOverConstrainedTest(unittest.TestCase):
    """``add_user_message`` validates options with the permissive options_object."""

    def test_chat_message_without_generative_fields_is_still_sent(self):
        client = build_client([ok({})])
        client.send_chat_message(
            chat_id=1, collection="docs", message="hello", options={}
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/add_user_message")
        self.assertEqual(call.json["options"], {})


if __name__ == "__main__":
    unittest.main()
