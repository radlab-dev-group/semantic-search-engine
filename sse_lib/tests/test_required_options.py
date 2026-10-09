"""
test_required_options.py
------------------------

The backend requires ``max_tokens_in_chunk`` / ``number_of_overlap_tokens`` for
indexing and ``generative_model`` / ``percentage_rank_mass`` for
``generative_answer``, and rejects them as ``null``.  The client used to send
exactly that and get an opaque validation error back; it now refuses before
touching the network.

The chat endpoints are covered from the other side: they validate ``options``
permissively, so the client must NOT demand those fields there.
"""

from __future__ import annotations

import unittest

from support import INDEXING, build_client, ok

from sse_lib import SSEValueError


class IndexingRequiredFieldsTest(unittest.TestCase):
    def test_missing_chunk_size_is_rejected_before_sending(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError) as caught:
            client.index_documents("docs", ["text"])
        self.assertIn("max_tokens_in_chunk", str(caught.exception))
        self.assertEqual(client.fake_session.calls, [])

    def test_missing_overlap_is_rejected_before_sending(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError) as caught:
            client.index_documents("docs", ["text"], max_tokens_in_chunk=128)
        self.assertIn("number_of_overlap_tokens", str(caught.exception))
        self.assertEqual(client.fake_session.calls, [])

    def test_upload_files_is_rejected_before_opening_files(self):
        client = build_client([ok({})])
        with self.assertRaises(SSEValueError):
            client.upload_files("docs", [("a.txt", b"text")])
        self.assertEqual(client.fake_session.calls, [])

    def test_complete_options_are_sent(self):
        client = build_client([ok({})])
        client.index_documents("docs", ["text"], **INDEXING)
        sent = client.fake_session.calls[0].json["indexing_options"]
        self.assertEqual(sent["max_tokens_in_chunk"], 128)
        self.assertEqual(sent["number_of_overlap_tokens"], 0)

    def test_client_defaults_satisfy_the_requirement(self):
        client = build_client([ok({})], default_indexing_options=INDEXING)
        client.index_documents("docs", ["text"])
        self.assertEqual(len(client.fake_session.calls), 1)

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
