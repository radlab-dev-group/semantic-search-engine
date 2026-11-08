"""
Wire-format tests of :meth:`sse_lib.SSEClient.delete_search_history`.

They pin the request the backend expects (endpoint, method, the ``query_ids[]``
field name) and the client-side validation which refuses to send an empty list
of ids.
"""

from __future__ import annotations

import unittest

from sse_lib import SSEValueError

from support import build_client, ok

BODY = {
    "deleted": {
        "answers": 1,
        "responses": 2,
        "queries": 2,
        "rag_states": 1,
    },
    "not_found": {"query_ids": [], "query_response_ids": []},
}


class DeleteSearchHistoryTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok(BODY)])
        self.session = self.client.fake_session

    def test_queries_are_removed_by_their_ids(self):
        result = self.client.delete_search_history([1, 2])

        self.assertEqual(result, BODY)
        call = self.session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, "http://sse.test:8271/api/delete_search_history")
        self.assertEqual(
            call.json,
            {"query_ids[]": [1, 2], "lang": "pl"},
        )

    def test_a_single_query_id_may_be_given(self):
        self.client.delete_search_history(5)
        self.assertEqual(self.session.calls[0].json["query_ids[]"], [5])

    def test_giving_no_identifier_at_all_is_rejected_early(self):
        for call in (
            lambda: self.client.delete_search_history(),
            lambda: self.client.delete_search_history([]),
            lambda: self.client.delete_search_history(
                query_ids=[], query_response_ids=[]
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaises(SSEValueError):
                    call()
        self.assertEqual(self.session.calls, [])

    def test_not_numeric_identifiers_are_rejected_early(self):
        for kwargs in (
            {"query_ids": ["abc"]},
            {"query_ids": [0]},
            {"query_ids": [-1]},
            {"query_response_ids": ["abc"]},
            {"query_response_ids": [0]},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(SSEValueError):
                    self.client.delete_search_history(**kwargs)
        self.assertEqual(self.session.calls, [])

    def test_query_response_ids_are_sent_on_their_own_field(self):
        self.client.delete_search_history(query_response_ids=[3, 4])

        call = self.client.fake_session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(
            call.json,
            {"query_response_ids[]": [3, 4], "lang": "pl"},
        )

    def test_both_identifiers_may_be_given_together(self):
        self.client.delete_search_history(query_ids=[1], query_response_ids=[7])

        self.assertEqual(
            self.client.fake_session.calls[0].json,
            {"query_ids[]": [1], "query_response_ids[]": [7], "lang": "pl"},
        )

    def test_a_single_query_response_id_may_be_given(self):
        self.client.delete_search_history(query_response_ids=5)
        self.assertEqual(
            self.client.fake_session.calls[0].json["query_response_ids[]"], [5]
        )

    def test_not_found_ids_are_reported_per_identifier(self):
        self.client.fake_session.responses.append(
            ok(
                {
                    "deleted": {
                        "answers": 0,
                        "responses": 1,
                        "queries": 1,
                        "rag_states": 0,
                    },
                    "not_found": {"query_ids": [7], "query_response_ids": []},
                }
            )
        )
        self.client.delete_search_history(1)
        result = self.client.delete_search_history([7])
        self.assertEqual(result["not_found"]["query_ids"], [7])


CHAT_BODY = {
    "deleted": {
        "chats": 2,
        "messages": 4,
        "message_states": 2,
        "rag_states": 2,
        "content_supervisor_states": 0,
    },
    "not_found": {"chat_ids": []},
}


class DeleteChatTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok(CHAT_BODY)])
        self.session = self.client.fake_session

    def test_chats_are_removed_by_their_ids(self):
        result = self.client.delete_chat([1, 2])

        self.assertEqual(result, CHAT_BODY)
        call = self.session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, "http://sse.test:8271/api/delete_chat")
        self.assertEqual(
            call.json,
            {"chat_ids[]": [1, 2], "lang": "pl"},
        )

    def test_a_single_chat_id_may_be_given(self):
        self.client.delete_chat(5)
        self.assertEqual(self.session.calls[0].json["chat_ids[]"], [5])

    def test_chat_id_keyword_is_accepted(self):
        self.client.delete_chat(chat_id=9)
        self.assertEqual(self.session.calls[0].json["chat_ids[]"], [9])

    def test_giving_no_identifier_is_rejected_early(self):
        for call in (
            lambda: self.client.delete_chat(),
            lambda: self.client.delete_chat([]),
        ):
            with self.subTest(call=call):
                with self.assertRaises(SSEValueError):
                    call()
        self.assertEqual(self.session.calls, [])

    def test_not_numeric_identifiers_are_rejected_early(self):
        for kwargs in (
            {"chat_ids": ["abc"]},
            {"chat_ids": [0]},
            {"chat_ids": [-1]},
            {"chat_id": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(SSEValueError):
                    self.client.delete_chat(**kwargs)
        self.assertEqual(self.session.calls, [])


if __name__ == "__main__":
    unittest.main()
