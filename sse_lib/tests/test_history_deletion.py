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
    "not_found": [],
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

    def test_empty_query_ids_are_rejected_early(self):
        for query_ids in ([], None):
            with self.subTest(query_ids=query_ids):
                with self.assertRaises(SSEValueError):
                    self.client.delete_search_history(query_ids)
        self.assertEqual(self.session.calls, [])

    def test_not_numeric_query_ids_are_rejected_early(self):
        for query_ids in (["abc"], [0], [-1]):
            with self.subTest(query_ids=query_ids):
                with self.assertRaises(SSEValueError):
                    self.client.delete_search_history(query_ids)
        self.assertEqual(self.session.calls, [])

    def test_not_found_ids_are_reported_to_the_caller(self):
        self.client.fake_session.responses.append(
            ok(
                {
                    "deleted": {
                        "answers": 0,
                        "responses": 1,
                        "queries": 1,
                        "rag_states": 0,
                    },
                    "not_found": [7],
                }
            )
        )
        self.client.delete_search_history(1)
        result = self.client.delete_search_history([7])
        self.assertEqual(result["not_found"], [7])


if __name__ == "__main__":
    unittest.main()
