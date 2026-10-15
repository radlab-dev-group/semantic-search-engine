"""
Wire-format tests of the visibility methods of :class:`sse_lib.SSEClient`.

They pin the request the backend expects (endpoint, method, field names) and
the client side validation which refuses to change an empty set of documents.
"""

from __future__ import annotations

import unittest

from sse_lib import SSEValueError

from support import build_client, ok

BODY = {"updated_documents": 1, "use_in_search": False, "not_found": []}


class SetDocumentsUseInSearchTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok(BODY)])
        self.session = self.client.fake_session

    def test_the_documents_are_disabled_by_their_names(self):
        result = self.client.set_documents_use_in_search(
            "my_docs", documents=["a.txt", "b.txt"], use_in_search=False
        )

        self.assertEqual(result, BODY)
        call = self.session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(
            call.url, "http://sse.test:8271/api/set_documents_use_in_search"
        )
        self.assertEqual(
            call.json,
            {
                "collection_name": "my_docs",
                "documents[]": ["a.txt", "b.txt"],
                "use_in_search": False,
                "lang": "pl",
            },
        )

    def test_the_documents_are_enabled_by_their_ids(self):
        self.client.set_documents_use_in_search(
            "my_docs", document_ids=[7, 8], use_in_search=True
        )
        payload = self.session.calls[0].json
        self.assertEqual(payload["document_ids[]"], [7, 8])
        self.assertTrue(payload["use_in_search"])

    def test_names_and_ids_may_be_combined(self):
        self.client.set_documents_use_in_search("my_docs", "a.txt", document_ids=[7])
        payload = self.session.calls[0].json
        self.assertEqual(payload["documents[]"], ["a.txt"])
        self.assertEqual(payload["document_ids[]"], [7])

    def test_something_has_to_be_given_to_change(self):
        for names, ids in ((None, None), ([], [])):
            with self.subTest(names=names, ids=ids):
                with self.assertRaises(SSEValueError):
                    self.client.set_documents_use_in_search(
                        "my_docs", documents=names, document_ids=ids
                    )
        self.assertEqual(self.session.calls, [])

    def test_the_flag_is_sent_as_a_boolean(self):
        self.client.set_documents_use_in_search(
            "my_docs", documents=["a.txt"], use_in_search=False
        )
        self.assertFalse(self.session.calls[0].json["use_in_search"])


class DeactivateDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok(BODY)])
        self.session = self.client.fake_session

    def test_deactivation_sets_the_flag_to_false(self):
        self.client.deactivate_documents("my_docs", documents=["a.txt"])

        call = self.session.calls[0]
        self.assertEqual(
            call.url, "http://sse.test:8271/api/set_documents_use_in_search"
        )
        self.assertFalse(call.json["use_in_search"])

    def test_nothing_has_to_be_given(self):
        with self.assertRaises(SSEValueError):
            self.client.deactivate_documents("my_docs")
        self.assertEqual(self.session.calls, [])


class ActivateDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok(dict(BODY, use_in_search=True))])
        self.session = self.client.fake_session

    def test_activation_sets_the_flag_to_true(self):
        self.client.activate_documents("my_docs", document_ids=[3])

        call = self.session.calls[0]
        self.assertTrue(call.json["use_in_search"])
        self.assertEqual(call.json["document_ids[]"], [3])


if __name__ == "__main__":
    unittest.main()
