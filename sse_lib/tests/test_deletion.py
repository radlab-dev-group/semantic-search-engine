"""
Wire-format tests of the deletion methods of :class:`sse_lib.SSEClient`.

They pin the request the backend expects (endpoint, method, field names) and
the client side validation which refuses to send an empty list of ids.
"""

from __future__ import annotations

import unittest

from sse_lib import SSEValueError

from support import build_client, ok


class DeleteTextsTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok({"deleted": {"texts": 2}})])
        self.session = self.client.fake_session

    def test_chunks_are_removed_by_their_ids(self):
        result = self.client.delete_texts("my_docs", [1, 2])

        self.assertEqual(result, {"deleted": {"texts": 2}})
        call = self.session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, "http://sse.test:8271/api/delete_texts")
        self.assertEqual(
            call.json,
            {"collection_name": "my_docs", "chunk_ids[]": [1, 2], "lang": "pl"},
        )

    def test_a_single_chunk_id_may_be_given(self):
        self.client.delete_texts("my_docs", 5)
        self.assertEqual(self.session.calls[0].json["chunk_ids[]"], [5])

    def test_empty_chunk_ids_are_rejected_early(self):
        for chunk_ids in ([], None):
            with self.subTest(chunk_ids=chunk_ids):
                with self.assertRaises(SSEValueError):
                    self.client.delete_texts("my_docs", chunk_ids)
        self.assertEqual(self.session.calls, [])

    def test_not_numeric_chunk_ids_are_rejected_early(self):
        for chunk_ids in (["abc"], [0], [-1]):
            with self.subTest(chunk_ids=chunk_ids):
                with self.assertRaises(SSEValueError):
                    self.client.delete_texts("my_docs", chunk_ids)
        self.assertEqual(self.session.calls, [])


class DeleteDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok({"deleted": {"documents": 1}})])
        self.session = self.client.fake_session

    def test_documents_are_removed_by_their_names(self):
        result = self.client.delete_documents(
            "my_docs", documents=["a.txt", "b.txt"]
        )

        self.assertEqual(result, {"deleted": {"documents": 1}})
        call = self.session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/delete_documents")
        self.assertEqual(
            call.json,
            {
                "collection_name": "my_docs",
                "documents[]": ["a.txt", "b.txt"],
                "lang": "pl",
            },
        )

    def test_documents_are_removed_by_their_ids(self):
        self.client.delete_documents("my_docs", document_ids=[7, 8])
        self.assertEqual(self.session.calls[0].json["document_ids[]"], [7, 8])

    def test_names_and_ids_may_be_combined(self):
        self.client.delete_documents("my_docs", "a.txt", document_ids=[7])
        payload = self.session.calls[0].json
        self.assertEqual(payload["documents[]"], ["a.txt"])
        self.assertEqual(payload["document_ids[]"], [7])

    def test_something_has_to_be_given_to_delete(self):
        for names, ids in ((None, None), ([], [])):
            with self.subTest(names=names, ids=ids):
                with self.assertRaises(SSEValueError):
                    self.client.delete_documents(
                        "my_docs", documents=names, document_ids=ids
                    )
        self.assertEqual(self.session.calls, [])


class DeleteCollectionTest(unittest.TestCase):
    def setUp(self):
        self.client = build_client([ok({"dropped_collection": True})])
        self.session = self.client.fake_session

    def test_the_collection_is_removed_by_its_name(self):
        result = self.client.delete_collection("my_docs")

        self.assertEqual(result, {"dropped_collection": True})
        call = self.session.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, "http://sse.test:8271/api/delete_collection")
        self.assertEqual(call.json, {"collection_name": "my_docs", "lang": "pl"})

    def test_a_collection_object_may_be_given_instead_of_a_name(self):
        from sse_lib import Collection

        self.client.delete_collection(Collection(id=1, name="my_docs"))
        self.assertEqual(self.session.calls[0].json["collection_name"], "my_docs")


if __name__ == "__main__":
    unittest.main()
