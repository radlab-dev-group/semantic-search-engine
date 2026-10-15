"""
Wire-format tests of the document listing of :class:`sse_lib.SSEClient`.

They pin the request the backend expects for the two views of
``GET documents`` (the plain list of names and the ``with_disabled`` view
which carries the identifiers used by the deletion and visibility endpoints)
and the :class:`sse_lib.models.DocumentRef` model.
"""

from __future__ import annotations

import unittest

from sse_lib import DocumentRef

from support import build_client, ok


class ListDocumentsTest(unittest.TestCase):
    def test_only_the_searchable_documents_are_listed_by_default(self):
        client = build_client([ok({"documents": [{"name": "a.pdf"}]})])
        names = client.list_documents("docs")

        self.assertEqual(names, ["a.pdf"])
        call = client.fake_session.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertEqual(call.url, "http://sse.test:8271/api/documents")
        self.assertEqual(call.params, {"collection_name": "docs", "lang": "pl"})

    def test_with_disabled_sends_the_flag_as_a_boolean_string(self):
        client = build_client([ok({"documents": [{"name": "a.pdf"}]})])
        client.list_documents("docs", with_disabled=True)

        call = client.fake_session.calls[0]
        self.assertEqual(
            call.params,
            {"collection_name": "docs", "with_disabled": "true", "lang": "pl"},
        )

    def test_plain_names_are_returned_for_a_plain_server_answer(self):
        # An older server answers with a list of strings, not objects.
        client = build_client([ok({"documents": ["a.pdf", "b.pdf"]})])
        self.assertEqual(client.list_documents("docs"), ["a.pdf", "b.pdf"])


class ListDocumentDetailsTest(unittest.TestCase):
    BODY = {
        "documents": [
            {"id": 1, "name": "a.pdf", "use_in_search": True},
            {"id": 2, "name": "off.pdf", "use_in_search": False},
        ]
    }

    def setUp(self):
        self.client = build_client([ok(self.BODY)])
        self.session = self.client.fake_session

    def test_the_with_disabled_view_is_used(self):
        references = self.client.list_document_details("docs")

        call = self.session.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertEqual(
            call.params,
            {"collection_name": "docs", "with_disabled": "true", "lang": "pl"},
        )

    def test_the_flags_and_identifiers_are_modelled(self):
        references = self.client.list_document_details("docs")

        self.assertEqual([ref.name for ref in references], ["a.pdf", "off.pdf"])
        self.assertEqual([ref.id for ref in references], [1, 2])
        self.assertEqual([ref.use_in_search for ref in references], [True, False])

    def test_the_raw_payload_is_kept(self):
        reference = self.client.list_document_details("docs")[1]
        self.assertIsInstance(reference, DocumentRef)
        self.assertEqual(
            reference.raw, {"id": 2, "name": "off.pdf", "use_in_search": False}
        )

    def test_the_identifiers_can_drive_the_visibility_endpoint(self):
        references = self.client.list_document_details("docs")
        self.client.fake_session.responses.append(
            ok({"updated_documents": 1, "use_in_search": True, "not_found": []})
        )
        self.client.activate_documents(
            "docs", document_ids=[ref.id for ref in references]
        )

        call = self.client.fake_session.calls[1]
        self.assertEqual(call.json["document_ids[]"], [1, 2])


if __name__ == "__main__":
    unittest.main()
