"""
Wire-format tests of the document listing of :class:`sse_lib.SSEClient`.

They pin the request the backend expects for the two views of
``GET documents`` (the plain list of names and the ``with_disabled`` view
which carries the identifiers used by the deletion and visibility endpoints)
and the :class:`sse_lib.models.DocumentRef` model.
"""

from __future__ import annotations

import unittest

from sse_lib import DocumentPreview, DocumentRef, IndexingJobStatus

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


class DocumentContentTest(unittest.TestCase):
    BODY = {
        "document_id": 7,
        "document_name": "a.pdf",
        "relative_path": "/a.pdf",
        "category": "kat",
        "use_in_search": True,
        "chunk_type": "clear_texts",
        "chunk_types": ["clear_texts", "raw_text"],
        "pages": [
            {"page": 1, "content": "first\n\nsecond", "chunks": 2},
            {"page": 2, "content": "third", "chunks": 1},
        ],
        "content": "first\n\nsecond\n\nthird",
        "characters": 24,
        "chunks": 3,
        "from_page": 1,
        "total_pages": 2,
        "truncated": False,
        "next_page": None,
    }

    def setUp(self):
        self.client = build_client([ok(dict(self.BODY))])

    def test_a_document_is_asked_for_by_its_name(self):
        preview = self.client.document_content("docs", "a.pdf")

        call = self.client.fake_session.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertEqual(call.url, "http://sse.test:8271/api/document_content")
        self.assertEqual(
            call.params,
            {"collection_name": "docs", "document_name": "a.pdf", "lang": "pl"},
        )
        self.assertEqual(preview.document_name, "a.pdf")
        self.assertEqual(preview.content, "first\n\nsecond\n\nthird")

    def test_an_identifier_and_a_window_of_pages_are_sent(self):
        self.client.document_content(
            "docs", 7, from_page=2, max_pages=5, chunk_type="raw_text"
        )

        call = self.client.fake_session.calls[0]
        self.assertEqual(
            call.params,
            {
                "collection_name": "docs",
                "document_id": 7,
                "from_page": 2,
                "max_pages": 5,
                "chunk_type": "raw_text",
                "lang": "pl",
            },
        )

    def test_the_pages_and_the_window_are_modelled(self):
        preview = self.client.document_content("docs", "a.pdf")

        self.assertIsInstance(preview, DocumentPreview)
        self.assertEqual(preview.document_id, 7)
        self.assertEqual(preview.chunk_type, "clear_texts")
        self.assertEqual(preview.chunk_types, ["clear_texts", "raw_text"])
        self.assertEqual(
            [(page.page, page.content, page.chunks) for page in preview.pages],
            [(1, "first\n\nsecond", 2), (2, "third", 1)],
        )
        self.assertEqual(preview.total_pages, 2)
        self.assertFalse(preview.truncated)
        self.assertIsNone(preview.next_page)

    def test_a_truncated_answer_announces_the_next_window(self):
        body = dict(self.BODY, truncated=True, next_page=3)
        client = build_client([ok(body)])

        preview = client.document_content("docs", "a.pdf", max_pages=2)

        self.assertTrue(preview.truncated)
        self.assertEqual(preview.next_page, 3)
        self.assertEqual(preview.raw["total_pages"], 2)


class IndexingStatusTest(unittest.TestCase):
    """The progress of an indexing (``GET indexing_status``): a job by its id
    or the newest job of a collection with its documents and statuses."""

    BODY = {
        "job": {
            "id": 3,
            "status": "running",
            "total_documents": 3,
            "documents_indexed": 1,
            "documents_failed": 0,
            "documents_pending": 2,
            "current_document": "queue-2.pdf",
        },
        "documents": [
            {"id": 1, "name": "queue-1.pdf", "indexing_status": "indexed"},
            {"id": 2, "name": "queue-2.pdf", "indexing_status": "running"},
        ],
        "pending_documents": 1,
    }

    def setUp(self):
        self.client = build_client([ok(dict(self.BODY))])

    def test_a_job_is_asked_for_by_its_id(self):
        status = self.client.indexing_status(job_id=3)

        call = self.client.fake_session.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertEqual(call.url, "http://sse.test:8271/api/indexing_status")
        self.assertEqual(call.params, {"job_id": 3, "lang": "pl"})
        self.assertEqual(status.job.job_id, 3)
        self.assertEqual(status.job.status, "running")
        self.assertEqual(status.job.documents_indexed, 1)
        self.assertEqual(status.documents, self.BODY["documents"])
        self.assertEqual(status.pending_documents, 1)

    def test_a_collection_is_asked_for_by_its_name(self):
        self.client.indexing_status("docs")

        call = self.client.fake_session.calls[0]
        self.assertEqual(call.params, {"collection_name": "docs", "lang": "pl"})

    def test_a_collection_without_a_job_is_modelled_as_empty(self):
        client = build_client(
            [ok({"job": None, "documents": [], "pending_documents": 2})]
        )

        status = client.indexing_status("docs")

        self.assertEqual(status.job, IndexingJobStatus())
        self.assertEqual(status.documents, [])
        self.assertEqual(status.pending_documents, 2)


if __name__ == "__main__":
    unittest.main()
