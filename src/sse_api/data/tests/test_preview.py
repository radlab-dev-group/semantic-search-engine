"""
Content of a whole document (``GET document_content``).

The preview is read from the relational database only: pages of a document,
the chunks of one ``text_chunk_type`` each, joined back into the text of the
document.  These tests cover the chunk type which wins when a document was
indexed more than once, the window of pages and the view with its errors.
"""

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import force_authenticate

from sse_api.data.api import GetDocumentContent
from sse_api.data.controllers.preview import ContentPreviewController
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
)
from sse_api.core.utils import compute_text_hash
from sse_api.system.models import Organisation, OrganisationUser
from sse_api.tests.input_support import EndpointInputMixin


class PreviewFixture:
    """Document with two pages and two chunk types of the same text."""

    def make_fixture(self):
        self.collection = CollectionOfDocuments.objects.create(
            name="preview", display_name="Preview", created_by=self.profile
        )
        self.document = Document.objects.create(
            name="document.txt",
            collection=self.collection,
            path="/tmp/document.txt",
            relative_path="/document.txt",
            document_hash="hash-1",
            category="kat",
            added_by=self.profile,
        )
        self.first = self.make_page(self.document, 1)
        self.second = self.make_page(self.document, 2)
        # The form the document was indexed with first, and the cleared one
        # stored later, which the preview shows by default.
        self.make_chunk(self.first, 1, "raw first", "raw_text")
        self.make_chunk(self.first, 1, "first chunk", "clear_texts")
        self.make_chunk(self.first, 2, "of the page", "clear_texts")
        self.make_chunk(self.second, 1, "second chunk", "clear_texts")

    def make_page(self, document, page_number):
        return DocumentPage.objects.create(
            document=document, page_number=page_number
        )

    def make_chunk(self, page, text_number, text, chunk_type):
        return DocumentPageText.objects.create(
            page=page,
            text_number=text_number,
            text_str=text,
            text_hash=compute_text_hash(f"{page.page_number}-{text_number}-{text}"),
            text_chunk_type=chunk_type,
            language="en",
        )


class DocumentPreviewControllerTests(EndpointInputMixin, PreviewFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.make_fixture()
        self.controller = ContentPreviewController()

    def test_the_document_is_found_by_its_name_and_id(self):
        by_name = self.controller.find_document(
            collection=self.collection, document_name="document.txt"
        )
        by_id = self.controller.find_document(
            collection=self.collection, document_id=self.document.pk
        )

        self.assertEqual(by_name, self.document)
        self.assertEqual(by_id, self.document)
        self.assertIsNone(
            self.controller.find_document(
                collection=self.collection, document_name="other.txt"
            )
        )

    def test_the_newest_document_of_a_duplicated_name_is_shown(self):
        newer = Document.objects.create(
            name="document.txt",
            collection=self.collection,
            path="/tmp/other/document.txt",
            relative_path="/other/document.txt",
            document_hash="hash-2",
            added_by=self.profile,
        )

        found = self.controller.find_document(
            collection=self.collection, document_name="document.txt"
        )

        self.assertEqual(found, newer)

    def test_the_whole_document_is_stitched_page_by_page(self):
        preview = self.controller.document_content(document=self.document)

        self.assertEqual(
            preview.as_dict(),
            {
                "document_id": self.document.pk,
                "document_name": "document.txt",
                "relative_path": "/document.txt",
                "category": "kat",
                "use_in_search": True,
                "chunk_type": "clear_texts",
                "chunk_types": ["clear_texts", "raw_text"],
                "pages": [
                    {
                        "page": 1,
                        "content": "first chunk\n\nof the page",
                        "chunks": 2,
                    },
                    {"page": 2, "content": "second chunk", "chunks": 1},
                ],
                "content": "first chunk\n\nof the page\n\nsecond chunk",
                "characters": len("first chunk\n\nof the page\n\nsecond chunk"),
                "chunks": 3,
                "from_page": 1,
                "total_pages": 2,
                "truncated": False,
                "next_page": None,
            },
        )

    def test_a_chunk_type_may_be_asked_for(self):
        preview = self.controller.document_content(
            document=self.document, chunk_type="raw_text"
        )

        self.assertEqual(preview.chunk_type, "raw_text")
        self.assertEqual(preview.content, "raw first")

    def test_an_unknown_chunk_type_falls_back_to_the_newest_one(self):
        preview = self.controller.document_content(
            document=self.document, chunk_type="there_is_no_such_type"
        )

        self.assertEqual(preview.chunk_type, "clear_texts")

    def test_a_window_of_pages_is_reported_as_truncated(self):
        preview = self.controller.document_content(
            document=self.document, from_page=2, max_pages=1
        )

        self.assertEqual([page["page"] for page in preview.pages], [2])
        self.assertEqual(preview.content, "second chunk")
        self.assertEqual(preview.total_pages, 2)
        self.assertFalse(preview.truncated)
        self.assertIsNone(preview.next_page)

    def test_the_next_window_is_announced(self):
        preview = self.controller.document_content(
            document=self.document, max_pages=1
        )

        self.assertEqual([page["page"] for page in preview.pages], [1])
        self.assertTrue(preview.truncated)
        self.assertEqual(preview.next_page, 2)
        self.assertEqual(preview.total_pages, 2)

    def test_max_pages_is_bounded(self):
        from sse_api.data.core.constants import (
            PREVIEW_MAX_PAGES,
            PREVIEW_MAX_PAGES_LIMIT,
        )

        bounded = self.controller._ContentPreviewController__bounded_pages
        self.assertEqual(bounded(0), 1)
        self.assertEqual(bounded(10**9), PREVIEW_MAX_PAGES_LIMIT)
        self.assertEqual(bounded("many"), PREVIEW_MAX_PAGES)

    def test_a_document_without_any_text_has_an_empty_preview(self):
        empty = Document.objects.create(
            name="empty.txt",
            collection=self.collection,
            path="/tmp/empty.txt",
            relative_path="/empty.txt",
            document_hash="hash-3",
            added_by=self.profile,
        )

        preview = self.controller.document_content(document=empty)

        self.assertEqual(preview.pages, [])
        self.assertEqual(preview.content, "")
        self.assertEqual(preview.chunk_type, "")
        self.assertFalse(preview.truncated)


class DocumentPreviewEndpointTests(EndpointInputMixin, PreviewFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.make_fixture()

    def invoke_view(self, payload, user=None):
        request = self.factory.get("/", payload or {})
        force_authenticate(request, user=(user or self.profile).auth_user)
        return GetDocumentContent.as_view()(request)

    def test_the_collection_of_another_user_is_not_found(self):
        other = User.objects.create_user(username="someone-else")
        other_profile = OrganisationUser.objects.create(
            auth_user=other,
            organisation=Organisation.objects.create(name="Other organisation"),
        )

        response = self.invoke_view(
            {"collection_name": "preview", "document_name": "document.txt"},
            user=other_profile,
        )

        self.assertFalse(response.data["status"])
        # The shared "collection not found" error of the chat app (the data
        # app reuses it; its code carries the ERROR_MARK_CHAT suffix).
        self.assertEqual(response.data["errors"][0]["error_code"], "000002___chat")

    def test_a_document_of_the_collection_is_returned(self):
        response = self.invoke_view(
            {"collection_name": "preview", "document_name": "document.txt"}
        )

        body = response.data["body"]
        self.assertTrue(response.data["status"])
        self.assertEqual(body["document_id"], self.document.pk)
        self.assertEqual(body["document_name"], "document.txt")
        self.assertEqual(body["chunk_type"], "clear_texts")
        self.assertEqual(
            body["content"], "first chunk\n\nof the page\n\nsecond chunk"
        )
        self.assertEqual(
            body["pages"],
            [
                {
                    "page": 1,
                    "content": "first chunk\n\nof the page",
                    "chunks": 2,
                },
                {"page": 2, "content": "second chunk", "chunks": 1},
            ],
        )
        self.assertEqual(body["total_pages"], 2)
        self.assertFalse(body["truncated"])

    def test_the_document_may_be_addressed_by_its_id(self):
        document = Document.objects.create(
            name="by-id.txt",
            collection=self.collection,
            path="/tmp/by-id.txt",
            relative_path="/by-id.txt",
            document_hash="hash-4",
            added_by=self.profile,
        )

        response = self.invoke_view(
            {"collection_name": "preview", "document_id": document.pk}
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(response.data["body"]["document_id"], document.pk)

    def test_neither_name_nor_id_is_refused(self):
        response = self.invoke_view({"collection_name": "preview"})

        self.assertFalse(response.data["status"])
        self.assertTrue(response.data["errors"])

    def test_an_unknown_document_is_reported(self):
        response = self.invoke_view(
            {"collection_name": "preview", "document_name": "missing.txt"}
        )

        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0]["error_code"],
            "000002_DATA",  # DOCUMENT_NOT_FOUND
        )

    def test_bad_page_numbers_are_refused(self):
        for value in ("0", "-3", "many"):
            with self.subTest(value=value):
                response = self.invoke_view(
                    {
                        "collection_name": "preview",
                        "document_name": "document.txt",
                        "from_page": value,
                    }
                )
                self.assertFalse(response.data["status"])
