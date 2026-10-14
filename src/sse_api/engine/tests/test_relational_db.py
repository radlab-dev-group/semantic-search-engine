from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from sse_api.data.models import Document
from sse_api.engine.controllers.database.relational_db import RelationalDBController


class RelationalDBReaderMetadataTests(SimpleTestCase):
    def test_pages_without_reader_page_metadata_get_sequential_numbers(self):
        controller = RelationalDBController(store_to_db=False)
        document = SimpleNamespace(upload_process=None)
        collection = object()
        organisation_user = object()
        document_dict = {
            "filepath": "document.md",
            "relative_filepath": "document.md",
            "category": "",
            "options": {},
            "pages": [
                {"metadata": {"source": "document.md"}, "page_content": "First"},
                {"metadata": {"source": "document.md"}, "page_content": "Second"},
                {"metadata": {"page": None}, "page_content": "Third"},
            ],
        }

        with (
            patch(
                "sse_api.engine.controllers.database.relational_db.Document.objects.get",
                side_effect=Document.DoesNotExist,
            ),
            patch(
                "sse_api.engine.controllers.database.relational_db.Document.objects.get_or_create",
                return_value=(document, True),
            ),
            patch(
                "sse_api.engine.controllers.database.relational_db.DocumentPage.objects.get_or_create",
                side_effect=[(object(), True), (object(), True), (object(), True)],
            ) as get_or_create_page,
            patch.object(controller, "add_document_chunk", return_value=object()),
        ):
            controller.add_document_from_dict(
                collection=collection,
                document_dict=document_dict,
                organisation_user=organisation_user,
            )

        self.assertEqual(
            [call.kwargs["page_number"] for call in get_or_create_page.call_args_list],
            [1, 2, 3],
        )

    def test_reader_page_metadata_takes_precedence_over_fallback(self):
        controller = RelationalDBController(store_to_db=False)
        document = object()

        with (
            patch(
                "sse_api.engine.controllers.database.relational_db.DocumentPage.objects.get_or_create",
                return_value=(object(), True),
            ) as get_or_create_page,
            patch.object(controller, "add_document_chunk", return_value=object()),
        ):
            controller.get_add_document_page(
                document,
                {"metadata": {"page": "7"}, "page_content": "Content"},
            )

        get_or_create_page.assert_called_once_with(document=document, page_number=7)