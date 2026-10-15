"""
Deactivation of documents in the search of a collection
(``Document.use_in_search``).

Tests use a real (in memory) relational database: the flag is set in the
relational db only, the texts and the embeddings are never touched.
"""

import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.chat.core.constants import ERROR_MARK_CHAT
from sse_api.data.api import ListDocumentsFromCollection, SetDocumentsUseInSearch
from sse_api.data.controllers.visibility import ContentVisibilityController
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
)
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.search.relational import DBTextSearchController
from sse_api.system.models import Organisation, OrganisationUser


class FixtureTestCase(TestCase):
    """
    Collection with two documents, one page and one text chunk each.
    """

    def setUp(self):
        self.organisation = Organisation.objects.create(name="visibility org")
        self.user = User.objects.create_user(username="visibility-user")
        self.organisation_user = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = CollectionOfDocuments.objects.create(
            name="collection",
            display_name="Collection",
            created_by=self.organisation_user,
        )
        self.document = self.add_document("document.txt", "hash-1")
        self.other_document = self.add_document("other.txt", "hash-2")
        self.page = DocumentPage.objects.create(
            document=self.document, page_number=1
        )
        self.chunk = self.add_chunk(self.page, 1, "first chunk")
        self.controller = ContentVisibilityController()

    def add_document(self, name, document_hash, collection=None):
        return Document.objects.create(
            name=name,
            collection=collection or self.collection,
            path=f"/tmp/{name}",
            relative_path=name,
            document_hash=document_hash,
            added_by=self.organisation_user,
        )

    def add_chunk(self, page, text_number, text):
        return DocumentPageText.objects.create(
            page=page,
            text_number=text_number,
            text_str=text,
            text_hash=f"hash-{text}",
            text_chunk_type="plain",
            language="en",
        )


class DocumentVisibilityControllerTests(FixtureTestCase):
    def test_document_is_disabled_by_its_name(self):
        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
        )

        self.document.refresh_from_db()
        self.assertFalse(self.document.use_in_search)
        self.assertEqual(
            result.as_dict(),
            {"updated_documents": 1, "use_in_search": False, "not_found": []},
        )
        # Nothing is removed: the page and the text chunk are still there.
        self.assertTrue(self.page.pk)
        self.assertTrue(self.chunk.pk)

    def test_document_is_enabled_again_by_its_id(self):
        self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_ids=[self.document.pk],
        )
        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=True,
            document_ids=[self.document.pk],
        )

        self.document.refresh_from_db()
        self.assertTrue(self.document.use_in_search)
        self.assertEqual(result.updated_documents, 1)
        self.assertEqual(result.as_dict()["use_in_search"], True)

    def test_names_and_ids_may_be_given_together(self):
        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
            document_ids=[self.other_document.pk],
        )

        self.document.refresh_from_db()
        self.other_document.refresh_from_db()
        self.assertFalse(self.document.use_in_search)
        self.assertFalse(self.other_document.use_in_search)
        self.assertEqual(result.updated_documents, 2)
        self.assertEqual(result.not_found, [])

    def test_a_document_given_by_name_and_id_is_updated_once(self):
        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
            document_ids=[self.document.pk],
        )

        self.assertEqual(result.updated_documents, 1)

    def test_the_names_which_do_not_exist_are_reported(self):
        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["missing.txt"],
            document_ids=[123456],
        )

        self.assertEqual(result.updated_documents, 0)
        self.assertEqual(result.not_found, ["missing.txt", "123456"])
        self.document.refresh_from_db()
        self.assertTrue(self.document.use_in_search)

    def test_the_documents_of_other_collections_are_not_touched(self):
        other_user = User.objects.create_user(username="visibility-other")
        other_collection = CollectionOfDocuments.objects.create(
            name="foreign",
            display_name="Foreign",
            created_by=OrganisationUser.objects.create(
                auth_user=other_user, organisation=self.organisation
            ),
        )
        foreign_document = self.add_document(
            "document.txt", "hash-3", other_collection
        )

        result = self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
        )

        self.assertEqual(result.updated_documents, 1)
        self.document.refresh_from_db()
        foreign_document.refresh_from_db()
        self.assertFalse(self.document.use_in_search)
        self.assertTrue(foreign_document.use_in_search)


class DocumentVisibilityEndpointTests(FixtureTestCase):
    """
    Who is allowed to deactivate what and how the answers look like.
    """

    def setUp(self):
        super().setUp()
        other_user = User.objects.create_user(username="visibility-other-user")
        self.foreign_collection = CollectionOfDocuments.objects.create(
            name="foreign",
            display_name="Foreign",
            created_by=OrganisationUser.objects.create(
                auth_user=other_user, organisation=self.organisation
            ),
        )
        self.factory = APIRequestFactory()

    def invoke(self, payload, method="post"):
        request = self.factory.generic(
            method.upper(), "/", json.dumps(payload), content_type="application/json"
        )
        force_authenticate(request, user=self.user)
        return SetDocumentsUseInSearch.as_view()(request)

    def test_the_document_is_deactivated_by_its_name(self):
        response = self.invoke(
            {
                "collection_name": "collection",
                "documents[]": ["document.txt"],
                "use_in_search": False,
            }
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(
            response.data["body"],
            {"updated_documents": 1, "use_in_search": False, "not_found": []},
        )
        self.document.refresh_from_db()
        self.assertFalse(self.document.use_in_search)

    def test_deactivation_is_the_default_flag_value(self):
        response = self.invoke(
            {
                "collection_name": "collection",
                "document_ids[]": [self.document.pk],
            }
        )

        self.assertTrue(response.data["status"])
        self.document.refresh_from_db()
        self.assertFalse(self.document.use_in_search)
        self.assertEqual(response.data["body"]["use_in_search"], False)

    def test_the_document_is_activated_by_its_id(self):
        self.document.use_in_search = False
        self.document.save()

        response = self.invoke(
            {
                "collection_name": "collection",
                "document_ids[]": [self.document.pk],
                "use_in_search": True,
            }
        )

        self.assertTrue(response.data["status"])
        self.document.refresh_from_db()
        self.assertTrue(self.document.use_in_search)

    def test_the_documents_of_the_collection_are_required(self):
        for payload in (
            {"collection_name": "collection"},
            {"collection_name": "collection", "documents[]": []},
        ):
            with self.subTest(payload=payload):
                response = self.invoke(payload)
                self.assertFalse(response.data["status"])
        self.document.refresh_from_db()
        self.assertTrue(self.document.use_in_search)

    def test_the_flag_must_be_a_boolean(self):
        for flag in ("nope", 2, None, ["False"]):
            with self.subTest(use_in_search=flag):
                response = self.invoke(
                    {
                        "collection_name": "collection",
                        "documents[]": ["document.txt"],
                        "use_in_search": flag,
                    }
                )
                self.assertFalse(response.data["status"])
        self.document.refresh_from_db()
        self.assertTrue(self.document.use_in_search)

    def test_the_documents_which_were_not_found_are_reported(self):
        response = self.invoke(
            {
                "collection_name": "collection",
                "documents[]": ["missing.txt"],
                "use_in_search": False,
            }
        )

        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000002_DATA")

    def test_the_documents_of_a_foreign_collection_cannot_be_changed(self):
        response = self.invoke(
            {
                "collection_name": "foreign",
                "documents[]": ["document.txt"],
                "use_in_search": False,
            }
        )

        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0]["error_code"],
            f"000002_{ERROR_MARK_CHAT}",
        )


class DeactivatedDocumentSearchTests(FixtureTestCase):
    """
    The chunks of a deactivated document are not returned by the search
    anymore, on the relational side of it.
    """

    def setUp(self):
        super().setUp()
        other_page = DocumentPage.objects.create(
            document=self.other_document, page_number=1
        )
        self.other_chunk = self.add_chunk(other_page, 1, "second chunk")

    def test_get_texts_skips_the_chunks_of_deactivated_documents(self):
        self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
        )

        results = DBTextSearchController().get_texts(
            texts_ids=[self.chunk.pk, self.other_chunk.pk],
            texts_scores=[0.9, 0.8],
        )
        self.assertEqual(
            [result["result"]["text"]["text_str"] for result in results],
            ["second chunk"],
        )

    def test_get_texts_keeps_the_chunks_without_the_flag(self):
        results = DBTextSearchController().get_texts(
            texts_ids=[self.chunk.pk, self.other_chunk.pk],
            texts_scores=[0.9, 0.8],
            only_used_to_search=False,
        )
        self.assertEqual(len(results), 2)

    def test_only_the_enabled_documents_are_searchable(self):
        self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
        )

        documents = RelationalDBController.get_documents_to_search_from_collection(
            collection=self.collection,
        )
        self.assertEqual([document.name for document in documents], ["other.txt"])


class ListDocumentsWithDisabledTests(FixtureTestCase):
    """
    ``GET documents`` lists the documents used in the search, with
    ``with_disabled=true`` also the deactivated ones (with their flag).
    """

    def setUp(self):
        super().setUp()
        self.factory = APIRequestFactory()
        self.controller.set_documents_use_in_search(
            collection=self.collection,
            use_in_search=False,
            document_names=["document.txt"],
        )

    def invoke(self, query):
        request = self.factory.get("/", query)
        force_authenticate(request, user=self.user)
        return ListDocumentsFromCollection.as_view()(request)

    def test_only_the_enabled_documents_are_listed_by_default(self):
        response = self.invoke({"collection_name": "collection"})

        self.assertTrue(response.data["status"])
        self.assertEqual(response.data["body"]["documents"], [{"name": "other.txt"}])

    def test_the_disabled_documents_are_listed_with_their_flag(self):
        response = self.invoke(
            {"collection_name": "collection", "with_disabled": "true"}
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(
            [dict(document) for document in response.data["body"]["documents"]],
            [
                {
                    "id": self.document.pk,
                    "name": "document.txt",
                    "use_in_search": False,
                },
                {
                    "id": self.other_document.pk,
                    "name": "other.txt",
                    "use_in_search": True,
                },
            ],
        )

    def test_the_with_disabled_flag_must_be_a_boolean(self):
        response = self.invoke(
            {"collection_name": "collection", "with_disabled": "x"}
        )

        self.assertFalse(response.data["status"])


if __name__ == "__main__":
    import unittest

    unittest.main()
