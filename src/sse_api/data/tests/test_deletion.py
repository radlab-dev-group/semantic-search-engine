"""
Deletion of the content of a collection: in the relational database and in
the vector database.

Tests use a real (in memory) relational database and a faked Milvus
controller, so both sides of the deletion are observable: which rows
disappeared, and what was asked to be removed from the vector database.
"""

import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.chat.models import Chat
from sse_api.chat.core.constants import ERROR_MARK_CHAT
from sse_api.data.api import (
    DeleteCollection,
    DeleteDocumentsFromCollection,
    DeleteTextsFromCollection,
)
from sse_api.data.controllers.deletion import (
    ContentDeletionController,
    ContentProtected,
    DeletionResult,
)
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
)
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.database.semantic_db import SemanticDBController
from sse_api.engine.models import UserQuery, UserQueryResponse
from sse_api.system.models import Organisation, OrganisationUser


class FixtureTestCase(TestCase):
    """
    Collection with one document, two pages and three text chunks.
    """

    def setUp(self):
        self.organisation = Organisation.objects.create(name="input organisation")
        self.user = User.objects.create_user(username="input-user")
        self.organisation_user = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = CollectionOfDocuments.objects.create(
            name="collection",
            display_name="Collection",
            created_by=self.organisation_user,
        )
        self.document = self.add_document("document.txt", "hash-1")
        self.page = DocumentPage.objects.create(
            document=self.document, page_number=1
        )
        self.other_page = DocumentPage.objects.create(
            document=self.document, page_number=2
        )
        self.chunk = self.add_chunk(self.page, 1, "first chunk")
        self.other_chunk = self.add_chunk(self.page, 2, "second chunk")
        self.third_chunk = self.add_chunk(self.other_page, 1, "third chunk")

    def add_document(self, name, document_hash, collection=None):
        return Document.objects.create(
            name=name,
            collection=collection or self.collection,
            path=f"/tmp/{name}",
            relative_path=name,
            document_hash=document_hash,
            added_by=self.organisation_user,
        )

    def add_chunk(self, page, text_number, text, paged_text=None):
        return DocumentPageText.objects.create(
            page=page,
            text_number=text_number,
            text_str=text,
            text_hash=f"hash-{text}",
            text_chunk_type="plain",
            language="en",
            paged_text=paged_text,
        )


class ContentDeletionTests(FixtureTestCase):
    def setUp(self):
        super().setUp()
        semantic_db = patch("sse_api.data.controllers.deletion.SemanticDBController")
        self.semantic_db = semantic_db.start().return_value
        self.addCleanup(semantic_db.stop)
        self.semantic_db.delete_texts.return_value = 3
        self.semantic_db.delete_documents.return_value = 3
        self.semantic_db.drop_collection.return_value = True
        self.controller = ContentDeletionController()

    def test_documents_are_removed_with_pages_and_texts(self):
        result = self.controller.delete_documents(
            collection=self.collection, document_names=["document.txt"]
        )

        self.assertEqual(result.deleted, {"texts": 3, "pages": 2, "documents": 1})
        self.assertEqual(result.deleted_vectors, 3)
        self.assertEqual(
            self.semantic_db.delete_documents.call_args.kwargs["external_ids"],
            [self.document.pk],
        )
        self.assertFalse(Document.objects.exists())
        self.assertFalse(DocumentPage.objects.exists())
        self.assertFalse(DocumentPageText.objects.exists())

    def test_documents_given_by_ids_are_removed(self):
        result = self.controller.delete_documents(
            collection=self.collection, document_ids=[self.document.pk]
        )

        self.assertEqual(result.deleted["documents"], 1)
        self.assertEqual(result.not_found, [])
        self.assertFalse(Document.objects.exists())

    def test_names_of_the_documents_which_do_not_exist_are_reported(self):
        result = self.controller.delete_documents(
            collection=self.collection,
            document_names=["document.txt", "missing.txt"],
        )

        self.assertEqual(result.not_found, ["missing.txt"])
        self.assertEqual(result.deleted["documents"], 1)

    def test_only_given_chunks_are_removed_with_pages_left_without_texts(self):
        result = self.controller.delete_texts(
            collection=self.collection,
            chunk_ids=[self.chunk.pk, self.other_chunk.pk],
        )

        self.assertEqual(result.deleted, {"texts": 2, "pages": 1, "documents": 0})
        self.assertEqual(
            list(DocumentPageText.objects.values_list("pk", flat=True)),
            [self.third_chunk.pk],
        )
        self.assertEqual(
            list(DocumentPage.objects.values_list("pk", flat=True)),
            [self.other_page.pk],
        )
        self.assertEqual(
            sorted(self.semantic_db.delete_texts.call_args.kwargs["external_ids"]),
            sorted([self.chunk.pk, self.other_chunk.pk]),
        )

    def test_chunks_of_other_collections_are_not_removed(self):
        result = self.controller.delete_texts(
            collection=self.collection, chunk_ids=["999999"]
        )

        self.assertEqual(result.deleted, {"texts": 0, "pages": 0, "documents": 0})
        self.assertEqual(result.not_found, ["999999"])
        self.semantic_db.delete_texts.assert_not_called()
        self.assertEqual(DocumentPageText.objects.count(), 3)
        self.semantic_db.delete_texts.assert_not_called()

    def test_chunks_referencing_given_ones_are_found(self):
        referring = self.add_chunk(
            self.other_page, 2, "merged chunk", paged_text=self.chunk
        )

        self.assertEqual(
            RelationalDBController.chunks_to_delete(chunk_ids=[self.chunk.pk]),
            sorted([self.chunk.pk, referring.pk]),
        )

    def test_failed_collection_deletion_leaves_the_relational_db_untouched(self):
        Chat.objects.create(
            organisation_user=self.organisation_user, collection=self.collection
        )

        with (
            patch(
                "sse_api.engine.controllers.database.relational_db"
                "._delete_collection_queries",
                side_effect=RuntimeError("the database is gone"),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.controller.delete_collection(collection=self.collection)

        self.assertTrue(
            CollectionOfDocuments.objects.filter(pk=self.collection.pk).exists()
        )
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(DocumentPageText.objects.count(), 3)
        self.assertTrue(Chat.objects.exists())
        self.semantic_db.drop_collection.assert_not_called()

    def test_content_referenced_through_a_protected_key_is_not_removed(self):
        self.add_chunk(self.other_page, 2, "merged chunk", paged_text=self.chunk)

        with self.assertRaises(ContentProtected):
            self.controller.delete_texts(
                collection=self.collection, chunk_ids=[self.chunk.pk]
            )

        self.assertEqual(DocumentPageText.objects.count(), 4)
        self.semantic_db.delete_texts.assert_not_called()

    def test_collection_is_removed_with_documents_chats_and_search_history(self):
        Chat.objects.create(
            organisation_user=self.organisation_user, collection=self.collection
        )
        user_query = UserQuery.objects.create(
            organisation_user=self.organisation_user,
            collection=self.collection,
            query_str_prompt="question",
            query_options={},
        )
        UserQueryResponse.objects.create(
            user_query=user_query,
            general_stats_json={},
            detailed_results_json={},
        )
        other_collection = CollectionOfDocuments.objects.create(
            name="other", display_name="Other", created_by=self.organisation_user
        )
        other_document = self.add_document(
            "other.txt", "hash-2", collection=other_collection
        )
        other_page = DocumentPage.objects.create(
            document=other_document, page_number=1
        )
        self.add_chunk(other_page, 1, "chunk of another collection")

        result = self.controller.delete_collection(collection=self.collection)

        self.assertEqual(
            result.deleted,
            {
                "texts": 3,
                "pages": 2,
                "documents": 1,
                "chats": 1,
                "queries": 1,
                "collection": 1,
            },
        )
        self.assertTrue(result.dropped_collection)
        self.assertEqual(
            self.semantic_db.drop_collection.call_args.kwargs["collection_name"],
            "collection",
        )
        self.assertFalse(
            CollectionOfDocuments.objects.filter(pk=self.collection.pk).exists()
        )
        self.assertFalse(Chat.objects.exists())
        self.assertFalse(UserQuery.objects.exists())
        self.assertFalse(UserQueryResponse.objects.exists())
        # the collection which was not given stays untouched
        self.assertTrue(
            CollectionOfDocuments.objects.filter(pk=other_collection.pk).exists()
        )
        self.assertEqual(DocumentPageText.objects.count(), 1)


class DeletionEndpointTests(FixtureTestCase):
    """
    Endpoints of the deletion. Removing itself is faked here, the tests check
    who is allowed to delete what and how the answers look like.
    """

    def setUp(self):
        super().setUp()
        other_user = User.objects.create_user(username="other-user")
        self.foreign_collection = CollectionOfDocuments.objects.create(
            name="foreign",
            display_name="Foreign",
            created_by=OrganisationUser.objects.create(
                auth_user=other_user, organisation=self.organisation
            ),
        )
        result = DeletionResult(
            deleted={"texts": 3, "pages": 2, "documents": 1},
            deleted_vectors=3,
            dropped_collection=True,
        )
        controller = patch("sse_api.data.api.ContentDeletionController")
        self.deletion = controller.start().return_value
        self.addCleanup(controller.stop)
        self.deletion.delete_texts.return_value = result
        self.deletion.delete_documents.return_value = result
        self.deletion.delete_collection.return_value = result
        self.factory = APIRequestFactory()

    def invoke(self, view, payload, method="post"):
        request = self.factory.generic(
            method.upper(), "/", json.dumps(payload), content_type="application/json"
        )
        force_authenticate(request, user=self.user)
        return view.as_view()(request)

    def test_collection_is_deleted_by_its_owner(self):
        response = self.invoke(DeleteCollection, {"collection_name": "collection"})

        self.assertTrue(response.data["status"])
        self.assertEqual(
            self.deletion.delete_collection.call_args.kwargs["collection"],
            self.collection,
        )
        self.assertEqual(response.data["body"]["deleted"]["documents"], 1)
        self.assertTrue(response.data["body"]["dropped_collection"])

    def test_collection_may_be_removed_with_the_delete_method(self):
        response = self.invoke(
            DeleteCollection, {"collection_name": "collection"}, method="delete"
        )

        self.assertTrue(response.data["status"])
        self.deletion.delete_collection.assert_called_once()

    def test_collection_of_another_user_cannot_be_deleted(self):
        response = self.invoke(DeleteCollection, {"collection_name": "foreign"})

        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0]["error_code"],
            f"000002_{ERROR_MARK_CHAT}",
        )
        self.deletion.delete_collection.assert_not_called()

    def test_documents_are_deleted_by_names(self):
        response = self.invoke(
            DeleteDocumentsFromCollection,
            {"collection_name": "collection", "documents[]": ["document.txt"]},
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(
            self.deletion.delete_documents.call_args.kwargs["document_names"],
            ["document.txt"],
        )

    def test_documents_are_deleted_by_ids(self):
        response = self.invoke(
            DeleteDocumentsFromCollection,
            {"collection_name": "collection", "document_ids[]": [self.document.pk]},
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(
            self.deletion.delete_documents.call_args.kwargs["document_ids"],
            [self.document.pk],
        )

    def test_documents_of_the_collection_are_required(self):
        response = self.invoke(
            DeleteDocumentsFromCollection, {"collection_name": "collection"}
        )

        self.assertFalse(response.data["status"])
        self.deletion.delete_documents.assert_not_called()

    def test_documents_which_were_not_found_are_reported(self):
        self.deletion.delete_documents.return_value = DeletionResult(
            deleted={"texts": 0, "pages": 0, "documents": 0},
            not_found=["unknown.txt"],
        )

        response = self.invoke(
            DeleteDocumentsFromCollection,
            {"collection_name": "collection", "documents[]": ["unknown.txt"]},
        )

        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000002_DATA")

    def test_chunks_are_deleted_by_ids(self):
        self.deletion.delete_texts.return_value = DeletionResult(
            deleted={"texts": 1, "pages": 0, "documents": 0},
            deleted_vectors=1,
        )

        response = self.invoke(
            DeleteTextsFromCollection,
            {"collection_name": "collection", "chunk_ids[]": [self.chunk.pk]},
        )

        self.assertTrue(response.data["status"])
        self.assertEqual(
            self.deletion.delete_texts.call_args.kwargs["chunk_ids"],
            [self.chunk.pk],
        )
        self.assertEqual(response.data["body"]["deleted"]["texts"], 1)

    def test_chunks_which_were_not_found_are_reported(self):
        self.deletion.delete_texts.return_value = DeletionResult(
            deleted={"texts": 0, "pages": 0, "documents": 0},
            not_found=[str(self.chunk.pk)],
        )

        response = self.invoke(
            DeleteTextsFromCollection,
            {"collection_name": "collection", "chunk_ids[]": [self.chunk.pk]},
        )

        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000004_DATA")

    def test_chunks_of_other_users_are_not_deleted(self):
        response = self.invoke(
            DeleteTextsFromCollection,
            {"collection_name": "foreign", "chunk_ids[]": [self.chunk.pk]},
        )

        self.assertFalse(response.data["status"])
        self.deletion.delete_texts.assert_not_called()

    def test_empty_and_not_numeric_chunk_ids_are_rejected(self):
        for chunk_ids in ([], ["abc"], [0], [-1], None, "1"):
            with self.subTest(chunk_ids=chunk_ids):
                response = self.invoke(
                    DeleteTextsFromCollection,
                    {"collection_name": "collection", "chunk_ids[]": chunk_ids},
                )
                self.assertFalse(response.data["status"])
        self.deletion.delete_texts.assert_not_called()

    def test_protected_content_is_reported_to_the_client(self):
        self.deletion.delete_collection.side_effect = ContentProtected("protected")

        response = self.invoke(DeleteCollection, {"collection_name": "collection"})

        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000003_DATA")


class FakeMilvusHandler:
    """
    Records the ids which the controller asks the vector store to remove and
    reports them as deleted.
    """

    def delete_texts_by_external_ids(self, external_ids):
        return len(list(external_ids))

    def delete_documents_by_external_ids(self, external_ids):
        return len(list(external_ids))

    def close_connection(self):
        return None


class SemanticControllerSignatureTests(FixtureTestCase):
    """
    The vector-side controller is called with keyword arguments from
    ``ContentDeletionController``.  A mismatch of the parameter names used to
    raise a ``TypeError`` (500) only in production, because the unit tests
    replace the whole controller with a mock.  These tests call the real
    ``SemanticDBController`` with a stubbed Milvus handler, so a drift of the
    signature is caught here.
    """

    def _run(self, method, external_ids):
        with patch.object(
            SemanticDBController,
            "_SemanticDBController__prepare_milvus_handler",
            return_value=FakeMilvusHandler(),
        ):
            controller = SemanticDBController(milvus_config_path="")
            return getattr(controller, method)(
                collection_name=self.collection.name, external_ids=external_ids
            )

    def test_delete_documents_accepts_external_ids(self):
        self.assertEqual(self._run("delete_documents", [self.document.pk]), 1)

    def test_delete_texts_accepts_external_ids(self):
        self.assertEqual(self._run("delete_texts", [self.chunk.pk]), 1)
