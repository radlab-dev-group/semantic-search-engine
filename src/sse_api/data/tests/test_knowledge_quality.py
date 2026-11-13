"""Integration coverage for the authenticated quality metadata endpoint."""

from django.test import TestCase

from sse_api.data.api import UpdateKnowledgeQuality
from sse_api.data.models import CollectionOfDocuments, Document
from sse_api.tests.input_support import EndpointInputMixin


class KnowledgeQualityTests(EndpointInputMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.collection = CollectionOfDocuments.objects.create(
            name="knowledge", display_name="Knowledge", created_by=self.profile
        )
        self.document = Document.objects.create(
            collection=self.collection,
            name="recipe.input_text",
            path="recipe.input_text",
            relative_path="recipe.input_text",
            document_hash="abc",
            category="recipe",
            metadata_json={"paths": ["auth.py"]},
        )

    def request(self, **overrides):
        body = dict(
            collection_name="knowledge",
            document_name="recipe.input_text",
            upvotes=3,
            downvotes=1,
        )
        body.update(overrides)
        return self.invoke(UpdateKnowledgeQuality, body)

    def test_updates_quality_and_preserves_other_metadata(self):
        self.assertTrue(self.request().data["status"])
        self.document.refresh_from_db()
        self.assertEqual(self.document.metadata_json["times_used"], 4)
        self.assertEqual(self.document.metadata_json["paths"], ["auth.py"])
        self.assertAlmostEqual(self.document.metadata_json["success_rate"], 4 / 6)

    def test_repeated_or_older_delivery_does_not_roll_back_votes(self):
        self.request()
        self.request(upvotes=1, downvotes=0)
        self.document.refresh_from_db()
        self.assertEqual(self.document.metadata_json["times_used"], 4)

    def test_unknown_collection_or_document_is_denied(self):
        self.assert_denial(self.request(collection_name="other"))
        self.assert_denial(self.request(document_name="other"))

    def test_invalid_counters_are_denied(self):
        for value in (-1, True, "1", 1.5):
            with self.subTest(value=value):
                self.assert_denial(self.request(upvotes=value))
