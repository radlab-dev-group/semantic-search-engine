from django.test import TestCase

from sse_api.data.api import GetDocumentContent
from sse_api.data.controllers.preview import ContentPreviewController
from sse_api.data.models import CollectionOfDocuments, Document
from sse_api.data.tests.test_preview import PreviewFixture
from sse_api.engine.core.search_engine import reformat_search_results_to_display
from sse_api.tests.input_support import EndpointInputMixin


class ChunkOffsetTests(EndpointInputMixin, PreviewFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.make_fixture()
        self.controller = ContentPreviewController()

    def preview(self, **params):
        return self.invoke(
            GetDocumentContent,
            {"collection_name": self.collection.name, **params},
            method="get",
        )

    def test_search_adapter_emits_the_same_range_as_preview_in_two_queries(self):
        from sse_api.engine.controllers.search.relational import (
            DBTextSearchController,
        )

        target = self.make_chunk(self.first, 3, "target 😀", "clear_texts")
        with self.assertNumQueries(2):
            raw = DBTextSearchController().get_texts([target.pk], [1.0])
        hit = reformat_search_results_to_display(raw)[0]
        body = self.preview(text_id=target.pk).data["body"]
        self.assertEqual(hit["char_start"], body["highlight"]["char_start"])
        self.assertEqual(hit["char_end"], body["highlight"]["char_end"])
        self.assertEqual(
            body["pages"][0]["content"][hit["char_start"] : hit["char_end"]],
            target.text_str,
        )

    def test_repeated_overlapping_unicode_chunks_have_distinct_exact_ranges(self):
        from sse_api.data.controllers.chunk_offsets import page_content_with_offsets

        chunks = [
            self.make_chunk(self.first, 1, "żółw 😀 wspólne", "overlap"),
            self.make_chunk(self.first, 2, "wspólne koniec", "overlap"),
            self.make_chunk(self.first, 3, "wspólne koniec", "overlap"),
            self.make_chunk(self.first, 4, "", "overlap"),
        ]
        content, offsets = page_content_with_offsets(chunks)
        self.assertEqual(
            content, "żółw 😀 wspólne\n\nwspólne koniec\n\nwspólne koniec"
        )
        for chunk in chunks[:3]:
            start, end = offsets[chunk.pk]
            self.assertEqual(content[start:end], chunk.text_str)
        self.assertNotEqual(offsets[chunks[1].pk], offsets[chunks[2].pk])
        self.assertEqual(offsets[chunks[3].pk], (None, None))

    def test_batch_offsets_reject_changed_or_normalized_text(self):
        from sse_api.data.controllers.chunk_offsets import chunk_offsets_for_texts

        first = self.make_chunk(self.second, 2, "A  B\nC", "other")
        second = self.make_chunk(self.second, 3, "A  B\nC", "other")
        with self.assertNumQueries(1):
            offsets = chunk_offsets_for_texts([first, second])
        self.assertEqual(offsets[first.pk], (0, len(first.text_str)))
        self.assertEqual(offsets[second.pk], (len(first.text_str) + 2, 14))
        first.text_str = "A B C"
        self.assertEqual(chunk_offsets_for_texts([first])[first.pk], (None, None))

    def test_search_display_preserves_verified_offsets_and_nulls(self):
        text = {
            "text_id": 9,
            "document_id": 8,
            "score": 1.0,
            "document_name": "x",
            "relative_path": "/x",
            "page_number": 2,
            "text_number": 1,
            "language": "pl",
            "text_str": "żółw",
            "char_start": 7,
            "char_end": 11,
        }
        raw = [{"result": {"text": text, "left_context": [], "right_context": []}}]
        formatted = reformat_search_results_to_display(raw)[0]
        self.assertEqual((formatted["char_start"], formatted["char_end"]), (7, 11))
        del text["char_start"]
        del text["char_end"]
        formatted = reformat_search_results_to_display(raw)[0]
        self.assertIsNone(formatted["char_start"])
        self.assertIsNone(formatted["char_end"])

    def test_text_id_selects_exact_page_and_type_despite_newer_type(self):
        target = self.make_chunk(self.second, 2, "cytat 😀", "clear_texts")
        self.make_chunk(self.second, 1, "different normalized text", "newest")
        response = self.preview(text_id=target.pk, from_page=1, max_pages=1)
        self.assertTrue(response.data["status"])
        body = response.data["body"]
        self.assertEqual(body["chunk_type"], "clear_texts")
        self.assertEqual(body["from_page"], 2)
        self.assertEqual(len(body["pages"]), 1)
        highlight = body["highlight"]
        self.assertEqual(highlight["text_id"], target.pk)
        self.assertEqual(highlight["page_number"], 2)
        self.assertEqual(
            body["pages"][0]["content"][
                highlight["char_start"] : highlight["char_end"]
            ],
            target.text_str,
        )

    def test_foreign_collection_text_id_and_missing_text_are_not_found(self):
        foreign = CollectionOfDocuments.objects.create(
            name="foreign", created_by=self.profile
        )
        document = Document.objects.create(
            name="foreign.txt",
            collection=foreign,
            path="/foreign",
            relative_path="/foreign",
            document_hash="foreign",
            added_by=self.profile,
        )
        chunk = self.make_chunk(self.make_page(document, 1), 1, "secret", "raw")
        for text_id in (chunk.pk, chunk.pk + 10000):
            with self.subTest(text_id=text_id):
                self.assert_denial(self.preview(text_id=text_id))

    def test_text_id_cannot_override_explicit_document_or_type(self):
        target = self.make_chunk(self.first, 2, "target", "raw_text")
        other = Document.objects.create(
            name="other.txt",
            collection=self.collection,
            path="/other",
            relative_path="/other",
            document_hash="other",
            added_by=self.profile,
        )
        for params in (
            {"document_id": other.pk},
            {"document_name": "missing.txt"},
            {"document_id": self.document.pk, "document_name": "missing.txt"},
            {"chunk_type": "clear_texts"},
        ):
            with self.subTest(params=params):
                self.assert_denial(self.preview(text_id=target.pk, **params))

    def test_text_id_is_validated_and_cannot_access_another_users_collection(self):
        target = self.make_chunk(self.first, 2, "target", "raw_text")
        for text_id in ("invalid", "0", "-1"):
            with self.subTest(text_id=text_id):
                self.assert_denial(self.preview(text_id=text_id))
        # A collection from another owner must fail before resolving a text id.
        from django.contrib.auth.models import User
        from sse_api.system.models import OrganisationUser

        other = OrganisationUser.objects.create(
            auth_user=User.objects.create_user(username="other-owner"),
            organisation=self.profile.organisation,
        )
        self.collection.created_by = other
        self.collection.save(update_fields=["created_by"])
        self.assert_denial(self.preview(text_id=target.pk))

    def test_controller_rejects_text_id_outside_the_document(self):
        from sse_api.data.models import DocumentPageText

        target = DocumentPageText.objects.filter(page=self.first).first()
        other = Document.objects.create(
            name="other.txt",
            collection=self.collection,
            path="/other",
            relative_path="/other",
            document_hash="other",
            added_by=self.profile,
        )
        with self.assertRaises(ValueError):
            self.controller.document_content(document=other, text_id=target.pk)

    def test_batch_mapping_and_preview_use_identical_coordinates(self):
        from sse_api.data.controllers.chunk_offsets import chunk_offsets_for_texts

        targets = [
            self.make_chunk(self.first, 2, "same 😀", "raw_text"),
            self.make_chunk(self.first, 3, "same 😀", "raw_text"),
            self.make_chunk(self.second, 2, "normalized  text", "clear_texts"),
        ]
        offsets = chunk_offsets_for_texts(targets)
        for target in targets:
            with self.subTest(text_id=target.pk):
                body = self.preview(
                    text_id=target.pk,
                    document_id=self.document.pk,
                    document_name=self.document.name,
                ).data["body"]
                start, end = offsets[target.pk]
                self.assertEqual(body["highlight"]["char_start"], start)
                self.assertEqual(body["highlight"]["char_end"], end)
                self.assertEqual(
                    body["pages"][0]["content"][start:end], target.text_str
                )
                self.assertEqual(
                    self.preview(text_id=target.pk).data["body"]["highlight"],
                    body["highlight"],
                )

    def test_empty_chunk_highlight_is_explicitly_unmappable(self):
        target = self.make_chunk(self.first, 2, "", "raw_text")
        response = self.preview(text_id=target.pk)
        self.assertTrue(response.data["status"])
        self.assertEqual(
            response.data["body"]["highlight"],
            {
                "text_id": target.pk,
                "page_number": 1,
                "char_start": None,
                "char_end": None,
            },
        )
