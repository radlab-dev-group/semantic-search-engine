"""
Tests for the token chunking through the model service.

The pure helpers are tested with a fake chunker; the wiring of the indexing
paths is tested with the external reader replaced, as in the other indexing
tests, so that no document is really read.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import force_authenticate

from sse_api.data.api import UploadAndIndexFilesToCollection
from sse_api.data.controllers import token_chunking
from sse_api.data.controllers.upload import UploadDocumentsController
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPageText,
)
from sse_api.tests.input_support import EndpointInputMixin, INDEXING_OPTIONS


class ChunkHelpersTests(TestCase):
    def test_chunk_type_suffix_mirrors_the_reader(self):
        self.assertEqual(
            token_chunking.chunk_type_suffix(400, 60),
            "_chunk_max_tokens_400_overlap_60",
        )
        self.assertEqual(
            token_chunking.chunk_type_suffix(400), "_chunk_max_tokens_400"
        )
        self.assertEqual(token_chunking.chunk_type_suffix(1), "")
        self.assertEqual(token_chunking.chunk_type_suffix(None), "")
        self.assertEqual(token_chunking.chunk_type_suffix(None, 60), "")

    def test_chunk_pages_numbers_the_chunks_inside_their_page(self):
        pages = [
            {
                "page_content": "a b c d",
                "metadata": {"text_chunk_type": "clear_texts", "page": 2},
                "page_number": 2,
            },
            {
                "page_content": "short",
                "metadata": {"text_chunk_type": "clear_texts", "page": 3},
                "page_number": 3,
            },
        ]
        chunker = lambda texts, max_tokens, overlap: [["a b", "c d"], ["short"]]
        chunked = token_chunking.chunk_pages(pages, chunker, 2, 1)

        self.assertEqual(
            [page["page_content"] for page in chunked], ["a b", "c d", "short"]
        )
        # the chunks of the first page are numbered from 1, and the second
        # one is chunk 1 of its page: every stored text carries a chunk number
        self.assertEqual([page["metadata"]["chunk"] for page in chunked], [1, 2, 1])
        # every page of the document shares the chunk type of its document
        self.assertEqual(
            [page["metadata"]["text_chunk_type"] for page in chunked],
            ["clear_texts_chunk_max_tokens_2_overlap_1"] * 3,
        )
        # the page identity is carried over
        self.assertEqual(chunked[0]["metadata"]["page"], 2)
        self.assertEqual(chunked[0]["page_number"], 2)

    def test_chunk_pages_copies_the_metadata_of_the_pages(self):
        pages = [
            {
                "page_content": "a b",
                "metadata": {"text_chunk_type": "raw_text", "page": 1},
                "page_number": 1,
            }
        ]
        before = json.loads(json.dumps(pages[0]["metadata"]))
        token_chunking.chunk_pages(pages, lambda texts, mt, ov: [["a b"]], 2)
        self.assertEqual(pages[0]["metadata"], before)

    def test_chunk_pages_rejects_a_miscounted_chunker(self):
        pages = [
            {
                "page_content": "a b",
                "metadata": {"text_chunk_type": "raw_text"},
                "page_number": 1,
            }
        ]
        with self.assertRaises(ValueError):
            token_chunking.chunk_pages(pages, lambda texts, mt, ov: [], 2)

    def test_chunk_document_dict_keeps_the_identity_of_the_document(self):
        doc = {
            "filepath": "f.txt",
            "relative_filepath": "f.txt",
            "category": "doc",
            "options": {"clear_texts": True},
            "pages": [
                {
                    "page_content": "a b c d",
                    "metadata": {"text_chunk_type": "clear_texts", "page": 1},
                    "page_number": 1,
                }
            ],
        }
        chunker = lambda texts, max_tokens, overlap: [["a b", "c d"]]
        chunked = token_chunking.chunk_document_dict(doc, chunker, 2)
        self.assertEqual(chunked["filepath"], "f.txt")
        self.assertEqual(chunked["category"], "doc")
        self.assertEqual(len(chunked["pages"]), 2)
        self.assertEqual(doc["pages"][0]["page_content"], "a b c d")


class ChunkerFactoryTests(TestCase):
    def setUp(self):
        super().setUp()
        # the service is not configured in the tests
        for name in (
            "SSE_MODEL_SERVICE_ENABLED",
            "SSE_MODEL_SERVICE_URL",
            "SSE_MODEL_SERVICE_TOKEN",
        ):
            self._saved = os.environ.get(name)
            os.environ.pop(name, None)

    def tearDown(self):
        for name in (
            "SSE_MODEL_SERVICE_ENABLED",
            "SSE_MODEL_SERVICE_URL",
            "SSE_MODEL_SERVICE_TOKEN",
        ):
            if self._saved is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self._saved
        super().tearDown()

    def test_no_chunking_without_a_chunk_size(self):
        self.assertIsNone(token_chunking.token_chunker_for("embedder", 0))
        self.assertIsNone(token_chunking.token_chunker_for("embedder", None))

    def test_no_chunker_without_a_configured_service(self):
        self.assertIsNone(token_chunking.token_chunker_for("embedder", 400))

    def test_chunker_batches_the_pages_it_is_given(self):
        os.environ["SSE_MODEL_SERVICE_ENABLED"] = "1"
        os.environ["SSE_MODEL_SERVICE_URL"] = "http://models:8280"
        calls = []

        def fake_chunk(texts, max_tokens, overlap=0):
            calls.append((list(texts), max_tokens, overlap))
            return [["c"] for _ in texts]

        with (
            patch(
                "sse_api.engine.core.models.backend.resolve_model_name",
                return_value="catalog/embedder",
            ),
            patch(
                "sse_api.engine.core.models.backend.build_chunk_client",
                return_value=fake_chunk,
            ),
        ):
            chunker = token_chunking.token_chunker_for("embedder", 400, batch_size=2)
            self.assertIsNotNone(chunker)
            self.assertEqual(
                chunker(["p1", "p2", "p3"], 400, 60), [["c"], ["c"], ["c"]]
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], (["p1", "p2"], 400, 60))
        self.assertEqual(calls[1], (["p3"], 400, 60))


class UploadChunkingWiringTests(EndpointInputMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.controller = UploadDocumentsController(upload_dir=self.temp.name)
        self.collection = CollectionOfDocuments.objects.create(
            name="chunked", display_name="Chunked", created_by=self.profile
        )

    def upload_with_reader(self, pages, chunker):
        load_options = {}

        def read_directory(main_dir_path, **kwargs):
            self.assertEqual(
                [path.name for path in Path(main_dir_path).iterdir()], ["doc.txt"]
            )
            reader = SimpleNamespace(
                load=lambda **options: load_options.update(options),
                documents=[
                    SimpleNamespace(
                        as_dict={
                            "filepath": "doc.txt",
                            "relative_filepath": "doc.txt",
                            "category": "",
                            "options": {},
                            "pages": [dict(page) for page in pages],
                        }
                    )
                ],
            )
            return reader

        data = {
            "files[]": SimpleUploadedFile("doc.txt", b"text", "text/plain"),
            "collection_name": self.collection.name,
            "indexing_options": json.dumps(INDEXING_OPTIONS),
        }
        request = self.factory.post("/", data, format="multipart")
        force_authenticate(request, user=self.user)
        with (
            patch(
                "sse_api.data.api.UploadDocumentsController",
                return_value=self.controller,
            ),
            patch("sse_api.data.indexing_queue.async_indexing", return_value=False),
            patch(
                "sse_api.engine.controllers.database.relational_db.directory_file_reader",
                side_effect=read_directory,
            ),
            patch(
                "sse_api.engine.controllers.database.relational_db.token_chunker_for",
                return_value=chunker,
            ),
            # the relational write is the stage under test; the embeddings of
            # the stored pages would need a vector database
            patch(
                "sse_api.data.indexing_pipeline.run_index_document",
                return_value=None,
            ),
        ):
            response = UploadAndIndexFilesToCollection.as_view()(request)
        self.assertTrue(response.data["status"], response.data)
        return load_options

    @staticmethod
    def _reader_pages():
        # the reader numbers every page it hands over (a whole page is chunk 1)
        return [
            {
                "page_content": "some words in a page",
                "metadata": {
                    "text_chunk_type": "clear_texts",
                    "page": 1,
                    "chunk": 1,
                },
                "page_number": 1,
            }
        ]

    def test_reader_skips_the_tokens_when_the_service_chunks(self):
        def chunker(texts, max_tokens, overlap=0):
            return [[text[:10]] for text in texts]

        load_options = self.upload_with_reader(
            self._reader_pages(),
            chunker,
        )
        # the reader got no token budget: the service owns the cutting now
        self.assertEqual(load_options["max_tokens_in_chunk"], None)
        self.assertEqual(load_options["number_of_overlap_tokens"], None)
        document = Document.objects.get(collection=self.collection)
        self.assertEqual(
            DocumentPageText.objects.filter(page__document=document)
            .first()
            .text_chunk_type,
            f"clear_texts{_suffix()}",
        )

    def test_reader_chunks_itself_without_a_service(self):
        load_options = self.upload_with_reader(self._reader_pages(), None)
        self.assertEqual(
            load_options["max_tokens_in_chunk"],
            INDEXING_OPTIONS["max_tokens_in_chunk"],
        )
        document = Document.objects.get(collection=self.collection)
        self.assertEqual(
            DocumentPageText.objects.filter(page__document=document)
            .first()
            .text_chunk_type,
            "clear_texts",
        )


def _suffix():
    return token_chunking.chunk_type_suffix(
        INDEXING_OPTIONS["max_tokens_in_chunk"],
        INDEXING_OPTIONS["number_of_overlap_tokens"],
    )
