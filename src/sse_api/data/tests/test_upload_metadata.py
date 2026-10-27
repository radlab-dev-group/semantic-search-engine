import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import force_authenticate

from sse_api.data import indexing_pipeline
from sse_api.data.api import UploadAndIndexFilesToCollection
from sse_api.data.controllers.upload import UploadDocumentsController
from sse_api.data.models import CollectionOfDocuments, Document, IndexingJob
from sse_api.tests.input_support import EndpointInputMixin, INDEXING_OPTIONS


class UploadMetadataTests(EndpointInputMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.controller = UploadDocumentsController(upload_dir=self.temp.name)
        self.collection = CollectionOfDocuments.objects.create(
            name="metadata", display_name="Metadata", created_by=self.profile
        )
        self.name = "doc:20261022:abc.txt"
        self.metadata = {
            self.name: {
                "category": "doc",
                "options": {
                    "source_file": "docs/guide.md",
                    "scope": "project",
                    "doc_type": "markdown",
                    "nested": {"tags": ["żółć"]},
                },
            }
        }

    def upload(self, metadata, asynchronous):
        data = {
            "files[]": SimpleUploadedFile(self.name, b"Document text", "text/plain"),
            "collection_name": self.collection.name,
            "indexing_options": json.dumps(INDEXING_OPTIONS),
        }
        if metadata is not None:
            data["file_metadata"] = metadata
        request = self.factory.post("/", data, format="multipart")
        force_authenticate(request, user=self.user)
        with (
            patch(
                "sse_api.data.api.UploadDocumentsController",
                return_value=self.controller,
            ),
            patch(
                "sse_api.data.indexing_queue.async_indexing",
                return_value=asynchronous,
            ),
            patch("sse_api.data.indexing_queue.dispatch_indexing_job") as dispatch,
        ):
            response = UploadAndIndexFilesToCollection.as_view()(request)
        return response, dispatch

    def test_metadata_survives_inline_and_queued_preparation(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                # Only the external reader is replaced (no chunks to embed);
                # storage, job reload, preparation and relational writes are real.
                def read_directory(main_dir_path, **kwargs):
                    paths = list(Path(main_dir_path).iterdir())
                    self.assertEqual([path.name for path in paths], [self.name])
                    self.assertEqual(paths[0].read_bytes(), b"Document text")
                    return SimpleNamespace(
                        load=lambda **options: None,
                        documents=[
                            SimpleNamespace(as_dict={
                                "filepath": str(paths[0]),
                                "relative_filepath": self.name,
                                "category": "reader category",
                                "options": {"reader_option": True},
                                "pages": [],
                            })
                        ],
                    )

                with patch(
                    "sse_api.engine.controllers.database.relational_db.DirectoryFileReader",
                    side_effect=read_directory,
                ):
                    response, dispatch = self.upload(
                        json.dumps(self.metadata), asynchronous
                    )
                    self.assertTrue(response.data["status"], response.data)
                    job = IndexingJob.objects.latest("pk")
                    self.assertEqual(job.request_payload["file_metadata"], self.metadata)
                    if asynchronous:
                        dispatch.assert_called_once()
                        indexing_pipeline.run_prepare(
                            IndexingJob.objects.get(pk=job.pk), 1
                        )
                    else:
                        dispatch.assert_not_called()
                    document = Document.objects.get(indexing_job=job)
                    self.assertEqual(document.category, "doc")
                    for key, value in self.metadata[self.name]["options"].items():
                        self.assertEqual(document.metadata_json[key], value)
                    self.assertTrue(document.metadata_json["reader_option"])

    def test_invalid_metadata_rejected_before_storage_or_dispatch(self):
        for metadata in (
            "{", "null", "[]", "{}",
            json.dumps({"missing.txt": {"category": "doc", "options": {}}}),
            json.dumps({self.name: {"category": [], "options": {}}}),
        ):
            with self.subTest(metadata=metadata):
                response, dispatch = self.upload(metadata, True)
                self.assertFalse(response.data["status"])
                dispatch.assert_not_called()
                self.assertFalse(IndexingJob.objects.exists())
                self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_same_basename_in_different_subdirectories_is_not_deduplicated(self):
        # A zip may hold two files of the same name in different sub
        # directories; the name is not the identity of a document, so both
        # of them have to reach the database.
        paths = [
            Path(self.temp.name) / "nested" / "report.txt",
            Path(self.temp.name) / "other" / "report.txt",
        ]
        reader = SimpleNamespace(
            load=lambda **options: None,
            documents=[
                SimpleNamespace(
                    as_dict={
                        "filepath": str(path),
                        "relative_filepath": "nested/report.txt"
                        if path.parent.name == "nested"
                        else "other/report.txt",
                        "category": "",
                        "options": {},
                        "pages": [],
                    }
                )
                for path in paths
            ],
        )
        with patch(
            "sse_api.engine.controllers.database.relational_db.DirectoryFileReader",
            return_value=reader,
        ):
            response, dispatch = self.upload(None, False)
        self.assertTrue(response.data["status"], response.data)
        job = IndexingJob.objects.latest("pk")
        documents = sorted(
            Document.objects.filter(indexing_job=job),
            key=lambda document: document.relative_path,
        )
        self.assertEqual(
            [document.relative_path for document in documents],
            ["nested/report.txt", "other/report.txt"],
        )
        self.assertEqual(
            [document.path for document in documents], [str(path) for path in paths]
        )

    def test_reader_omission_rejects_inline_job_instead_of_losing_metadata(self):
        reader = SimpleNamespace(load=lambda **options: None, documents=[])
        with patch(
            "sse_api.engine.controllers.database.relational_db.DirectoryFileReader",
            return_value=reader,
        ):
            response, dispatch = self.upload(json.dumps(self.metadata), False)
        self.assertFalse(response.data["status"])
        self.assertEqual(IndexingJob.objects.get().status, "failed")
        self.assertFalse(Document.objects.exists())
        dispatch.assert_not_called()