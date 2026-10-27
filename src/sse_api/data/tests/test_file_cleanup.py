"""
The files of the uploads on the disk.

A deletion of a document or of a collection removes the files the documents
were indexed from, and an indexing call may ask for the same once its
embeddings are stored (``delete_files_after_indexing``).  Only the files
inside the root of the uploads may disappear, and a directory only when its
last file is gone -- which is what the tests here look at, with a temporary
root and a real (in memory) relational database.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.test import TestCase

from sse_api.core.utils import compute_text_hash
from sse_api.core.validation import indexing_options as indexing_options_validator
from sse_api.data import indexing_pipeline
from sse_api.data.controllers import file_cleanup
from sse_api.data.controllers.deletion import ContentDeletionController
from sse_api.data.core import indexing_progress
from sse_api.data.core.constants import (
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_SKIPPED,
    INDEXING_JOB_FILES,
)
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
    UploadedDocuments,
)
from sse_api.engine.controllers.database.semantic_db import SemanticDBController
from sse_api.system.models import Organisation, OrganisationUser
from sse_api.tests.input_support import INDEXING_OPTIONS

DELETE_FILES_OPTIONS = dict(INDEXING_OPTIONS, delete_files_after_indexing=True)


class UploadsTestCase(TestCase):
    """Organisation, collections and a root of the uploads in a temporary dir."""

    def setUp(self):
        self.uploads = tempfile.TemporaryDirectory()
        self.addCleanup(self.uploads.cleanup)
        root = Path(self.uploads.name).resolve()
        patcher = patch.object(file_cleanup, "DEFAULT_UPLOAD_ROOT", str(root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = root

        self.organisation = Organisation.objects.create(name="files organisation")
        self.user = User.objects.create_user(username="files-user")
        self.profile = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = self.make_collection("collection")
        self.other_collection = self.make_collection("other-collection")

    def make_collection(self, name):
        return CollectionOfDocuments.objects.create(
            name=name,
            display_name=name.title(),
            created_by=self.profile,
            model_embedder="embedder",
        )

    def upload_directory(self, collection, name="20261026_120000_1"):
        directory = (
            self.root / "organisation" / "files-user" / collection.name / name
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def make_document(
        self,
        name,
        collection=None,
        directory=None,
        with_chunk=True,
        content="A text of the file.",
    ):
        """Document of a real file below the root of the uploads."""
        directory = directory or self.upload_directory(collection or self.collection)
        path = directory / name
        path.write_text(content)
        document = Document.objects.create(
            name=name,
            collection=collection or self.collection,
            path=str(path),
            relative_path=f"/{name}",
            document_hash=compute_text_hash(name),
            added_by=self.profile,
        )
        if with_chunk:
            page = DocumentPage.objects.create(document=document, page_number=1)
            DocumentPageText.objects.create(
                page=page,
                text_number=1,
                text_str="A chunk of text.",
                text_hash=compute_text_hash(name + "-chunk"),
                text_chunk_type="plain",
                language="en",
            )
        return document


class FileRemovalTests(UploadsTestCase):
    def test_files_and_the_directories_they_leave_behind_disappear(self):
        document = self.make_document("notes.txt")

        result = file_cleanup.delete_document_files([document])

        self.assertEqual(result.deleted, 1)
        self.assertEqual(result.errors, [])
        self.assertFalse(Path(document.path).exists())
        # the directories of the upload, of the collection and of the user
        # are empty now, so they are gone; only the root is left
        self.assertFalse(self.root.joinpath("organisation").exists())
        self.assertTrue(self.root.exists())

    def test_a_directory_which_still_holds_a_file_stays(self):
        directory = self.upload_directory(self.collection)
        document = self.make_document("notes.txt", directory=directory)
        neighbour = self.make_document("other.txt", directory=directory)

        file_cleanup.delete_document_files([document])

        self.assertFalse(Path(document.path).exists())
        self.assertTrue(Path(neighbour.path).exists())
        self.assertTrue(directory.exists())

    def test_files_outside_of_the_uploads_are_left_alone(self):
        outside = self.root.parent / "somewhere-else.txt"
        outside.write_text("not an upload")
        self.addCleanup(outside.unlink, missing_ok=True)
        document = self.make_document("elsewhere.txt")
        document.path = str(outside)

        result = file_cleanup.delete_document_files([document])

        self.assertEqual(result.deleted, 0)
        self.assertTrue(outside.exists())

    def test_a_file_which_cannot_be_removed_is_reported(self):
        document = self.make_document("notes.txt")
        patcher = patch.object(
            file_cleanup.Path, "unlink", side_effect=OSError("read-only")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        result = file_cleanup.delete_document_files([document])

        self.assertEqual(result.deleted, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("read-only", result.errors[0])
        self.assertTrue(Path(document.path).exists())

    def test_only_the_files_of_the_collection_disappear(self):
        directory = self.upload_directory(self.collection)
        document = self.make_document("notes.txt", directory=directory)
        other = self.make_document("notes.txt", collection=self.other_collection)

        result = file_cleanup.delete_collection_files(self.collection)

        self.assertEqual(result.deleted, 1)
        self.assertFalse(Path(document.path).exists())
        self.assertTrue(Path(other.path).exists())


class DeletionWithFilesTests(UploadsTestCase):
    """The controllers of the deletion remove the files as their last step."""

    def setUp(self):
        super().setUp()
        semantic_db = patch.object(
            ContentDeletionController,
            "_ContentDeletionController__semantic_db_controller",
        )
        controller = semantic_db.start().return_value
        self.addCleanup(semantic_db.stop)
        controller.delete_documents.return_value = 1
        controller.drop_collection.return_value = True
        self.controller = ContentDeletionController()
        self.semantic = controller

    def test_documents_lose_their_files_with_their_rows(self):
        document = self.make_document("notes.txt")

        result = self.controller.delete_documents(
            collection=self.collection, document_names=["notes.txt"]
        )

        self.assertEqual(result.deleted["documents"], 1)
        self.assertEqual(result.deleted_files, 1)
        self.assertFalse(Path(document.path).exists())
        self.assertEqual(result.as_dict()["deleted_files"], 1)

    def test_the_collection_takes_its_files_and_its_directories(self):
        first = self.upload_directory(self.collection, "upload-one")
        second = self.upload_directory(self.collection, "upload-two")
        document = self.make_document("notes.txt", directory=first)
        other = self.make_document("more.txt", directory=second)
        kept = self.make_document("notes.txt", collection=self.other_collection)
        UploadedDocuments.objects.create(
            dir_hash="upload-hash",
            dir_path=str(first),
            uploaded_by=self.profile,
        )

        result = self.controller.delete_collection(collection=self.collection)

        self.assertEqual(result.deleted["collection"], 1)
        self.assertEqual(result.deleted_files, 2)
        self.assertFalse(Path(document.path).exists())
        self.assertFalse(Path(other.path).exists())
        self.assertFalse(first.exists())
        self.assertFalse(second.parent.exists())  # the collection dir is empty
        self.assertTrue(Path(kept.path).exists())
        self.assertTrue(self.root.joinpath("organisation").exists())


class IndexingKeepsNoFilesTests(UploadsTestCase):
    """The option of an indexing call which asks for its files back."""

    def make_job(self, options=None):
        return indexing_progress.create_job(
            collection=self.collection,
            organisation_user=self.profile,
            kind=INDEXING_JOB_FILES,
            indexing_options=dict(INDEXING_OPTIONS if options is None else options),
        )

    def without_an_embedder(self, error=None):
        controller = Mock()
        if error is not None:
            controller.index_texts_from_list.side_effect = error
        patcher = patch.object(
            indexing_pipeline, "semantic_controller_for", return_value=controller
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return controller

    def index(self, document, options=None, error=None):
        job = self.make_job(options)
        indexing_progress.mark_job_indexing(job, 1)
        self.without_an_embedder(error)
        status = indexing_pipeline.run_index_document(job, document)
        document.refresh_from_db()
        return status, document

    def test_the_file_of_an_indexed_document_is_removed(self):
        document = self.make_document("notes.txt")

        status, document = self.index(document, DELETE_FILES_OPTIONS)

        self.assertEqual(status, DOCUMENT_STATUS_INDEXED)
        self.assertEqual(document.indexing_status, DOCUMENT_STATUS_INDEXED)
        self.assertFalse(Path(document.path).exists())

    def test_the_file_stays_when_the_option_was_not_given(self):
        document = self.make_document("notes.txt")

        status, document = self.index(document)

        self.assertEqual(status, DOCUMENT_STATUS_INDEXED)
        self.assertTrue(Path(document.path).exists())

    def test_a_document_without_text_loses_its_file_too(self):
        document = self.make_document("empty.txt", with_chunk=False)

        status, _ = self.index(document, DELETE_FILES_OPTIONS)

        self.assertEqual(status, DOCUMENT_STATUS_SKIPPED)
        self.assertFalse(Path(document.path).exists())

    def test_a_failed_document_keeps_the_file_a_retry_needs(self):
        document = self.make_document("broken.txt")

        self.assertRaises(
            RuntimeError,
            lambda: self.index(
                document,
                DELETE_FILES_OPTIONS,
                error=RuntimeError("embedder is gone"),
            ),
        )

        self.assertTrue(Path(document.path).exists())

    def test_the_option_is_remembered_on_the_job(self):
        job = self.make_job(DELETE_FILES_OPTIONS)

        self.assertTrue(job.indexing_options["delete_files_after_indexing"])
        self.assertTrue(indexing_pipeline.deletes_files_after_indexing(job))
        self.assertFalse(
            indexing_pipeline.deletes_files_after_indexing(self.make_job())
        )


class OptionValidationTests(TestCase):
    def test_the_option_is_an_optional_boolean(self):
        validated = indexing_options_validator(
            dict(INDEXING_OPTIONS, delete_files_after_indexing="true")
        )
        self.assertIs(validated["delete_files_after_indexing"], True)

        validated = indexing_options_validator(dict(INDEXING_OPTIONS))
        self.assertNotIn("delete_files_after_indexing", validated)

    def test_a_value_which_is_not_a_boolean_is_rejected(self):
        with self.assertRaises(ValueError):
            indexing_options_validator(
                dict(INDEXING_OPTIONS, delete_files_after_indexing="sometimes")
            )
