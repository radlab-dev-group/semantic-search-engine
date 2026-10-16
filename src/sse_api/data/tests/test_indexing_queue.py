"""
Progress of the indexing and the queue it runs in.

Every document belongs to an indexing job and carries a status of its own, so a
client can ask what is already indexed and how much is still waiting.  The
vector database is never touched: the controller of the embeddings is replaced
with a stand-in (``semantic_controller_for``).
"""

import json
import unittest
from importlib.util import find_spec
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.core.utils import compute_text_hash
from sse_api.data import indexing_pipeline, indexing_queue
from sse_api.data.api import (
    AddAndIndexTextsFromEP,
    IndexingStatus,
    UploadAndIndexFilesToCollection,
)
from sse_api.data.core import indexing_progress
from sse_api.data.core.constants import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_INDEXING,
    DOCUMENT_STATUS_PENDING,
    DOCUMENT_STATUS_SKIPPED,
    INDEXING_JOB_FILES,
    INDEXING_JOB_TEXTS,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_INDEXING,
    JOB_STATUS_PARTIAL,
)
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
    IndexingJob,
    UploadedDocuments,
)
from sse_api.system.models import Organisation, OrganisationUser
from sse_api.tests.input_support import EndpointInputMixin, INDEXING_OPTIONS

#: The tasks can only be imported when Celery is installed (ENV_USE_CELERY=1).
HAS_CELERY = find_spec("celery") is not None

TEXT_PAYLOAD = [
    {
        "filepath": "notes.txt",
        "relative_filepath": "notes.txt",
        "category": "Notes",
        "pages": [{"page_content": "A text long enough to be indexed."}],
    }
]


class IndexingFixtureTestCase(TestCase):
    """Organisation, user, collection, upload process and the helpers."""

    def setUp(self):
        self.organisation = Organisation.objects.create(name="queue org")
        self.user = User.objects.create_user(username="queue-user")
        self.profile = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = CollectionOfDocuments.objects.create(
            name="collection",
            display_name="Collection",
            created_by=self.profile,
            model_embedder="embedder",
        )
        self.upload = UploadedDocuments.objects.create(
            dir_hash="upload-hash",
            dir_path="/tmp/sse-upload",
            uploaded_by=self.profile,
        )

    def make_job(self, kind=INDEXING_JOB_FILES):
        return indexing_progress.create_job(
            collection=self.collection,
            organisation_user=self.profile,
            kind=kind,
            indexing_options=dict(INDEXING_OPTIONS),
            upload_process=self.upload if kind == INDEXING_JOB_FILES else None,
        )

    def make_document(self, name, indexing_job=None, with_chunk=True):
        document = Document.objects.create(
            name=name,
            collection=self.collection,
            path=f"/tmp/{name}",
            relative_path=name,
            document_hash=compute_text_hash(name),
            added_by=self.profile,
            indexing_job=indexing_job,
            indexing_status=DOCUMENT_STATUS_PENDING if indexing_job else None,
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

    def without_an_embedder(self):
        """Replace the controller of the vector database with a stand-in."""
        controller = Mock()
        patcher = patch.object(
            indexing_pipeline, "semantic_controller_for", return_value=controller
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return controller


class IndexingEndpointTestCase(EndpointInputMixin, IndexingFixtureTestCase):
    """The fixtures of ``IndexingFixtureTestCase`` and the request helpers."""

    def setUp(self):
        # EndpointInputMixin does not call super(), so both are called here.
        EndpointInputMixin.setUp(self)
        IndexingFixtureTestCase.setUp(self)
        self.factory = APIRequestFactory()

    def status_of(self, payload=None):
        return self.invoke(IndexingStatus, payload, method="get")


class IndexingProgressTests(IndexingFixtureTestCase):
    def test_documents_are_counted_and_the_job_closes_itself(self):
        job = self.make_job()
        documents = [
            self.make_document(f"a-{number}.txt", job) for number in range(3)
        ]
        indexing_progress.mark_job_indexing(job, len(documents))
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_INDEXING)
        self.assertEqual(job.percent(), 0)
        self.assertFalse(job.is_finished)

        indexing_progress.finish_document(job, documents[0], DOCUMENT_STATUS_INDEXED)
        job.refresh_from_db()
        self.assertEqual(job.documents_indexed, 1)
        self.assertEqual(job.percent(), 33)
        self.assertFalse(job.is_finished)

        for document in documents[1:]:
            indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_COMPLETED)
        self.assertEqual(job.percent(), 100)
        self.assertIsNotNone(job.finished_on)
        self.assertIsNone(job.current_document)

    def test_upload_process_is_closed_with_the_job(self):
        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 1)
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)

        self.upload.refresh_from_db()
        self.assertTrue(self.upload.is_indexed)
        self.assertIsNotNone(self.upload.end_indexing_time)

    def test_partly_failed_job_is_partial(self):
        job = self.make_job()
        good = self.make_document("good.txt", job)
        broken = self.make_document("broken.txt", job)
        indexing_progress.mark_job_indexing(job, 2)

        indexing_progress.finish_document(job, good, DOCUMENT_STATUS_INDEXED)
        indexing_progress.finish_document(
            job, broken, DOCUMENT_STATUS_FAILED, error="model is gone"
        )
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_PARTIAL)
        self.assertEqual(job.documents_indexed, 1)
        self.assertEqual(job.documents_failed, 1)
        self.assertIn("broken.txt", job.error_message)
        self.upload.refresh_from_db()
        self.assertFalse(self.upload.is_indexed)

    def test_job_whose_documents_all_failed(self):
        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 1)
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_FAILED)
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_FAILED)

    def test_finalizing_twice_changes_nothing(self):
        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 1)
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)
        job.refresh_from_db()
        finished_on = job.finished_on

        indexing_progress.finalize_job_if_done(job)
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_COMPLETED)
        self.assertEqual(job.documents_indexed, 1)
        self.assertEqual(job.finished_on, finished_on)

    def test_job_without_documents_is_finished_after_preparing(self):
        job = self.make_job(kind=INDEXING_JOB_TEXTS)
        indexing_progress.mark_job_indexing(job, 0)
        indexing_progress.finalize_job_if_done(job)
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_COMPLETED)

    def test_failing_a_job_fails_the_documents_which_never_started(self):
        job = self.make_job()
        started = self.make_document("started.txt", job)
        waiting = self.make_document("waiting.txt", job)
        indexing_progress.mark_job_indexing(job, 2)
        indexing_progress.mark_document_indexing(job, started)

        indexing_progress.fail_job(job, "broker is down")
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_FAILED)
        self.assertEqual(job.error_message, "broker is down")
        waiting.refresh_from_db()
        self.assertEqual(waiting.indexing_status, DOCUMENT_STATUS_FAILED)
        started.refresh_from_db()
        self.assertEqual(started.indexing_status, DOCUMENT_STATUS_INDEXING)

    def test_stale_jobs_are_reset(self):
        from datetime import timedelta

        from django.utils import timezone

        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 2)
        long_ago = timezone.now() - timedelta(minutes=90)
        IndexingJob.objects.filter(pk=job.pk).update(created_on=long_ago)
        Document.objects.filter(pk=document.pk).update(created_on=long_ago)

        reset = indexing_progress.reset_stale_jobs(minutes=60)
        job.refresh_from_db()
        document.refresh_from_db()

        self.assertEqual(reset, {"jobs": 1, "documents": 1})
        self.assertEqual(job.status, JOB_STATUS_FAILED)
        self.assertEqual(document.indexing_status, DOCUMENT_STATUS_FAILED)

    def test_recent_jobs_are_left_alone(self):
        job = self.make_job()
        self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 2)

        self.assertEqual(
            indexing_progress.reset_stale_jobs(minutes=60),
            {"jobs": 0, "documents": 0},
        )
        job.refresh_from_db()
        self.assertEqual(job.status, JOB_STATUS_INDEXING)


class IndexingPipelineTests(IndexingFixtureTestCase):
    def test_documents_of_a_job_are_indexed_one_by_one(self):
        job = self.make_job()
        first = self.make_document("a.txt", job)
        second = self.make_document("b.txt", job)
        indexing_progress.mark_job_indexing(job, 2)
        controller = self.without_an_embedder()

        indexing_pipeline.run_index_document(job, first)
        job.refresh_from_db()
        first.refresh_from_db()
        second.refresh_from_db()

        self.assertEqual(job.documents_indexed, 1)
        self.assertEqual(job.current_document, "a.txt")
        self.assertEqual(first.indexing_status, DOCUMENT_STATUS_INDEXED)
        self.assertEqual(second.indexing_status, DOCUMENT_STATUS_PENDING)

        indexing_pipeline.run_index_document(job, second)
        job.refresh_from_db()
        second.refresh_from_db()

        self.assertEqual(second.indexing_status, DOCUMENT_STATUS_INDEXED)
        self.assertEqual(job.status, JOB_STATUS_COMPLETED)
        self.assertEqual(controller.index_texts_from_list.call_count, 2)

    def test_document_without_text_is_skipped(self):
        job = self.make_job()
        document = self.make_document("empty.txt", job, with_chunk=False)
        indexing_progress.mark_job_indexing(job, 1)
        controller = self.without_an_embedder()

        status = indexing_pipeline.run_index_document(job, document)
        document.refresh_from_db()
        job.refresh_from_db()

        self.assertEqual(status, DOCUMENT_STATUS_SKIPPED)
        self.assertEqual(document.indexing_status, DOCUMENT_STATUS_SKIPPED)
        self.assertEqual(job.status, JOB_STATUS_COMPLETED)
        controller.index_texts_from_list.assert_not_called()

    def test_embedder_is_loaded_once_per_process(self):
        indexing_pipeline.clear_controller_cache()
        self.addCleanup(indexing_pipeline.clear_controller_cache)

        with patch.object(
            indexing_pipeline, "DBSemanticSearchController"
        ) as controller_class:
            first = indexing_pipeline.semantic_controller_for(self.collection)
            second = indexing_pipeline.semantic_controller_for(self.collection)
            self.assertIs(first, second)
            indexing_pipeline.semantic_controller_for(
                self.collection, jsonl_config_path="other.json"
            )

        self.assertEqual(controller_class.call_count, 2)

    def test_preparing_stores_the_documents_of_pasted_texts(self):
        job = self.make_job(kind=INDEXING_JOB_TEXTS)
        job.request_payload = {"texts": TEXT_PAYLOAD}
        job.save(update_fields=["request_payload"])
        fixture = self

        def add_documents(controller, **kwargs):
            document = fixture.make_document("notes.txt", kwargs["indexing_job"])
            return [{"document": document, "pages_chunks": []}]

        with patch(
            "sse_api.engine.controllers.database.relational_db."
            "RelationalDBController.add_documents_from_list_of_doc_as_dict",
            new=add_documents,
        ):
            indexing_pipeline.run_prepare(job)

        job.refresh_from_db()
        document = job.documents.get()

        self.assertEqual(job.total_documents, 1)
        self.assertEqual(job.status, JOB_STATUS_INDEXING)
        self.assertEqual(document.indexing_status, DOCUMENT_STATUS_PENDING)
        self.assertEqual(indexing_pipeline.count_job_chunks(job), 1)

    def test_preparing_failure_reaches_the_caller(self):
        job = self.make_job(kind=INDEXING_JOB_TEXTS)
        with patch(
            "sse_api.engine.controllers.database.relational_db."
            "RelationalDBController.add_documents_from_list_of_doc_as_dict",
            side_effect=OSError("disk is gone"),
        ):
            with self.assertRaises(OSError):
                indexing_pipeline.run_prepare(job)

    def test_reading_processes_survive_a_daemon(self):
        """A daemonic process may not create the reading pool of the files."""
        with patch.object(
            indexing_pipeline.multiprocessing, "current_process"
        ) as current_process:
            current_process.return_value.daemon = False
            self.assertEqual(indexing_pipeline.reading_processes(5), 5)

            current_process.return_value.daemon = True
            self.assertEqual(indexing_pipeline.reading_processes(5), 1)
            self.assertEqual(indexing_pipeline.reading_processes(1), 1)

    def test_files_of_a_daemon_are_read_in_one_process(self):
        """The job is prepared slowly rather than not at all in a prefork task."""
        job = self.make_job()
        with patch.object(
            indexing_pipeline.multiprocessing, "current_process"
        ) as current_process:
            current_process.return_value.daemon = True
            with patch(
                "sse_api.engine.controllers.database.relational_db."
                "RelationalDBController.add_uploaded_documents_to_db"
            ) as add_documents:
                indexing_pipeline.run_prepare(job)

        self.assertEqual(add_documents.call_args.kwargs["number_of_process"], 1)


class IndexingStatusEndpointTests(IndexingEndpointTestCase):
    def test_without_job_and_collection(self):
        self.assertFalse(self.status_of().data["status"])

    def test_collection_without_indexing_has_no_job(self):
        body = self.status_of({"collection_name": "collection"}).data["body"]

        self.assertIsNone(body["job"])
        self.assertEqual(body["documents"], [])
        self.assertEqual(body["pending_documents"], 0)

    def test_the_newest_job_of_the_collection_is_reported(self):
        older = self.make_job()
        indexing_progress.mark_job_indexing(older, 0)
        indexing_progress.finalize_job_if_done(older)
        newest = self.make_job()
        document = self.make_document("a.txt", newest)
        indexing_progress.mark_job_indexing(newest, 2)
        indexing_progress.mark_document_indexing(newest, document)

        body = self.status_of({"collection_name": "collection"}).data["body"]

        self.assertEqual(body["job"]["id"], newest.pk)
        self.assertEqual(body["job"]["status"], JOB_STATUS_INDEXING)
        self.assertEqual(body["job"]["total_documents"], 2)
        self.assertEqual(body["job"]["documents_pending"], 2)
        self.assertEqual(body["job"]["current_document"], "a.txt")
        # One document row is waiting; one more is expected but not stored yet.
        self.assertEqual(body["pending_documents"], 1)
        self.assertEqual(
            body["documents"],
            [{"id": document.pk, "name": "a.txt", "indexing_status": "indexing"}],
        )

    def test_one_job_by_its_id(self):
        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 1)
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)

        body = self.status_of({"job_id": job.pk}).data["body"]

        self.assertEqual(body["job"]["status"], JOB_STATUS_COMPLETED)
        self.assertEqual(body["job"]["percent"], 100)
        self.assertEqual(
            body["documents"],
            [{"id": document.pk, "name": "a.txt", "indexing_status": "indexed"}],
        )

    def test_unknown_and_foreign_jobs_are_not_found(self):
        self.assertFalse(self.status_of({"job_id": 999999}).data["status"])

        somebody = User.objects.create_user(username="somebody-else")
        profile = OrganisationUser.objects.create(
            auth_user=somebody, organisation=self.organisation
        )
        private = CollectionOfDocuments.objects.create(
            name="private", display_name="Private", created_by=profile
        )
        foreign = indexing_progress.create_job(
            collection=private,
            organisation_user=profile,
            kind=INDEXING_JOB_TEXTS,
            indexing_options={},
        )

        self.assertFalse(self.status_of({"job_id": foreign.pk}).data["status"])

    def test_unknown_collection(self):
        self.assertFalse(
            self.status_of({"collection_name": "no-such"}).data["status"]
        )


class QueuedIndexingEndpointTests(IndexingEndpointTestCase):
    """With ``use_celery`` on, the request stores the content and queues it."""

    def queue_only(self):
        patch(
            "sse_api.data.api.indexing_queue.async_indexing", return_value=True
        ).start()
        dispatch = patch(
            "sse_api.data.api.indexing_queue.dispatch_indexing_job"
        ).start()
        self.addCleanup(patch.stopall)
        return dispatch

    def upload_files(self):
        request = self.factory.post(
            "/",
            {
                "files[]": SimpleUploadedFile("note.txt", b"a note"),
                "collection_name": "collection",
                "indexing_options": json.dumps(INDEXING_OPTIONS),
            },
            format="multipart",
        )
        force_authenticate(request, user=self.user)
        return UploadAndIndexFilesToCollection.as_view()(request)

    def test_files_are_stored_and_queued(self):
        dispatch = self.queue_only()
        with (
            patch(
                "sse_api.data.api.UploadDocumentsController.store_uploaded_files",
                return_value=self.upload,
            ) as store,
            patch(
                "sse_api.data.api.UploadDocumentsController."
                "store_and_index_files_rel_db_post_request"
            ) as index_in_request,
        ):
            response = self.upload_files()

        store.assert_called_once()
        index_in_request.assert_not_called()
        job = dispatch.call_args.args[0]
        self.assertEqual(job.kind, INDEXING_JOB_FILES)
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.upload_process_id, self.upload.pk)
        self.assertTrue(response.data["body"]["async"])
        self.assertEqual(response.data["body"]["indexing_job"]["id"], job.pk)

    def test_texts_are_queued_with_their_payload(self):
        dispatch = self.queue_only()
        with patch(
            "sse_api.engine.controllers.database.relational_db."
            "RelationalDBController.add_documents_from_list_of_doc_as_dict"
        ) as add_documents:
            response = self.invoke(
                AddAndIndexTextsFromEP,
                {
                    "texts[]": TEXT_PAYLOAD,
                    "collection_name": "collection",
                    "indexing_options": INDEXING_OPTIONS,
                },
            )

        add_documents.assert_not_called()
        job = dispatch.call_args.args[0]
        self.assertEqual(job.kind, INDEXING_JOB_TEXTS)
        self.assertEqual(job.request_payload, {"texts": TEXT_PAYLOAD})
        self.assertEqual(job.indexing_options, INDEXING_OPTIONS)
        self.assertTrue(response.data["body"]["async"])

    def test_the_queue_is_off_without_the_configuration(self):
        self.assertFalse(indexing_queue.async_indexing())


@unittest.skipUnless(HAS_CELERY, "Celery is installed only with the queue")
class CeleryTasksTests(IndexingFixtureTestCase):
    def setUp(self):
        super().setUp()
        from sse_api.data import tasks

        self.tasks = tasks

    def test_preparing_fans_the_job_out_into_documents(self):
        job = self.make_job()
        documents = [self.make_document(f"a-{n}.txt", job) for n in range(2)]
        for document in documents:
            indexing_progress.mark_document_indexing(job, document)
        indexing_progress.mark_job_indexing(job, len(documents))

        with (
            patch.object(indexing_pipeline, "run_prepare") as prepare,
            patch.object(self.tasks.index_document, "delay") as queue_document,
        ):
            self.tasks.prepare_indexing_job.run(job.pk)

        prepare.assert_called_once()
        self.assertEqual(
            [call.args for call in queue_document.call_args_list],
            [(job.pk, document.pk) for document in documents],
        )

    def test_a_finished_job_is_not_prepared_again(self):
        job = self.make_job()
        indexing_progress.mark_job_indexing(job, 0)
        indexing_progress.finalize_job_if_done(job)

        with (
            patch.object(indexing_pipeline, "run_prepare") as prepare,
            patch.object(self.tasks.index_document, "delay"),
        ):
            self.tasks.prepare_indexing_job.run(job.pk)

        prepare.assert_not_called()

    def test_a_document_of_a_finished_job_is_not_indexed_again(self):
        job = self.make_job()
        document = self.make_document("a.txt", job)
        indexing_progress.mark_job_indexing(job, 1)
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)

        with patch.object(indexing_pipeline, "run_index_document") as index:
            self.tasks.index_document.run(job.pk, document.pk)

        index.assert_not_called()

    def test_failure_of_the_preparing_closes_the_job(self):
        job = self.make_job()
        error = OSError("the disk is gone")

        self.tasks.prepare_indexing_job.on_failure(
            error, "task-id", (job.pk,), {}, None
        )
        job.refresh_from_db()

        self.assertEqual(job.status, JOB_STATUS_FAILED)
        self.assertIn("disk is gone", job.error_message)

    def test_failure_of_one_document_marks_only_that_document(self):
        job = self.make_job()
        broken = self.make_document("broken.txt", job)
        waiting = self.make_document("waiting.txt", job)
        indexing_progress.mark_job_indexing(job, 2)

        self.tasks.index_document.on_failure(
            OSError("no embeddings"), "task-id", (job.pk, broken.pk), {}, None
        )
        broken.refresh_from_db()
        waiting.refresh_from_db()
        job.refresh_from_db()

        self.assertEqual(broken.indexing_status, DOCUMENT_STATUS_FAILED)
        self.assertEqual(waiting.indexing_status, DOCUMENT_STATUS_PENDING)
        self.assertEqual(job.documents_failed, 1)
        self.assertEqual(job.status, JOB_STATUS_INDEXING)
