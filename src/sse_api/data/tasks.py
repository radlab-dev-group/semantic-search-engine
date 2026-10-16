"""
Celery tasks of the indexing.

Two tasks, both thin wrappers over ``sse_api.data.indexing_pipeline``:

``prepare_indexing_job``
    Reads the uploaded files / pasted texts of one job, stores its documents and
    asks for one ``index_document`` per document of the job.

``index_document``
    Embeds one document into the vector database.  A document is the unit of
    work: it is retried on its own, its status is what a client reads as the
    progress, and one broken document does not stop the others.

Both acknowledge late, so a worker killed in the middle of a document puts the
message back on the queue.  When the retries of a task are used up, ``on_failure``
writes the failure into the job (and into the document) instead of leaving them
running forever.
"""

import logging

from celery import shared_task
from celery import Task

from sse_api.data import indexing_pipeline
from sse_api.data.core import indexing_progress
from sse_api.data.core.constants import DOCUMENT_STATUS_FAILED
from sse_api.data.models import Document, IndexingJob

MAIN_LOGGER = logging.getLogger("sse_api")

#: How often a task is retried before it is reported as failed.
MAX_RETRIES = 3

#: Seconds before the first retry; every next one waits twice as long.
RETRY_BACKOFF_SECONDS = 30
RETRY_BACKOFF_MAX_SECONDS = 600

TASK_OPTIONS = {
    "acks_late": True,
    "autoretry_for": (Exception,),
    "retry_backoff": RETRY_BACKOFF_SECONDS,
    "retry_backoff_max": RETRY_BACKOFF_MAX_SECONDS,
    "retry_jitter": True,
    "max_retries": MAX_RETRIES,
}


class PrepareIndexingTask(Task):
    """Keeps the job alive when the preparing cannot be done at all."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job = IndexingJob.objects.filter(pk=args[0]).first()
        if job is None:
            return
        MAIN_LOGGER.error("Indexing job %s failed: %s", job.pk, exc)
        indexing_progress.fail_job(job, exc)


class IndexDocumentTask(Task):
    """Reports a document which cannot be embedded after all the retries."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job = IndexingJob.objects.filter(pk=args[0]).first()
        document = Document.objects.filter(pk=args[1]).first()
        if job is None or document is None:
            return
        MAIN_LOGGER.error(
            "Document %s of indexing job %s failed: %s", document.pk, job.pk, exc
        )
        indexing_progress.finish_document(
            job, document, DOCUMENT_STATUS_FAILED, error=exc
        )


@shared_task(
    bind=True,
    base=PrepareIndexingTask,
    name="sse_api.data.prepare_indexing_job",
    **TASK_OPTIONS,
)
def prepare_indexing_job(self, job_id: int) -> None:
    """Store the documents of one indexing job and queue them one by one."""
    job = (
        IndexingJob.objects.select_related(
            "collection", "upload_process", "created_by"
        )
        .filter(pk=job_id)
        .first()
    )
    if job is None:
        MAIN_LOGGER.warning("Indexing job %s no longer exists", job_id)
        return
    if job.is_finished:
        MAIN_LOGGER.info("Indexing job %s is finished already", job.pk)
        return

    indexing_pipeline.run_prepare(job)

    document_ids = indexing_pipeline.job_document_ids(job)
    MAIN_LOGGER.info(
        "Indexing job %s: %s documents to embed", job.pk, len(document_ids)
    )
    for document_id in document_ids:
        index_document.delay(job.pk, document_id)


@shared_task(
    bind=True,
    base=IndexDocumentTask,
    name="sse_api.data.index_document",
    **TASK_OPTIONS,
)
def index_document(self, job_id: int, document_id: int) -> None:
    """Embed one document of an indexing job into the vector database."""
    job = IndexingJob.objects.select_related("created_by").filter(pk=job_id).first()
    document = (
        Document.objects.select_related("collection").filter(pk=document_id).first()
    )
    if job is None or document is None:
        MAIN_LOGGER.warning(
            "Indexing job %s / document %s no longer exists", job_id, document_id
        )
        return
    if job.is_finished or indexing_pipeline.is_indexed_already(document):
        MAIN_LOGGER.info(
            "Document %s of job %s was indexed already", document.pk, job.pk
        )
        return

    indexing_pipeline.run_index_document(job, document)
