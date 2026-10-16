"""
Bookkeeping of one indexing request.

Every document which is indexed belongs to an :class:`IndexingJob`.  The job
row is the single source of truth for the progress: it is written by the API
request which starts the indexing and then updated by the Celery tasks (or by
the request itself when ``use_celery`` is off).  Keeping the counters in the
relational database, and not in the result backend of Celery, means that the
progress survives a restart of a worker and that several workers may update
the same job at the same time.

Counters are changed with ``F()`` expressions only, and every transition into a
terminal status goes through :func:`finalize_job_if_done`, which locks the row
so that exactly one worker closes the job.
"""

import logging
from datetime import timedelta

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from sse_api.data.core.constants import (
    DEFAULT_INDEXING_STALE_MINUTES,
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_INDEXING,
    DOCUMENT_STATUS_PENDING,
    DOCUMENT_STATUS_SKIPPED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_INDEXING,
    JOB_STATUS_PARTIAL,
    JOB_STATUS_PREPARING,
    JOB_STATUS_QUEUED,
    JOB_TERMINAL_STATUSES,
)
from sse_api.data.models import Document, IndexingJob, UploadedDocuments

MAIN_LOGGER = logging.getLogger("sse_api")

#: Longest error message kept on a job.
MAX_ERROR_LENGTH = 1500

#: Documents which are waiting for their embeddings.
RUNNING_DOCUMENT_STATUSES = (DOCUMENT_STATUS_PENDING, DOCUMENT_STATUS_INDEXING)

#: Which counter of the job one final document status increments.
_DOCUMENT_COUNTER = {
    DOCUMENT_STATUS_INDEXED: "documents_indexed",
    DOCUMENT_STATUS_FAILED: "documents_failed",
    DOCUMENT_STATUS_SKIPPED: "documents_skipped",
}


def _shorten(error) -> str | None:
    if error is None:
        return None
    return str(error)[:MAX_ERROR_LENGTH]


def create_job(
    collection,
    organisation_user,
    kind: str,
    indexing_options: dict,
    upload_process: UploadedDocuments = None,
    request_payload: dict = None,
) -> IndexingJob:
    """A new job in status ``queued``, before any indexing work is done."""
    return IndexingJob.objects.create(
        collection=collection,
        upload_process=upload_process,
        kind=kind,
        status=JOB_STATUS_QUEUED,
        indexing_options=indexing_options,
        request_payload=request_payload,
        created_by=organisation_user,
    )


def mark_job_preparing(job: IndexingJob) -> None:
    """The files of the job are being read and split into chunks."""
    job.status = JOB_STATUS_PREPARING
    job.started_on = job.started_on or timezone.now()
    job.save(update_fields=["status", "started_on"])


def mark_job_indexing(job: IndexingJob, total_documents: int) -> None:
    """Documents are stored; ``total_documents`` of them wait for embeddings."""
    job.status = JOB_STATUS_INDEXING
    job.started_on = job.started_on or timezone.now()
    job.total_documents = total_documents
    job.save(update_fields=["status", "started_on", "total_documents"])


def mark_document_indexing(job: IndexingJob, document: Document) -> None:
    Document.objects.filter(pk=document.pk).update(
        indexing_status=DOCUMENT_STATUS_INDEXING
    )
    IndexingJob.objects.filter(pk=job.pk).update(current_document=document.name)


def finish_document(
    job: IndexingJob, document: Document, status: str, error=None
) -> None:
    """Final status of one document plus the counters of its job."""
    Document.objects.filter(pk=document.pk).update(indexing_status=status)

    counter = _DOCUMENT_COUNTER.get(status)
    if counter is not None:
        IndexingJob.objects.filter(pk=job.pk).update(**{counter: F(counter) + 1})
    if status == DOCUMENT_STATUS_FAILED:
        # The first error is the one worth showing; the rest are in the logs.
        message = _shorten(f"{document.name}: {error}")
        IndexingJob.objects.filter(pk=job.pk, error_message__isnull=True).update(
            error_message=message
        )

    fresh = finalize_job_if_done(job)
    job.status = fresh.status


def fail_job(job: IndexingJob, error) -> None:
    """Nothing more can be indexed; the documents which never started failed."""
    with transaction.atomic():
        fresh = IndexingJob.objects.select_for_update().get(pk=job.pk)
        if fresh.status in JOB_TERMINAL_STATUSES:
            return
        fresh.status = JOB_STATUS_FAILED
        fresh.finished_on = timezone.now()
        fresh.current_document = None
        if fresh.error_message is None:
            fresh.error_message = _shorten(error)
        fresh.save(
            update_fields=[
                "status",
                "finished_on",
                "current_document",
                "error_message",
            ]
        )
        Document.objects.filter(
            indexing_job=fresh, indexing_status=DOCUMENT_STATUS_PENDING
        ).update(indexing_status=DOCUMENT_STATUS_FAILED)
        _close_upload_process(fresh)
        job.refresh_from_db()


def finalize_job_if_done(job: IndexingJob) -> IndexingJob:
    """Close ``job`` when all of its documents are through.

    The row is locked while the counters are read, so exactly one of the
    workers which finish the last documents switches the job into a terminal
    status and closes the upload process of the files.
    """
    with transaction.atomic():
        fresh = IndexingJob.objects.select_for_update().get(pk=job.pk)
        if fresh.status in JOB_TERMINAL_STATUSES:
            return fresh

        total = fresh.total_documents
        if total is None or fresh.documents_processed < total:
            return fresh

        if fresh.documents_failed:
            everything_failed = not (
                fresh.documents_indexed or fresh.documents_skipped
            )
            fresh.status = (
                JOB_STATUS_FAILED if everything_failed else JOB_STATUS_PARTIAL
            )
        else:
            fresh.status = JOB_STATUS_COMPLETED

        fresh.finished_on = timezone.now()
        fresh.current_document = None
        fresh.save(update_fields=["status", "finished_on", "current_document"])
        _close_upload_process(fresh)
        return fresh


def _close_upload_process(job: IndexingJob) -> None:
    if not job.upload_process_id:
        return
    UploadedDocuments.objects.filter(pk=job.upload_process_id).update(
        end_indexing_time=job.finished_on,
        is_indexed=job.status == JOB_STATUS_COMPLETED,
    )


def latest_job(collection) -> IndexingJob | None:
    """The newest job of ``collection``, finished or not."""
    return (
        IndexingJob.objects.filter(collection=collection)
        .order_by("-created_on", "-id")
        .first()
    )


def pending_documents(collection) -> int:
    """Documents of the collection waiting for their embeddings."""
    return Document.objects.filter(
        collection=collection, indexing_status__in=RUNNING_DOCUMENT_STATUSES
    ).count()


def stale_jobs(minutes: int = DEFAULT_INDEXING_STALE_MINUTES):
    """Jobs which were created before ``minutes`` and are still running."""
    return IndexingJob.objects.filter(created_on__lt=_deadline(minutes)).exclude(
        status__in=JOB_TERMINAL_STATUSES
    )


def stale_documents(minutes: int = DEFAULT_INDEXING_STALE_MINUTES):
    """Documents which were created before ``minutes`` and are still waiting."""
    return Document.objects.filter(
        created_on__lt=_deadline(minutes),
        indexing_status__in=RUNNING_DOCUMENT_STATUSES,
    )


def reset_stale_jobs(minutes: int = DEFAULT_INDEXING_STALE_MINUTES) -> dict:
    """Fail jobs and documents which were left in a running status.

    A worker killed in the middle of a document would keep a job open forever,
    so everything which is still running after ``minutes`` is reported as
    failed and the client stops asking about it.
    """
    now = timezone.now()
    jobs = stale_jobs(minutes).update(status=JOB_STATUS_FAILED, finished_on=now)
    documents = stale_documents(minutes).update(
        indexing_status=DOCUMENT_STATUS_FAILED
    )
    return {"jobs": jobs, "documents": documents}


def _deadline(minutes: int):
    return timezone.now() - timedelta(minutes=minutes)


def job_payload(job: IndexingJob | None) -> dict | None:
    """The job as it leaves the API; ``None`` when nothing was indexed yet."""
    if job is None:
        return None
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "total_documents": job.total_documents,
        "documents_indexed": job.documents_indexed,
        "documents_failed": job.documents_failed,
        "documents_skipped": job.documents_skipped,
        "documents_pending": _documents_pending(job),
        "documents_processed": job.documents_processed,
        "current_document": job.current_document,
        "error_message": job.error_message,
        "percent": job.percent(),
        "finished": job.is_finished,
        "created_on": _isoformat(job.created_on),
        "started_on": _isoformat(job.started_on),
        "finished_on": _isoformat(job.finished_on),
    }


def _documents_pending(job: IndexingJob) -> int:
    if job.total_documents is None:
        return 0
    return max(0, job.total_documents - job.documents_processed)


def _isoformat(value) -> str | None:
    return value.isoformat() if value is not None else None
