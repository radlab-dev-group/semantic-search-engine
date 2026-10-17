"""
Which process indexes, and how a job gets to the queue.

``use_celery`` (``ENV_USE_CELERY``) decides whether indexing runs in the
request which uploaded the files or in a Celery worker.  Both paths execute the
very same pipeline (``sse_api.data.indexing_pipeline``) and write the same
progress, so a client does not have to know which of them was used.

Celery itself is imported only when a job is really dispatched: an engine
without ``ENV_USE_CELERY`` does not have to install it at all.
"""

import logging

from django.conf import settings

from sse_api.data.core.constants import (
    INDEXING_JOB_FILES,
    INDEXING_JOB_TEXTS,
)
from sse_api.data.models import IndexingJob

MAIN_LOGGER = logging.getLogger("sse_api")


def async_indexing() -> bool:
    """Whether indexing is handed over to Celery."""
    handler = getattr(settings, "SYSTEM_HANDLER", None)
    return bool(getattr(handler, "use_celer_tasks", False))


def dispatch_indexing_job(job: IndexingJob) -> None:
    """Put ``job`` on the queue; documents are fanned out by the prepare task."""
    from sse_api.data import tasks

    if job.kind not in (INDEXING_JOB_FILES, INDEXING_JOB_TEXTS):
        raise ValueError(f"unknown kind of indexing job: {job.kind}")

    MAIN_LOGGER.info("Queueing indexing job %s (%s)", job.pk, job.kind)
    tasks.prepare_indexing_job.delay(job.pk)
