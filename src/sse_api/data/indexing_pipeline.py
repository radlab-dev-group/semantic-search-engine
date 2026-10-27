"""
The indexing work itself, without anything Celery specific.

Two steps, both safe to call from a Celery task, from a request (when
``use_celery`` is off) or from a management command:

``run_prepare(job)``
    Read the files (or the pasted texts) of the job, split them into chunks and
    store ``Document`` / ``DocumentPage`` / ``DocumentPageText`` rows.  Every new
    document starts in status ``pending`` and the total number of documents of
    the job becomes known.

``run_index_document(job, document)``
    Calculate the embeddings of one document and put them into the vector
    database.  One document is the unit of work of the queue: it can be
    retried on its own and its status is what a client sees.  A job created
    with ``delete_files_after_indexing`` loses the file of the document here,
    once its embeddings are safely stored (a failed document keeps its file).

The controller of the vector database keeps the embedder model in memory, so
one instance per worker process and collection is cached instead of being
loaded for every document.
"""

import logging
import multiprocessing

from sse_api.data.controllers import file_cleanup
from sse_api.data.core import indexing_progress
from sse_api.data.core.constants import (
    DB_INDEXING_PROCESS_COUNT,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_SKIPPED,
    INDEXING_JOB_FILES,
    INDEXING_JOB_TEXTS,
    MILVUS_CONFIG_PATH,
)
from sse_api.data.models import Document, DocumentPageText, IndexingJob
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.search.semantic import DBSemanticSearchController

MAIN_LOGGER = logging.getLogger("sse_api")

# (collection pk, index type, embedder, reranker) -> controller
_CONTROLLERS: dict = {}


def reading_processes(number_of_process: int = DB_INDEXING_PROCESS_COUNT) -> int:
    """How many processes may read the files of a job in parallel.

    ``radlab_data`` reads with a ``multiprocessing.Pool`` of its own, and Python
    refuses to create one inside a process which is itself the child of a daemon
    ("daemonic processes are not allowed to have children") -- which is exactly
    what a task of a Celery ``prefork`` worker is.  Reading then falls back to
    the current process instead of failing the whole job; a worker started with
    ``--pool solo`` (the default of ``run.sh``, ``CELERY_WORKER_POOL``) keeps
    the parallel reading.
    """
    if number_of_process > 1 and multiprocessing.current_process().daemon:
        MAIN_LOGGER.warning(
            "Reading the files of a job inside a daemon process: one process "
            "instead of %s (start the worker with --pool solo to read in "
            "parallel)",
            number_of_process,
        )
        return 1
    return number_of_process


def semantic_controller_for(collection, jsonl_config_path: str = MILVUS_CONFIG_PATH):
    """The vector database controller of ``collection``, cached per process."""
    key = (
        collection.pk,
        collection.embedder_index_type,
        collection.model_embedder,
        collection.model_reranker,
        jsonl_config_path,
    )
    controller = _CONTROLLERS.get(key)
    if controller is None:
        controller = DBSemanticSearchController(
            collection_name=collection.name,
            index_name=collection.embedder_index_type,
            batch_size=100,
            embedder_model=collection.model_embedder,
            cross_encoder_model=collection.model_reranker,
            jsonl_config_path=jsonl_config_path,
        )
        _CONTROLLERS[key] = controller
    return controller


def clear_controller_cache() -> None:
    """Forget the cached controllers (tests, or a model of the collection changed)."""
    _CONTROLLERS.clear()


def run_prepare(
    job: IndexingJob, number_of_process: int = DB_INDEXING_PROCESS_COUNT
) -> IndexingJob:
    """Store the documents of ``job`` and count them.

    Raises whatever reading or storing the content raises; the job is then
    failed by the caller which knows about retries.
    """
    indexing_progress.mark_job_preparing(job)

    options = job.indexing_options or {}
    relational = RelationalDBController(store_to_db=True)

    if job.kind == INDEXING_JOB_FILES:
        relational.add_uploaded_documents_to_db(
            collection_name=job.collection.name,
            organisation_user=job.created_by,
            uploaded_document=job.upload_process,
            prepare_proper_pages=options.get("prepare_proper_pages", True),
            merge_document_pages=options.get("merge_document_pages", False),
            clear_texts=options.get("clear_text", True),
            use_text_denoiser=options.get("use_text_denoiser", False),
            max_tokens_in_chunk=options.get("max_tokens_in_chunk"),
            number_of_overlap_tokens=options.get("number_of_overlap_tokens"),
            check_text_language=options.get("check_text_lang", True),
            number_of_process=reading_processes(number_of_process),
            indexing_job=job,
            file_metadata=(job.request_payload or {}).get("file_metadata"),
        )
    elif job.kind == INDEXING_JOB_TEXTS:
        relational.add_documents_from_list_of_doc_as_dict(
            collection=job.collection,
            texts=(job.request_payload or {}).get("texts", []),
            organisation_user=job.created_by,
            indexing_options=options,
            indexing_job=job,
        )
    else:
        raise ValueError(f"unknown kind of indexing job: {job.kind}")

    job.refresh_from_db()
    indexing_progress.mark_job_indexing(job, job.documents.count())
    # A job without documents is already done when the preparing ends.
    return indexing_progress.finalize_job_if_done(job)


def index_chunks(
    collection,
    chunks,
    jsonl_config_path: str = MILVUS_CONFIG_PATH,
) -> None:
    """Embed a ready list of chunks; used when there is no indexing job."""
    if not len(chunks):
        return
    controller = semantic_controller_for(collection, jsonl_config_path)
    controller.index_texts_from_list(all_texts=chunks, collection=collection)


def deletes_files_after_indexing(job: IndexingJob) -> bool:
    """
    Whether the job asks for the files of its uploads to disappear from the
    disk once their documents are indexed.

    The answer comes from ``job.indexing_options``, where the request which
    started the indexing left it: a worker of the queue knows nothing else
    about that call.
    """
    indexing_options = job.indexing_options or {}
    return bool(indexing_options.get("delete_files_after_indexing"))


def remove_document_file(job: IndexingJob, document: Document) -> None:
    """
    Remove the file of one document when the job asked for it.

    Called for a document which is through with indexing, never for a failed
    one: the file of a document whose embeddings are not in the vector
    database is the only thing a retry can still work on.  A problem with the
    disk is written into the log and never interrupts the job.
    """
    if not deletes_files_after_indexing(job):
        return
    result = file_cleanup.delete_document_file(document)
    for error in result.errors:
        MAIN_LOGGER.warning(
            "Could not remove the file of %s: %s", document.name, error
        )


def is_indexed_already(document) -> bool:
    """Whether a message about ``document`` may be skipped as a duplicate."""
    return document.indexing_status in (
        DOCUMENT_STATUS_INDEXED,
        DOCUMENT_STATUS_SKIPPED,
    )


def run_index_document(
    job: IndexingJob,
    document: Document,
    jsonl_config_path: str = MILVUS_CONFIG_PATH,
) -> str:
    """Embed one document into the vector database and report its status.

    Returns the final status of the document: ``indexed`` or ``skipped`` when
    the document has no text long enough to be worth indexing.
    """
    indexing_progress.mark_document_indexing(job, document)

    chunks = DocumentPageText.objects.filter(page__document=document).select_related(
        "page__document"
    )
    if not chunks.exists():
        indexing_progress.finish_document(job, document, DOCUMENT_STATUS_SKIPPED)
        # A document without text has nothing left to wait for: what the job
        # asked for happens here too.
        remove_document_file(job, document)
        return DOCUMENT_STATUS_SKIPPED

    controller = semantic_controller_for(document.collection, jsonl_config_path)
    controller.index_texts_from_list(
        all_texts=chunks, collection=document.collection
    )

    indexing_progress.finish_document(job, document, DOCUMENT_STATUS_INDEXED)
    # The embeddings are in the vector database, so the file from which they
    # were made is not needed anymore.
    remove_document_file(job, document)
    return DOCUMENT_STATUS_INDEXED


def job_document_ids(job: IndexingJob) -> list:
    """Documents of the job, in the order in which they were stored."""
    return list(job.documents.order_by("pk").values_list("pk", flat=True))


def count_job_chunks(job: IndexingJob) -> int:
    """Text chunks which were stored for the documents of ``job``."""
    return DocumentPageText.objects.filter(page__document__indexing_job=job).count()


def run_job(
    job: IndexingJob,
    jsonl_config_path: str = MILVUS_CONFIG_PATH,
    number_of_process: int = DB_INDEXING_PROCESS_COUNT,
) -> IndexingJob:
    """Whole pipeline of one job, in the current process (no Celery involved)."""
    run_prepare(job, number_of_process=number_of_process)
    document_ids = job_document_ids(job)
    job.refresh_from_db()
    for document_id in document_ids:
        document = Document.objects.get(pk=document_id)
        run_index_document(job, document, jsonl_config_path)
        job.refresh_from_db()
    return job
