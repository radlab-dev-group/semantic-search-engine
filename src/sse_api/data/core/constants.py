"""
Defaults of the chunking of a collection.

A collection decides once how its texts are split; the values below are used
when a collection is created without saying otherwise.  They are also what a
UI can prefill its form with, see the ``filtering_options`` endpoint.
"""

#: Key of the chunking settings inside ``CollectionOfDocuments.metadata_json``.
CHUNKING_METADATA_KEY = "chunking"

#: Key of the strict-chunking flag inside the same dictionary.
STRICT_CHUNKING_METADATA_KEY = "strict_chunking"

#: Sub-keys of the chunking settings, named exactly like in ``indexing_options``.
MAX_TOKENS_IN_CHUNK_KEY = "max_tokens_in_chunk"
NUMBER_OF_OVERLAP_TOKENS_KEY = "number_of_overlap_tokens"

#: Default size of a text chunk, in tokens.
DEFAULT_MAX_TOKENS_IN_CHUNK = 500

#: Default number of tokens repeated between two neighbouring chunks.
DEFAULT_NUMBER_OF_OVERLAP_TOKENS = 0

#: Upper bound of a chunk when the embedder does not declare its own limit.
#: It only guards against typos such as ``500000``; a model which declares
#: ``max_model_length`` in ``configs/embedders.json`` is always stricter.
MAX_TOKENS_IN_CHUNK_LIMIT = 100000

# ---------------------------------------------------------------------------
# Statusy kolejki indeksacji.
#
# Wdrożenie może indeksować w wątku żądania (``use_celery=false``) albo przez
# kolejkę Celery.  Niezależnie od trybu postęp jest opisywany tymi samymi
# wartościami, tak aby jeden klient (UI) obsłużył oba tryby.
# ---------------------------------------------------------------------------

#: Rodzaje zgłoszenia indeksacji.
INDEXING_JOB_FILES = "files"
INDEXING_JOB_TEXTS = "texts"
INDEXING_JOB_KINDS = (INDEXING_JOB_FILES, INDEXING_JOB_TEXTS)

#: Stona, w których job nigdy się nie zmieni.
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PREPARING = "preparing"
JOB_STATUS_INDEXING = "indexing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_PARTIAL = "partial"
JOB_STATUS_FAILED = "failed"
JOB_STATUSES = (
    JOB_STATUS_QUEUED,
    JOB_STATUS_PREPARING,
    JOB_STATUS_INDEXING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_PARTIAL,
    JOB_STATUS_FAILED,
)
JOB_TERMINAL_STATUSES = (JOB_STATUS_COMPLETED, JOB_STATUS_PARTIAL, JOB_STATUS_FAILED)

#: Statusy pojedynczego dokumentu w trakcie indeksacji.
DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_INDEXING = "indexing"
DOCUMENT_STATUS_INDEXED = "indexed"
DOCUMENT_STATUS_FAILED = "failed"
DOCUMENT_STATUS_SKIPPED = "skipped"
DOCUMENT_STATUSES = (
    DOCUMENT_STATUS_PENDING,
    DOCUMENT_STATUS_INDEXING,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_SKIPPED,
)

#: Jobe i dokumenty starsze niż tyle minut, a wciąż w fazie czynnej, są
#: traktowane jako porzucone (worker został zabity w połowie pracy).
DEFAULT_INDEXING_STALE_MINUTES = 60

#: Processes used to read the files of one upload into the relational database.
DB_INDEXING_PROCESS_COUNT = 5

#: Vector database configuration of the collections.
MILVUS_CONFIG_PATH = "./configs/milvus_config.json"

#: Documents of one indexing request reported by ``api/indexing_status``.
MAX_STATUS_DOCUMENTS = 200
