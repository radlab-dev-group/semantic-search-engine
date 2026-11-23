"""
Defaults of the chunking of a collection.

A collection decides once how its texts are split; the values below are used
when a collection is created without saying otherwise.  They are also what a
UI can prefill its form with, see the ``filtering_options`` endpoint.
"""

import os

from sse_api.core.constants import CONFIG_DIR

#: Key of the chunking settings inside ``CollectionOfDocuments.metadata_json``.
CHUNKING_METADATA_KEY = "chunking"

#: Key of the strict-chunking flag inside the same dictionary.
STRICT_CHUNKING_METADATA_KEY = "strict_chunking"

#: Sub-keys of the chunking settings, named exactly like in ``indexing_options``.
MAX_TOKENS_IN_CHUNK_KEY = "max_tokens_in_chunk"
NUMBER_OF_OVERLAP_TOKENS_KEY = "number_of_overlap_tokens"

#: Default size of a text chunk, in tokens.
DEFAULT_MAX_TOKENS_IN_CHUNK = 400

#: Default number of tokens repeated between two neighbouring chunks.
DEFAULT_NUMBER_OF_OVERLAP_TOKENS = 60

#: Upper bound of a chunk when the embedder does not declare its own limit.
#: It only guards against typos such as ``500000``; a model which declares
#: ``max_model_length`` in ``configs/embedders.json`` is always stricter.
MAX_TOKENS_IN_CHUNK_LIMIT = 100000

# ---------------------------------------------------------------------------
# Domyślne hiperparametry wyszukiwania i generacji per kolekcja.
#
# Kolekcja może trwale zapisać w ``metadata_json`` swój profil
# wyszukiwania (``default_search_options``) i profil generacji
# (``default_generation_options``); poniższe wartości są tym, co silnik
# podkłada, gdy kolekcja czegoś nie deklaruje.
# ---------------------------------------------------------------------------

#: Klucz opcji wyszukiwania w ``CollectionOfDocuments.metadata_json``.
SEARCH_OPTIONS_METADATA_KEY = "default_search_options"

#: Klucz opcji generacji (model + parametry LLM) w tym samym słowniku.
GENERATION_OPTIONS_METADATA_KEY = "default_generation_options"

#: Domyślne hiperparametry wyszukiwania kolekcji.
GLOBAL_DEFAULT_SEARCH_OPTIONS = {
    "max_results": 40,
    "hybrid_search": True,
    "rerank_results": False,
    "rerank_max_results": 80,
    "dense_weight": 0.6,
    "lexical_weight": 0.4,
    "rrf_k": 60,
    "min_similarity": None,
    "max_chunks_per_document": 2,
}

#: Domyślne parametry generacji odpowiedzi RAG kolekcji.
GLOBAL_DEFAULT_GENERATION_OPTIONS = {
    "generative_model": "",
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "top_p": 0.99,
    "top_k": 50,
    "typical_p": 1.0,
    "max_new_tokens": 1024,
    "percentage_rank_mass": 40,
    "system_prompt": None,
    "answer_language": "pl",
    "translate_answer": False,
}

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
MILVUS_CONFIG_PATH = os.path.join(CONFIG_DIR, "milvus_config.json")

#: Documents of one indexing request reported by ``api/indexing_status``.
MAX_STATUS_DOCUMENTS = 200

# ---------------------------------------------------------------------------
# Preview of a whole document (``api/document_content``).
#
# The text of a document is read page by page, so an answer about a huge
# document is cut after a window of pages and the client may ask for the next
# one (``from_page``) instead of receiving everything at once.
# ---------------------------------------------------------------------------

#: Pages of the document returned by one request, when the client asks for
#: none (and the largest ``max_pages`` the server accepts).
PREVIEW_MAX_PAGES = 200

#: Hard ceiling of ``max_pages``; a client may ask for less, not for more.
PREVIEW_MAX_PAGES_LIMIT = 1000
