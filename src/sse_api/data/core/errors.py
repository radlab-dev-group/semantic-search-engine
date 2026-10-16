from sse_api.core.errors import ALL_ERRORS, MSG, MSG_PL, MSG_EN, ECODE

UPLOAD_REJECTED = "UPLOAD_REJECTED"
DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
DELETE_PROTECTED = "DELETE_PROTECTED"
CHUNKING_TOO_LONG = "CHUNKING_TOO_LONG"
CHUNKING_MISMATCH = "CHUNKING_MISMATCH"
INDEXING_JOB_NOT_FOUND = "INDEXING_JOB_NOT_FOUND"

ALL_ERRORS_DATA = {
    DOCUMENT_NOT_FOUND: {
        ECODE: "000002_DATA",
        MSG: {
            MSG_PL: "Nie znaleziono dokumentów o podanych nazwach.",
            MSG_EN: "Documents with the given names were not found.",
        },
    },
    CHUNK_NOT_FOUND: {
        ECODE: "000004_DATA",
        MSG: {
            MSG_PL: "Nie znaleziono tekstów o podanych identyfikatorach.",
            MSG_EN: "Texts with the given ids were not found.",
        },
    },
    DELETE_PROTECTED: {
        ECODE: "000003_DATA",
        MSG: {
            MSG_PL: "Nie można usunąć: inne dane są powiązane z usuwanymi.",
            MSG_EN: "Cannot delete: other data are related to the deleted ones.",
        },
    },
    CHUNKING_TOO_LONG: {
        ECODE: "000005_DATA",
        MSG: {
            MSG_PL: (
                "Chunk jest dłuższy niż maksymalna długość wejścia embedera "
                "użytego w kolekcji; teksty byłyby ucinane bez ostrzeżenia."
            ),
            MSG_EN: (
                "The chunk is longer than the maximum input length of the "
                "embedder of the collection; texts would be truncated silently."
            ),
        },
    },
    CHUNKING_MISMATCH: {
        ECODE: "000006_DATA",
        MSG: {
            MSG_PL: (
                "Parametry fragmentowania są niezgodne z ustawionymi dla kolekcji "
                "(kolekcja ma włączony rygorystyczny podział tekstów)."
            ),
            MSG_EN: (
                "Chunking settings differ from the ones of the collection "
                "(the collection enforces its own chunking)."
            ),
        },
    },
    INDEXING_JOB_NOT_FOUND: {
        ECODE: "000007_DATA",
        MSG: {
            MSG_PL: "Nie znaleziono takiego procesu indeksacji.",
            MSG_EN: "This indexing process was not found.",
        },
    },
    UPLOAD_REJECTED: {
        ECODE: "000001_DATA",
        MSG: {
            MSG_PL: "Odrzucono pliki: niepoprawne archiwum, ścieżki lub przekroczone limity.",
            MSG_EN: "Upload rejected: invalid archive, unsafe paths or exceeded limits.",
        },
    },
}

ALL_ERRORS += [ALL_ERRORS_DATA]
