from sse_api.core.errors import ALL_ERRORS, MSG, MSG_PL, MSG_EN, ECODE

UPLOAD_REJECTED = "UPLOAD_REJECTED"
DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
DELETE_PROTECTED = "DELETE_PROTECTED"

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
    UPLOAD_REJECTED: {
        ECODE: "000001_DATA",
        MSG: {
            MSG_PL: "Odrzucono pliki: niepoprawne archiwum, ścieżki lub przekroczone limity.",
            MSG_EN: "Upload rejected: invalid archive, unsafe paths or exceeded limits.",
        },
    },
}

ALL_ERRORS += [ALL_ERRORS_DATA]
