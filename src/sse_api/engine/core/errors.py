from sse_api.core.errors import ALL_ERRORS
from sse_api.core.errors_constants import MSG, MSG_PL, MSG_EN, ECODE

COLLECTION_ACCESS_DENIED = "COLLECTION_ACCESS_DENIED"
RESPONSE_ACCESS_DENIED = "RESPONSE_ACCESS_DENIED"
ANSWER_ACCESS_DENIED = "ANSWER_ACCESS_DENIED"
GENERATION_FAILED = "GENERATION_FAILED"
QUERY_NOT_FOUND = "QUERY_NOT_FOUND"
DELETE_PROTECTED = "DELETE_PROTECTED"

ENGINE_ERRORS = {
    COLLECTION_ACCESS_DENIED: {
        ECODE: "engine_001",
        MSG: {
            MSG_PL: "Kolekcja nie istnieje lub brak dostępu!",
            MSG_EN: "Collection not found or access denied!",
        },
    },
    RESPONSE_ACCESS_DENIED: {
        ECODE: "engine_002",
        MSG: {
            MSG_PL: "Odpowiedź nie istnieje lub brak dostępu!",
            MSG_EN: "Response not found or access denied!",
        },
    },
    ANSWER_ACCESS_DENIED: {
        ECODE: "engine_003",
        MSG: {
            MSG_PL: "Odpowiedź nie istnieje lub brak dostępu!",
            MSG_EN: "Answer not found or access denied!",
        },
    },
    GENERATION_FAILED: {
        ECODE: "engine_004",
        MSG: {
            MSG_PL: "Błąd modelu lub konfiguracji!",
            MSG_EN: "Model or configuration error occurred!",
        },
    },
    QUERY_NOT_FOUND: {
        ECODE: "engine_005",
        MSG: {
            MSG_PL: "Nie znaleziono zapytań o podanych identyfikatorach.",
            MSG_EN: "Queries with the given ids were not found.",
        },
    },
    DELETE_PROTECTED: {
        ECODE: "engine_006",
        MSG: {
            MSG_PL: "Nie można usunąć: inne dane są powiązane z usuwanymi.",
            MSG_EN: "Cannot delete: other data are related to the deleted ones.",
        },
    },
}

ALL_ERRORS += [ENGINE_ERRORS]
