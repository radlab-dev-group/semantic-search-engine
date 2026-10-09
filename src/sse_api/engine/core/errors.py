from sse_api.core.errors import ALL_ERRORS
from sse_api.core.errors_constants import MSG, MSG_PL, MSG_EN, ECODE

COLLECTION_ACCESS_DENIED = "COLLECTION_ACCESS_DENIED"
RESPONSE_ACCESS_DENIED = "RESPONSE_ACCESS_DENIED"
ANSWER_ACCESS_DENIED = "ANSWER_ACCESS_DENIED"
GENERATION_FAILED = "GENERATION_FAILED"

ENGINE_ERRORS = {
    COLLECTION_ACCESS_DENIED: {ECODE: "engine_001", MSG: {
        MSG_PL: "Kolekcja nie istnieje lub brak dostępu!",
        MSG_EN: "Collection not found or access denied!"}},
    RESPONSE_ACCESS_DENIED: {ECODE: "engine_002", MSG: {
        MSG_PL: "Odpowiedź nie istnieje lub brak dostępu!",
        MSG_EN: "Response not found or access denied!"}},
    ANSWER_ACCESS_DENIED: {ECODE: "engine_003", MSG: {
        MSG_PL: "Odpowiedź nie istnieje lub brak dostępu!",
        MSG_EN: "Answer not found or access denied!"}},
    GENERATION_FAILED: {ECODE: "engine_004", MSG: {
        MSG_PL: "Błąd modelu lub konfiguracji!",
        MSG_EN: "Model or configuration error occurred!"}},
}

ALL_ERRORS += [ENGINE_ERRORS]