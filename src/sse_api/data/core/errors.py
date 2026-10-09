from sse_api.core.errors import ALL_ERRORS, MSG, MSG_PL, MSG_EN, ECODE


UPLOAD_REJECTED = "UPLOAD_REJECTED"

ALL_ERRORS_DATA = {
    UPLOAD_REJECTED: {
        ECODE: "000001_DATA",
        MSG: {
            MSG_PL: "Odrzucono pliki: niepoprawne archiwum, ścieżki lub przekroczone limity.",
            MSG_EN: "Upload rejected: invalid archive, unsafe paths or exceeded limits.",
        },
    },
}

ALL_ERRORS += [ALL_ERRORS_DATA]