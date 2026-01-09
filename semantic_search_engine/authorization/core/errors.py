from main.src.errors import ALL_ERRORS
from main.src.errors_constants import MSG, ECODE, MSG_EN, MSG_PL

STATE_GEN_PROBLEM = "STATE_GEN_PROBLEM"
LOGIN_URL_GEN_PROBLEM = "LOGIN_URL_GEN_PROBLEM"
TOKEN_LOGIN_PROBLEM = "TOKEN_LOGIN_PROBLEM"
TOKEN_REFRESH_PROBLEM = "REFRESH_TOKEN_PROBLEM"
TOKEN_DISABLE_FAILED = "TOKEN_DISABLE_FAILED"
NO_USER_FOR_TOKEN = "NO_USER_FOR_TOKEN"

error_str = "authorization"

RDL_AUTH_ERRORS = {
    STATE_GEN_PROBLEM: {
        ECODE: f"e__{error_str}_001",
        MSG: {
            MSG_PL: "Wystąpił problem podczas generowanie state!",
            MSG_EN: "Error during state generation",
        },
    },
    LOGIN_URL_GEN_PROBLEM: {
        ECODE: f"e__{error_str}_002",
        MSG: {
            MSG_PL: "Błąd podczas generowania linku do logowania!",
            MSG_EN: "Error during login ling generation!",
        },
    },
    TOKEN_LOGIN_PROBLEM: {
        ECODE: f"e__{error_str}_003",
        MSG: {
            MSG_PL: "Problem podczas uzyskiwania tokenu (grant)",
            MSG_EN: "Problem during token accessing (grant)",
        },
    },
    NO_USER_FOR_TOKEN: {
        ECODE: f"e__{error_str}_004",
        MSG: {
            MSG_PL: "Nie można odnaleźć tokenu dla użytkownika!",
            MSG_EN: "Cannot find user for token!",
        },
    },
    TOKEN_REFRESH_PROBLEM: {
        ECODE: f"e__{error_str}_005",
        MSG: {
            MSG_PL: "Błąd podczas odświeżania tokenu!",
            MSG_EN: "Problem during token refreshing",
        },
    },
    TOKEN_DISABLE_FAILED: {
        ECODE: f"e__{error_str}_006",
        MSG: {
            MSG_PL: "Błąd podczas usuwania tokenu!",
            MSG_EN: "Error during token disabling!",
        },
    },
}

ALL_ERRORS += [RDL_AUTH_ERRORS]
