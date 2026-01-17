from main.src.errors_constants import MSG, MSG_PL, MSG_EN, ECODE

UER_NO_PERMIT = "UER_NO_PERMIT"
NO_REQUIRED_PARAMS = "NO_REQUIRED_PARAMS"
NO_LOGIN_PARAMS = "NO_LOGIN_PARAMS"
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"

BUILT_IN_GENERAL_ERRORS = {
    NO_LOGIN_PARAMS: {
        ECODE: f"e__login_400",
        MSG: {
            MSG_PL: "Nie podano wymaganych danych do logowania!",
            MSG_EN: "No login required data are given!",
        },
    },
    UER_NO_PERMIT: {
        ECODE: "e__data_500",
        MSG: {
            MSG_PL: "Nie posiadasz uprawnień do tej operacji!",
            MSG_EN: "You dont have permission to call this function!",
        },
    },
    NO_REQUIRED_PARAMS: {
        ECODE: "e__data_501",
        MSG: {
            MSG_PL: "Nie podano wymaganych parametrów!",
            MSG_EN: "No required params are given",
        },
    },
    UNSUPPORTED_LANGUAGE: {
        ECODE: "e__data_555",
        MSG: {
            MSG_PL: "Ten język nie jest wspierany!",
            MSG_EN: "This language is not supported",
        },
    },
}
