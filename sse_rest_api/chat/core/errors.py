from main.src.errors import ALL_ERRORS, MSG, MSG_PL, MSG_EN, ECODE
from chat.core.constants import ERROR_MARK_CHAT


COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
NOT_SUPPORTED_SYSTEM_ROLE = "NOT_SUPPORTED_SYSTEM_ROLE"
CHAT_ID_NOT_FOUND = "CHAT_ID_NOT_FOUND"
USER_DENIED_TO_CHAT = "USER_DENIED_TO_CHAT"
CANNOT_ADD_MESSAGE_CHAT_RO = "CANNOT_ADD_MESSAGE_CHAT_RO"


ALL_ERRORS_CHATS = {
    NOT_SUPPORTED_SYSTEM_ROLE: {
        ECODE: f"000001_{ERROR_MARK_CHAT}",
        MSG: {
            MSG_PL: "Podana rola nie jest obsługiwana!",
            MSG_EN: "Given role is not supported!",
        },
    },
    COLLECTION_NOT_FOUND: {
        ECODE: f"000002_{ERROR_MARK_CHAT}",
        MSG: {
            MSG_PL: "Nie odnaleziono kolekcji!",
            MSG_EN: "Given collection does not exist!",
        },
    },
    CHAT_ID_NOT_FOUND: {
        ECODE: f"000003_{ERROR_MARK_CHAT}",
        MSG: {
            MSG_PL: "Nie odnaleziono czatu o przekazanym idetyfikatorze!",
            MSG_EN: "Chat with given id is not found!",
        },
    },
    USER_DENIED_TO_CHAT: {
        ECODE: f"000004_{ERROR_MARK_CHAT}",
        MSG: {
            MSG_PL: "Nie masz uprawnień do odczytania czatu!",
            MSG_EN: "You don't have permission to read the chat!",
        },
    },
    CANNOT_ADD_MESSAGE_CHAT_RO: {
        ECODE: f"000004_{ERROR_MARK_CHAT}",
        MSG: {
            MSG_PL: "Nie możesz dodać wiadomości do czatu tylko do odczytu!",
            MSG_EN: "You cannot add messages to this chat because is read only!",
        },
    },
}

ALL_ERRORS += [ALL_ERRORS_CHATS]
