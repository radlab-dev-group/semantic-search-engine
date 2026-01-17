from main.src.errors import ALL_ERRORS, MSG, MSG_PL, MSG_EN, ECODE
from system.core.constants import ERROR_MARK_SYSTEM


GROUP_NAME_NOT_EXIST = "GROUP_NAME_NOT_EXIST"


ALL_ERRORS_DATA = {
    GROUP_NAME_NOT_EXIST: {
        ECODE: f"000001_{ERROR_MARK_SYSTEM}",
        MSG: {
            MSG_PL: "Podana grupa nie istnieje!",
            MSG_EN: "Given group name does not exist!",
        },
    },
}

ALL_ERRORS += [ALL_ERRORS_DATA]
