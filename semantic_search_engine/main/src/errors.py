from rest_framework.response import Response

from main.src.errors_list import BUILT_IN_GENERAL_ERRORS
from main.src.errors_constants import (
    MSG,
    MSG_PL,
    MSG_EN,
    ECODE,
    RPARAMS,
    OPARAMS,
    NGPARAMS,
)

# List of errors must be updated into modules!
ALL_ERRORS = []
ERRORS_MAP = {}


def build_errors_map():
    for err in ALL_ERRORS + [BUILT_IN_GENERAL_ERRORS]:
        if err is None:
            continue
        for error_code_name, error in err.items():
            ERRORS_MAP[error_code_name] = error
    if not len(ERRORS_MAP):
        raise Exception("No errors found! Cannot build errors map!")


def get_error(
    error_name: str,
    language: str,
    required_params: list = None,
    optional_params: list = None,
    not_given_params: list = None,
) -> dict:
    """
    Simple error getter, return error as wrapped dictionary
    :param error_name:
    :param language:
    :param required_params:
    :param optional_params:
    :param not_given_params:
    :return: Single error as the dictionary
    """
    error = ERRORS_MAP[error_name].copy()
    error[MSG] = error[MSG][language]
    if required_params is not None:
        error[RPARAMS] = required_params
    if optional_params is not None and len(optional_params):
        error[OPARAMS] = optional_params
    if not_given_params is not None and len(not_given_params):
        error[NGPARAMS] = not_given_params
    return error


def error_response(
    error_name: str,
    language: str,
    not_given_params: list = None,
    required_params: list = None,
    optional_params: list = None,
):
    """
    Simple method to wrap any error to error message in given language
    """
    if error_name is None:
        return Response({"status": False})

    if not len(ERRORS_MAP):
        build_errors_map()

    edict = {
        "status": False,
        "errors": [
            get_error(
                error_name=error_name,
                language=language,
                required_params=required_params,
                optional_params=optional_params,
                not_given_params=not_given_params,
            )
        ],
    }
    return Response(edict)

    #
    # m_error = ERRORS_MAP[error_name]
    # error_json = {"code": m_error["code"], "msg": m_error["msg"][language]}
    #
    # if required_params is not None and len(required_params):
    #     error_json["required_params"] = ", ".join(required_params)
    #
    # return Response({"status": False, "errors": [error_json]})
