from functools import wraps
from collections.abc import Mapping
from rest_framework.request import Request

from sse_api.core.errors import error_response
from sse_api.core.errors_list import (
    NO_REQUIRED_PARAMS,
    UNSUPPORTED_LANGUAGE,
    INVALID_PARAMS,
)
from sse_api.core.constants import default_app_language, AVAILABLE_LANGUAGES
from sse_api.core.validation import request_params


def required_params_exists(required_params: list, optional_params: list = None):
    """
    Decorator to check if required parameters are passed
    :param required_params: List of names of required params
    :param optional_params: List of names of optional params
    """

    def _check_required_params_wrap(method):
        @wraps(method)
        def _check_required_params(self, *method_args, **method_kwargs):
            request = method_args[0]
            params = request_params(request)
            use_language = params.get("lang", default_app_language())
            if (
                not isinstance(use_language, str)
                or use_language not in AVAILABLE_LANGUAGES
            ):
                return error_response(UNSUPPORTED_LANGUAGE, default_app_language())
            if request.method != "GET" and not isinstance(request.data, Mapping):
                return error_response(INVALID_PARAMS, use_language)
            not_given_params = []
            for param in required_params:
                if param not in params or params[param] is None:
                    not_given_params.append(param)
                elif isinstance(params[param], str) and not params[param].strip():
                    not_given_params.append(param)
            if len(not_given_params):
                return error_response(
                    error_name=NO_REQUIRED_PARAMS,
                    language=use_language,
                    required_params=required_params,
                    optional_params=optional_params,
                    not_given_params=not_given_params,
                )
            try:
                for key, validator in getattr(self, "input_validators", {}).items():
                    if key in params:
                        params[key] = validator(params[key])
            except (ValueError, TypeError):
                return error_response(INVALID_PARAMS, use_language)
            request.validated_params = params
            return method(self, *method_args, **method_kwargs)

        return _check_required_params

    return _check_required_params_wrap


def get_default_language(method):
    """
    Returns language given as request parameter
    :return: Language / default language
    """

    @wraps(method)
    def _get_default_language(self, *method_args, **method_kwargs) -> str:
        default_lang = default_app_language()
        for req_arg in method_args:
            if isinstance(req_arg, Request):
                default_lang = request_params(req_arg).get(
                    "lang", default_app_language()
                )
                if (
                    not isinstance(default_lang, str)
                    or default_lang not in AVAILABLE_LANGUAGES
                ):
                    return error_response(
                        error_name=UNSUPPORTED_LANGUAGE,
                        language=default_app_language(),
                    )
                if req_arg.method != "GET" and not isinstance(req_arg.data, Mapping):
                    return error_response(INVALID_PARAMS, default_lang)
        return method(self, default_lang, *method_args, **method_kwargs)

    return _get_default_language
