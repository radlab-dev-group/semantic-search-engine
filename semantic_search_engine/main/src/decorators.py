from functools import wraps
from rest_framework.request import Request

from main.src.errors import error_response
from main.src.errors_list import NO_REQUIRED_PARAMS, UNSUPPORTED_LANGUAGE
from main.src.constants import default_app_language, AVAILABLE_LANGUAGES


def required_params_exists(required_params: list, optional_params: list = None):
    """
    Decorator to check if required parameters are passed
    :param required_params: List of names of required params
    :param optional_params: List of names of optional params
    """

    def _check_required_params_wrap(method):
        @wraps(method)
        def _check_required_params(self, *method_args, **method_kwargs):
            not_given_params = []
            for param in required_params:
                if param not in method_args[0].data:
                    not_given_params.append(param)
                else:
                    value = str(method_args[0].data[param]).strip()
                    if not len(value):
                        not_given_params.append(param)
            if len(not_given_params):
                if "lang" in method_args[0].data:
                    use_language = method_args[0].data["lang"].strip()
                    if use_language not in AVAILABLE_LANGUAGES:
                        return error_response(
                            error_name=UNSUPPORTED_LANGUAGE,
                            language=default_app_language(),
                        )
                    if use_language is None or not len(use_language):
                        use_language = default_app_language()
                else:
                    use_language = default_app_language()
                return error_response(
                    error_name=NO_REQUIRED_PARAMS,
                    language=use_language,
                    required_params=required_params,
                    optional_params=optional_params,
                    not_given_params=not_given_params,
                )
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
            if type(req_arg) is Request:
                default_lang = req_arg.data.get("lang", default_app_language())
                if default_lang not in AVAILABLE_LANGUAGES:
                    return error_response(
                        error_name=UNSUPPORTED_LANGUAGE,
                        language=default_app_language(),
                    )
        return method(self, default_lang, *method_args, **method_kwargs)

    return _get_default_language
