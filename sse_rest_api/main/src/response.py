from rest_framework.response import Response

from main.src.errors import error_response


def response_with_status(
    status: bool, language: str, error_name: str = None, response_body: dict = None
) -> Response:
    if status is False:
        return error_response(error_name=error_name, language=language)

    return Response({"status": True, "body": response_body})
