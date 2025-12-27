from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken

from main.src.errors import error_response
from main.src.errors_list import NO_LOGIN_PARAMS
from main.src.constants import default_app_language
from main.src.decorators import required_params_exists

DEFAULT_APP_LANGUAGE = default_app_language()


class OrganisationLogin(ObtainAuthToken):
    """
    Main endpoint to organisation login
    ```{
      "username": "user",
      "password": "User0123",
    }```
    """

    required_params = ["username", "password"]

    @required_params_exists(required_params=required_params)
    def post(self, request, *args, **kwargs):
        username = request.data.get("username", False)
        password = request.data.get("password", False)
        if not username or not password:
            return error_response(
                error_name=NO_LOGIN_PARAMS, language=DEFAULT_APP_LANGUAGE
            )
        serializer = self.get_serializer(
            data={"username": username, "password": password}
        )
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        token, created = Token.objects.get_or_create(user=user)
        return Response({"token": token.key})
