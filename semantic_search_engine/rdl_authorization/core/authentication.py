from main.src.constants import get_logger

from rdl_authorization.models import Token
from rdl_authorization.core.handlers import (
    RdlAuthUserTokenHandler,
    RdlAuthGrantAccTokenHandler,
)

DEFAULT_AUTH_TYPES = ["Token", "Bearer"]
DEFAULT_PATH_INFO_KEY = "PATH_INFO"
DEFAULT_AUTH_KEY = "HTTP_AUTHORIZATION"
URLS_DONT_PANIC_PATH_INFO = ["login", "logout", "login_url"]


class GATokenAuthentication:
    model = None

    def __init__(self):
        self._logger = get_logger()
        self.token_handler = RdlAuthGrantAccTokenHandler()

    def get_token_model(self):
        if self.model is not None:
            return self.model
        return Token

    def authenticate(self, request):
        return self.authenticate_credentials(request)

    def authenticate_credentials(self, request):
        """
        Adds user to the request when authorized user is found in the session
        :param django.http.request.HttpRequest request: django request
        """
        token_authentication = self.__parse_auth_header_for_token(request)
        if not token_authentication:
            return self.__no_authenticated_user()

        user_token = self.__find_user_token_for_token_str(token_authentication)
        if not user_token:
            return self.__no_authenticated_user()

        decoded_token = self.token_handler.verify_and_decode_token(
            token=user_token, token_str=None
        )
        if decoded_token is None or not len(decoded_token):
            return self.__no_authenticated_user()

        if not self.__is_user_role_ok(decoded_token=decoded_token):
            self._logger.error(
                f"User {user_token.auth_user.username} "
                f"is not granted to application access!"
            )
            return self.__no_authenticated_user()

        self._logger.info(f"User {user_token.auth_user.username} is authenticated!")
        return user_token.auth_user, token_authentication

    def get_user_token_for_request(self, request):
        token_from_request = self.__parse_auth_header_for_token(request)
        if token_from_request is not None and len(token_from_request.strip()):
            return self.__find_user_token_for_token_str(token_str=token_from_request)

        session_state_str = self.__parse_request_for_token_session_state(request)
        return self.__find_user_token_for_state(session_state_str)

    def introspect_token(self, request):
        token_from_request = self.__parse_auth_header_for_token(request)
        if token_from_request is None or not len(token_from_request.strip()):
            return self.__no_authenticated_user()
        return self.token_handler.introspect_user_token(
            user_token=token_from_request
        )

    def __is_user_role_ok(self, decoded_token: dict) -> bool:
        kc_config = self.token_handler.rdl_auth_state_handler.rdl_auth_config
        if "*" in kc_config.accepted_user_roles:
            return True

        scope = kc_config.user_role_main_scope
        user_scope = decoded_token.get(scope, None)
        if user_scope is None:
            return False

        variable = kc_config.user_role_scope_variable
        user_variable = user_scope.get(variable, None)
        if user_variable is None:
            return False

        for user_role in user_variable.get("roles", []):
            if user_role in kc_config.accepted_user_roles:
                return True
        return False

    @staticmethod
    def __parse_request_for_token_session_state(request) -> str | None:
        request_data = None
        if len(request.POST):
            request_data = request.POST
        elif len(request.GET):
            request_data = request.GET
        if request_data is None:
            return None
        if "session_state" not in request_data:
            return None
        return request_data["session_state"]

    @staticmethod
    def __parse_auth_header_for_token(request):
        if request.headers is None or not len(request.headers):
            return None

        auth_header = request.headers.get("Authorization", None)
        if auth_header is None or not len(auth_header):
            return None

        auth_header_spl = auth_header.split()
        if not len(auth_header_spl) == 2:
            return None

        auth_type = auth_header_spl[0].lower()
        found_auth_type = False
        for a_type in DEFAULT_AUTH_TYPES:
            if a_type.lower() == auth_type:
                found_auth_type = True
                break

        if not found_auth_type:
            return None

        auth_token = auth_header_spl[1]
        return auth_token

    @staticmethod
    def __find_user_token_for_state(state_str: str | None):
        if state_str is None or not len(state_str.strip()):
            return None
        return RdlAuthUserTokenHandler.find_active_token_for_session_state(
            session_state=state_str
        )

    @staticmethod
    def __find_user_token_for_token_str(token_str: str | None):
        if token_str is None or not len(token_str):
            return None
        return RdlAuthUserTokenHandler.find_active_token_for_token_str(
            token_str=token_str
        )

    @staticmethod
    def __check_request_header(request):
        path_info = request.META.get(DEFAULT_PATH_INFO_KEY, None)
        path_info = path_info.lower() if path_info else None
        if path_info is None:
            return False
        for url in URLS_DONT_PANIC_PATH_INFO:
            if url.lower() in path_info:
                return True
        return False

    @staticmethod
    def __no_authenticated_user():
        """
        Default empty response for bad authentication
        """
        return None, ""
