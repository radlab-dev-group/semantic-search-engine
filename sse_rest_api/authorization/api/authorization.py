from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from authorization.core.errors import (
    STATE_GEN_PROBLEM,
    LOGIN_URL_GEN_PROBLEM,
    TOKEN_LOGIN_PROBLEM,
    NO_USER_FOR_TOKEN,
    TOKEN_REFRESH_PROBLEM,
    TOKEN_DISABLE_FAILED,
)

from main.src.constants import get_logger
from main.src.response import response_with_status
from main.src.decorators import get_default_language, required_params_exists

from authorization.core.handlers import RdlAuthStateHandler
from authorization.core.handlers import RdlAuthGrantAccTokenHandler


class GenerateLoginUrl(APIView):
    state_handler = RdlAuthStateHandler()
    permission_classes = (AllowAny,)

    @get_default_language
    def post(self, language, request):
        state = self.state_handler.generate_state()
        if not state:
            return response_with_status(
                status=False, error_name=STATE_GEN_PROBLEM, language=language
            )

        login_url = self.state_handler.generate_login_url(state)
        if not login_url or not len(login_url):
            return response_with_status(
                status=False, error_name=LOGIN_URL_GEN_PROBLEM, language=language
            )

        return response_with_status(
            status=True,
            response_body={"login_url": login_url},
            language=language,
        )


class CreateRdlAuthToken(APIView):
    """
    Based on the given grant code, request for access token
    """

    required_params = ["code", "state"]
    optional_params = ["session_state"]

    permission_classes = (AllowAny,)

    logger = get_logger()
    token_handler = RdlAuthGrantAccTokenHandler()

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def post(self, language, request):
        grant_code = request.data.get("code")
        state_str = request.data.get("state")
        session_state = request.data.get("session_state", None)

        self.logger.debug(f"Trying to authenticate user with state: {state_str}")
        token_str, token_opts, errors = self.token_handler.get_token_with_options(
            grant_code=grant_code, refresh_token=None
        )
        if len(errors) or token_str is None or not len(token_str.strip()):
            for error in errors:
                self.logger.error(f"Error during login: {error}")
            return response_with_status(
                status=False, error_name=TOKEN_LOGIN_PROBLEM, language=language
            )

        user, user_token = self.token_handler.get_rdl_auth_user_for_token(
            token_str, state_str, session_state, grant_code, token_opts
        )
        if user is None or user_token is None:
            return response_with_status(
                status=False, error_name=NO_USER_FOR_TOKEN, language=language
            )

        return response_with_status(
            status=True,
            response_body={
                "token": user_token.acc_token,
                "refresh_token": user_token.refresh_token,
                "decoded_token": user_token.decoded_token,
            },
            language=language,
        )


class RefreshToken(APIView):
    required_params = ["refresh_token"]

    permission_classes = (AllowAny,)

    logger = get_logger()
    token_handler = RdlAuthGrantAccTokenHandler()

    @required_params_exists(required_params=required_params)
    @get_default_language
    def post(self, language, request):
        refresh_token = request.data.get("refresh_token")
        decoded_token = self.token_handler.verify_and_decode_token(
            token=None, token_str=refresh_token
        )
        if decoded_token is None:
            return response_with_status(
                status=False, error_name=TOKEN_REFRESH_PROBLEM, language=language
            )

        state = self.token_handler.rdl_auth_state_handler.generate_state()
        token_str, token_opts, errors = self.token_handler.get_token_with_options(
            grant_code=None, refresh_token=refresh_token
        )
        user, user_token = self.token_handler.get_rdl_auth_user_for_token(
            token_str=token_str,
            state_str=state.state,
            session_state="",
            grant_code="",
            token_opts=token_opts,
        )

        return response_with_status(
            status=True,
            response_body={
                "token": user_token.acc_token,
                "refresh_token": user_token.refresh_token,
                "decoded_token": user_token.decoded_token,
            },
            language=language,
        )


class DisableRdlAuthToken(APIView):
    logger = get_logger()
    token_handler = RdlAuthGrantAccTokenHandler()

    @get_default_language
    def post(self, language: str, request: Request) -> Response:
        if request.user.is_authenticated:
            try:
                self.token_handler.disable_all_user_tokens(request.user)
            except Exception as exc:
                self.logger.error(f"Failed to disable tokens: {exc}")
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=TOKEN_DISABLE_FAILED,
                )
        return response_with_status(status=True, language=language)


obtain_auth_token = CreateRdlAuthToken.as_view()
remove_auth_token = DisableRdlAuthToken.as_view()
