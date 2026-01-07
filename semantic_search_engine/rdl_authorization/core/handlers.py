import jwt
import json
import random
import string
import requests

from django.conf import settings
from django.contrib.auth.models import User

from main.src.constants import get_logger
from rdl_authorization.models import Token, SessionState
from rdl_authorization.utils.config import RdlAuthConfig


DEFAULT_AUTH = "Bearer"
DEFAULT_STATE_LENGTH = 16
DEFAULT_STATE_CHARACTERS = string.ascii_letters + string.digits

# =============================================================================
# Setting which depends on used authenticator (openid/oauth)
# Mapping from token field to application field
MAPPINGS_REQUIRED = {}
MAPPINGS_OPTIONAL = {"preferred_username": ("note",), "email": ("email",)}

if settings.SYSTEM_HANDLER.use_keycloak_authentication():
    MAPPINGS_REQUIRED["given_name"] = ("first_name",)
    MAPPINGS_REQUIRED["family_name"] = ("last_name",)

    DEFAULT_USER_ID = "email"
    DEFAULT_PATH = "{}/realms/{}/protocol/openid-connect"
    DEFAULT_AUTH_LOGIN_PATH = DEFAULT_PATH + (
        "/auth?client_id={}&response_type=code&state={}&redirect_uri={}"
    )
elif settings.SYSTEM_HANDLER.use_oauth_v1_authentication():
    DEFAULT_USER_ID = "unique_name"
    DEFAULT_PATH = "{}/{}/oauth2"
    DEFAULT_AUTH_LOGIN_PATH = DEFAULT_PATH + (
        "/authorize?client_id={}&response_type=code"
        "&state={}&redirect_uri={}&scope={}"
    )
elif settings.SYSTEM_HANDLER.use_oauth_v2_authentication():
    DEFAULT_USER_ID = "unique_name"
    DEFAULT_PATH = "{}/{}/oauth2/v2.0"
    DEFAULT_AUTH_LOGIN_PATH = DEFAULT_PATH + (
        "/authorize?client_id={}&response_type=code"
        "&state={}&redirect_uri={}&scope={}"
    )
else:
    raise Exception("KeyCloak or OAuth/OAuth V2 is required!")

MAPPINGS_REQUIRED[DEFAULT_USER_ID] = ("name",)

# =============================================================================

DEFAULT_AUTH_LOGOUT_PATH = DEFAULT_PATH + "/logout"
DEFAULT_TOKEN_REQUEST_PATH = DEFAULT_PATH + "/token"
DEFAULT_TOKEN_INTROSPECT_PATH = DEFAULT_TOKEN_REQUEST_PATH + "/introspect"

ADDITIONAL_TOKEN_OPTIONS = [
    "not-before-policy",
    "expires_in",
    "token_type",
    "refresh_expires_in",
    "refresh_token",
    "scope",
]


class RdlAuthUserTokenHandler(object):
    """
    Handler for RdlAuthToken model.
    """

    @staticmethod
    def find_active_token_for_session_state(
        session_state: str,
    ) -> Token | None:
        try:
            return Token.objects.get(
                is_active=True, state__session_state=session_state
            )
        except Token.DoesNotExist:
            return None

    @staticmethod
    def find_active_token_for_token_str(token_str: str) -> Token | None:
        try:
            return Token.objects.get(is_active=True, acc_token=token_str)
        except Token.DoesNotExist:
            return None


class RdlAuthStateHandler(object):
    def __init__(self, auth_config_path: str = "configs/auth-config.json"):
        random.seed(a=None)
        self.logger = get_logger()
        self.rdl_auth_config = RdlAuthConfig(cfg_path=auth_config_path)

    def generate_state(self) -> SessionState:
        while True:
            state_str = self.__generate_state_str()
            if not self.__exists_state_str(state_str):
                return self.__add_state(state_str)

    def generate_login_url(self, state: SessionState) -> str:
        if (
            settings.SYSTEM_HANDLER.use_oauth_v1_authentication()
            or settings.SYSTEM_HANDLER.use_oauth_v2_authentication()
        ):
            login_url = DEFAULT_AUTH_LOGIN_PATH.format(
                self.rdl_auth_config.auth_host,
                self.rdl_auth_config.realm,
                self.rdl_auth_config.client_id,
                state.state,
                self.rdl_auth_config.redirect_uri,
                self.rdl_auth_config.scope,
            )
        else:
            login_url = DEFAULT_AUTH_LOGIN_PATH.format(
                self.rdl_auth_config.auth_host,
                self.rdl_auth_config.realm,
                self.rdl_auth_config.client_id,
                state.state,
                self.rdl_auth_config.redirect_uri,
            )
        return login_url

    def generate_logout_url(self):
        logout_url = DEFAULT_AUTH_LOGOUT_PATH.format(
            self.rdl_auth_config.auth_host, self.rdl_auth_config.realm
        )
        return logout_url

    def get_state(self, state_str: str):
        if self.__exists_state_str(state_str):
            return SessionState.objects.get(state=state_str)
        return None

    def add_state_str(self, state_str: str):
        state = self.get_state(state_str=state_str)
        if state is None:
            return self.__add_state(state_str=state_str)
        return state

    @staticmethod
    def __generate_state_str() -> str:
        state_str = "".join(
            random.choice(DEFAULT_STATE_CHARACTERS)
            for i in range(DEFAULT_STATE_LENGTH)
        )
        return state_str

    @staticmethod
    def __exists_state_str(state_str: str) -> bool:
        return SessionState.objects.filter(state=state_str).exists()

    @staticmethod
    def __add_state(state_str: str) -> SessionState:
        return SessionState.objects.create(state=state_str)


class RdlAuthGrantAccTokenHandler(object):
    logger = get_logger()

    def __init__(self, auth_config_path: str = "configs/auth-config.json"):
        self.rdl_auth_state_handler = RdlAuthStateHandler(
            auth_config_path=auth_config_path
        )

    def get_token_with_options(
        self, grant_code: str | None, refresh_token: str | None
    ) -> (str | None, dict | None | list):
        self.logger.debug(f"Trying to authenticate with grant code: {grant_code}")
        acc_token, token_options, errors = (
            self.__request_rdl_auth_token_with_options(
                grant_code=grant_code, refresh_token=refresh_token
            )
        )
        return acc_token, token_options, errors

    def get_rdl_auth_user_for_token(
        self,
        token_str: str,
        state_str: str,
        session_state: str,
        grant_code: str,
        token_opts: dict,
    ) -> (User | None, Token | None):
        self.logger.debug("Preparing user for token: {token_str}")
        if not state_str or not len(state_str):
            self.logger.error("state_str is not given!")
            return None, None

        state = self.rdl_auth_state_handler.get_state(state_str=state_str)
        if not state:
            self.logger.error(f"Cannot find state: {state_str}")
            return None, None

        state.grant_code = grant_code
        if session_state is not None and len(session_state):
            state.session_state = session_state
        state.save()

        kc_config = self.rdl_auth_state_handler.rdl_auth_config
        public_sign_key = kc_config.public_sign_key
        audience = kc_config.audience
        algorithms = kc_config.algorithms
        try:
            if public_sign_key is not None and len(public_sign_key):
                decoded_token = self.__decode_jwt_token_with_publ_key(
                    token_str=token_str,
                    public_sign_key=public_sign_key,
                    audience=audience,
                    algorithms=algorithms,
                )
            else:
                decoded_token = self.__decode_jwt_token_no_signature(
                    token_str=token_str, audience=audience
                )
        except Exception as e:
            self.logger.error("Error during token verification" + str(e))
            return None, None

        self.logger.debug(
            f"Decoded user token (user info): {str(json.dumps(decoded_token))}"
        )

        errors = self.__check_mappings_errors(user_info=decoded_token)
        if len(errors):
            for e in errors:
                self.logger.error(f"Error while accessing user info: {e}")
            return None, None

        user = self.__get_or_add_user_from_user_info(user_info=decoded_token)
        if not user:
            self.logger.error(f"Error while get/add user")
            return None, None

        token = self.__add_token_for_user(
            user, token_str, state, token_opts, decoded_token
        )
        if not token:
            self.logger.error(f"Error while adding token for user {user.username}")
            return None, None
        return user, token

    def disable_all_user_tokens(self, user: User) -> None:
        """
        Disable all active tokens for given user
        """
        kc_tokens = [t for t in Token.objects.filter(auth_user=user, is_active=True)]
        for token in kc_tokens:
            token.is_active = False
            token.save()

            logout_url = self.rdl_auth_state_handler.generate_logout_url()
            token_request_data = {
                "client_id": self.rdl_auth_state_handler.rdl_auth_config.client_id,
                "client_secret": self.rdl_auth_state_handler.rdl_auth_config.client_secret,
                "refresh_token": token.refresh_token,
            }
            response = requests.post(logout_url, data=token_request_data)
            if not response.ok:
                self.logger.error(f"Error while logout token {response.text}")

    def introspect_user_token(self, user_token: str) -> User or None:
        intro_response = self.__call_introspection_ep(user_token=user_token)
        if intro_response is None:
            return None

        if settings.SYSTEM_HANDLER.use_introspect:
            is_verified = self.__introspection_is_email_verified(
                intro_response=intro_response
            )
            if not is_verified:
                return True

        intro_user = self.__introspection_get_or_add_user(
            intro_response=intro_response
        )
        if intro_user is None:
            return None

        token = self.__introspect_get_add_token(
            user_token=user_token,
            intro_response=intro_response,
            intro_user=intro_user,
        )
        if token is None:
            return None

        return intro_user

    def __introspect_get_add_token(
        self, user_token: str, intro_response: dict, intro_user
    ):
        # Session state have to be consistent with token
        state_str = intro_response.get("session_state", "")
        if not len(state_str):
            state_str = self.rdl_auth_state_handler.generate_state()
        session_state = self.rdl_auth_state_handler.add_state_str(
            state_str=state_str
        )
        if session_state is None:
            self.logger.error(
                f"Introspection problem while adding state "
                f"{intro_response['session_state']}"
            )
            return None

        # Prepare token and store actual token to database
        decoded_token = self.__decode_jwt_token_no_signature(
            token_str=user_token,
            audience=self.rdl_auth_state_handler.rdl_auth_config.audience,
        )
        token = self.__add_token_for_user(
            user=intro_user,
            token_str=user_token,
            state=session_state,
            token_opts={},
            decoded_token=decoded_token,
        )
        if token is None:
            self.logger.error(
                f"Introspection problem with adding token for user "
                f"{intro_response[DEFAULT_USER_ID]}"
            )
            return None

        return token

    def __introspection_is_email_verified(self, intro_response: dict):
        """
        Check user email is verified
        """
        email = intro_response.get("email", None)
        if email is not None and not intro_response.get("email_verified", False):
            self.logger.error(
                f"Token introspection: User email {email} is not verified!"
            )
            return False
        return True

    def __introspection_get_or_add_user(self, intro_response: dict):
        """
        :param intro_response:
        :return:
        """
        user_id = intro_response.get(DEFAULT_USER_ID, None)
        if user_id is None:
            self.logger.error(
                f"Token introspection: User scope info '{DEFAULT_USER_ID}' "
                f"is not found during introspection!"
            )
            return None

        # Make sure user exists in database, if not then will be created
        user_info = {
            DEFAULT_USER_ID: user_id,
            "email": intro_response.get("email", None),
            "given_name": intro_response.get("given_name", None),
            "family_name": intro_response.get("family_name", None),
        }
        intro_user = self.__get_or_add_user_from_user_info(user_info=user_info)
        if intro_user is None:
            self.logger.error("Problem with adding user while introspection!")
            return None

        return intro_user

    def __call_introspection_ep(self, user_token: str):
        api_introspect_url, api_introspect_body = (
            self.__prepare_token_introspect_request(token_str=user_token)
        )
        if api_introspect_url is None or api_introspect_body is None:
            return None

        response = requests.post(api_introspect_url, data=api_introspect_body)
        if not response.ok:
            self.logger.error(f"Error while token introspection {response.text}")
            return None

        # Check intro requirements
        intro_response = response.json()
        if "active" in intro_response and not intro_response["active"]:
            # If token is not active
            self.logger.warning(f"Error while introspection: token is not active!")
            return None
        elif "error" in intro_response:
            # If error occurred
            self.logger.error(
                f"Error while introspection {intro_response['error']}. "
                f"{intro_response.get('error_description', '')}"
            )
            return None
        return intro_response

    def __request_rdl_auth_token_with_options(
        self, grant_code: str | None, refresh_token: str | None = None
    ) -> (str | None, dict | None | list):
        if grant_code is not None and len(grant_code.strip()):
            token_url, request_data = self.__prepare_token_request_grant_code(
                grant_code=grant_code
            )
        else:
            assert refresh_token is not None
            token_url, request_data = self.__request_token_with_refresh_token(
                refresh_token=refresh_token
            )

        if token_url is None or request_data is None:
            return None, None, []

        response = requests.post(token_url, data=request_data)
        if not response.ok:
            self.logger.error(response.text)

            return None, None, []
        j_response = response.json()

        token_options = self.__manage_token_options(json_response=j_response)
        acc_token = j_response.get("access_token", None)
        if acc_token is not None:
            return acc_token, token_options, []

        return None, None, [j_response["error"] if "error" in j_response else ""]

    def __request_token_with_refresh_token(
        self, refresh_token: str
    ) -> (str | None, dict | None):
        if not self.rdl_auth_state_handler:
            self.logger.error("RdlAuth is not initialized!")
            return None, None

        token_url = DEFAULT_TOKEN_REQUEST_PATH.format(
            self.rdl_auth_state_handler.rdl_auth_config.auth_host,
            self.rdl_auth_state_handler.rdl_auth_config.realm,
        )
        token_request_data = {
            "client_id": self.rdl_auth_state_handler.rdl_auth_config.client_id,
            "client_secret": self.rdl_auth_state_handler.rdl_auth_config.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }

        return token_url, token_request_data

    def __prepare_token_request_grant_code(
        self, grant_code: str
    ) -> (str | None, dict | None):
        if not self.rdl_auth_state_handler:
            self.logger.error("RdlAuth is not initialized!")
            return None, None

        token_url = DEFAULT_TOKEN_REQUEST_PATH.format(
            self.rdl_auth_state_handler.rdl_auth_config.auth_host,
            self.rdl_auth_state_handler.rdl_auth_config.realm,
        )

        token_request_data = {
            "client_id": self.rdl_auth_state_handler.rdl_auth_config.client_id,
            "client_secret": self.rdl_auth_state_handler.rdl_auth_config.client_secret,
            "grant_type": self.rdl_auth_state_handler.rdl_auth_config.grant_type,
            "redirect_uri": self.rdl_auth_state_handler.rdl_auth_config.redirect_uri,
            "code": grant_code,
        }
        return token_url, token_request_data

    def __prepare_token_introspect_request(
        self, token_str: str
    ) -> (str | None, dict | None):
        if not self.rdl_auth_state_handler:
            self.logger.error("RdlAuth is not initialized!")
            return None, None

        introspect_url = DEFAULT_TOKEN_INTROSPECT_PATH.format(
            self.rdl_auth_state_handler.rdl_auth_config.auth_host,
            self.rdl_auth_state_handler.rdl_auth_config.realm,
        )

        token_request_data = {
            "client_id": self.rdl_auth_state_handler.rdl_auth_config.client_id,
            "client_secret": self.rdl_auth_state_handler.rdl_auth_config.client_secret,
            "token": token_str,
        }
        return introspect_url, token_request_data

    def __add_token_for_user(
        self,
        user: User,
        token_str: str,
        state: SessionState,
        token_opts: dict,
        decoded_token: dict,
    ) -> Token | None:
        """
        In case when token exists, then error message is logged
        and instead of user None will be returned!
        :param user:
        :param token_str:
        :param state:
        :param token_opts:
        :return:
        """
        if Token.objects.filter(acc_token=token_str).exists():
            for token_obj in Token.objects.filter(acc_token=token_str):
                token_obj.is_active = False
                token_obj.save()
            self.logger.error(
                "Trying to create token object into database, but token "
                "with given token_str exists. Cannot continue."
                "Token is: {}".format(token_str)
            )
            return None

        Token.objects.filter(auth_user=user, is_active=True).update(is_active=False)

        token_obj = Token.objects.create(
            acc_token=token_str,
            auth_user=user,
            state=state,
            is_active=True,
            decoded_token=decoded_token,
        )

        # Try to update any of new added token field(s)
        was_changed = False
        if token_opts and len(token_opts):
            for topt, tval in token_opts.items():
                topt = topt.replace("-", "_")
                if hasattr(token_obj, topt):
                    setattr(token_obj, topt, tval)
                    was_changed = True
        if was_changed:
            token_obj.save()
        return token_obj

    # =========================================================================
    @staticmethod
    def __manage_token_options(json_response: dict) -> dict:
        token_options = {}
        for option in ADDITIONAL_TOKEN_OPTIONS:
            opt = json_response.get(option, None)
            if opt is not None:
                token_options[option] = opt
        return token_options

    @staticmethod
    def __check_mappings_errors(user_info: dict) -> list:
        errors = []
        for map_field in MAPPINGS_REQUIRED.keys():
            if map_field not in user_info:
                errors.append(f"Cannot find mapping {map_field}!")
        if DEFAULT_USER_ID not in user_info:
            errors.append(f"Cannot find '{DEFAULT_USER_ID}'!")
        return errors

    @staticmethod
    def __get_or_add_user_from_user_info(user_info: dict) -> User:
        user = None
        user_name = user_info[DEFAULT_USER_ID]
        user_tokens = Token.objects.filter(auth_user__username=user_name)
        if not len(user_tokens):
            try:
                user = User.objects.get(username=user_name)
            except User.DoesNotExist:
                pass
        else:
            user = user_tokens[0].auth_user

        if user is None:
            user = User.objects.create_user(
                username=user_name,
                first_name=user_info.get("given_name", ""),
                last_name=user_info.get("family_name", ""),
                email=user_info.get("email", ""),
                password=None,
                is_superuser=False,
                is_active=True,
            )

        return user

    @staticmethod
    def __decode_jwt_token_no_signature(token_str, audience: str) -> dict:
        decode_options = {}
        if len(audience.strip()):
            decode_options["audience"] = audience

        decoded = jwt.decode(
            token_str,
            **decode_options,
            options={"verify_signature": False, "verify_exp": True},
        )
        return decoded

    @staticmethod
    def __decode_jwt_token_with_publ_key(
        token_str: str, public_sign_key: str, audience: str, algorithms: list
    ) -> dict:
        decode_params = {}
        if len(public_sign_key.strip()):
            decode_params["key"] = public_sign_key
            if len(algorithms):
                decode_params["algorithms"] = algorithms
        else:
            decode_params["options"] = {
                "verify_signature": False,
                "verify_exp": True,
            }
        if len(audience.strip()):
            decode_params["audience"] = audience
        decoded = jwt.decode(token_str, **decode_params)
        return decoded

    def verify_and_decode_token(
        self, token: Token | None, token_str: str | None
    ) -> dict | None:
        kc_config = self.rdl_auth_state_handler.rdl_auth_config
        public_sign_key = kc_config.public_sign_key
        audience = kc_config.audience
        algorithms = kc_config.algorithms
        try:
            acc_token = token_str
            if token is not None:
                acc_token = token.acc_token

            decoded_acc_token = self.__decode_jwt_token_with_publ_key(
                token_str=acc_token,
                public_sign_key=public_sign_key,
                audience=audience,
                algorithms=algorithms,
            )

            return self.__verify_user_token(
                token=token, decoded_token=decoded_acc_token
            )
        except jwt.ExpiredSignatureError:
            return None
        except Exception as e:
            self.logger.error(e)
            return None

    @staticmethod
    def __verify_user_token(token: Token, decoded_token) -> dict | None:
        if token is not None:
            return token.decoded_token
        return decoded_token
