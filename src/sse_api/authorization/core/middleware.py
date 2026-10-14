from django.conf import settings

from typing import Callable

from django.contrib.auth.models import AnonymousUser
from django.utils.deprecation import MiddlewareMixin

from sse_api.core.constants import get_logger
from sse_api.authorization.core.authentication import GATokenAuthentication

middleware_callback: Callable | None = None


class AuthAuthenticationMiddleware(MiddlewareMixin):
    def __init__(self, get_response):
        super().__init__(get_response)
        # Built here rather than as class attributes: GATokenAuthentication
        # reads configs/auth-config.json, and class attributes would run that
        # at import time, which makes this module unimportable without a
        # deployment config file. Django builds middleware once per process,
        # so this is still a single instance — just created after settings are
        # guaranteed to be configured.
        self.ga_token = GATokenAuthentication()
        self.logger = get_logger()

    def process_request(self, request):
        """
        Adds user to the request when authorized user is found in the session
        In case when user is not authorized and introspection is activated
        introspection will be used to determine user in central KC.

        :param django.http.request.HttpRequest request: django request
        """
        introspect_log = False
        user_token = self.ga_token.get_user_token_for_request(request=request)
        if not user_token:
            if settings.SYSTEM_HANDLER.use_introspect:
                intro_user = self.ga_token.introspect_token(request)
                if not intro_user:
                    request.user = AnonymousUser()
                    self.logger.info("Anonymous user after introspection")
                else:
                    request.user = intro_user
                    self.logger.info(
                        f"Setting introspection middleware "
                        f"logged user as {request.user.username}"
                    )
                    introspect_log = True
            else:
                request.user = AnonymousUser()
                self.logger.info("Anonymous user")
        else:
            claims = self.ga_token.token_handler.verify_and_decode_token(
                token=user_token, token_str=None
            )
            request.user = user_token.auth_user if claims else AnonymousUser()
            self.logger.info(
                f"Setting middleware logged user as {request.user.username}"
            )

        if introspect_log and middleware_callback is not None:
            middleware_callback(user=request.user)

        return self.get_response(request)

    def process_response(self, request, response):
        if response.status_code == 403 and request.user is None:
            response.status_code = 401
        return response
