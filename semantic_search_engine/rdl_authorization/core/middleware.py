from django.conf import settings

from typing import Callable

from django.contrib.auth.models import AnonymousUser
from django.utils.deprecation import MiddlewareMixin

from main.src.constants import get_logger
from rdl_authorization.core.authentication import GATokenAuthentication


middleware_callback: Callable | None = None


class RdlAuthAuthenticationMiddleware(MiddlewareMixin):
    ga_token = GATokenAuthentication()
    logger = get_logger()

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
            request.user = user_token.auth_user
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
