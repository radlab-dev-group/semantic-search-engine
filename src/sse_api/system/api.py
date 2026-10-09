from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken

from sse_api.core.errors import error_response
from sse_api.core.errors_list import NO_LOGIN_PARAMS
from sse_api.core.constants import default_app_language, get_logger
from sse_api.core.decorators import required_params_exists
from sse_api.system.core.health import dependencies_status

DEFAULT_APP_LANGUAGE = default_app_language()

SERVICE_UNAVAILABLE = 503


class SSEHealthCheck(APIView):
    """
    Readiness/liveness probe used by nginx, the load balancer and the
    ``HEALTHCHECK`` of the Docker image.

    Deliberately public: a probe has no token to present, and the payload is
    only booleans (``health.py`` keeps tracebacks and connection strings in the
    application logger).  Returns 200 when every dependency answers and 503
    otherwise, so ``curl -f`` and ``depends_on: service_healthy`` both fail on
    an unhealthy instance.

    The body is ``{"healthy": bool, "checks": {...}}`` rather than the usual
    envelope: ``sse_lib`` detects the envelope by ``payload["status"] is True``,
    so a probe replying ``{"status": "ok"}`` would be read as a failed API call
    by its own client.  The HTTP status code stays the signal probes use.
    """

    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            checks = dependencies_status()
        except Exception as exc:  # pylint: disable=broad-except
            # Individual checks already swallow their own errors; this guards
            # against a check blowing up outright, which would otherwise make
            # the probe answer 500 with a traceback (and leak internals under
            # DEBUG) instead of a plain "unhealthy".
            get_logger().error(f"Health check failed: {exc}")
            return Response({"healthy": False}, status=SERVICE_UNAVAILABLE)

        healthy = bool(checks) and all(checks.values())
        return Response(
            {"healthy": healthy, "checks": checks},
            status=200 if healthy else SERVICE_UNAVAILABLE,
        )


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
