from django.urls import path

from sse_api.system.api import OrganisationLogin, SSEHealthCheck
from sse_api.core.constants import prepare_api_url

urlpatterns = [
    path(
        prepare_api_url("login"),
        OrganisationLogin.as_view(),
        name="login",
    ),
    # The probe is reachable under the configured API root *and* as a bare
    # /healthz.  ``nginx/sse.conf`` and the image HEALTHCHECK hardcode
    # ``/api/healthz``; keeping an unversioned alias means enabling
    # ``api.major_version`` (which turns the API root into ``api/v1``) or
    # renaming ``root_url`` cannot silently break container health reporting.
    path(
        prepare_api_url("healthz"),
        SSEHealthCheck.as_view(),
        name="healthz",
    ),
    path("healthz", SSEHealthCheck.as_view(), name="healthz_bare"),
]
