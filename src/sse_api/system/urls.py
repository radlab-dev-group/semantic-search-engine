from django.urls import path

from sse_api.system.api import OrganisationLogin
from sse_api.core.constants import prepare_api_url


urlpatterns = [
    path(
        prepare_api_url("login"),
        OrganisationLogin.as_view(),
        name="login",
    ),
]
