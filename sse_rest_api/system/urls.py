from django.urls import path

from system.api import OrganisationLogin
from main.src.constants import prepare_api_url


urlpatterns = [
    path(
        prepare_api_url("login"),
        OrganisationLogin.as_view(),
        name="login",
    ),
]
