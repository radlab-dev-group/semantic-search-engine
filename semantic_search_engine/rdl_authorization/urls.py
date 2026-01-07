from django.urls import path
from main.src.constants import prepare_api_url

from rdl_authorization.api.authorization import (
    obtain_auth_token,
    remove_auth_token,
    GenerateLoginUrl,
    RefreshToken,
)

urlpatterns = [
    path(
        prepare_api_url("login_url"),
        GenerateLoginUrl.as_view(),
        name="generate_login_url",
    ),
    path(prepare_api_url("login"), obtain_auth_token, name="login"),
    path(prepare_api_url("logout"), remove_auth_token, name="logout"),
    path(
        prepare_api_url("refresh_token"),
        RefreshToken.as_view(),
        name="refresh_token",
    ),
]
