from django.urls import path
from django.conf import settings
from rest_framework.authtoken.views import obtain_auth_token

from chat.urls import urlpatterns as chat_urlpatterns
from data.urls import urlpatterns as data_urlpatterns
from engine.urls import urlpatterns as engine_urlpatterns
from system.urls import urlpatterns as system_urlpatterns


urlpatterns = []

if settings.SYSTEM_HANDLER.show_admin_window():
    from django.contrib import admin

    urlpatterns.append(path("admin/", admin.site.urls))

if (
    settings.SYSTEM_HANDLER.use_keycloak_authentication()
    or settings.SYSTEM_HANDLER.use_oauth_v1_authentication()
    or settings.SYSTEM_HANDLER.use_oauth_v2_authentication()
):
    _kc = settings.SYSTEM_HANDLER.use_keycloak_authentication()
    _ov1 = settings.SYSTEM_HANDLER.use_oauth_v1_authentication()
    _ov2 = settings.SYSTEM_HANDLER.use_oauth_v2_authentication()
    if sum([_kc, _ov1, _ov2]) > 1:
        raise Exception("Choose between Keycloak or OAuth/OAuth V2 authentication!")

    from authorization import urls as rdl_auth_urls

    urlpatterns += rdl_auth_urls.urlpatterns
else:
    urlpatterns.append(
        path("api-token-auth/", obtain_auth_token, name="api_token_auth")
    )


urlpatterns += chat_urlpatterns
urlpatterns += data_urlpatterns
urlpatterns += engine_urlpatterns
urlpatterns += system_urlpatterns
