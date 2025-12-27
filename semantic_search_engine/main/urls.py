from django.urls import path
from django.conf import settings
from rest_framework.authtoken.views import obtain_auth_token

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

    from rdl_authorization import urls as rdl_auth_urls

    urlpatterns += rdl_auth_urls.urlpatterns
else:
    urlpatterns.append(
        path("api-token-auth/", obtain_auth_token, name="api_token_auth")
    )

if settings.SYSTEM_HANDLER.use_radlab_proxy():
    if settings.SYSTEM_HANDLER.use_radlab_proxy_sse():
        from proxy.sse import urls as sse_urls

        urlpatterns += sse_urls.urlpatterns

# Auto added urls during installation
# chat_urls.py
from chat.urls import urlpatterns as chat_urlpatterns

urlpatterns += chat_urlpatterns

# data_urls.py
from data.urls import urlpatterns as data_urlpatterns

urlpatterns += data_urlpatterns

# engine_urls.py
from engine.urls import urlpatterns as engine_urlpatterns

urlpatterns += engine_urlpatterns

# system_urls.py
from system.urls import urlpatterns as system_urlpatterns

urlpatterns += system_urlpatterns

# End of auto added urls
