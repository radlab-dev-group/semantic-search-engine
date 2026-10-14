import logging
from types import SimpleNamespace

SECRET_KEY = "authorization-tests-only"
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "sse_api.authorization",
]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
MIGRATION_MODULES = {"authorization": None}
MAIN_LOGGER = logging.getLogger("authorization.tests")
SYSTEM_HANDLER = SimpleNamespace(
    use_keycloak_authentication=lambda: True,
    use_oauth_v1_authentication=lambda: False,
    use_oauth_v2_authentication=lambda: False,
    use_introspect=False,
)
