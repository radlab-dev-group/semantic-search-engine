"""
settings_django.py
------------------

Django settings for the automated test suite.

Deliberately minimal and self-contained: in-memory SQLite, no external
authentication, no Celery, no AWS.  It mirrors the production
``sse_api.config.settings`` layout closely enough that the ``system_handler`` based
modules keep working, but every value the suite depends on is spelled out
here so a test can never depend on a real deployment file.
"""

import logging

SECRET_KEY = "sse-test-secret-key"
DEBUG = False
ALLOWED_HOSTS = ["testserver", "localhost", "127.0.0.1"]
DEFAULT_APP_LANGUAGE = "en"
MAIN_API_URL = "api"
MAIN_LOGGER = logging.getLogger("sse.tests")
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
STATIC_URL = "static/"
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "rest_framework",
    "sse_api.authorization",
    "sse_api.chat",
    "sse_api.data",
    "sse_api.engine",
    "sse_api.system",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication"
    ],
}

SYSTEM_HANDLER = type(
    "SystemHandlerStub",
    (),
    {
        # authorization.core.handlers raises at import unless one provider is
        # active, so Keycloak is the active provider in the test suite.
        "use_keycloak_authentication": staticmethod(lambda: True),
        "use_oauth_v1_authentication": staticmethod(lambda: False),
        "use_oauth_v2_authentication": staticmethod(lambda: False),
        "use_introspect": False,
        "show_admin_window": staticmethod(lambda: False),
    },
)
