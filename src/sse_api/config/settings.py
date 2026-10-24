import os.path
from pathlib import Path

from sse_api.core.settings import system_handler
from sse_api.core.constants import STATIC_DIR, CONFIG_DIR, INDEXING_QUEUE

# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = STATIC_DIR

ROOT_URLCONF = "sse_api.config.urls"
WSGI_APPLICATION = "sse_api.config.wsgi.application"

LANGUAGE_CODE = "pl-PL"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# =============================================================================

SECRET_KEY = system_handler.load_secret_key(
    filepath=system_handler.secret_key_filename
)
DEBUG = system_handler.debug()
DATABASES = system_handler.database()
DEFAULT_APP_LANGUAGE = system_handler.default_language()
MAIN_API_URL = system_handler.api_version_url()
ALLOWED_HOSTS = system_handler.allowed_hosts()
MAIN_LOGGER = system_handler.logger()

# =============================================================================

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "rest_framework",
]

INSTALLED_APPS += system_handler.installed_apps()

# =============================================================================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# =============================================================================

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

# =============================================================================

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# =============================================================================

REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
    "PAGE_SIZE": 10,
    # 'DATETIME_FORMAT': '%s000',
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    # Throttling.  Anonymous clients are budgeted per IP (an anonymous
    # request cannot burn the embedding/LLM budget of the deployment),
    # authenticated users per user, and the login endpoints carry a
    # dedicated, tighter throttle (see StrictAuthThrottle) so a script
    # cannot brute-force a password.  The rates are the burst budgets;
    # raise them per deployment through the environment if needed.
    "DEFAULT_THROTTLE_CLASSES": [
        "sse_api.core.throttles.BurstAnonRateThrottle",
        "sse_api.core.throttles.BurstUserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.environ.get("SSE_THROTTLE_ANON", "60/minute"),
        "user": os.environ.get("SSE_THROTTLE_USER", "300/minute"),
        "auth": os.environ.get("SSE_THROTTLE_AUTH", "10/minute"),
    },
}

# SimpleRateThrottle counts hits in the cache; without a CACHES entry
# Django falls back to its own LocMemCache definition.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "sse-throttles",
    }
}


def _production_security_settings():
    """
    Django security settings applied whenever ``DEBUG`` is off.

    nginx (``nginx/sse.conf``) already terminates TLS and sends
    ``X-Forwarded-Proto: https``; the settings below are what Django needs
    on the other side of that proxy.  With ``DEBUG`` on the SSL redirect is
    dropped so a local ``http://localhost`` dev server keeps working.
    """
    return {
        "SECURE_PROXY_SSL_HEADER": ("HTTP_X_FORWARDED_PROTO", "https"),
        "SECURE_SSL_REDIRECT": not DEBUG,
        "SECURE_CONTENT_TYPE_NOSNIFF": True,
        "X_FRAME_OPTIONS": "DENY",
        "SECURE_HSTS_SECONDS": 31536000,
        "SECURE_HSTS_INCLUDE_SUBDOMAINS": True,
        "SECURE_HSTS_PRELOAD": True,
        "SESSION_COOKIE_SECURE": not DEBUG,
        "SESSION_COOKIE_HTTPONLY": True,
        "CSRF_COOKIE_SECURE": not DEBUG,
    }


for _name, _value in _production_security_settings().items():
    globals()[_name] = _value

_env_csrf_origins = os.environ.get("ENV_CSRF_TRUSTED_ORIGINS", "").strip()
if _env_csrf_origins:
    CSRF_TRUSTED_ORIGINS = [
        origin.strip()
        for origin in _env_csrf_origins.split(",")
        if origin.strip()
    ]

DATA_UPLOAD_MAX_MEMORY_SIZE = int(
    os.environ.get("DATA_UPLOAD_MAX_MEMORY_SIZE", 5242880)
)


if (
    system_handler.use_kc_auth
    or system_handler.use_oauth_v1_auth
    or system_handler.use_oauth_v2_auth
):
    from sse_api import authorization

    INSTALLED_APPS.append(authorization.app_name)
    MIDDLEWARE.append(authorization.default_middleware_class)
    REST_FRAMEWORK["DEFAULT_AUTHENTICATION_CLASSES"] = [
        authorization.default_auth_class
    ]
else:
    INSTALLED_APPS.append("rest_framework.authtoken")
    REST_FRAMEWORK["DEFAULT_AUTHENTICATION_CLASSES"] = [
        "rest_framework.authentication.TokenAuthentication"
    ]


if system_handler.use_celer_tasks:
    # Celery is configured through the CELERY_* namespace (see
    # sse_api/config/celery.py); it is not a Django application.
    CELERY_BROKER_URL = system_handler.celery_broker_url()
    # Results are not kept in the backend: the progress of the indexing lives
    # in the relational database, which also survives a restart of a worker.
    CELERY_RESULT_BACKEND = system_handler.celery_result_backend()
    CELERY_TASK_DEFAULT_QUEUE = INDEXING_QUEUE
    CELERY_TASK_SERIALIZER = "json"
    CELERY_TASK_TRACK_STARTED = True
    # One document at a time per process, and only acknowledge a task after it
    # was really done: a worker killed while embedding does not lose documents.
    CELERY_WORKER_PREFETCH_MULTIPLIER = 1
    CELERY_TASK_ACKS_LATE = True
    CELERY_TASK_TIME_LIMIT = system_handler.celery_task_time_limit()
    CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
    CELERY_BROKER_CONNECTION_MAX_RETRIES = None

if system_handler.use_radlab_proxy():
    INSTALLED_APPS.append("proxy")


AWS_S3_HANDLER = None
if system_handler.use_aws_storage:
    AWS_S3_HANDLER = system_handler.aws_handler

# Additionally system handler may be available from sse_api.config settings
SYSTEM_HANDLER = system_handler
