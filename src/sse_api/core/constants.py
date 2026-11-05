import os
from pathlib import Path
from django.conf import settings

# Root of the repository (sse_api lives in <repo>/src/sse_api).  The config
# and the runtime data (uploads, static) live in the repository root, so the
# defaults below are resolved against it: the server must work no matter
# which directory the WSGI server changed into (gunicorn chdirs into
# src/sse_api).
REPO_ROOT = Path(__file__).resolve().parents[3]

STATIC_DIR = "static/"
# Directory holding runtime config files. Resolved relative to the
# repository root by default; override with SSE_CONFIG_DIR (used by the
# Docker image, the start scripts and by tests).
CONFIG_DIR = os.getenv("SSE_CONFIG_DIR", str(REPO_ROOT / "configs"))

SYSTEM_CONFIG_FILENAME = "django-config.json"
RDL_AUTH_CONFIG_FILENAME = "auth-config.json"
SYSTEM_CONFIG_FILENAME_DEFAULT = "django-config-default.json"
AWS_CONFIG_FILENAME = "django-config.json"

SECRET_KEY_FILENAME = "secret-key.txt"
SECRET_KEY_FILENAME_DEFAULT = "secret-key-default.txt"

# Directory where the engine stores the uploaded files.
UPLOAD_DIR = os.getenv("SSE_UPLOAD_DIR", str(REPO_ROOT / "upload_sse"))

AVAILABLE_CONFIG_DIRS = [
    STATIC_DIR,
    CONFIG_DIR,
    os.path.join(STATIC_DIR, CONFIG_DIR),
]

AVAILABLE_LANGUAGES = ["pl", "en"]

ENV_DEBUG = "ENV_DEBUG"
ENV_SHOW_ADMIN_WINDOW = "ENV_SHOW_ADMIN_WINDOW"
ENV_PUBLIC_API_AVAILABLE = "ENV_PUBLIC_API_AVAILABLE"
ENV_DEFAULT_LANGUAGE = "ENV_DEFAULT_LANGUAGE"
ENV_SECRET_KEY = "ENV_SECRET_KEY"
ENV_ALLOWED_HOSTS = "ENV_ALLOWED_HOSTS"
ENV_INSTALLED_APPS = "ENV_INSTALLED_APPS"
ENV_CELERY_BROKER_URL = "ENV_CELERY_BROKER_URL"
ENV_CELERY_TASK_TIME_LIMIT = "ENV_CELERY_TASK_TIME_LIMIT"
ENV_CELERY_RESULT_BACKEND = "ENV_CELERY_RESULT_BACKEND"

#: Name of the Celery queue the indexing tasks are put on.
INDEXING_QUEUE = "indexing"


ENV_DB_NAME = "ENV_DB_NAME"
ENV_DB_HOST = "ENV_DB_HOST"
ENV_DB_PORT = "ENV_DB_PORT"
ENV_DB_ENGINE = "ENV_DB_ENGINE"
ENV_DB_USERNAME = "ENV_DB_USERNAME"
ENV_DB_PASSWORD = "ENV_DB_PASSWORD"

ENV_LOGGER = "ENV_LOGGER"

ENV_EXTERNAL_API = "ENV_EXTERNAL_API"

ENV_USE_KC_AUTH = "ENV_USE_KC_AUTH"
ENV_USE_OAUTH_V1_AUTH = "ENV_USE_OAUTH_V1_AUTH"
ENV_USE_OAUTH_V2_AUTH = "ENV_USE_OAUTH_V2_AUTH"
ENV_USE_INTROSPECT = "ENV_USE_INTROSPECT"
ENV_INTROSPECT_EMAIL_VERIFICATION = "ENV_INTROSPECT_EMAIL_VERIFICATION"

ENV_USE_AWS = "ENV_USE_AWS"
ENV_USE_CELERY = "ENV_USE_CELERY"

ENV_USE_RADLAB_PROXY = "ENV_USE_RADLAB_PROXY"
ENV_USE_RADLAB_PROXY_SEE = "ENV_USE_PROXY_SEE"


def default_app_language():
    """
    Returns default application language
    :return: String-like application language
    """
    return settings.DEFAULT_APP_LANGUAGE


def get_logger(name: str = None):
    """
    Returns logger defined into settings as global logger
    :param name: Optionally specify logger name
    :return: Built logger object
    """
    return settings.MAIN_LOGGER


def prepare_api_url(endpoint_url: str) -> str:
    """
    Based on main_api_endpoint url, prepare whole api url with (optionally) version
    :param endpoint_url: Endpoint url (without root call, only ep name)
    :return: Prepared whole api url
    """
    main_url = settings.MAIN_API_URL
    if main_url.endswith("/"):
        main_url = main_url[:-1]

    endpoint_url = endpoint_url.strip()
    if endpoint_url.startswith("/"):
        endpoint_url = endpoint_url[1:]
    whole_url = f"{main_url}/{endpoint_url}"

    get_logger().info(f"Sharing api url {whole_url}")

    return whole_url
