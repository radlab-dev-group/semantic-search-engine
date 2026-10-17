"""
test_settings_handler.py
------------------------

Unit tests for SystemSettingsHandler env parsing and config handling.
"""

import pytest

from sse_api.core.constants import (
    ENV_ALLOWED_HOSTS,
    ENV_INSTALLED_APPS,
    ENV_DEBUG,
    ENV_SHOW_ADMIN_WINDOW,
    ENV_PUBLIC_API_AVAILABLE,
    ENV_USE_KC_AUTH,
    ENV_USE_OAUTH_V1_AUTH,
    ENV_USE_OAUTH_V2_AUTH,
    ENV_USE_INTROSPECT,
    ENV_INTROSPECT_EMAIL_VERIFICATION,
    ENV_USE_CELERY,
    ENV_USE_AWS,
    ENV_USE_RADLAB_PROXY,
    ENV_USE_RADLAB_PROXY_SEE,
    ENV_CELERY_BROKER_URL,
    ENV_DEFAULT_LANGUAGE,
    ENV_DB_NAME,
    ENV_DB_HOST,
    ENV_DB_PORT,
    ENV_DB_ENGINE,
    ENV_DB_USERNAME,
    ENV_DB_PASSWORD,
    ENV_CELERY_RESULT_BACKEND,
    ENV_CELERY_TASK_TIME_LIMIT,
)
from sse_api.core.settings import SystemSettingsHandler


@pytest.fixture
def dummy_handler(tmp_path):
    """Create a SystemSettingsHandler instance with mock json config."""
    config_file = tmp_path / "django-config.json"
    config_file.write_text(
        """{
        "main": {
            "debug": false,
            "allowed_hosts": ["localhost", "127.0.0.1"],
            "admin_window": false,
            "public_api_available": true,
            "default_language": "pl",
            "use_keycloak_authentication": false,
            "use_oauth_v1_authentication": false,
            "use_oauth_v2_authentication": false,
            "use_authorization_introspect": false,
            "introspect_email_verification": false,
            "use_celery": false,
            "use_aws": false
        },
        "database": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": "test_db",
            "USER": "test_user",
            "PASSWORD": "test_password",
            "HOST": "localhost",
            "PORT": 5471
        },
        "installed_apps": [
            "sse_api.system",
            "sse_api.data"
        ],
        "celery": {
            "broker_url": "amqp://guest:guest@localhost:5672/"
        },
        "logger": {
            "use_logger": "dev",
            "config": {
                "version": 1,
                "disable_existing_loggers": false,
                "dev": {"handlers": []},
                "prod": {"handlers": []}
            }
        },
        "aws": {
            "region": "eu-central-1"
        }
    }"""
    )
    secret_key_file = tmp_path / "secret-key.txt"
    secret_key_file.write_text("test-secret-key-12345")

    handler = SystemSettingsHandler(
        system_cfg_filename=str(config_file),
        secret_key_filename=str(secret_key_file),
        load_config=True,
        use_environment=True,
    )
    return handler


def test_allowed_hosts_fallback_to_json(dummy_handler, monkeypatch):
    monkeypatch.delenv(ENV_ALLOWED_HOSTS, raising=False)
    assert dummy_handler.allowed_hosts() == ["localhost", "127.0.0.1"]


@pytest.mark.parametrize(
    ("env_val", "expected"),
    [
        (
            "localhost,127.0.0.1,192.168.100.70",
            ["localhost", "127.0.0.1", "192.168.100.70"],
        ),
        ("192.168.100.70", ["192.168.100.70"]),
        ("*", ["*"]),
        (
            "192.168.100.70:8271, localhost:8000",
            ["192.168.100.70:8271", "localhost:8000"],
        ),
        ("  localhost ,  127.0.0.1  ", ["localhost", "127.0.0.1"]),
    ],
)
def test_allowed_hosts_from_env(dummy_handler, monkeypatch, env_val, expected):
    monkeypatch.setenv(ENV_ALLOWED_HOSTS, env_val)
    assert dummy_handler.allowed_hosts() == expected


def test_installed_apps_from_env(dummy_handler, monkeypatch):
    monkeypatch.setenv(
        ENV_INSTALLED_APPS, "sse_api.system, sse_api.engine, sse_api.chat"
    )
    assert dummy_handler.installed_apps() == [
        "sse_api.system",
        "sse_api.engine",
        "sse_api.chat",
    ]


def test_installed_apps_fallback_to_json(dummy_handler, monkeypatch):
    monkeypatch.delenv(ENV_INSTALLED_APPS, raising=False)
    assert dummy_handler.installed_apps() == ["sse_api.system", "sse_api.data"]


@pytest.mark.parametrize(
    ("env_val", "expected"),
    [
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_debug_from_env(dummy_handler, monkeypatch, env_val, expected):
    monkeypatch.setenv(ENV_DEBUG, env_val)
    assert dummy_handler.debug() is expected


def test_debug_fallback_to_json(dummy_handler, monkeypatch):
    monkeypatch.delenv(ENV_DEBUG, raising=False)
    assert dummy_handler.debug() is False


@pytest.mark.parametrize(
    ("flag", "env_name", "getter"),
    [
        ("admin_window", ENV_SHOW_ADMIN_WINDOW, lambda h: h.show_admin_window()),
        (
            "public_api_available",
            ENV_PUBLIC_API_AVAILABLE,
            lambda h: h.is_public_api_available(),
        ),
        (
            "use_keycloak_authentication",
            ENV_USE_KC_AUTH,
            lambda h: h.use_keycloak_authentication(),
        ),
        (
            "use_oauth_v1_authentication",
            ENV_USE_OAUTH_V1_AUTH,
            lambda h: h.use_oauth_v1_authentication(),
        ),
        (
            "use_oauth_v2_authentication",
            ENV_USE_OAUTH_V2_AUTH,
            lambda h: h.use_oauth_v2_authentication(),
        ),
        (
            "use_authorization_introspect",
            ENV_USE_INTROSPECT,
            lambda h: h.use_authorization_introspect(),
        ),
        (
            "introspect_email_verification",
            ENV_INTROSPECT_EMAIL_VERIFICATION,
            lambda h: h.introspect_check_user_email_is_verified(),
        ),
        ("use_celery", ENV_USE_CELERY, lambda h: h.use_celery()),
        ("use_aws", ENV_USE_AWS, lambda h: h.use_aws()),
    ],
)
def test_boolean_flags(dummy_handler, monkeypatch, flag, env_name, getter):
    monkeypatch.setenv(env_name, "1")
    assert getter(dummy_handler) is True
    monkeypatch.setenv(env_name, "0")
    assert getter(dummy_handler) is False


def test_static_proxy_methods(monkeypatch):
    monkeypatch.setenv(ENV_USE_RADLAB_PROXY, "1")
    assert SystemSettingsHandler.use_radlab_proxy() is True
    monkeypatch.setenv(ENV_USE_RADLAB_PROXY, "0")
    assert SystemSettingsHandler.use_radlab_proxy() is False

    monkeypatch.setenv(ENV_USE_RADLAB_PROXY_SEE, "1")
    assert SystemSettingsHandler.use_radlab_proxy_sse() is True
    monkeypatch.setenv(ENV_USE_RADLAB_PROXY_SEE, "0")
    assert SystemSettingsHandler.use_radlab_proxy_sse() is False


def test_celery_broker_url(dummy_handler, monkeypatch):
    monkeypatch.setenv(ENV_CELERY_BROKER_URL, "amqp://user:pass@remote:5672/")
    assert dummy_handler.celery_broker_url() == "amqp://user:pass@remote:5672/"
    monkeypatch.delenv(ENV_CELERY_BROKER_URL, raising=False)
    assert dummy_handler.celery_broker_url() == "amqp://guest:guest@localhost:5672/"


def test_celery_result_backend_is_optional(dummy_handler, monkeypatch):
    monkeypatch.delenv(ENV_CELERY_RESULT_BACKEND, raising=False)
    assert dummy_handler.celery_result_backend() is None
    monkeypatch.setenv(ENV_CELERY_RESULT_BACKEND, "redis://redis:6379/0")
    assert dummy_handler.celery_result_backend() == "redis://redis:6379/0"


def test_celery_task_time_limit(dummy_handler, monkeypatch):
    monkeypatch.delenv(ENV_CELERY_TASK_TIME_LIMIT, raising=False)
    assert dummy_handler.celery_task_time_limit() == 14400
    monkeypatch.setenv(ENV_CELERY_TASK_TIME_LIMIT, "600")
    assert dummy_handler.celery_task_time_limit() == 600


def test_default_language(dummy_handler, monkeypatch):
    monkeypatch.setenv(ENV_DEFAULT_LANGUAGE, "en")
    assert dummy_handler.default_language() == "en"
    monkeypatch.delenv(ENV_DEFAULT_LANGUAGE, raising=False)
    assert dummy_handler.default_language() == "pl"


def test_database_config_env_override(dummy_handler, monkeypatch):
    # pymilvus reads a ``.env`` of the working directory on import, so the
    # variables of this test have to start from an environment of its own.
    for name in (ENV_DB_ENGINE, ENV_DB_USERNAME, ENV_DB_PASSWORD):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_DB_NAME, "custom_db")
    monkeypatch.setenv(ENV_DB_HOST, "pg.example.com")
    monkeypatch.setenv(ENV_DB_PORT, "5432")

    db = dummy_handler.database()["default"]
    assert db["NAME"] == "custom_db"
    assert db["HOST"] == "pg.example.com"
    assert db["PORT"] == 5432
    assert db["USER"] == "test_user"
