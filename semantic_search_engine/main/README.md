### Overview

The **main** package contains the global project configuration, utilities, and the Django entry‑point files. It glues
together all sub‑applications (`system`, `data`, `engine`, `chat`) and provides common services such as logging,
environment handling, error handling, and settings management.

### Package Layout

```
main/
├─ src/
│   ├─ __init__.py
│   ├─ aws_handler.py            # Thin wrapper around boto3 for S3 interactions
│   ├─ constants.py              # Global constants (static dirs, config filenames, env keys)
│   ├─ decorators.py             # Request validation & language extraction helpers
│   ├─ env_utils.py              # Helper for boolean environment variables
│   ├─ errors.py                 # Central error response builder
│   ├─ errors_constants.py
│   ├─ errors_list.py            # Built‑in error definitions
│   ├─ logger.py                 # Simple logger wrapper (configurable via JSON)
│   ├─ response.py               # `response_with_status()` helper
│   └─ settings.py               # Project‑wide Django settings (loads system config)
├─ __init__.py
├─ asgi.py                        # ASGI entry point
├─ celery.py                      # Celery app configuration
├─ settings.py                    # Wrapper that imports `src.settings`
├─ urls.py                        # Root URL dispatcher (includes sub‑app URLs)
└─ wsgi.py                        # WSGI entry point
```

### Core Concepts

| Module                         | Responsibility                                                                                                                                                                                                                                                 |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **constants.py**               | Centralizes file‑system paths (`STATIC_DIR`, `CONFIG_DIR`), config filenames, and environment variable names (`ENV_DEBUG`, `ENV_USE_AWS`, etc.). Provides helpers `default_app_language()`, `get_logger()`, and `prepare_api_url()` (adds API version prefix). |
| **settings.py (src)**          | `SystemSettingsHandler` reads `configs/django-config.json` (or defaults) and optionally overrides values from environment variables. It configures database, logger, AWS handler, Celery broker, authentication back‑ends, and more.                           |
| **settings.py (project)**      | Imports the system handler, exposes Django settings such as `SECRET_KEY`, `DEBUG`, `DATABASES`, `INSTALLED_APPS`, `MIDDLEWARE`, and `REST_FRAMEWORK`. Dynamically adds apps (`system`, `data`, `engine`, `chat`, optional `rdl_authorization`).                |
| **decorators.py**              | - `required_params_exists` validates required/optional request parameters and returns a structured error if missing. <br> - `get_default_language` extracts the requested language or falls back to the default.                                               |
| **errors.py / errors_list.py** | Central error catalogue. `error_response()` builds a DRF `Response` with a status flag and a list of error dictionaries (code + localized message).                                                                                                            |
| **aws_handler.py**             | Minimal wrapper around `boto3` that can create directories, delete objects, and upload files to an S3 bucket defined in the system config.                                                                                                                     |
| **logger.py**                  | Simple wrapper that can load a logging configuration dictionary and apply it via `logging.config.dictConfig`.                                                                                                                                                  |
| **response.py**                | `response_with_status(status, language, error_name, response_body)` – returns either a success payload (`{"status": True, "body": …}`) or forwards to `error_response`.                                                                                        |
| **env_utils.py**               | `bool_env_value()` parses common truthy strings (`"true"`, `"1"`, `"yes"`, etc.) into a boolean.                                                                                                                                                               |
| **asgi.py / wsgi.py**          | Standard Django entry points that set the default settings module (`main.settings`).                                                                                                                                                                           |
| **celery.py**                  | Configures a Celery application using Django settings (`CELERY_` namespace).                                                                                                                                                                                   |
| **urls.py**                    | Root URL dispatcher. Includes URLs from `chat`, `data`, `engine`, `system`. Handles admin site, authentication (Token auth or external Keycloak/OAuth), and the API token endpoint.                                                                            |

### Error Handling Flow

1. **Validation** – Views decorate methods with `@required_params_exists`. If any required param is missing,
   `error_response()` is invoked with `NO_REQUIRED_PARAMS`.
2. **Localization** – The `language` argument (derived from request or default) selects the appropriate message (
   `MSG_PL` vs. `MSG_EN`).
3. **Custom Errors** – Each sub‑app (e.g., `system.errors`, `chat.errors`) extends `ALL_ERRORS` with its own namespace (
   `ERROR_MARK_SYSTEM`, `ERROR_MARK_CHAT`). `build_errors_map()` merges them into a global map.

### Configuration Lifecycle

- **Startup** – `SystemSettingsHandler` loads the JSON config (`django-config.json`) and, if `use_environment=True`,
  overlays values from environment variables (e.g., `ENV_DB_HOST`, `ENV_USE_AWS`).
- **AWS** – If `ENV_USE_AWS` is true, the handler creates an `AwsHandler` instance (`aws_handler`) that can be accessed
  via `settings.AWS_S3_HANDLER`.
- **Logging** – The logger configuration (`logger` section in the config) is applied on first call to
  `system_handler.logger()`.
- **Authentication** – Depending on flags (`ENV_USE_KC_AUTH`, `ENV_USE_OAUTH_V1_AUTH`, `ENV_USE_OAUTH_V2_AUTH`), the
  project either loads `rdl_authorization` middleware or falls back to DRF token authentication.

### Extending the Core

- **Add a new environment variable** – Define a constant in `constants.py`, add handling in
  `SystemSettingsHandler._get_configuration_for_variable`, and reference it where needed.
- **Custom error** – Create a new error entry in a sub‑app’s `errors.py` (e.g., `MY_NEW_ERROR = "MY_NEW_ERROR"`), add a
  dictionary with `ECODE`, `MSG`, then append to `ALL_ERRORS`. Run `build_errors_map()` once at startup (already done in
  `error_response`).
- **New logger sink** – Extend `Logger.load_from_dict()` to accept a custom logging configuration and call it from
  `SystemSettingsHandler.logger()`.
