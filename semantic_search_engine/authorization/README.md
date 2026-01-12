# authorization module – documentation

## 1. Purpose

The **authorization** module implements the complete OAuth / OpenID‑Connect (Keycloak‑compatible) authentication flow
for the Semantic Search Engine system. It is responsible for:

* Generating a CSRF‑protected *state* token and the corresponding login URL.
* Exchanging an authorization *code* for access and refresh tokens.
* Verifying, decoding and (optionally) introspecting JWTs.
* Persisting tokens and their associated session state in the database.
* Creating or locating a Django `User` based on the claims contained in the token.
* Providing a DRF‑based API that the rest of the system can call (`/auth/login_url/`, `/auth/login/`, `/auth/logout/`,
  `/auth/refresh_token/`).
* Supplying a middleware that attaches the authenticated user (or an anonymous user) to every incoming request.

All of the above is driven by a **single JSON configuration file** (`configs/auth-config.json`).

---

## 2. Configuration

The module expects a JSON object under the key **`authorization`**.  
Below is the *real* configuration that matches the current deployment:

```json
{
  "authorization": {
    "auth_host": "https://login.radlab.dev",
    "realm": "",
    "client_id": "",
    "client_secret": "",
    "grant_type": "authorization_code",
    "redirect_uri": "http://0.0.0.0:8000/",
    "public_sign_key": "",
    "audience": "account",
    "scope": ".default",
    "algorithms": [
      "RS256",
      "AES",
      "HS512",
      "RSA-QEAP"
    ],
    "user_role_main_scope": "resource_access",
    "user_role_scope_variable": "account",
    "accepted_user_roles": [
      "superadmin"
    ],
    "default_organisation": {
      "name": "sse_default",
      "display_name": "SSE Default Organisation",
      "description": "Semantic search engine default organisation"
    },
    "default_group": {
      "name": "sse_default_users",
      "display_name": "SSE default user group",
      "description": "Semantic search engine default user group"
    }
  }
}
```

* `auth_host` – base URL of the authentication provider.
* `realm`, `client_id`, `client_secret` – values supplied by the IdP.
* `grant_type` – must be `"authorization_code"` for the standard flow.
* `redirect_uri` – URL the IdP redirects back to after a successful login.
* `public_sign_key` – optional public key used to verify JWT signatures.
* `audience`, `scope`, `algorithms` – passed to the token request / verification logic.
* `user_role_main_scope` & `user_role_scope_variable` – where the module looks for the list of roles inside the decoded
  JWT.
* `accepted_user_roles` – list of roles that are allowed to use the system (`["superadmin"]` in the example).
* `default_organisation` / `default_group` – metadata used when the module creates a new Django user.

The configuration file is read once at start‑up by `authorization.utils.config.RdlAuthConfig` and its values are exposed
as read‑only properties (`auth_host`, `client_id`, …).

If you need to load a different configuration (e.g., for tests), instantiate the class with an explicit path:

```python
from authorization.utils.config import RdlAuthConfig

cfg = RdlAuthConfig(cfg_path="tests/configs/auth-test.json")
```

---

## 3. Database models

| Model            | Fields (relevant)                                                                                                                                                     | Description                                                                                                                                              |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SessionState** | `state` (PK), `session_state`, `grant_code`, `state_created`                                                                                                          | Stores the random *state* string generated for the login URL and, after the callback, the `grant_code` and optional `session_state` returned by the IdP. |
| **Token**        | `acc_token`, `refresh_token`, `decoded_token`, `is_active`, `state` (FK → `SessionState`), `auth_user` (FK → `User`), plus optional fields (`expires_in`, `scope`, …) | Persists the access token and related metadata. Only one token per user is active at a time; older tokens are de‑activated automatically.                |

Both models are defined in `authorization/models.py`.

---

## 4. Core classes

### 4.1 `RdlAuthStateHandler` ( `authorization.core.handlers` )

* Generates a random 16‑character alphanumeric *state* (`DEFAULT_STATE_LENGTH`).
* Guarantees uniqueness by checking the DB before persisting (`SessionState`).
* Builds the login URL using the configuration values and the selected authentication mode (Keycloak, OAuth v1/v2).

### 4.2 `RdlAuthGrantAccTokenHandler`

* Performs the **token request** (grant‑code flow or refresh‑token flow).
* Verifies the JWT signature when a `public_sign_key` is supplied; otherwise decodes without verification (
  `verify_signature=False`).
* Calls `__get_or_add_user_from_user_info` to obtain a Django `User` (creates one if missing).
* Stores the token in the `Token` model, de‑activating any previous token for the same user.
* Provides helper methods:

| Method                        | Purpose                                                                              |
|-------------------------------|--------------------------------------------------------------------------------------|
| `get_token_with_options`      | Request token from IdP (grant‑code or refresh).                                      |
| `get_rdl_auth_user_for_token` | Resolve a `User` + `Token` from a freshly‑obtained access token.                     |
| `disable_all_user_tokens`     | De‑activate every active token belonging to a user and call the IdP logout endpoint. |
| `introspect_user_token`       | Optional introspection (when `SYSTEM_HANDLER.use_introspect` is true).               |
| `verify_and_decode_token`     | Verify signature (if possible) and return the decoded payload.                       |

### 4.3 `GATokenAuthentication` (`authorization.core.authentication`)

* Extracts the bearer token from the `Authorization` header (`Bearer <token>`).
* If a token is present, looks it up in the DB (`Token.objects.get(is_active=True, acc_token=…)`).
* If a `session_state` query or POST parameter is supplied, looks up a token by that value.
* Validates that the decoded token contains at least one role from `accepted_user_roles`.
* Returns `(user, token_string)` or `(None, "")` for unauthenticated requests.

### 4.4 `AuthAuthenticationMiddleware` (`authorization.core.middleware`)

* Runs **before** any view.
* Calls `GATokenAuthentication.get_user_token_for_request`.
* If a token is found, sets `request.user` to `token.auth_user`.
* If no token is found and introspection is enabled, attempts introspection and sets `request.user` accordingly.
* Falls back to `AnonymousUser` when no authentication data can be resolved.

---

## 5. API endpoints

The module registers its URLs in `authorization/urls.py`.  
Add the module’s URL namespace to the main router (e.g., `path("auth/", include("authorization.urls"))`).

| HTTP method | URL (relative to the `auth/` prefix) | View class / callable                                 | Result                                                               |
|-------------|--------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------|
| `POST`      | `login_url/`                         | `GenerateLoginUrl`                                    | `{ "login_url": "<generated URL>" }`                                 |
| `POST`      | `login/`                             | `obtain_auth_token` (`CreateRdlAuthToken.as_view()`)  | `{ "token": "...", "refresh_token": "...", "decoded_token": { … } }` |
| `POST`      | `logout/`                            | `remove_auth_token` (`DisableRdlAuthToken.as_view()`) | `{ "status": true }` (all user tokens disabled)                      |
| `POST`      | `refresh_token/`                     | `RefreshToken`                                        | New access‑token set, same shape as *login* response.                |

All responses are produced by `main.src.response.response_with_status` and follow the unified schema:

```json
{
  "status": true
  |
  false,
  "response_body": {
    …
  }
  // present only on success
  "error_name": "TOKEN_LOGIN_PROBLEM",
  // present only on failure
  "language": "en"
}
```

---

## 6. Error handling

Error identifiers are defined in `authorization/core/errors.py` and merged into the global `ALL_ERRORS` list.

Key identifiers used by the module:

| Identifier              | Meaning                                                 |
|-------------------------|---------------------------------------------------------|
| `STATE_GEN_PROBLEM`     | Failed to generate a unique state token.                |
| `LOGIN_URL_GEN_PROBLEM` | Could not build the login URL.                          |
| `TOKEN_LOGIN_PROBLEM`   | Token request (grant‑code) failed.                      |
| `NO_USER_FOR_TOKEN`     | No user could be matched/created for the decoded token. |
| `TOKEN_REFRESH_PROBLEM` | Refresh‑token request failed.                           |
| `TOKEN_DISABLE_FAILED`  | Error while trying to deactivate user tokens.           |

The API returns the identifier in the `error_name` field; the human‑readable message is taken from the language‑specific
dictionary inside `RDL_AUTH_ERRORS`.

---

## 7. Extending the module

* **Custom user model** – replace the import of `django.contrib.auth.models.User` in `handlers.py` and `models.py` with
  your own model and adjust the foreign‑key definitions.
* **Additional token attributes** – add new fields to `Token` and extend
  `RdlAuthGrantAccTokenHandler.__manage_token_options` to copy them from the IdP response.
* **Different role‑mapping logic** – modify `GATokenAuthentication.__is_user_role_ok` to use a different claim path or
  to implement more complex role aggregation.

All changes should keep the public API (the DRF view classes, the middleware, and the config loader) unchanged so that
the rest of the system continues to interact with the module transparently.

---

## 8. Summary

The **authorization** module is the authentication backbone of the Semantic Search Engine system. By configuring the
JSON file correctly and wiring the URLs and middleware into the main Django project, the system gains:

* Secure, CSRF‑protected login URL generation.
* Automatic handling of token exchange, refresh, and revocation.
* Seamless user creation and role‑based access control.
* A unified response format for all authentication‑related endpoints.

All of this lives inside the `authorization` package and integrates tightly with the surrounding Django components.  