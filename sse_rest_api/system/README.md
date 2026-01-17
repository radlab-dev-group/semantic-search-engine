# 📦 `system` Django App – Quick Overview & Guide

This app provides the core **organisation management** functionality for the Semantic Search Engine project.  
It defines the data model, helper utilities, API endpoints, and controller logic needed to:

* Create and retrieve **Organizations**, **Organization Groups**, and **Organization Users**.
* Authenticate users belonging to an organization via a token‑based login endpoint.
* Enforce sensible defaults (default organisation, group, and user) through constants.

---  

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Concepts](#key-concepts)
3. [Configuration & Defaults](#configuration--defaults)
4. [Main Components](#main-components)
    - `models.py`
    - `controllers.py`
    - `api.py`
    - `decorators.py`
    - `errors.py` & `constants.py`
5. [URL Routing](#url-routing)
6. [How to Use the API](#how-to-use-the-api)
7. [Setup & Migrations](#setup--migrations)
8. [Running Tests (if any)](#running-tests)
9. [Further Reading](#further-reading)

---  

## Project Structure

```
system/
├── api.py           # REST API view(s)
├── apps.py          # Django AppConfig
├── controllers.py   # Business‑logic helpers
├── models.py        # DB schema
├── urls.py          # URL patterns for this app
├── decorators.py    # Request‑level decorators
├── errors.py        # Custom error catalogue
├── constants.py     # Default values & error markers
└── migrations/      # Auto‑generated DB migrations
```

---  

## Key Concepts

| Concept                  | Description                                                                                                                                                           |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Organisation**         | Top‑level entity (e.g., a company or research group).                                                                                                                 |
| **OrganisationGroup**    | Sub‑division inside an organisation (e.g., “admin”, “analysts”).                                                                                                      |
| **OrganisationUser**     | A Django `User` linked to a specific organisation and optionally to one or more groups.                                                                               |
| **Token authentication** | Uses Django‑REST‑Framework’s `Token` model; the login endpoint returns a token that must be supplied in the `Authorization: Token <key>` header for subsequent calls. |

---  

## Configuration & Defaults

The module ships with sensible defaults defined in **`constants.py`**:

```python
DEFAULT_ORGANISATION_NAME = "Default"
DEFAULT_ORGANISATION_DESCRIPTION = "Auto-created default organisation"

DEFAULT_USER_GROUP_NAME = "default"
DEFAULT_USER_GROUP_DESCRIPTION = "Auto-created default group"

DEFAULT_USER_NAME = "default_user"
DEFAULT_USER_PASS = "default_password"
DEFAULT_USER_EMAIL = "default@user.email"
```

These are used by any bootstrap scripts or management commands that need a ready‑to‑use organisation, group, and user.

Error handling uses a unique marker (`ERROR_MARK_SYSTEM = "__system"`) to namespace system‑level errors.

---  

## Main Components

### `models.py`

Defines three Django models:

* **`Organisation`** – primary key `name`, optional `description`.
* **`OrganisationGroup`** – `group_name`, `description`, and a foreign key to `Organisation`. Enforced uniqueness on
  `(group_name, organisation)`.
* **`OrganisationUser`** – links a Django `User` (`auth_user`) to an `Organisation` and holds a many‑to‑many
  relationship to `OrganisationGroup`.

All foreign keys use `PROTECT` to avoid accidental cascade deletions.

### `controllers.py`

A collection of static helper methods encapsulated in the `SystemController` class:

| Method                                                                    | Purpose                                                                                    |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `add_organisation(name, description)`                                     | Create (or fetch) an `Organisation`.                                                       |
| `add_organisation_group(organisation, name, description)`                 | Create (or fetch) a group within an organisation.                                          |
| `add_organisation_user(name, email, password, organisation, user_groups)` | Create a Django auth user and its `OrganisationUser` wrapper, optionally assigning groups. |
| `get_organisation_user(username)`                                         | Retrieve the `OrganisationUser` linked to a given auth username.                           |
| `get_organisation(organisation_name)`                                     | Fetch an `Organisation` by its name.                                                       |
| `get_organisation_group(organisation, group_name)`                        | Fetch a specific `OrganisationGroup`.                                                      |

A small utility class (`UserConfigReader`) reads a JSON configuration file (default: `configs/default-user.json`) that
can be used for initial data seeding.

### `api.py`

Provides a **login endpoint** (`OrganisationLogin`) derived from DRF’s `ObtainAuthToken`.

* Expected payload:

```json
{
  "username": "user",
  "password": "User0123"
}
```

* Returns:

```json
{
  "token": "<generated-token>"
}
```

The view validates required parameters using the `required_params_exists` decorator (imported from the main project).

### `decorators.py`

Contains `get_organisation_user`, a wrapper that extracts the `OrganisationUser` associated with the authenticated
request and injects it into the wrapped view method.

### `errors.py` & `constants.py`

`errors.py` builds a structured error catalogue (`ALL_ERRORS_DATA`) and merges it into the global `ALL_ERRORS` list
defined elsewhere in the project.  
Each error contains an error code (`ECODE`) and multilingual messages (`MSG_PL`, `MSG_EN`).

---  

## URL Routing

`system/urls.py` registers the login endpoint:

```python
urlpatterns = [
    path(
        prepare_api_url("login"),
        OrganisationLogin.as_view(),
        name="login",
    ),
]
```

`prepare_api_url` (from the main project) prefixes the path with the appropriate API version or base URL.

---  

## How to Use the API

1. **Obtain a token**

```shell script
curl -X POST http://<host>/api/v1/login/ \
        -H "Content-Type: application/json" \
        -d '{"username":"default_user","password":"default_password"}'
```

Response:

```json
{
  "token": "abcd1234efgh5678"
}
```

2. **Authenticated requests** – include the token in the `Authorization` header:

```shell script
curl -H "Authorization: Token abcd1234efgh5678" http://<host>/api/v1/your-protected-endpoint/
```

> **Note**: The current module only exposes the login endpoint; other resources are expected to be added by the broader
> application (e.g., search, indexing).

---  

## Setup & Migrations

1. **Add the app to `INSTALLED_APPS`** in your Django settings:

```python
INSTALLED_APPS = [
    # …
    "system",
]
```

2. **Run migrations**

```shell script
python manage.py makemigrations system
   python manage.py migrate
```

3. **(Optional) Seed defaults** – you can write a simple management command or script that uses `SystemController` to
   create the default organisation, group, and user based on the constants.

