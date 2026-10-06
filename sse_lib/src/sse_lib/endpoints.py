"""
endpoints.py
------------

Every REST endpoint exposed by the SSE backend, in one place.

The Django views live in ``sse_rest_api/{data,engine,chat,system}/api.py`` and
their paths are built with ``main.src.constants.prepare_api_url``, i.e.
``<root_url>/<endpoint>`` where ``root_url`` comes from the ``api`` section of
``configs/django-config.json`` (``api`` by default, ``api/v1`` once the
deployment is versioned).  Only the endpoint name is hardcoded here; the prefix
is a client setting.

Paths are deliberately stored **without** a trailing slash because that is how
Django registers them (``path("api/collections", ...)``); a trailing slash is
not redirected and ends with a 404.
"""

from __future__ import annotations

# Prefix prepended to every endpoint when the server config does not say
# otherwise (``main.api.root_url`` in ``django-config.json``).
DEFAULT_API_PREFIX = "api"


# --- authentication -----------------------------------------------------------
LOGIN = "login"
LOGOUT = "logout"
LOGIN_URL = "login_url"
REFRESH_TOKEN = "refresh_token"


# --- collections & indexing (sse_rest_api/data) -------------------------------
NEW_COLLECTION = "new_collection"
COLLECTIONS = "collections"
CATEGORIES = "categories"
DOCUMENTS = "documents"
UPLOAD_AND_INDEX_FILES = "upload_and_index_files"
ADD_AND_INDEX_TEXTS = "add_and_index_texts"
QUESTION_TEMPLATES = "question_templates"
FILTER_OPTIONS = "filter_options"


# --- search & generation (sse_rest_api/engine) --------------------------------
SEARCH_WITH_OPTIONS = "search_with_options"
GENERATIVE_ANSWER = "generative_answer"
RATE_GENERATIVE_ANSWER = "rate_generative_answer"
GENERATIVE_MODELS = "generative_models"
EMBEDDERS = "embedders"
RERANKERS = "rerankers"


# --- chat (sse_rest_api/chat) -------------------------------------------------
NEW_CHAT = "new_chat"
ADD_USER_MESSAGE = "add_user_message"
SAVE_CHAT = "save_chat"
GET_CHAT_BY_HASH = "get_chat_by_hash"
CHATS = "chats"


ALL_ENDPOINTS = frozenset(
    {
        LOGIN,
        LOGOUT,
        LOGIN_URL,
        REFRESH_TOKEN,
        NEW_COLLECTION,
        COLLECTIONS,
        CATEGORIES,
        DOCUMENTS,
        UPLOAD_AND_INDEX_FILES,
        ADD_AND_INDEX_TEXTS,
        QUESTION_TEMPLATES,
        FILTER_OPTIONS,
        SEARCH_WITH_OPTIONS,
        GENERATIVE_ANSWER,
        RATE_GENERATIVE_ANSWER,
        GENERATIVE_MODELS,
        EMBEDDERS,
        RERANKERS,
        NEW_CHAT,
        ADD_USER_MESSAGE,
        SAVE_CHAT,
        GET_CHAT_BY_HASH,
        CHATS,
    }
)
