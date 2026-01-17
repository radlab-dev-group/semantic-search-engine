"""
urls.py
-------

URL configuration for the public API endpoints of the search engine.
Each view class implements a distinct operation (search, generative
answer, rating, model listing, …).  The ``prepare_api_url`` helper
adds the appropriate prefix (e.g. ``/api/v1/``) defined in project
settings.
"""

from django.urls import path

from main.src.constants import prepare_api_url
from engine.api import (
    SearchWithOptions,
    GenerativeAnswerForQuestion,
    ListGenerativeModels,
    ListEmbeddersModels,
    ListRerankersModels,
    SetRateForQueryResponseAnswer,
)

urlpatterns = [
    path(
        prepare_api_url("search_with_options"),
        SearchWithOptions.as_view(),
        name="search_with_options",
    ),
    path(
        prepare_api_url("generative_answer"),
        GenerativeAnswerForQuestion.as_view(),
        name="generative_answer",
    ),
    path(
        prepare_api_url("rate_generative_answer"),
        SetRateForQueryResponseAnswer.as_view(),
        name="rate_generative_answer",
    ),
    path(
        prepare_api_url("generative_models"),
        ListGenerativeModels.as_view(),
        name="generative_models",
    ),
    path(
        prepare_api_url("embedders"),
        ListEmbeddersModels.as_view(),
        name="embedders",
    ),
    path(
        prepare_api_url("rerankers"),
        ListRerankersModels.as_view(),
        name="embedders",
    ),
]
