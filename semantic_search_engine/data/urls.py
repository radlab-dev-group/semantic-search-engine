from django.urls import path

from data.api import (
    NewCollection,
    UploadAndIndexFilesToCollection,
    AddAndIndexTextsFromEP,
    ListCollections,
    ListCategoriesFromCollection,
    ListDocumentsFromCollection,
    ListQuestionTemplates,
    ListFilteringOptions,
)
from main.src.constants import prepare_api_url

urlpatterns = [
    path(
        prepare_api_url("new_collection"),
        NewCollection.as_view(),
        name="new_collection",
    ),
    path(
        prepare_api_url("collections"),
        ListCollections.as_view(),
        name="collections",
    ),
    path(
        prepare_api_url("categories"),
        ListCategoriesFromCollection.as_view(),
        name="categories",
    ),
    path(
        prepare_api_url("documents"),
        ListDocumentsFromCollection.as_view(),
        name="documents",
    ),
    path(
        prepare_api_url("upload_and_index_files"),
        UploadAndIndexFilesToCollection.as_view(),
        name="upload_and_index_files",
    ),
    path(
        prepare_api_url("add_and_index_texts"),
        AddAndIndexTextsFromEP.as_view(),
        name="add_and_index_texts",
    ),
    path(
        prepare_api_url("question_templates"),
        ListQuestionTemplates.as_view(),
        name="question_templates",
    ),
    path(
        prepare_api_url("filter_options"),
        ListFilteringOptions.as_view(),
        name="question_templates",
    ),
]
