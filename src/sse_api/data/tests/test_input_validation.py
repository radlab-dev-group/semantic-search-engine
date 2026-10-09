import json
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import force_authenticate

from sse_api.data.api import (
    NewCollection,
    ListCategoriesFromCollection,
    ListDocumentsFromCollection,
    UploadAndIndexFilesToCollection,
    AddAndIndexTextsFromEP,
)
from sse_api.tests.input_support import EndpointInputMixin, INDEXING_OPTIONS


class DataInputTests(EndpointInputMixin, TestCase):
    def test_unknown_index_and_bad_types_before_creation(self):
        payload = {key: "test" for key in NewCollection.required_params}
        with (
            patch("sse_api.data.api.SemanticDBController") as semantic,
            patch("sse_api.data.api.RelationalDBController") as relational,
        ):
            for key, value in (
                ("embedder_index_type", "unknown"),
                ("embedder_index_type", []),
                ("collection_name", None),
                ("collection_description", {}),
                ("group_name", False),
            ):
                with self.subTest(key=key, value=value):
                    self.assert_denial(
                        self.invoke(NewCollection, {**payload, key: value})
                    )
            semantic.assert_not_called()
            relational.assert_not_called()

    def test_get_query_body_fallback_and_per_key_precedence(self):
        with (
            patch(
                "sse_api.data.api.RelationalDBController.get_collection",
                return_value=object(),
            ) as lookup,
            patch(
                "sse_api.data.api.RelationalDBController.get_all_categories_from_collection",
                return_value=["a"],
            ),
        ):
            cases = [
                (
                    "/?collection_name=url&lang=en",
                    {"collection_name": "body", "lang": "pl"},
                    "url",
                ),
                ("/?lang=en", {"collection_name": "body", "lang": "pl"}, "body"),
                ("/", {"collection_name": "body", "lang": "en"}, "body"),
            ]
            for url, body, expected in cases:
                with self.subTest(url=url):
                    response = self.invoke(
                        ListCategoriesFromCollection, url=url, body=body
                    )
                    self.assertTrue(response.data["status"])
                    self.assertEqual(
                        lookup.call_args.kwargs["collection_name"], expected
                    )
            self.assertTrue(
                self.invoke(
                    ListCategoriesFromCollection,
                    {"collection_name": "url"},
                    method="get",
                ).data["status"]
            )

    def test_get_bad_params_language_and_missing_collection(self):
        with (
            patch(
                "sse_api.data.api.RelationalDBController.get_collection",
                return_value=None,
            ) as lookup,
            patch(
                "sse_api.data.api.RelationalDBController.get_documents_to_search_from_collection"
            ) as documents,
        ):
            for view in (ListCategoriesFromCollection, ListDocumentsFromCollection):
                self.assert_denial(
                    self.invoke(view, {"collection_name": "missing"}, method="get")
                )
                lookup.reset_mock()
                self.assert_denial(
                    self.invoke(
                        view,
                        url="/?collection_name=&lang=en",
                        body={"collection_name": "body"},
                    )
                )
                self.assert_denial(
                    self.invoke(
                        view, url="/?lang=bad", body={"collection_name": "body"}
                    )
                )
                lookup.assert_not_called()
            documents.assert_not_called()

    def test_upload_invalid_options_before_storage(self):
        with patch("sse_api.data.api.UploadDocumentsController") as upload:
            for options in ("{", "[]", "null", "{}", "3"):
                with self.subTest(options=options):
                    request = self.factory.post(
                        "/",
                        {
                            "files[]": SimpleUploadedFile("test.txt", b"test"),
                            "collection_name": "docs",
                            "indexing_options": options,
                        },
                        format="multipart",
                    )
                    force_authenticate(request, user=self.user)
                    self.assert_denial(
                        UploadAndIndexFilesToCollection.as_view()(request)
                    )
            upload.assert_not_called()

    def test_text_input_bad_structure_or_options_before_write(self):
        document = {
            "filepath": "text",
            "relative_filepath": "text",
            "category": "Notes",
            "pages": [{"page_content": "text"}],
        }
        payload = {
            "texts[]": [document],
            "collection_name": "docs",
            "indexing_options": INDEXING_OPTIONS,
        }
        with (
            patch("sse_api.data.api.RelationalDBController") as relational,
            patch("sse_api.data.api.DBSemanticSearchController") as semantic,
        ):
            for key, value in (
                ("texts[]", []),
                ("texts[]", ["text"]),
                ("texts[]", [{}]),
                ("indexing_options", {}),
                ("indexing_options", "[]"),
                (
                    "indexing_options",
                    {**INDEXING_OPTIONS, "max_tokens_in_chunk": False},
                ),
                (
                    "indexing_options",
                    {**INDEXING_OPTIONS, "number_of_overlap_tokens": 128},
                ),
            ):
                with self.subTest(key=key, value=value):
                    self.assert_denial(
                        self.invoke(AddAndIndexTextsFromEP, {**payload, key: value})
                    )
            relational.assert_not_called()
            semantic.assert_not_called()

    def test_text_dict_contract_passes_normalized_options(self):
        payload = {
            "texts[]": [
                {
                    "filepath": "text",
                    "relative_filepath": "text",
                    "category": None,
                    "pages": [{"page_content": "text"}],
                }
            ],
            "collection_name": "docs",
            "indexing_options": INDEXING_OPTIONS,
        }
        with patch("sse_api.data.api.RelationalDBController") as relational:
            relational.return_value.add_documents_from_list_of_doc_as_dict.return_value = (
                []
            )
            for options in (INDEXING_OPTIONS, json.dumps(INDEXING_OPTIONS)):
                with self.subTest(options=options):
                    self.assertTrue(
                        self.invoke(
                            AddAndIndexTextsFromEP,
                            {
                                **payload,
                                "indexing_options": options,
                            },
                        ).data["status"]
                    )
                    add_documents = (
                        relational.return_value.add_documents_from_list_of_doc_as_dict
                    )
                    self.assertEqual(
                        add_documents.call_args.kwargs["indexing_options"],
                        INDEXING_OPTIONS,
                    )
