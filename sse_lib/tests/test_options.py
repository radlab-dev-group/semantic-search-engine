"""
test_options.py
---------------

Unit tests for the payload builders in :mod:`sse_lib.options`.
"""

from __future__ import annotations

import json
import unittest

from sse_lib import SSEValueError, build_options
from sse_lib.options import (
    GenerativeOptions,
    IndexingOptions,
    SearchOptions,
    text_document,
)

INDEXING_KEYS = {
    "prepare_proper_pages",
    "merge_document_pages",
    "clear_text",
    "use_text_denoiser",
    "max_tokens_in_chunk",
    "number_of_overlap_tokens",
    "check_text_lang",
}


class SearchOptionsTest(unittest.TestCase):
    def test_only_set_options_are_sent(self):
        options = SearchOptions(
            categories=["Law"], max_results=20, rerank_results=True
        )
        self.assertEqual(
            options.to_dict(),
            {"categories": ["Law"], "max_results": 20, "rerank_results": True},
        )

    def test_extra_keys_are_merged(self):
        options = SearchOptions(max_results=5, extra={"rrf_k": 42})
        self.assertEqual(options.to_dict(), {"max_results": 5, "rrf_k": 42})

    def test_to_json_is_the_wire_format_of_the_api(self):
        options = SearchOptions(hybrid_search=False)
        self.assertEqual(json.loads(options.to_json()), {"hybrid_search": False})


class IndexingOptionsTest(unittest.TestCase):
    def test_all_keys_required_by_the_server_are_present(self):
        self.assertEqual(set(IndexingOptions().to_dict()), INDEXING_KEYS)

    def test_values_are_kept(self):
        options = IndexingOptions(max_tokens_in_chunk=200, use_text_denoiser=True)
        payload = options.to_dict()
        self.assertEqual(payload["max_tokens_in_chunk"], 200)
        self.assertTrue(payload["use_text_denoiser"])
        self.assertFalse(payload["merge_document_pages"])

    def test_json_string_for_multipart_forms(self):
        payload = IndexingOptions().to_json()
        self.assertEqual(set(json.loads(payload)), INDEXING_KEYS)


class GenerativeOptionsTest(unittest.TestCase):
    def test_none_values_are_dropped(self):
        options = GenerativeOptions(
            generative_model="radlab/model", percentage_rank_mass=0.3
        )
        self.assertEqual(
            options.to_dict(),
            {"generative_model": "radlab/model", "percentage_rank_mass": 0.3},
        )


class BuildOptionsTest(unittest.TestCase):
    def test_kwargs_win_over_the_options_mapping(self):
        options = build_options(
            SearchOptions, {"max_results": 10}, {"max_results": 3}
        )
        self.assertEqual(options.max_results, 3)

    def test_dict_and_dataclass_are_accepted(self):
        from_dataclass = build_options(
            SearchOptions, SearchOptions(categories=["A"]), None
        )
        from_dict = build_options(SearchOptions, {"categories": ["A"]}, None)
        self.assertEqual(from_dataclass.to_dict(), from_dict.to_dict())

    def test_unknown_names_land_in_extra_and_are_still_sent(self):
        options = build_options(SearchOptions, None, {"brand_new_option": 1})
        self.assertEqual(options.to_dict(), {"brand_new_option": 1})

    def test_unknown_names_are_rejected_without_extra(self):
        with self.assertRaises(SSEValueError):
            build_options(IndexingOptions, None, {"definitely_not_an_option": 1})

    def test_wrong_type_is_rejected(self):
        with self.assertRaises(SSEValueError):
            build_options(SearchOptions, 42, None)

    def test_extra_of_a_dataclass_is_preserved(self):
        options = build_options(
            SearchOptions, SearchOptions(extra={"rrf_k": 7}), None
        )
        self.assertEqual(options.to_dict(), {"rrf_k": 7})


class TextDocumentTest(unittest.TestCase):
    def test_default_shape_matches_the_api(self):
        document = text_document("some text")
        self.assertEqual(
            document,
            {
                "filepath": "text-document",
                "relative_filepath": "text-document",
                "category": "",
                "pages": [{"page_content": "some text"}],
            },
        )

    def test_relative_filepath_is_never_empty(self):
        """The server validates it with required_text; "" was rejected."""
        for kwargs in ({}, {"name": "report"}, {"relative_path": "a/b.txt"}):
            with self.subTest(**kwargs):
                document = text_document("text", **kwargs)
                self.assertTrue(document["relative_filepath"].strip())

    def test_name_category_and_metadata(self):
        document = text_document(
            "text", name="report", category="Finance", options={"author": "nk"}
        )
        self.assertEqual(document["filepath"], "report")
        self.assertEqual(document["category"], "Finance")
        self.assertEqual(document["options"], {"author": "nk"})

    def test_pages_override_the_single_text(self):
        document = text_document("ignored", pages=["a", {"page_content": "b"}])
        self.assertEqual(
            document["pages"], [{"page_content": "a"}, {"page_content": "b"}]
        )


if __name__ == "__main__":
    unittest.main()
