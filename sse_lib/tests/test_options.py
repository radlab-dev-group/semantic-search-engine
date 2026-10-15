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
    chunking_payload,
    text_document,
)

#: The options the server reads with ``indexing_options["..."]``.
INDEXING_KEYS = {
    "prepare_proper_pages",
    "merge_document_pages",
    "clear_text",
    "use_text_denoiser",
    "check_text_lang",
}
CHUNK_KEYS = {"max_tokens_in_chunk", "number_of_overlap_tokens"}


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

    def test_chunk_keys_are_sent_only_when_set(self):
        # Chunking belongs to the collection: what is not set there is not
        # sent, so that the settings of the collection apply.
        self.assertEqual(
            set(IndexingOptions().to_dict()) & CHUNK_KEYS,
            set(),
        )
        payload = IndexingOptions(max_tokens_in_chunk=200).to_dict()
        self.assertEqual(payload["max_tokens_in_chunk"], 200)
        self.assertNotIn("number_of_overlap_tokens", payload)

    def test_values_are_kept(self):
        options = IndexingOptions(max_tokens_in_chunk=200, use_text_denoiser=True)
        payload = options.to_dict()
        self.assertEqual(payload["max_tokens_in_chunk"], 200)
        self.assertTrue(payload["use_text_denoiser"])
        self.assertFalse(payload["merge_document_pages"])

    def test_json_string_for_multipart_forms(self):
        payload = IndexingOptions().to_json()
        self.assertEqual(set(json.loads(payload)), INDEXING_KEYS)


class ChunkingPayloadTest(unittest.TestCase):
    def test_complete_chunking_is_normalized(self):
        self.assertEqual(
            chunking_payload(
                {"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64}
            ),
            {"max_tokens_in_chunk": 500, "number_of_overlap_tokens": 64},
        )

    def test_overlap_defaults_to_none(self):
        self.assertEqual(
            chunking_payload({"max_tokens_in_chunk": 300}),
            {"max_tokens_in_chunk": 300, "number_of_overlap_tokens": 0},
        )

    def test_indexing_options_are_accepted(self):
        self.assertEqual(
            chunking_payload(IndexingOptions(max_tokens_in_chunk=128)),
            {"max_tokens_in_chunk": 128, "number_of_overlap_tokens": 0},
        )

    def test_chunk_size_is_required(self):
        with self.assertRaises(SSEValueError) as caught:
            chunking_payload({"number_of_overlap_tokens": 10})
        self.assertIn("max_tokens_in_chunk", str(caught.exception))

    def test_overlapping_must_stay_below_the_chunk(self):
        for chunking in (
            {"max_tokens_in_chunk": 128, "number_of_overlap_tokens": 128},
            {"max_tokens_in_chunk": 0},
            {"max_tokens_in_chunk": -5},
            {"max_tokens_in_chunk": "many"},
        ):
            with self.subTest(chunking=chunking):
                with self.assertRaises(SSEValueError):
                    chunking_payload(chunking)


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
