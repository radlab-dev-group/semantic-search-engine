"""Boolean keyword composition and API enum validation, without a DB query."""

from unittest import TestCase

from sse_api.core.validation import search_options
from sse_api.data.core.knowledge_extensions import build_fts_query


class Query:
    def __init__(self, value, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def __or__(self, other):
        return Query(("OR", self.value, other.value))

    def __and__(self, other):
        return Query(("AND", self.value, other.value))


class KeywordsMatchTests(TestCase):
    def test_default_mode_is_and(self):
        options = search_options({"keywords": ["TypeError", "pool"]})
        self.assertNotIn("keywords_match", options)
        query = build_fts_query(
            "ignored",
            options["keywords"],
            options.get("keywords_match", "and"),
            "simple",
            Query,
        )
        self.assertEqual(query.value, ("AND", "TypeError", "pool"))

    def test_explicit_modes_compose_between_entries(self):
        for mode, expected in (("and", "AND"), ("any", "OR")):
            with self.subTest(mode=mode):
                self.assertEqual(
                    build_fts_query(
                        "unused", ["pool timeout", "TypeError"], mode, "simple", Query
                    ).value,
                    (expected, "pool timeout", "TypeError"),
                )

    def test_no_keywords_preserves_plain_query(self):
        for words in (None, []):
            for mode in ("and", "any"):
                query = build_fts_query("original query", words, mode, "simple", Query)
                self.assertEqual(query.value, "original query")
                self.assertEqual(query.kwargs["search_type"], "plain")

    def test_invalid_enum_is_rejected(self):
        for value in ("all", "or", "AND", "", None, True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                search_options({"keywords_match": value})

    def test_real_django_query_uses_safe_plain_parameters(self):
        from django.contrib.postgres.search import SearchQuery

        for mode, operator in (("and", "AND"), ("any", "OR")):
            query = build_fts_query(
                "unused", ["foo | !bar", "pool timeout"], mode, "simple", SearchQuery
            )
            self.assertEqual(query.connector, "&&" if mode == "and" else "||")
            self.assertEqual(query.lhs.function, "plainto_tsquery")
            self.assertEqual(query.rhs.function, "plainto_tsquery")
