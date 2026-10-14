from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from sse_api.engine.controllers.search.relational import DBTextSearchController
from sse_api.engine.core.search_engine import reformat_search_results_to_display

# (id, document_id, text_chunk_type, page_number, text_number, text_str)
# Document 10 (md) has no reader page metadata, so every chunk sits on its own
# page; document 20 (pdf) has two chunks per page; text 9 has another type.
TEXTS = [
    (1, 10, "chunk_400", 1, 1, "md one"),
    (2, 10, "chunk_400", 2, 2, "md two"),
    (3, 10, "chunk_400", 3, 3, "md three"),
    (4, 10, "chunk_400", 4, 4, "md four"),
    (9, 10, "chunk_200", 2, 1, "other type"),
    (5, 20, "chunk_400", 1, 1, "pdf p1 c1"),
    (6, 20, "chunk_400", 1, 2, "pdf p1 c2"),
    (7, 20, "chunk_400", 2, 1, "pdf p2 c1"),
]


def _text(row):
    text_id, document_id, chunk_type, page_number, text_number, text_str = row
    document = SimpleNamespace(name=f"doc{document_id}", relative_path="/")
    return SimpleNamespace(
        id=text_id,
        text_number=text_number,
        text_str=text_str,
        language="pl",
        text_chunk_type=chunk_type,
        page=SimpleNamespace(
            document_id=document_id, document=document, page_number=page_number
        ),
    )


def _fake_objects():
    def filter_(**kwargs):
        queryset = MagicMock()
        if "id__in" in kwargs:
            ids = set(kwargs["id__in"])
            rows = [r for r in TEXTS if r[0] in ids]
            queryset.select_related.return_value = [_text(r) for r in rows]
            queryset.values_list.return_value = [(r[0], r[5]) for r in rows]
        else:
            documents = set(kwargs["page__document_id__in"])
            rows = sorted(
                (r for r in TEXTS if r[1] in documents),
                key=lambda r: (r[1], r[3], r[4], r[0]),
            )
            queryset.order_by.return_value.values_list.return_value = [
                (r[0], r[1], r[2], r[4]) for r in rows
            ]
        return queryset

    objects = MagicMock()
    objects.filter.side_effect = filter_
    return objects


class SurroundingContextTests(SimpleTestCase):
    def _get(self, ids, surrounding_chunks=2):
        with patch(
            "sse_api.engine.controllers.search.relational.DocumentPageText.objects",
            _fake_objects(),
        ):
            results = DBTextSearchController().get_texts(
                texts_ids=ids,
                texts_scores=[1.0] * len(ids),
                surrounding_chunks=surrounding_chunks,
            )
        return [
            (
                [c["text_str"] for c in r["result"]["left_context"]],
                r["result"]["text"]["text_str"],
                [c["text_str"] for c in r["result"]["right_context"]],
            )
            for r in results
        ]

    def test_context_crosses_pages_of_documents_without_page_metadata(self):
        self.assertEqual(
            self._get([3]), [(["md one", "md two"], "md three", ["md four"])]
        )

    def test_context_follows_reading_order_and_document_boundaries(self):
        self.assertEqual(
            self._get([6, 5]),
            [
                (["pdf p1 c1"], "pdf p1 c2", ["pdf p2 c1"]),
                ([], "pdf p1 c1", ["pdf p1 c2", "pdf p2 c1"]),
            ],
        )

    def test_chunks_of_another_chunk_type_are_not_context(self):
        self.assertEqual(self._get([2], 1), [(["md one"], "md two", ["md three"])])

    def test_no_context_when_not_requested(self):
        self.assertEqual(self._get([2], 0), [([], "md two", [])])

    def test_display_format_uses_the_nearest_neighbours(self):
        with patch(
            "sse_api.engine.controllers.search.relational.DocumentPageText.objects",
            _fake_objects(),
        ):
            results = DBTextSearchController().get_texts(
                texts_ids=[2], texts_scores=[1.0], surrounding_chunks=2
            )
        display = reformat_search_results_to_display(results)
        self.assertEqual(display[0]["left_context"], "md one")
        self.assertEqual(display[0]["right_context"], "md three")
