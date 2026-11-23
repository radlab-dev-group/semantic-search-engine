"""
test_context_stitching.py
-------------------------

The context stitcher of plan 05: the neighbouring chunks of one document
are joined into one continuous passage before they are read by a model or
by a human, so a sentence cut by the chunking window is not cut twice.

The helpers are pure; a minimal Django configuration is applied only so
that the engine package can be imported.
"""

import os
import sys

_APP_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir
    )
)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import django  # noqa: E402
from django.conf import settings  # noqa: E402

if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.postgres",
            "sse_api.data",
            "sse_api.system",
            "sse_api.engine",
            "sse_api.chat",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_APP_LANGUAGE="pl",
        MAIN_API_URL="/api/",
        MAIN_LOGGER=__import__("logging").getLogger("test_context_stitching"),
    )
    django.setup()

from sse_api.engine.core.search_engine import (  # noqa: E402
    convert_search_results_to_doc2answer,
    convert_search_results_to_enumerated_documents,
    stitch_adjacent_chunks,
)


def _chunk(
    document_name: str,
    page_number: int,
    text_number: int,
    text_str: str,
    text_id: int,
    left_context: object = "",
    **extra,
) -> dict:
    chunk = {
        "document_name": document_name,
        "page_number": page_number,
        "text_number": text_number,
        "text_str": text_str,
        "text_id": text_id,
        "left_context": left_context,
    }
    chunk.update(extra)
    return chunk


# ---------------------------------------------------------------------------
# The proofs of adjacency
# ---------------------------------------------------------------------------


def test_chunks_of_the_same_page_are_stitched():
    chunks = [
        _chunk("a.pdf", 1, 2, "B", 12),
        _chunk("a.pdf", 1, 0, "A", 10),
        _chunk("a.pdf", 1, 1, "A2", 11),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert len(blocks) == 1
    assert blocks[0]["text_str"] == "A\nA2\nB"
    # the block speaks with the identity of its first part in the text
    assert blocks[0]["text_id"] == 10
    assert blocks[0]["page_number"] == 1


def test_chunks_of_two_pages_of_a_paged_document_are_not_stitched():
    """``text_number`` starts over on the next page, so what lies between
    the last chunk of one page and the first of the other is unknown."""
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10),
        _chunk("a.pdf", 2, 1, "next", 11),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert [block["text_id"] for block in blocks] == [10, 11]


def test_the_attached_context_proves_adjacency_across_pages():
    """Readers without page metadata (txt, md, html) put every chunk on a
    page of its own; the context the pipeline attaches to a hit is the
    proof that it follows another hit."""
    chunks = [
        _chunk("m.md", 3, 3, "three", 30, left_context="two"),
        _chunk("m.md", 1, 1, "one", 10),
        _chunk("m.md", 2, 2, "two", 20, left_context="one"),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert len(blocks) == 1
    assert blocks[0]["text_str"] == "one\ntwo\nthree"
    assert blocks[0]["text_id"] == 10


def test_the_context_of_another_text_does_not_prove_adjacency():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10),
        _chunk("a.pdf", 3, 3, "third", 12, left_context="second"),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert [block["text_id"] for block in blocks] == [10, 12]


def test_chunks_of_two_documents_are_never_stitched():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10),
        _chunk("b.pdf", 1, 2, "first", 20),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert len(blocks) == 2
    assert {block["document_name"] for block in blocks} == {"a.pdf", "b.pdf"}


# ---------------------------------------------------------------------------
# The shape of the merged block
# ---------------------------------------------------------------------------


def test_the_span_of_a_merged_block_is_the_widest_proven_span():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10, char_start=0, char_end=9),
        _chunk("a.pdf", 1, 2, "second", 11, char_start=9, char_end=18),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert blocks[0]["char_start"] == 0
    assert blocks[0]["char_end"] == 18


def test_an_unknown_span_of_a_part_leaves_the_span_unknown():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10, char_start=0, char_end=9),
        _chunk("a.pdf", 1, 2, "second", 11),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert blocks[0]["char_end"] is None


def test_the_score_of_a_merged_block_is_the_best_of_its_parts():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10, score=0.4),
        _chunk("a.pdf", 1, 2, "second", 11, score=0.9),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert blocks[0]["score"] == 0.9


def test_the_order_of_the_answer_is_preserved():
    """A merged block keeps the place of the first of its parts which the
    answer showed, and reads in the order of the document."""
    chunks = [
        _chunk("a.pdf", 1, 2, "B", 12),
        _chunk("b.pdf", 1, 1, "other", 20),
        _chunk("a.pdf", 1, 1, "A", 10),
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert [block["text_str"] for block in blocks] == ["A\nB", "other"]


def test_chunks_without_position_information_are_copied_not_merged():
    chunks = [
        {"text_str": "raw one", "document_name": "x"},
        {"text_str": "raw two", "document_name": "x"},
    ]
    blocks = stitch_adjacent_chunks(chunks)
    assert len(blocks) == 2
    assert blocks[0] == chunks[0]
    assert blocks[1] == chunks[1]


def test_the_input_is_not_modified():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10),
        _chunk("a.pdf", 1, 2, "second", 11),
    ]
    import copy

    before = copy.deepcopy(chunks)
    stitch_adjacent_chunks(chunks)
    assert chunks == before


def test_the_separator_is_configurable():
    chunks = [
        _chunk("a.pdf", 1, 1, "first", 10),
        _chunk("a.pdf", 1, 2, "second", 11),
    ]
    blocks = stitch_adjacent_chunks(chunks, separator=" ")
    assert blocks[0]["text_str"] == "first second"


# ---------------------------------------------------------------------------
# The converters of the RAG context
# ---------------------------------------------------------------------------


def _search_results():
    return {
        0: _chunk(
            "a.pdf",
            1,
            1,
            "first",
            10,
            document_id=1,
            char_start=0,
            char_end=9,
            score=0.1,
        ),
        1: _chunk(
            "a.pdf",
            1,
            2,
            "second",
            11,
            left_context="first",
            document_id=1,
            char_start=9,
            char_end=18,
            score=0.2,
        ),
        2: _chunk("b.pdf", 1, 1, "other", 20, document_id=2, score=0.3),
    }


def test_enumerated_documents_stitch_and_renumber():
    documents = convert_search_results_to_enumerated_documents(
        _search_results(), None
    )
    # one source per merged passage, numbered after the stitching
    assert [doc["n"] for doc in documents] == [1, 2]
    assert documents[0]["text_str"] == "first\nsecond"
    assert documents[0]["char_start"] == 0
    assert documents[0]["char_end"] == 18
    # the provenance of the merging is not part of a citation
    assert "text_number" not in documents[0]
    assert "left_context" not in documents[0]


def test_enumerated_documents_without_adjacency_keep_one_citation_per_chunk():
    results = {
        0: _chunk("a.pdf", 1, 1, "first", 10),
        1: _chunk("a.pdf", 3, 3, "third", 12),
    }
    documents = convert_search_results_to_enumerated_documents(results, None)
    assert [doc["n"] for doc in documents] == [1, 2]
    assert [doc["text_str"] for doc in documents] == ["first", "third"]


def test_doc2answer_stitching_keeps_the_documents_apart():
    answers = convert_search_results_to_doc2answer(_search_results(), None)
    assert list(answers.keys()) == ["a.pdf", "b.pdf"]
    assert answers["a.pdf"] == ["first\nsecond"]
    assert answers["b.pdf"] == ["other"]


def test_doc2answer_with_document_names():
    answers = convert_search_results_to_doc2answer(
        _search_results(), None, use_doc_names_in_response=True
    )
    assert answers["a.pdf"] == ["a.pdf: first\nsecond"]
