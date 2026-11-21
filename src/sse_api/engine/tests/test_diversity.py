"""
test_diversity.py
-----------------

The result diversification of plan 05: a single document may not occupy the
whole answer, and a repeated document pays for it with its rank.

The helper is pure, so these tests are pure as well -- no Django, no
database, no models.  A minimal settings configuration is applied only so
that the engine package can be imported at all.
"""

import copy
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
        MAIN_LOGGER=__import__("logging").getLogger("test_diversity"),
    )
    django.setup()

from sse_api.engine.core.diversity import (  # noqa: E402
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    apply_result_diversity,
    document_name_of,
    relevance_of,
)


def _hit(document_name: str, score, text_number: int = 0) -> dict:
    return {
        "result": {
            "text": {
                "document_name": document_name,
                "score": score,
                "text_number": text_number,
                "text_str": f"{document_name} #{text_number}",
            }
        }
    }


def _names(results) -> list:
    return [document_name_of(result) for result in results]


def test_defaults_of_the_module():
    assert DEFAULT_MAX_CHUNKS_PER_DOCUMENT == 2


# ---------------------------------------------------------------------------
# The cap
# ---------------------------------------------------------------------------


def test_cap_keeps_the_answer_of_the_search():
    """Nothing is lost: a capped result is moved back, never dropped."""
    results = [_hit("manual.pdf", 0.99, number) for number in range(6)]
    diversified = apply_result_diversity(
        results, max_results=6, max_chunks_per_document=2
    )
    assert len(diversified) == 6
    assert len({id(hit) for hit in diversified}) == 6


def test_cap_lets_other_documents_through():
    results = [
        _hit("manual.pdf", 0.99, 0),
        _hit("manual.pdf", 0.98, 1),
        _hit("manual.pdf", 0.97, 2),
        _hit("note.pdf", 0.70, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=2
    )
    assert _names(diversified) == ["manual.pdf", "manual.pdf", "note.pdf"]
    assert len(diversified) == 3


def test_more_than_the_cap_are_backfilled_behind_the_others():
    results = [
        _hit("manual.pdf", 0.99, 0),
        _hit("manual.pdf", 0.98, 1),
        _hit("manual.pdf", 0.97, 2),
        _hit("note.pdf", 0.70, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=4, max_chunks_per_document=2
    )
    # note.pdf is promoted into the first pass, the surplus chunk of the
    # manual comes back behind it
    assert _names(diversified) == [
        "manual.pdf",
        "manual.pdf",
        "note.pdf",
        "manual.pdf",
    ]


def test_cap_below_one_behaves_like_one():
    results = [_hit("a.pdf", 0.9, 0), _hit("b.pdf", 0.8, 0)]
    assert len(apply_result_diversity(results, 2, 0)) == 2
    assert _names(apply_result_diversity(results, 2, 0))[0] == "a.pdf"


def test_cap_does_not_modify_the_input():
    results = [_hit("a.pdf", 0.9, number) for number in range(5)]
    before = copy.deepcopy(results)
    apply_result_diversity(results, max_results=2, max_chunks_per_document=1)
    assert results == before


# ---------------------------------------------------------------------------
# The decay
# ---------------------------------------------------------------------------


def test_decay_demotes_a_repeated_document_below_a_close_one():
    """A second chunk of a document stays on top only when its relevance
    is clearly above the chunk of another document.

    With the default decay of 0.85 a repeat pays 0.15 of the relevance
    range; a gap of 0.1 in the raw scores (0.2 of the range here) is not
    enough.  With a stronger decay of 0.5 it is.
    """
    results = [
        _hit("manual.pdf", 0.90, 0),
        _hit("manual.pdf", 0.86, 1),
        _hit("note.pdf", 0.85, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=3, diversity_decay=0.85
    )
    assert _names(diversified) == ["manual.pdf", "manual.pdf", "note.pdf"]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=3, diversity_decay=0.5
    )
    assert _names(diversified) == ["manual.pdf", "note.pdf", "manual.pdf"]


def test_decay_keeping_the_order_leaves_the_ranking_alone():
    results = [
        _hit("manual.pdf", 0.90, 0),
        _hit("manual.pdf", 0.86, 1),
        _hit("note.pdf", 0.85, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=3, diversity_decay=1.0
    )
    assert _names(diversified) == ["manual.pdf", "manual.pdf", "note.pdf"]


def test_a_clearly_better_second_chunk_stays_on_top():
    results = [
        _hit("manual.pdf", 1.0, 0),
        _hit("manual.pdf", 0.99, 1),
        _hit("note.pdf", 0.10, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=2, max_chunks_per_document=2, diversity_decay=0.5
    )
    assert _names(diversified) == ["manual.pdf", "manual.pdf"]


def test_without_scores_the_ranking_of_the_pipeline_is_kept():
    """A reranker logit, a cosine and an RRF sum are not comparable with a
    missing score, so the decay stands aside and only the cap applies."""
    results = [
        _hit("manual.pdf", None, 0),
        _hit("manual.pdf", None, 1),
        _hit("note.pdf", None, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=3
    )
    assert _names(diversified) == ["manual.pdf", "manual.pdf", "note.pdf"]


def test_negative_scores_keep_their_ordering():
    """Reranker logits are negative; a penalty is subtracted from the
    normalised relevance, so it can never reward a worse logit."""
    results = [
        _hit("manual.pdf", -1.0, 0),
        _hit("manual.pdf", -1.2, 1),
        _hit("note.pdf", -1.1, 0),
    ]
    diversified = apply_result_diversity(
        results, max_results=2, max_chunks_per_document=2, diversity_decay=0.5
    )
    assert _names(diversified) == ["manual.pdf", "note.pdf"]


# ---------------------------------------------------------------------------
# Shapes and edges
# ---------------------------------------------------------------------------


def test_flat_display_shape_is_understood():
    results = [
        {"document_name": "a.pdf", "score": 0.9},
        {"document_name": "a.pdf", "score": 0.8},
        {"document_name": "b.pdf", "score": 0.7},
    ]
    diversified = apply_result_diversity(
        results, max_results=2, max_chunks_per_document=1
    )
    assert [hit["document_name"] for hit in diversified] == ["a.pdf", "b.pdf"]


def test_results_without_a_document_name_are_never_capped():
    results = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]
    diversified = apply_result_diversity(
        results, max_results=3, max_chunks_per_document=1
    )
    assert len(diversified) == 3


def test_empty_input_and_empty_budget():
    assert apply_result_diversity([], max_results=5) == []
    assert apply_result_diversity([_hit("a.pdf", 0.5)], max_results=0) == []


def test_more_chunks_than_the_list_holds():
    results = [_hit("a.pdf", 0.5, number) for number in range(4)]
    diversified = apply_result_diversity(
        results, max_results=10, max_chunks_per_document=2
    )
    assert len(diversified) == 4


def test_relevance_of_reads_both_shapes():
    assert relevance_of(_hit("a.pdf", 0.42)) == 0.42
    assert relevance_of({"score": 0.11}) == 0.11
    assert relevance_of({"score": "0.5"}) is None
    assert relevance_of({"score": True}) is None
    assert relevance_of({}) is None
