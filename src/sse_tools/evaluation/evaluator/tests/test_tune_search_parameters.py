"""
test_tune_search_parameters.py
------------------------------

The pure parts of the grid-search tuner: the shape of a golden dataset,
the enumeration of the grid, and the scoring of one question.  The tuner
itself talks to a database and a collection; the helpers below are the
ones which decide what is measured, so they are tested without either.
"""

import os
import sys

_APP_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from sse_tools.evaluation.evaluator.tune_search_parameters import (  # noqa: E402
    DEFAULT_GRID,
    combinations,
    load_golden_dataset,
    score_ranking,
)


def _write(tmp_path, content: str) -> str:
    path = os.path.join(str(tmp_path), "golden.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


# ---------------------------------------------------------------------------
# The golden dataset
# ---------------------------------------------------------------------------


def test_a_list_of_question_objects(tmp_path):
    path = _write(
        tmp_path,
        '[{"question": "pytanie", "documents": ["a.pdf", "b.pdf"]}]',
    )
    assert load_golden_dataset(path) == [
        {"question": "pytanie", "documents": ["a.pdf", "b.pdf"]}
    ]


def test_a_mapping_of_a_question_to_its_documents(tmp_path):
    path = _write(tmp_path, '{"pytanie": ["a.pdf"], "inne": []}')
    examples = load_golden_dataset(path)
    assert {"question": "pytanie", "documents": ["a.pdf"]} in examples


def test_the_other_names_of_the_relevant_documents(tmp_path):
    path = _write(
        tmp_path,
        '[{"question": "q1", "relevant_documents": ["a.pdf"]},'
        ' {"question": "q2", "files": [{"file_name": "b.pdf"}]}]',
    )
    examples = load_golden_dataset(path)
    assert examples[0]["documents"] == ["a.pdf"]
    assert examples[1]["documents"] == ["b.pdf"]


def test_questions_without_a_text_are_not_run(tmp_path):
    path = _write(tmp_path, '[{"question": "", "documents": ["a.pdf"]}]')
    assert load_golden_dataset(path) == []


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_the_grid_of_the_plan():
    assert DEFAULT_GRID["rrf_k"] == [20, 40, 60, 100]
    assert DEFAULT_GRID["rerank_results"] == [True, False]


def test_combinations_enumerate_the_whole_grid_without_the_weight_duplicates():
    """20 ratios (only the ratio of the weights decides a fusion), each of
    them with every other parameter of the grid."""
    combos = list(combinations(DEFAULT_GRID))
    unique_ratios = {
        round(combo["dense_weight"] / combo["lexical_weight"], 6) for combo in combos
    }
    assert len(unique_ratios) == 20
    assert len(combos) == 20 * len(DEFAULT_GRID["rrf_k"]) * len(
        DEFAULT_GRID["rerank_results"]
    )


def test_a_ratio_is_a_duplicate_only_of_the_same_other_parameters():
    grid = {
        "dense_weight": [0.6, 0.3],
        "lexical_weight": [0.4, 0.2],
        "rrf_k": [60, 100],
    }
    combos = list(combinations(grid))
    # 0.6/0.4 and 0.3/0.2 are one search, so that ratio is measured once
    # per rrf_k, not once per pair of weights; the other ratios of the
    # grid (1.5 and 3) are kept, every one of them with every rrf_k
    assert len(combos) == 3 * 2
    assert len([c for c in combos if c["rrf_k"] == 60]) == 3
    pairs = {
        (combo["dense_weight"], combo["lexical_weight"]) for combo in combos
    }
    # the duplicate is gone, the search it duplicates is not
    assert (0.3, 0.2) not in pairs
    assert (0.6, 0.4) in pairs



def test_a_grid_without_the_weights_is_enumerated_as_it_is():
    grid = {"rerank_results": [True, False], "rrf_k": [40, 60]}
    assert len(list(combinations(grid))) == 4


# ---------------------------------------------------------------------------
# The scoring of one question
# ---------------------------------------------------------------------------


def test_score_ranking_of_a_perfect_search():
    scores = score_ranking(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"], k=10)
    assert scores["ndcg"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["average_precision"] == 1.0


def test_score_ranking_of_a_search_which_missed_everything():
    scores = score_ranking(["x.pdf", "y.pdf"], ["a.pdf"], k=10)
    assert scores["ndcg"] == 0.0
    assert scores["mrr"] == 0.0
    assert scores["recall"] == 0.0


def test_score_ranking_reports_the_depth_it_was_given():
    # the relevant document stands at the second position: seen at the
    # depth 1 it is not there at all
    assert score_ranking(["x.pdf", "a.pdf"], ["a.pdf"], k=1)["mrr"] == 0.0
    assert score_ranking(["x.pdf", "a.pdf"], ["a.pdf"], k=2)["mrr"] == 0.5
