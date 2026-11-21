"""
test_ir_metrics.py
------------------

The standard metrics of Information Retrieval of ``ir_metrics``: pure
functions over a ranked list and a set of ground truth, tested against
hand-computed values, so a change of the formula is visible at a glance.
"""

import math
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

from sse_tools.evaluation.evaluator.ir_metrics import (
    average_precision_at_k,
    dcg_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_at_k,
)

#: A ground truth of three documents and a ranking which finds them all,
#: interleaved with noise.
GROUND_TRUTH = {"a", "b", "c"}
RANKED = ["a", "x", "b", "y", "z", "c"]


# ---------------------------------------------------------------------------
# DCG / NDCG
# ---------------------------------------------------------------------------


def test_dcg_discounts_the_gains_by_the_position():
    # position 1 counts fully, position 2 by 1/log2(3), position 3 by
    # 1/log2(4) = 1/2
    assert math.isclose(dcg_at_k([1.0, 1.0, 1.0], 3), 1.0 + 1.0 / math.log2(3) + 0.5)
    # a depth below the length of the list reads only the top of it
    assert dcg_at_k([1.0, 1.0], 1) == 1.0


def test_dcg_of_an_empty_or_shallow_ranking():
    assert dcg_at_k([], 5) == 0.0
    assert dcg_at_k([1.0, 1.0], 0) == 0.0


def test_ndcg_of_the_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], GROUND_TRUTH, k=3) == 1.0


def test_ndcg_of_a_ranking_without_any_relevant_hit_is_zero():
    assert ndcg_at_k(["x", "y", "z"], GROUND_TRUTH, k=10) == 0.0


def test_ndcg_is_normalised_by_the_perfect_ranking():
    # the ranking finds a, b and c at the positions 1, 3 and 6 of a
    # ground truth of three: the gain is real, the order is not perfect
    score = ndcg_at_k(RANKED, GROUND_TRUTH, k=6)
    actual = 1.0 + 0.5 + 1.0 / math.log2(7)
    ideal = 1.0 + 1.0 / math.log2(3) + 0.5
    assert math.isclose(score, actual / ideal, rel_tol=1e-12)
    assert 0.0 < score < 1.0


def test_ndcg_of_an_empty_ground_truth_is_zero():
    assert ndcg_at_k(["a"], set(), k=5) == 0.0


def test_ndcg_respects_the_depth():
    # a relevant document past the depth is not seen at all
    assert ndcg_at_k(["x", "a"], {"a"}, k=1) == 0.0
    # at the depth 2 the hit stands at position 2: the discount of that
    # position against the perfect one, which stands at position 1
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == 1.0 / math.log2(3)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_of_the_first_position():
    assert mrr_at_k(["a", "x"], {"a"}, k=10) == 1.0


def test_mrr_of_a_later_position():
    assert mrr_at_k(["x", "y", "a"], {"a"}, k=10) == 1.0 / 3


def test_mrr_without_a_relevant_hit_in_the_depth():
    assert mrr_at_k(["x", "y"], {"a"}, k=2) == 0.0
    assert mrr_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert mrr_at_k(["x"], {"a"}, k=0) == 0.0


def test_mrr_of_an_empty_ranking():
    assert mrr_at_k([], {"a"}, k=5) == 0.0
    assert mrr_at_k(["a"], set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------


def test_precision_recall_of_a_mixed_top():
    # the top-4 shows a, x, b, y: two relevant of four shown, two of three
    # relevant found
    precision, recall = precision_recall_at_k(RANKED, GROUND_TRUTH, k=4)
    assert math.isclose(precision, 0.5)
    assert math.isclose(recall, 2.0 / 3.0)


def test_precision_recall_of_a_full_ranking():
    precision, recall = precision_recall_at_k(RANKED, GROUND_TRUTH, k=10)
    assert math.isclose(precision, 3.0 / 6.0)
    assert math.isclose(recall, 1.0)


def test_precision_recall_measures_a_short_list_on_its_own_length():
    # three relevant documents, all found, on a list of three: a short,
    # complete answer is not penalised for not filling the top-10
    precision, recall = precision_recall_at_k(["a", "b", "c"], GROUND_TRUTH, k=10)
    assert precision == 1.0
    assert recall == 1.0


def test_precision_recall_of_the_empty_cases():
    assert precision_recall_at_k([], GROUND_TRUTH, k=5) == (0.0, 0.0)
    assert precision_recall_at_k(["a"], set(), k=5) == (0.0, 0.0)


def test_duplicate_hits_are_counted_once():
    precision, recall = precision_recall_at_k(["a", "a"], {"a"}, k=2)
    assert math.isclose(precision, 0.5)
    assert recall == 1.0


# ---------------------------------------------------------------------------
# Average Precision / MAP
# ---------------------------------------------------------------------------


def test_average_precision_of_a_perfect_ranking():
    assert average_precision_at_k(["a", "b", "c"], GROUND_TRUTH, k=3) == 1.0


def test_average_precision_averages_the_relevant_documents():
    # a at 1 (1/1), b at 3 (2/3), c at 6 (3/6): the sum, divided by the
    # three relevant documents of the query
    expected = (1.0 + 2.0 / 3.0 + 0.5) / 3.0
    assert math.isclose(average_precision_at_k(RANKED, GROUND_TRUTH, k=6), expected)


def test_average_precision_of_a_partial_find():
    # one of the three relevant documents, found first: the denominator is
    # the ground truth, not the find
    assert math.isclose(
        average_precision_at_k(["a", "x", "y"], GROUND_TRUTH, k=3), 1.0 / 3.0
    )


def test_average_precision_without_any_relevant_hit():
    assert average_precision_at_k(["x", "y"], GROUND_TRUTH, k=5) == 0.0
    assert average_precision_at_k([], GROUND_TRUTH, k=5) == 0.0
    assert average_precision_at_k(["a"], set(), k=5) == 0.0


def test_average_precision_respects_the_depth():
    # c at 6 is past the depth 5: the finds are a at 1 (1/1) and b at 3
    # (2/3), averaged over the three relevant documents of the query
    expected = (1.0 + 2.0 / 3.0) / 3.0
    assert math.isclose(average_precision_at_k(RANKED, GROUND_TRUTH, k=5), expected)


def test_map_is_the_mean_over_the_queries():
    per_query = [
        (["a", "b", "c"], GROUND_TRUTH),  # perfect: (1/1 + 2/2 + 3/3) / 3
        (["x", "y"], GROUND_TRUTH),  # nothing
        (["b", "a", "x", "c"], GROUND_TRUTH),  # (1/1 + 2/2 + 3/4) / 3
    ]
    expected = (1.0 + 0.0 + (1.0 + 1.0 + 0.75) / 3.0) / 3.0
    assert math.isclose(map_at_k(per_query, k=10), expected)


def test_map_of_an_empty_set():
    assert map_at_k([], k=10) == 0.0
