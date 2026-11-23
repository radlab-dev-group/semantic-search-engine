"""
ir_metrics.py
-------------

Standard metrics of Information Retrieval, as pure functions.

The evaluation of a search is a question of ranking: whether the most
relevant documents stand at the top of the list, and where the first
relevant one of a question stands.  The metrics below answer it without
PyTorch and without a running engine -- a ranking is a list, and the
ground truth is a set, which is all these functions need:

* ``dcg_at_k`` / ``ndcg_at_k`` -- the gain of the ranking at a depth, and
  the same gain normalised by the gain of the perfect ranking, so a score
  of ``1.0`` means "as good as it can be" and a score of ``0.0`` means
  "nothing relevant in the top-K";
* ``mrr_at_k`` -- the reciprocal of the position of the first relevant
  document;
* ``precision_recall_at_k`` -- how much of the top-K is relevant, and how
  much of the relevant was found;
* ``map_at_k`` -- the mean, over the relevant documents, of how early each
  of them is found.

The metrics are *per query*: a query has one ranked list and one ground
truth.  A corpus average is the mean of the per-query scores, which the
caller does when it has many queries (see ``tests_loader``).
"""

import math
from typing import Any, Hashable, Iterable, List, Sequence, Set

#: A document is either relevant or not: the binary gain of one hit.
_BINARY_RELEVANT = 1.0
_BINARY_NOT_RELEVANT = 0.0


def _binary_relevances(
    actual_hits: Sequence[Any], ground_truth: Set[Hashable]
) -> List[float]:
    """``1.0`` where the hit is relevant, ``0.0`` otherwise, in rank order."""
    return [
        _BINARY_RELEVANT if hit in ground_truth else _BINARY_NOT_RELEVANT
        for hit in actual_hits
    ]


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """
    Discounted Cumulative Gain at a depth.

    The gain of a hit is divided by ``log2(position + 1)``, so the first
    position (``log2(2) = 1``) counts fully and every next position counts
    less: a perfect ranking is worth more for having the best at the top
    than for having it at the bottom.  The positions are one-based.

    Parameters
    ----------
    relevance_scores : sequence of float
        The relevance of every hit, in the rank order of the list.
    k : int
        How deep the ranking is read; a depth below ``1`` reads nothing.

    Returns
    -------
    float
        The discounted sum of the gains of the first ``k`` hits.
    """
    if k <= 0:
        return 0.0
    total = 0.0
    for position, relevance in enumerate(list(relevance_scores)[:k], start=1):
        total += relevance / math.log2(position + 1)
    return total


def ndcg_at_k(
    actual_hits: Sequence[Any], ground_truth: Set[Hashable], k: int = 10
) -> float:
    """
    Normalized Discounted Cumulative Gain at a depth.

    The DCG of the ranking divided by the DCG of the perfect one (every
    relevant document first, in any order), on a binary relevance.  The
    result is in ``[0, 1]`` and comparable across queries of different
    sizes of the ground truth: ``1.0`` is the best ranking of these
    documents, ``0.0`` is a top-K without a single one of them.

    Parameters
    ----------
    actual_hits : sequence
        The ranked hits of one query, in the order of the list.
    ground_truth : set
        The documents relevant for the query.
    k : int, default 10
        How deep the ranking is read.

    Returns
    -------
    float
        ``0.0`` when the ground truth is empty or nothing of it is in the
        top-K.
    """
    if not ground_truth or k <= 0:
        return 0.0
    actual_dcg = dcg_at_k(_binary_relevances(actual_hits, ground_truth), k)
    # the perfect ranking: as many relevant documents as the depth allows
    ideal_dcg = dcg_at_k([_BINARY_RELEVANT] * min(len(ground_truth), k), k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr_at_k(
    actual_hits: Sequence[Any], ground_truth: Set[Hashable], k: int = 10
) -> float:
    """
    Reciprocal Rank of the first relevant hit at a depth.

    ``1 / position`` of the first hit of the ground truth, ``0.0`` when the
    top-K does not contain any.  It is the single number which answers "did
    the answer come early?": the first position is ``1.0``, the tenth is
    ``0.1``.

    Parameters
    ----------
    actual_hits : sequence
        The ranked hits of one query.
    ground_truth : set
        The documents relevant for the query.
    k : int, default 10
        How deep the ranking is read.

    Returns
    -------
    float
    """
    if k <= 0:
        return 0.0
    for position, hit in enumerate(list(actual_hits)[:k], start=1):
        if hit in ground_truth:
            return 1.0 / position
    return 0.0


def precision_recall_at_k(
    actual_hits: Sequence[Any], ground_truth: Set[Hashable], k: int = 10
) -> tuple:
    """
    Precision and recall at a depth.

    Precision is the share of the shown hits which are relevant -- it asks
    "can I trust the top of the list?"; recall is the share of the relevant
    documents which were found at all -- it asks "did I see the whole
    answer?".  A list shorter than the depth is measured on its own length,
    so a short, complete answer is not penalised for not filling the top-K.

    Parameters
    ----------
    actual_hits : sequence
        The ranked hits of one query.
    ground_truth : set
        The documents relevant for the query.
    k : int, default 10
        How deep the ranking is read.

    Returns
    -------
    tuple of (float, float)
        ``(precision, recall)``; ``(0.0, 0.0)`` when there is nothing to
        measure -- an empty list or an empty ground truth.
    """
    if not actual_hits or not ground_truth or k <= 0:
        return 0.0, 0.0
    retrieved_at_k = set(list(actual_hits)[:k])
    relevant_retrieved = len(retrieved_at_k & ground_truth)
    precision = relevant_retrieved / min(k, len(actual_hits))
    recall = relevant_retrieved / len(ground_truth)
    return precision, recall


def average_precision_at_k(
    actual_hits: Sequence[Any], ground_truth: Set[Hashable], k: int = 10
) -> float:
    """
    Average Precision of one query at a depth.

    Every relevant hit found in the top-K contributes the precision at its
    own position -- "of what was shown up to here, how much was
    relevant?" -- and the contributions are averaged over the relevant
    documents of the query, not over the ones which happened to be found: a
    system which finds one of three relevant documents early scores about
    ``0.33``, not ``1.0``.  The result is in ``[0, 1]``; a list without a
    single relevant hit scores ``0.0``, a list which finds every relevant
    document early scores ``1.0``.

    Parameters
    ----------
    actual_hits : sequence
        The ranked hits of one query.
    ground_truth : set
        The documents relevant for the query.
    k : int, default 10
        How deep the ranking is read.

    Returns
    -------
    float
    """
    if not actual_hits or not ground_truth or k <= 0:
        return 0.0
    hits_up_to_here = 0
    relevant_found = 0
    precision_at_hits = 0.0
    for position, hit in enumerate(list(actual_hits)[:k], start=1):
        hits_up_to_here += 1
        if hit in ground_truth:
            relevant_found += 1
            precision_at_hits += relevant_found / hits_up_to_here
    if relevant_found == 0:
        return 0.0
    # the standard denominator: the relevant documents of the query, not
    # the ones which happened to be retrieved
    return precision_at_hits / min(len(ground_truth), k)


def map_at_k(per_query: Iterable[tuple], k: int = 10) -> float:
    """
    Mean Average Precision of a set of queries.

    The mean of :func:`average_precision_at_k` over the queries: one number
    for the corpus, comparable between two configurations of the search.

    Parameters
    ----------
    per_query : iterable of (actual_hits, ground_truth)
        Every query of the set, with its ranked hits and its ground truth.
    k : int, default 10
        How deep the ranking is read.

    Returns
    -------
    float
        ``0.0`` for an empty set.
    """
    values = [
        average_precision_at_k(actual_hits, ground_truth, k=k)
        for actual_hits, ground_truth in per_query
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)
