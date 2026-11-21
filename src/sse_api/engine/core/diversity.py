"""
diversity.py
------------

Diversification of the search results before they are shown or read.

When one chapter of one document fits the query perfectly, several almost
identical chunks of that chapter occupy the top positions and push every
other, complementary document out of the top-K.  A human reads the same
paragraph five times, and a generative model spends its context window on
repetition instead of on the rest of the answer.

Two mechanisms counter it here, in the spirit of Maximal Marginal
Relevance -- with the document of a chunk as the only similarity the
engine needs to know:

* **rank decay** -- every chunk of a document which is already represented
  pays ``1 - diversity_decay`` of the relevance range for each of its
  document's chunks taken before it (the redundancy penalty of MMR).  A
  second chunk of the same document stays above another document only when
  its relevance is clearly higher;
* **a per-document cap** -- at most ``max_chunks_per_document`` chunks of
  one document are taken in the first pass.

The penalty is subtracted from the relevance of the list min-max normalised
to ``[0, 1]``, because the scores of a result list share a scale which is
not known here (an RRF sum is not a cosine and neither is a reranker logit)
and a negative score must not be rewarded by a penalty.  A penalty is
therefore a fraction of the relevance *range* of the answer: how much a
document pays for repeating itself relative to how much its results differ
from each other.

The function never discards a result it could have shown: when the cap
leaves the list shorter than ``max_results``, it is topped up with the
chunks skipped in the first pass.

The input list is not modified.
"""

from typing import Any, Dict, List, Optional, Tuple

#: Default: how many chunks of one document the first pass may take.
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2

#: Default relevance cost of repeating a document, in the spirit of the
#: lambda of MMR: ``1 - DEFAULT_DIVERSITY_DECAY`` of the relevance range is
#: subtracted from every chunk of a document which is already represented.
DEFAULT_DIVERSITY_DECAY = 0.85

#: Default cap of the diversified list.
DEFAULT_MAX_RESULTS = 10

#: Where the document name lives in the ``get_texts`` shape of a result.
_DOCUMENT_NAME_PATH = ("result", "text", "document_name")
#: Where the score lives in the same shape.
_SCORE_PATH = ("result", "text", "score")


def _nested(hit: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    node: Any = hit
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def document_name_of(hit: Dict[str, Any]) -> str:
    """
    Document name of a result, ``""`` when it carries none.

    Understands the nested ``get_texts`` shape
    (``{"result": {"text": {"document_name": ...}}}``) and the flat
    display shape (``{"document_name": ...}``).  A result without a name is
    a document of its own, which the diversification never caps.
    """
    for value in (_nested(hit, _DOCUMENT_NAME_PATH), hit.get("document_name")):
        if isinstance(value, str) and value:
            return value
    return ""


def relevance_of(hit: Dict[str, Any]) -> Optional[float]:
    """
    Relevance of a result as a number, ``None`` when it has none.

    Reads the nested score first, then the flat one; a non-numeric value
    means "unknown".
    """
    for value in (_nested(hit, _SCORE_PATH), hit.get("score")):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalized_relevances(
    results: List[Dict[str, Any]],
) -> List[Optional[float]]:
    """
    Per-result relevance in ``[0, 1]``, ``None`` where it is unknown.

    A min-max of the whole list; when every known score is the same there
    is no range to speak of and every known score becomes ``1.0``.
    """
    scores = [relevance_of(hit) for hit in results]
    known = [score for score in scores if score is not None]
    if not known:
        return [None] * len(results)
    lowest, highest = min(known), max(known)
    spread = highest - lowest
    normalized: List[Optional[float]] = []
    for score in scores:
        if score is None:
            normalized.append(None)
        elif spread <= 0.0:
            normalized.append(1.0)
        else:
            normalized.append((score - lowest) / spread)
    return normalized


def apply_result_diversity(
    results: List[Dict[str, Any]],
    max_results: int = DEFAULT_MAX_RESULTS,
    max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    diversity_decay: float = DEFAULT_DIVERSITY_DECAY,
) -> List[Dict[str, Any]]:
    """
    Spread the results of a search over more of its documents.

    Every chunk is ranked by its relevance minus the redundancy penalty of
    its document (``1 - diversity_decay`` of the relevance range, once for
    every chunk of that document before it), and the first
    ``max_chunks_per_document`` chunks of every document are taken from
    that order.  The list is then topped up -- in the same order -- with
    the chunks the cap skipped, so the caller still gets ``max_results``
    results whenever the search found that many.

    Parameters
    ----------
    results : list of dict
        Ranked search results, the ``get_texts`` shape or the flat display
        shape.
    max_results : int, default 10
        Maximum length of the returned list.
    max_chunks_per_document : int, default 2
        How many chunks of one document the first pass may take. A value
        below ``1`` behaves like ``1``.
    diversity_decay : float, default 0.85
        Relevance kept by a repeated document, in ``(0, 1]``. ``1.0``
        switches the penalty off and leaves only the cap.

    Returns
    -------
    list of dict
        At most ``max_results`` of the input results, the same objects.
    """
    if not results or max_results <= 0:
        return []

    cap = max(int(max_chunks_per_document), 1)
    decay = diversity_decay if 0.0 < diversity_decay <= 1.0 else 1.0
    penalty = 1.0 - decay

    normalized = _normalized_relevances(results)
    if penalty > 0.0 and any(value is None for value in normalized):
        # The penalty compares the relevance of two chunks of two documents;
        # when one of them has no score the comparison is meaningless, so
        # the ranking of the pipeline is kept and only the cap applies.
        penalty = 0.0

    # The original rank breaks every tie, so the sort is deterministic and
    # a decay of 1.0 changes nothing at all.
    seen_before: Dict[str, int] = {}
    keyed: List[Tuple[float, int, int]] = []
    for position, hit in enumerate(results):
        document_name = document_name_of(hit)
        occurrences = seen_before.get(document_name, 0)
        if document_name:
            # A result without a name is a document of its own.
            seen_before[document_name] = occurrences + 1
        relevance = normalized[position] or 0.0
        keyed.append((relevance - penalty * occurrences, -position, position))
    keyed.sort(reverse=True)

    kept: List[Dict[str, Any]] = []
    skipped: List[Tuple[float, int, int]] = []
    taken_per_document: Dict[str, int] = {}
    for entry in keyed:
        hit = results[entry[2]]
        document_name = document_name_of(hit)
        taken = taken_per_document.get(document_name, 0)
        if document_name and taken >= cap:
            skipped.append(entry)
            continue
        if document_name:
            taken_per_document[document_name] = taken + 1
        kept.append(hit)
        if len(kept) >= max_results:
            break

    for entry in skipped:
        if len(kept) >= max_results:
            break
        kept.append(results[entry[2]])

    return kept
