"""Pure helpers for searching more than one collection at once.

The module deliberately has no Django, Milvus or PostgreSQL imports: the API
endpoint owns the database work, while the naming, merging and ranking rules
live here so they can be tested on plain dictionaries.

Two aggregation strategies are supported:

``per_collection``
    Every collection contributes up to ``max_results`` hits.  The hits are
    pooled, so ``N`` collections can produce ``N * max_results`` chunks.

``global_limit``
    Every collection contributes candidates which are ranked together by
    ``score`` and cut off at ``max_results`` chunks in total.

Every hit carries its origin (``collection_name``,
``collection_display_name`` and ``original_document_name``).  In a multi
collection search the ``document_name`` is additionally prefixed with the
collection, e.g. ``[Kolekcja B] doc_b1.pdf``, so that two collections can
hold a document with the same file name without the RAG context (and the
citations) becoming ambiguous.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from sse_api.core.validation import (
    DEFAULT_MULTI_COLLECTION_MODE,
    MULTI_COLLECTION_MODES,
)

#: Search options which name documents, categories or templates of *one*
#: collection.  Applying them to every collection of a multi-collection query
#: would silently drop results (the same document rarely exists everywhere),
#: so only the options shared by all collections are sent to each search.
COLLECTION_SCOPED_OPTIONS = frozenset(
    {
        "categories",
        "documents",
        "documents_names",
        "relative_paths",
        "relative_path_contains",
        "metadata_filters",
        "metadata_filter_groups",
        "templates",
        "only_template_documents",
        "use_and_operator",
    }
)

PROVENANCE_KEYS = (
    "collection_name",
    "collection_display_name",
    "original_document_name",
)


def dedupe_collection_names(names: Iterable[Any]) -> list[str]:
    """Trim and de-duplicate collection names, keeping the original order."""
    unique: list[str] = []
    seen: set[str] = set()
    for name in names or []:
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def collection_display_name(collection: Any) -> str:
    """Human readable name of a collection, falling back to its identifier."""
    display = getattr(collection, "display_name", None)
    name = getattr(collection, "name", None)
    return str(display or name or "")


def document_label(display_name: str, document_name: str) -> str:
    """Prefix a document with the collection it came from."""
    return f"[{display_name}] {document_name}"


def numeric_score(value: Any) -> float:
    """A finite float score; missing, non-numeric or infinite scores are 0.0."""
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def result_score(item: dict) -> float:
    """Relevance used when hits from several collections compete.

    The display results normally carry ``score`` (the reranker overwrites it
    with its own value).  ``rerank_score`` and ``score_normalized`` are
    accepted as fallbacks for callers which keep those fields separately.
    """
    for key in ("score", "rerank_score", "score_normalized"):
        if item.get(key) is not None:
            return numeric_score(item.get(key))
    return 0.0


def shared_search_options(options: dict | None) -> dict:
    """Copy of ``options`` without filters bound to a single collection."""
    return {
        key: value
        for key, value in (options or {}).items()
        if key not in COLLECTION_SCOPED_OPTIONS
    }


def annotate_item(
    item: dict,
    collection: Any,
    *,
    rename_document: bool,
    document_name: str | None = None,
) -> dict:
    """Copy of one detailed result / stats row with its collection provenance.

    ``document_name`` is the key of a stats row; for detailed results it is
    read from the item itself.  When ``rename_document`` is true the returned
    item also carries an annotated ``document_name``.
    """
    annotated = dict(item)
    original = document_name or str(item.get("document_name") or "")
    display = collection_display_name(collection)
    annotated["original_document_name"] = original
    annotated["collection_name"] = getattr(collection, "name", None)
    annotated["collection_display_name"] = display
    if rename_document:
        annotated["document_name"] = document_label(display, original)
    return annotated


def annotate_results(
    results: dict | None,
    collection: Any,
    *,
    rename_document: bool = False,
) -> dict:
    """Annotate ``stats`` and ``detailed_results`` of one collection search.

    The input is never modified; controllers sometimes keep their own
    reference to the result dictionary.  A single-collection search keeps the
    original document names (``rename_document=False``) so the UI behaves as
    before, but still records where each fragment came from.
    """
    if not isinstance(results, dict):
        return {}
    annotated_results = dict(results)

    detailed = results.get("detailed_results")
    if isinstance(detailed, dict):
        annotated_results["detailed_results"] = {
            key: (
                annotate_item(value, collection, rename_document=rename_document)
                if isinstance(value, dict)
                else value
            )
            for key, value in detailed.items()
        }
    elif isinstance(detailed, (list, tuple)):
        annotated_results["detailed_results"] = [
            annotate_item(value, collection, rename_document=rename_document)
            if isinstance(value, dict)
            else value
            for value in detailed
        ]

    stats = results.get("stats")
    if isinstance(stats, dict):
        annotated_stats = {}
        for doc_name, value in stats.items():
            key = (
                document_label(collection_display_name(collection), str(doc_name))
                if rename_document
                else str(doc_name)
            )
            if isinstance(value, dict):
                annotated_stats[key] = annotate_item(
                    value,
                    collection,
                    rename_document=rename_document,
                    document_name=str(doc_name),
                )
            else:
                annotated_stats[key] = value
        annotated_results["stats"] = annotated_stats

    return annotated_results


def detailed_items(results: dict | None) -> list[dict]:
    """Detailed results of one collection as a list of dictionaries."""
    if not isinstance(results, dict):
        return []
    detailed = results.get("detailed_results") or {}
    values = detailed.values() if isinstance(detailed, dict) else detailed
    return [dict(item) for item in values if isinstance(item, dict)]


def _rescale_stats(stats: dict) -> dict:
    """Recalculate the share of the total score mass over all collections.

    ``get_accumulated_docs_by_rank_perc`` expects every stats row to have
    ``score_weighted_scaled``; leaving the per-collection values would make a
    five-collection search claim 500% of the rank mass.  A zero total gets a
    uniform share so the generation step still sees a complete mapping.
    """
    rows = {key: dict(value) for key, value in stats.items() if isinstance(value, dict)}
    total = sum(numeric_score(row.get("score_weighted")) for row in rows.values())
    uniform = 1.0 / len(rows) if rows else 0.0
    for row in rows.values():
        row["score_weighted_scaled"] = (
            numeric_score(row.get("score_weighted")) / total if total > 0 else uniform
        )
    return rows


def aggregate_results(
    detailed: list[dict],
    stats: dict,
    *,
    mode: str,
    max_results: int,
) -> tuple[list[dict], dict]:
    """Merge annotated hits of all collections into one result set.

    Returns ``(detailed_results, stats)``.  In ``global_limit`` mode the hits
    are sorted globally by score (stable, with the collection order as the
    tie-breaker) and cut to ``max_results``.  Stats are restricted to the
    documents which survived the cut and their score shares are recomputed
    globally.
    """
    if mode not in MULTI_COLLECTION_MODES:
        mode = DEFAULT_MULTI_COLLECTION_MODE
    limit = max(0, int(max_results or 0))
    items = [dict(item) for item in detailed if isinstance(item, dict)]

    if mode == "global_limit":
        indexed = list(enumerate(items))
        indexed.sort(key=lambda pair: (-result_score(pair[1]), pair[0]))
        items = [item for _, item in indexed[:limit]]

    kept_documents = {
        item.get("document_name") for item in items if item.get("document_name")
    }
    final_stats = {
        key: dict(value)
        for key, value in (stats or {}).items()
        if key in kept_documents and isinstance(value, dict)
    }

    # A collection can return a chunk whose document did not make it into the
    # display stats (e.g. a page-less chunk).  Generation selects documents
    # from stats, so such a chunk would otherwise never reach the model.
    for item in items:
        name = item.get("document_name")
        if not name or name in final_stats:
            continue
        final_stats[name] = {
            "document_name": name,
            "original_document_name": item.get("original_document_name") or name,
            "collection_name": item.get("collection_name"),
            "collection_display_name": item.get("collection_display_name"),
            "hits": 1,
            "pages_count": 1,
            "score_weighted": result_score(item),
        }

    return items, _rescale_stats(final_stats)
