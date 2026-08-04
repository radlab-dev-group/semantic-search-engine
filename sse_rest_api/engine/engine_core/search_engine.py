"""
search_engine.py
----------------

Pure‑computation helper functions extracted from ``DBSemanticSearchController``
and the protocol‑based ``SearchEngineCore`` orchestrator.

All functions here have **zero Django / zero database dependencies** — they
operate exclusively on plain Python dicts, lists and tuples.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from engine.engine_core.protocols import VectorStoreProtocol


# ---------------------------------------------------------------------------
# Pure computation helpers (previously methods on DBSemanticSearchController)
# ---------------------------------------------------------------------------


def rrf_merge(
    milvus_results: list[dict],
    postgres_results: list[tuple[int, float]],
    k: int = 60,
    max_results: int = 100,
) -> tuple[list[int], list[float]]:
    """
    Merge results using Reciprocal Rank Fusion.

    Parameters
    ----------
    milvus_results : list[dict]
        Results from vector search (each hit dict must have ``"metadata["external_text_id"]``).
    postgres_results : list[tuple[int, float]]
        Results from text search as ``(text_id, rank)`` tuples.
    k : int, default 60
        RRF constant.
    max_results : int, default 100
        Maximum number of merged results to return.

    Returns
    -------
    tuple[int list, float list]
        ``(sorted_ids, sorted_scores)``
    """
    scores: dict[int, float] = {}

    # Process Milvus results
    for rank, hit in enumerate(milvus_results, 1):
        try:
            text_id = int(hit["metadata"]["external_text_id"])
            scores[text_id] = scores.get(text_id, 0) + 1.0 / (k + rank)
        except (KeyError, ValueError, TypeError):
            continue

    # Process Postgres results
    for rank, (text_id, _) in enumerate(postgres_results, 1):
        scores[text_id] = scores.get(text_id, 0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
        :max_results
    ]

    return sorted_ids, [scores[tid] for tid in sorted_ids]


def rerank_results_with_vector_store(
    query_text: str,
    results: list[dict],
    max_results: int,
    vector_store: VectorStoreProtocol,
) -> list[dict]:
    """
    Rerank provided results using the cross‑encoder model exposed by a
    ``VectorStoreProtocol`` implementation.

    Parameters
    ----------
    query_text : str
        The query string.
    results : list[dict]
        List of result dictionaries (each must have a ``"text_str"`` key).
    max_results : int
        Maximum number of results to return after reranking.
    vector_store : VectorStoreProtocol
        A vector store that implements ``rerank()``.

    Returns
    -------
    list[dict]
        Reranked and truncated list of results.
    """
    if not results:
        return results

    try:
        reranked = vector_store.rerank(query_text, results, max_results)
        return reranked
    except Exception as exc:
        # Log via the caller or a logger passed in — here we fall back gracefully.
        return results[:max_results]


def prepare_documents_stats(
    postgres_docs: list[dict], smooth_factor: float = 0.0001
) -> dict:
    """
    Compute per‑document statistics (hits, average score, weighted
    score, etc.) from a list of PostgreSQL document fragments.

    Parameters
    ----------
    postgres_docs : list[dict]
        List of document fragment dictionaries returned by
        ``_prepare_document_page``.
    smooth_factor : float, default 0.0001
        Minimum value used when scaling weighted scores.

    Returns
    -------
    dict
        Mapping ``{document_name: stats_dict}``.
    """
    doc_stats: dict[str, dict] = {}
    for result in postgres_docs:
        result_inner = result["result"]
        text_info = result_inner["text"]
        doc_name = text_info["document_name"]
        doc_score = text_info["score"]
        doc_page_number = text_info["page_number"]

        if doc_name not in doc_stats:
            doc_stats[doc_name] = {
                "score": 0.0,
                "score_weighted": 0.0,
                "hits": 0,
                "pages": [],
                "pages_count": 0,
                "relative_path": text_info["relative_path"],
            }

        doc_stats[doc_name]["hits"] += 1
        doc_stats[doc_name]["score"] += doc_score
        doc_stats[doc_name]["pages"].append(doc_page_number)

    w_scores: list[float] = []
    for doc, res in doc_stats.items():
        res["score"] /= res["hits"]
        res["pages"] = sorted(set(res["pages"]))
        res["pages_count"] = len(res["pages"])
        res["score_weighted"] = math.log(
            float(res["score"] * res["hits"] * res["pages_count"])
        )
        w_scores.append(res["score_weighted"])

    min_w_scores = abs(min(w_scores))
    sum_w_scores = sum(s + min_w_scores for s in w_scores)

    for doc, res in doc_stats.items():
        v = res["score_weighted"]
        if sum_w_scores > 0.0:
            doc_stats[doc]["score_weighted_scaled"] = max(
                smooth_factor, (v + min_w_scores) / sum_w_scores
            )
        else:
            doc_stats[doc]["score_weighted_scaled"] = 1.0

    return doc_stats


def filter_stats_to_display_results(
    doc_stats: dict, remove_under_hits: int = 5, remove_under_pages: int = 3
) -> dict:
    """
    Filter out documents that do not meet minimal hit or page count
    thresholds.

    Parameters
    ----------
    doc_stats : dict
        Statistics dictionary produced by ``prepare_documents_stats``.
    remove_under_hits : int, default 5
        Minimum number of hits required.
    remove_under_pages : int, default 3
        Minimum number of distinct pages required.

    Returns
    -------
    dict
        Filtered statistics dictionary.
    """
    return {
        doc: res
        for doc, res in doc_stats.items()
        if res["hits"] >= remove_under_hits and res["pages_count"] >= remove_under_pages
    }


def convert_search_results_to_doc2answer(
    search_results: dict,
    which_docs: list | None,
    use_doc_names_in_response: bool = False,
) -> dict:
    """
    Convert raw search results into a mapping ``{document_name: [answers]}``.

    Parameters
    ----------
    search_results : dict
        Mapping of result identifiers to result dictionaries.
    which_docs : list | None
        If provided, only include results from these documents.
    use_doc_names_in_response : bool, default False
        Prefix each answer with its document name.

    Returns
    -------
    dict
        ``{document_name: [answer strings]}``.
    """
    doc2answers: dict[str, list[str]] = {}
    for _, result in search_results.items():
        doc_name = result["document_name"]
        if which_docs is not None and len(which_docs) and doc_name not in which_docs:
            continue

        if doc_name not in doc2answers:
            doc2answers[doc_name] = []

        text_str = result["text_str"]
        if use_doc_names_in_response:
            text_str = f"{doc_name}: {text_str}"
        doc2answers[doc_name].append(text_str)

    return doc2answers


def get_accumulated_docs_by_rank_perc(results: dict, perc_rank_gen_qa: float) -> list:
    """
    Return document names whose cumulative weighted score reaches the
    given percentage of the total.

    Parameters
    ----------
    results : dict
        Search results containing a ``stats`` mapping.
    perc_rank_gen_qa : float
        Target cumulative percentage (e.g., ``0.2`` for 20 %).

    Returns
    -------
    list
        Ordered list of document names.
    """
    doc_stats = results["stats"]
    sorted_doc_names = sorted(
        list(doc_stats.keys()),
        key=lambda x: doc_stats[x]["score_weighted_scaled"],
        reverse=True,
    )

    perc_max = perc_rank_gen_qa
    if perc_rank_gen_qa > 1:
        perc_max = float(perc_rank_gen_qa / 100)

    ret_docs: list[str] = []
    accum_perc = 0.0
    for doc_name in sorted_doc_names:
        if accum_perc >= perc_max:
            break
        accum_perc += results["stats"][doc_name]["score_weighted_scaled"]
        ret_docs.append(doc_name)

    return ret_docs


def reformat_search_results_to_display(
    search_results: list | dict,
) -> dict | list:
    """
    Transform raw search results into a UI‑friendly structure that
    contains scores, context snippets and document identifiers.

    Parameters
    ----------
    search_results : list | dict
        Raw results as returned by ``search``.

    Returns
    -------
    dict | list
        Reformatted results ready for presentation.
    """
    ans_dict: dict[int, dict] = {}
    for idx, answer in enumerate(search_results):
        a = answer["result"]
        left_ctx_str = ""
        right_ctx_str = ""
        if len(a["left_context"]):
            left_ctx_str = a["left_context"][-1]["text_str"]
        if len(a["right_context"]):
            right_ctx_str = a["right_context"][-1]["text_str"]

        text_info = a["text"]
        ans_dict[idx] = {
            "score": text_info["score"],
            "document_name": text_info["document_name"],
            "relative_filepath": text_info["relative_path"],
            "page_number": text_info["page_number"],
            "text_number": text_info["text_number"],
            "language": text_info["language"],
            "text_str": text_info["text_str"],
            "left_context": left_ctx_str,
            "right_context": right_ctx_str,
        }

    return ans_dict


# ---------------------------------------------------------------------------
# SearchEngineCore — protocol‑based orchestrator
# ---------------------------------------------------------------------------


class SearchEngineCore:
    """
    Protocol‑based search engine that orchestrates vector search, FTS,
    RRF merge, and reranking without knowing the concrete database
    implementations.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        data_source: Any = None,  # RelationalDataProtocol — avoid hard import
        model_config: Any = None,  # ModelConfigProtocol
        logger: Any = None,
    ):
        """
        Create a search engine core.

        Parameters
        ----------
        vector_store : VectorStoreProtocol
            Vector store adapter implementing the protocol interface.
        data_source : RelationalDataProtocol | None
            Relational data adapter for FTS and text retrieval.
        model_config : ModelConfigProtocol | None
            Embedder/reranker model configuration.
        logger : Any | None
            Optional logging object with ``info()`` / ``error()`` methods.
        """
        self._vector_store = vector_store
        self._data_source = data_source
        self._model_config = model_config
        self._logger = logger

    def hybrid_search(
        self,
        query_text: str,
        collection_id: int,
        filter_doc_names: list[str],
        max_results: int,
        rerank: bool = True,
        language: str | None = None,
        rrf_k: int = 60,
        hybrid_search: bool = True,
    ) -> dict:
        """
        Execute the full hybrid search pipeline.

        Parameters
        ----------
        query_text : str
            The user's query.
        collection_id : int
            Primary key of the document collection.
        filter_doc_names : list[str]
            List of document names to filter by (OR union).
        max_results : int
            Maximum number of results.
        rerank : bool, default True
            Whether to apply cross‑encoder reranking.
        language : str | None
            FTS language config (e.g., ``"polish"``, ``"english"``).
        rrf_k : int, default 60
            RRF constant for rank fusion.
        hybrid_search : bool, default True
            Enable PostgreSQL full‑text search in addition to vector search.

        Returns
        -------
        dict
            ``{"query": ..., "stats": {...}, "detailed_results": [...]}``
        """
        # Step 1: Vector store search
        metadata_filter = (
            {"filenames": filter_doc_names} if filter_doc_names else None
        )

        milvus_results = self._vector_store.search(
            query_text=query_text,
            max_results=max_results * 10,
            metadata_filter=metadata_filter,
        )

        # Step 2: FTS search (via data source protocol)
        postgres_fts_results: list[tuple[int, float]] = []
        if hybrid_search and self._data_source:
            postgres_fts_results = self._data_source.fts_search(
                query_str=query_text,
                collection_id=collection_id,
                max_results=max_results,
                language=language,
            )

        # Step 3: RRF merge
        if hybrid_search and postgres_fts_results:
            texts_ids, texts_scores = rrf_merge(
                milvus_results, postgres_fts_results, k=rrf_k, max_results=max_results * 10
            )
        else:
            texts_ids = [int(hit["metadata"]["external_text_id"]) for hit in milvus_results]
            texts_scores = [hit["score"] for hit in milvus_results]

        # Step 4: Get text details via data source protocol
        if texts_ids:
            texts_with_context = self._data_source.get_text_for_id(texts_ids, surrounding_chunks=2)
        else:
            texts_with_context = []

        # Step 5: Rerank if needed
        if rerank and texts_with_context:
            texts_with_context = rerank_results_with_vector_store(
                query_text, texts_with_context, max_results, self._vector_store
            )

        # Step 6: Compute stats
        docs_stats = prepare_documents_stats(texts_with_context)
        filtered_stats = filter_stats_to_display_results(docs_stats)

        return {
            "query": query_text,
            "stats": filtered_stats,
            "detailed_results": texts_with_context,
        }

    def index_collection(self, collection_id: int) -> None:
        """
        Index all text chunks from a collection into the vector store.

        Parameters
        ----------
        collection_id : int
            Primary key of the collection to index.
        """
        if not self._data_source:
            return

        chunks = self._data_source.get_collection_chunks(collection_id, min_char_len=300)
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            # The data source returns ORM objects; access their attributes.
            texts.append(chunk.text_str)
            metadatas.append({
                "external_text_id": str(chunk.id),
                "external_document_id": str(chunk.page.document.pk),
                "page_number": getattr(chunk, "page_number", 1),
                "text_number": getattr(chunk, "text_number", 0),
                "document_name": chunk.page.document.name,
                "relative_path": getattr(chunk.page.document, "relative_path", ""),
                "text_language": getattr(chunk, "language", ""),
            })

        if texts:
            self._vector_store.add_texts(texts, metadatas)
