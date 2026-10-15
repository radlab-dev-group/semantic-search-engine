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

from sse_api.engine.core.protocols import VectorStoreProtocol

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


def cosine_scores_by_text_id(milvus_results: list[dict]) -> dict:
    """
    Map ``text_id`` to the cosine similarity reported by the vector store.

    Both the hybrid and the plain vector search need the similarity of a
    chunk independently of the score used for ordering: the fusion and the
    reranker overwrite ``score``, and a chunk coming only from the
    full‑text list has no similarity at all.  Hits already carrying a
    ``similarity`` key (they were reranked, so ``score`` holds a logit)
    keep it.

    Parameters
    ----------
    milvus_results : list[dict]
        Vector search hits.

    Returns
    -------
    dict
        ``{text_id: cosine_similarity}``; hits without a usable
        identifier or score are skipped.
    """
    similarities: dict[int, float] = {}
    for hit in milvus_results or []:
        if not isinstance(hit, dict):
            continue
        metadata = hit.get("metadata") or {}
        try:
            text_id = int(metadata["external_text_id"])
            raw_score = hit.get("similarity")
            if raw_score is None:
                raw_score = hit.get("score")
            similarity = _as_float(raw_score)
            if similarity is None:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        similarities[text_id] = similarity
    return similarities


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
    except Exception:
        # Log via the caller or a logger passed in — here we fall back gracefully.
        return results[:max_results]


# ---------------------------------------------------------------------------
# Relevance score scales
# ---------------------------------------------------------------------------

#: Milvus vector search (``COSINE`` metric) -- scores in ``[-1, 1]``.
SCORE_SCALE_COSINE = "cosine"
#: Cross‑encoder reranker -- unbounded logits.
SCORE_SCALE_RERANKER = "reranker"
#: Reciprocal Rank Fusion -- small positive values, at most ``2 / (k + 1)``.
SCORE_SCALE_RRF = "rrf"
#: Already normalized scores -- only clamped to ``[0, 1]``.
SCORE_SCALE_RAW = "raw"
SCORE_SCALES = (
    SCORE_SCALE_COSINE,
    SCORE_SCALE_RERANKER,
    SCORE_SCALE_RRF,
    SCORE_SCALE_RAW,
)
DEFAULT_RRF_K = 60


def _sigmoid(value: float) -> float:
    """Logistic squash guarded against ``math.exp`` overflow."""
    if value <= -60.0:
        return 0.0
    if value >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def normalize_relevance(
    score: Any,
    score_scale: str = SCORE_SCALE_COSINE,
    rrf_k: int = DEFAULT_RRF_K,
) -> float:
    """
    Map a raw retrieval score onto the closed interval ``[0, 1]``.

    Every retrieval stage reports relevance on its own scale, so the
    document level weights have to be built from a common basis:

    * ``"cosine"`` -- cosine similarity in ``[-1, 1]`` is mapped linearly
      with ``(1 + score) / 2``;
    * ``"reranker"`` -- cross‑encoder logits are squashed with the logistic
      function, which keeps the ordering of negative logits intact;
    * ``"rrf"`` -- Reciprocal Rank Fusion values are rescaled by
      ``(k + 1) / 2`` (two fused lists), so a hit ranked first in both
      lists maps to ``1.0`` and first in a single list to ``0.5``;
    * ``"raw"`` -- the score is assumed to be a fraction already and is
      only clamped.

    Parameters
    ----------
    score : Any
        Raw score of a single hit. Values that are not real numbers become
        ``0.0``.
    score_scale : str, default "cosine"
        One of :data:`SCORE_SCALES`.
    rrf_k : int, default 60
        Reciprocal Rank Fusion constant, used by the ``"rrf"`` scale.

    Returns
    -------
    float
        Normalized relevance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``score_scale`` is not a known scale.
    """
    if score_scale not in SCORE_SCALES:
        raise ValueError(
            "Unknown score scale %r; expected one of %s"
            % (score_scale, ", ".join(SCORE_SCALES))
        )
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0

    if score_scale == SCORE_SCALE_RERANKER:
        normalized = _sigmoid(value)
    elif score_scale == SCORE_SCALE_RRF:
        # Reciprocal Rank Fusion sums one 1/(k + rank) term per fused list,
        # so two lists put a perfect hit at 2 / (k + 1).
        normalized = value * (max(1.0, float(rrf_k)) + 1.0) / 2.0
    elif score_scale == SCORE_SCALE_RAW:
        normalized = value
    else:
        normalized = (1.0 + value) / 2.0

    return min(1.0, max(0.0, normalized))


def retrieval_score_scale(rerank: bool = False, hybrid_search: bool = False) -> str:
    """
    Return the score scale produced by a given search configuration.

    Reranking overwrites the retrieval scores of every kept hit, so it wins
    over the hybrid (RRF) scale; a hybrid search without reranking scores
    with Reciprocal Rank Fusion values, and a plain vector search reports
    cosine similarity.

    Parameters
    ----------
    rerank : bool, default False
        Whether a cross‑encoder reranker was applied to the results.
    hybrid_search : bool, default False
        Whether Reciprocal Rank Fusion merged vector and full‑text results.

    Returns
    -------
    str
        One of :data:`SCORE_SCALES`.
    """
    if rerank:
        return SCORE_SCALE_RERANKER
    if hybrid_search:
        return SCORE_SCALE_RRF
    return SCORE_SCALE_COSINE


def _as_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` when it is not a number."""
    try:
        similarity = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(similarity) or math.isinf(similarity):
        return None
    return similarity


def annotate_chunk_scores(
    postgres_docs: list[dict],
    score_scale: str = SCORE_SCALE_COSINE,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    Attach the normalized relevance to every chunk of a ``get_texts`` result.

    ``score`` stays exactly what ordered the results — cosine similarity, a
    Reciprocal Rank Fusion value or a reranker logit — which makes its raw
    value hard to compare.  ``score_normalized`` is the same relevance on
    the ``[0, 1]`` scale of :func:`normalize_relevance`, and ``similarity``
    is the cosine similarity of the chunk when the vector search produced
    one.

    Parameters
    ----------
    postgres_docs : list[dict]
        Document fragments as returned by ``get_texts`` (modified in place).
    score_scale : str, default "cosine"
        Scale of the ``score`` of every chunk.
    rrf_k : int, default 60
        Reciprocal Rank Fusion constant used by the ``"rrf"`` scale.

    Returns
    -------
    list[dict]
        The same list that was passed in.
    """
    for doc in postgres_docs or []:
        text_info = (doc.get("result") or {}).get("text")
        if not isinstance(text_info, dict):
            continue
        text_info["score_normalized"] = normalize_relevance(
            text_info.get("score"), score_scale, rrf_k
        )
        text_info.setdefault("similarity", None)
    return postgres_docs


def prepare_documents_stats(
    postgres_docs: list[dict],
    smooth_factor: float = 0.0001,
    score_scale: str = SCORE_SCALE_COSINE,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict:
    """
    Compute per‑document statistics (hits, average score, weighted
    score, etc.) from a list of document fragments.

    Chunk scores are reported on whatever scale the retrieval stage used
    (cosine similarity, RRF values or cross‑encoder logits), so every score
    is first mapped onto ``[0, 1]`` by :func:`normalize_relevance`.  The
    document weights are therefore always defined, non‑negative and
    comparable between queries.

    Parameters
    ----------
    postgres_docs : list[dict]
        List of document fragment dictionaries returned by
        ``_prepare_document_page``.
    smooth_factor : float, default 0.0001
        Floor applied to ``weight_normalized`` of documents that carry
        some but no measurable relevance.
    score_scale : str, default "cosine"
        Scale of the incoming chunk scores; see :func:`normalize_relevance`.
    rrf_k : int, default 60
        Reciprocal Rank Fusion constant used by the ``"rrf"`` scale.

    Returns
    -------
    dict
        Mapping ``{document_name: stats_dict}`` with, among the hit/page
        counters:

        * ``score`` -- mean of the raw chunk scores,
        * ``score_normalized`` -- the same mean in ``[0, 1]``,
        * ``similarity`` -- mean cosine similarity of the hits, ``None``
          when none of them came from the vector search,
        * ``score_weighted`` -- document mass, i.e.
          ``score_normalized * hits * pages_count``,
        * ``score_weighted_scaled`` -- share of the total mass; all shares
          sum up to ``1.0``,
        * ``weight_normalized`` -- mass relative to the best document, so
          the best document is ``1.0`` and the rest is proportionally below.
    """
    if not postgres_docs:
        return {}

    doc_stats: dict[str, dict] = {}
    cosine_samples: dict[str, list[float]] = {}
    for result in postgres_docs:
        result_inner = result["result"]
        text_info = result_inner["text"]
        doc_name = text_info["document_name"]
        doc_score = text_info["score"]
        doc_page_number = text_info["page_number"]

        if doc_name not in doc_stats:
            doc_stats[doc_name] = {
                "score": 0.0,
                "score_normalized": 0.0,
                "score_weighted": 0.0,
                "score_weighted_scaled": 0.0,
                "weight_normalized": 0.0,
                "hits": 0,
                "pages": [],
                "pages_count": 0,
                "relative_path": text_info["relative_path"],
            }

        doc_stats[doc_name]["hits"] += 1
        doc_stats[doc_name]["score"] += doc_score
        doc_stats[doc_name]["score_normalized"] += normalize_relevance(
            doc_score, score_scale, rrf_k
        )
        doc_stats[doc_name]["pages"].append(doc_page_number)
        cosine = _as_float(text_info.get("similarity"))
        if cosine is not None:
            cosine_samples.setdefault(doc_name, []).append(cosine)

    for doc_name, res in doc_stats.items():
        hits = res["hits"]
        res["score"] /= hits
        res["score_normalized"] /= hits
        res["pages"] = sorted(set(res["pages"]))
        res["pages_count"] = len(res["pages"])
        # Mean cosine similarity of the hits; None when the vector search
        # did not contribute any of them (pure full-text matches).
        samples = cosine_samples.get(doc_name)
        res["similarity"] = sum(samples) / len(samples) if samples else None
        # Document mass keeps the original intuition -- mean relevance times
        # the amount of evidence -- but is built from normalized scores, so
        # it is defined for every metric and can never be negative.
        res["score_weighted"] = res["score_normalized"] * hits * res["pages_count"]

    total_mass = sum(res["score_weighted"] for res in doc_stats.values())
    top_mass = max(
        (res["score_weighted"] for res in doc_stats.values()), default=0.0
    )
    # Nothing carries any measurable relevance: fall back to a uniform
    # distribution so that cumulative document selection stays usable.
    uniform_share = 1.0 / len(doc_stats)

    for res in doc_stats.values():
        mass = res["score_weighted"]
        if total_mass > 0.0:
            res["score_weighted_scaled"] = mass / total_mass
        else:
            res["score_weighted_scaled"] = uniform_share
        if top_mass > 0.0:
            res["weight_normalized"] = max(smooth_factor, mass / top_mass)
        else:
            res["weight_normalized"] = 0.0

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
        if res["hits"] >= remove_under_hits
        and res["pages_count"] >= remove_under_pages
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


def get_accumulated_docs_by_rank_perc(
    results: dict, perc_rank_gen_qa: float
) -> list:
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
            right_ctx_str = a["right_context"][0]["text_str"]

        text_info = a["text"]
        ans_dict[idx] = {
            "score": text_info["score"],
            "score_normalized": text_info.get("score_normalized"),
            "similarity": text_info.get("similarity"),
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
        min_similarity: float | None = None,
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
        min_similarity : float | None, default None
            Optional cosine similarity cutoff for the vector search.

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
            min_similarity=min_similarity,
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
        texts_scores: list[float] = []
        fused_with_fts = hybrid_search and bool(postgres_fts_results)
        # A cosine similarity is reported only when the vector search is the
        # single ranking criterion. In a hybrid search the ranking is decided
        # by the fused ranks and full-text only chunks have no similarity,
        # so a cosine next to such a score would be misleading.
        vector_similarity = (
            {} if fused_with_fts else cosine_scores_by_text_id(milvus_results)
        )
        if fused_with_fts:
            texts_ids, texts_scores = rrf_merge(
                milvus_results,
                postgres_fts_results,
                k=rrf_k,
                max_results=max_results * 10,
            )
        else:
            texts_ids = [
                int(hit["metadata"]["external_text_id"]) for hit in milvus_results
            ]
            texts_scores = [float(hit.get("score", 0.0)) for hit in milvus_results]
        texts_similarity = None
        if vector_similarity:
            texts_similarity = [
                vector_similarity.get(text_id) for text_id in texts_ids
            ]

        # Step 4: Get text details via data source protocol
        if texts_ids:
            texts_with_context = self._data_source.get_text_for_id(
                texts_ids,
                surrounding_chunks=2,
                text_scores=texts_scores,
                text_similarity=texts_similarity,
            )
        else:
            texts_with_context = []

        # Step 5: Rerank if needed
        if rerank and texts_with_context:
            texts_with_context = rerank_results_with_vector_store(
                query_text, texts_with_context, max_results, self._vector_store
            )

        # Step 6: Compute stats
        score_scale = retrieval_score_scale(
            rerank=rerank, hybrid_search=fused_with_fts
        )
        annotate_chunk_scores(texts_with_context, score_scale, rrf_k)
        docs_stats = prepare_documents_stats(
            texts_with_context,
            score_scale=score_scale,
            rrf_k=rrf_k,
        )
        filtered_stats = filter_stats_to_display_results(docs_stats)

        return {
            "query": query_text,
            "score_scale": score_scale,
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

        chunks = self._data_source.get_collection_chunks(
            collection_id, min_char_len=300
        )
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            # The data source returns ORM objects; access their attributes.
            texts.append(chunk.text_str)
            metadatas.append(
                {
                    "external_text_id": str(chunk.id),
                    "external_document_id": str(chunk.page.document.pk),
                    "page_number": getattr(chunk, "page_number", 1),
                    "text_number": getattr(chunk, "text_number", 0),
                    "document_name": chunk.page.document.name,
                    "relative_path": getattr(
                        chunk.page.document, "relative_path", ""
                    ),
                    "text_language": getattr(chunk, "language", ""),
                }
            )

        if texts:
            self._vector_store.add_texts(texts, metadatas)
