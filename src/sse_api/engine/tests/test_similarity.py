"""
test_similarity.py
------------------

Tests for the COSINE metric alignment and the cosine similarity cutoff
(``min_similarity``) introduced in the search engine.

These tests are intentionally DB‑free:

* pure helpers from ``engine.core.search_engine`` are tested
  directly (no Django, no Milvus),
* ``MilvusHandler.search`` is exercised with a fake Milvus client,
* the rerank result-shape helpers on the controller are tested as static
  methods.

A minimal Django settings configuration is applied *only* if Django is
not configured yet, so that the controller modules can be imported.  No
database connection is required or made.
"""

import math
import os
import sys

# ---------------------------------------------------------------------------
# Make the application packages importable when running from the repo root.
# ---------------------------------------------------------------------------
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
        MAIN_LOGGER=__import__("logging").getLogger("test_similarity"),
    )
    django.setup()

from sse_api.data.controllers.constants import DEFAULT_MIN_SIMILARITY  # noqa: E402
from sse_api.engine.controllers.database.milvus import (  # noqa: E402
    INDEX_QUERY_PARAMS,
    MilvusHandler,
)
from sse_api.engine.controllers.search.semantic import (  # noqa: E402
    DBSemanticSearchController,
)
from sse_api.engine.core.search_engine import (  # noqa: E402
    SCORE_SCALE_COSINE,
    SCORE_SCALE_RAW,
    SCORE_SCALE_RERANKER,
    SCORE_SCALE_RRF,
    SearchEngineCore,
    get_accumulated_docs_by_rank_perc,
    normalize_relevance,
    prepare_documents_stats,
    retrieval_score_scale,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_postgres_doc(name: str, score: float, page: int = 1) -> dict:
    """Build a ``get_texts`` shaped result dict."""
    return {
        "result": {
            "left_context": [],
            "right_context": [],
            "text": {
                "score": score,
                "document_name": name,
                "relative_path": f"/docs/{name}",
                "page_number": page,
                "text_number": 0,
                "language": "pl",
                "text_str": f"Tekst dokumentu {name}",
            },
        }
    }


class _FakeMilvusClient:
    """Records search calls and returns pre-canned hits."""

    def __init__(self, distances, metric_type="COSINE"):
        self._distances = distances
        self._metric_type = metric_type
        self.last_kwargs = None

    def list_indexes(self, collection_name, field_name=""):
        return ["emb_idx"]

    def describe_index(self, collection_name, index_name):
        return {"metric_type": self._metric_type}

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        hits = [
            {
                "distance": d,
                "entity": {
                    "text": f"text-{d}",
                    "metadata": {"external_text_id": str(100 + i)},
                },
            }
            for i, d in enumerate(self._distances)
        ]
        return [hits]


def _make_handler(distances, metric_type="COSINE"):
    handler = MilvusHandler(
        jsonl_config_path=os.path.join(
            _APP_DIR, "configs", "milvus_config.example.json"
        ),
        collection_name="test_collection",
        create_db_if_not_exists=False,
        load_embedder=False,
        load_collection_and_schema=False,
    )
    handler._milvus_client = _FakeMilvusClient(distances, metric_type)
    handler.prepare_embeddings = lambda texts: [[0.0] * 4]
    return handler, handler._milvus_client


class _RecordingVectorStore:
    """Minimal VectorStoreProtocol implementation for propagation tests."""

    def __init__(self, hits):
        self._hits = hits
        self.last_min_similarity = object()  # sentinel: "not passed"

    def search(
        self,
        query_text,
        max_results,
        metadata_filter=None,
        min_similarity=None,
    ):
        self.last_min_similarity = min_similarity
        return list(self._hits)

    def rerank(self, query_text, results, max_results):
        return results


# ---------------------------------------------------------------------------
# Constants & metric alignment
# ---------------------------------------------------------------------------


def test_default_min_similarity_value():
    assert DEFAULT_MIN_SIMILARITY == 0.5


def test_index_query_params_use_cosine():
    for index_type in ("HNSW", "IVF_FLAT"):
        params = INDEX_QUERY_PARAMS[index_type]
        assert params["INDEX_PARAMS"]["metric_type"] == "COSINE"
        assert params["QUERY_PARAMS"]["metric_type"] == "COSINE"


def test_default_index_params_not_shared_with_registry():
    """``DEFAULT_INDEX_PARAMS`` must be a copy — the registry dict must not
    gain a ``field_name`` key as a side effect."""
    assert MilvusHandler.DEFAULT_INDEX_PARAMS["field_name"] == "embedding"
    assert "field_name" not in INDEX_QUERY_PARAMS["IVF_FLAT"]["INDEX_PARAMS"]
    assert "field_name" not in INDEX_QUERY_PARAMS["HNSW"]["INDEX_PARAMS"]


# ---------------------------------------------------------------------------
# MilvusHandler.search — similarity cutoff
# ---------------------------------------------------------------------------


def test_search_filters_hits_below_min_similarity():
    handler, _fake_client = _make_handler([0.9, 0.5, 0.2, 0.05])
    results = handler.search(
        search_text="pytanie",
        max_results=10,
        min_similarity=0.5,
    )
    assert len(results) == 1
    kept = results[0]
    assert len(kept) == 2  # 0.9 and 0.5 kept; 0.2 and 0.05 dropped
    assert all(hit["score"] >= 0.5 for hit in kept)


def test_search_without_cutoff_keeps_all_hits():
    handler, _fake_client = _make_handler([0.9, 0.5, 0.2, 0.05])
    results = handler.search(search_text="pytanie", max_results=10)
    assert len(results) == 1
    assert len(results[0]) == 4


def test_search_uses_cosine_query_params():
    handler, fake_client = _make_handler([0.9])
    handler.search(search_text="pytanie", max_results=5, min_similarity=0.5)
    assert fake_client.last_kwargs["search_params"]["metric_type"] == "COSINE"


def test_search_uses_existing_index_metric_type():
    handler, fake_client = _make_handler([0.9], metric_type="IP")
    handler.search(search_text="pytanie", max_results=5)
    assert fake_client.last_kwargs["search_params"]["metric_type"] == "IP"


def test_search_cutoff_excludes_everything():
    handler, _fake_client = _make_handler([0.9, 0.5, 0.2])
    results = handler.search(
        search_text="pytanie", max_results=10, min_similarity=0.95
    )
    assert results == [[]]


# ---------------------------------------------------------------------------
# prepare_documents_stats — cosine-safe (no math domain error)
# ---------------------------------------------------------------------------


def test_stats_negative_similarity_does_not_crash():
    docs = [
        _make_postgres_doc("docA", -0.3, page=1),
        _make_postgres_doc("docA", -0.1, page=2),
    ]
    stats = prepare_documents_stats(docs)
    assert "docA" in stats
    assert math.isfinite(stats["docA"]["score_weighted"])


def test_stats_zero_similarity_does_not_crash():
    docs = [_make_postgres_doc("docZ", 0.0, page=1)]
    stats = prepare_documents_stats(docs)
    assert math.isfinite(stats["docZ"]["score_weighted"])


def test_stats_ranking_positive_scores():
    docs = [
        _make_postgres_doc("good", 0.9, page=1),
        _make_postgres_doc("good", 0.8, page=2),
        _make_postgres_doc("bad", 0.51, page=1),
    ]
    stats = prepare_documents_stats(docs)
    assert stats["good"]["score_weighted"] > stats["bad"]["score_weighted"]
    # scaled weights are positive and finite
    assert stats["good"]["score_weighted_scaled"] > 0.0
    assert math.isfinite(stats["good"]["score_weighted_scaled"])


# ---------------------------------------------------------------------------
# Rerank helpers (flat and nested result shapes)
# ---------------------------------------------------------------------------


def test_extract_rerank_text_flat():
    assert (
        DBSemanticSearchController._extract_rerank_text({"text_str": "abc"}) == "abc"
    )


def test_extract_rerank_text_nested():
    res = _make_postgres_doc("docA", 0.5)
    assert (
        DBSemanticSearchController._extract_rerank_text(res)
        == "Tekst dokumentu docA"
    )


def test_extract_rerank_text_missing():
    assert DBSemanticSearchController._extract_rerank_text({}) == ""
    assert (
        DBSemanticSearchController._extract_rerank_text({"result": {"text": {}}})
        == ""
    )
    assert DBSemanticSearchController._extract_rerank_text({"result": {}}) == ""


def test_write_rerank_score_nested_location():
    res = _make_postgres_doc("docA", 0.5)
    DBSemanticSearchController._write_rerank_score(res, 0.77)
    assert res["result"]["text"]["score"] == 0.77


def test_write_rerank_score_flat_location():
    res = {"text_str": "abc"}
    DBSemanticSearchController._write_rerank_score(res, 0.42)
    assert res["score"] == 0.42


def test_read_result_score_nested_and_flat():
    res_nested = _make_postgres_doc("docA", 0.5)
    DBSemanticSearchController._write_rerank_score(res_nested, 0.9)
    assert DBSemanticSearchController._read_result_score(res_nested) == 0.9

    res_flat = {"text_str": "abc"}
    DBSemanticSearchController._write_rerank_score(res_flat, 0.3)
    assert DBSemanticSearchController._read_result_score(res_flat) == 0.3


def test_read_result_score_invalid_value():
    assert DBSemanticSearchController._read_result_score({"score": "junk"}) == 0.0
    assert DBSemanticSearchController._read_result_score({}) == 0.0


# ---------------------------------------------------------------------------
# SearchEngineCore — min_similarity propagation
# ---------------------------------------------------------------------------


def test_search_engine_core_propagates_min_similarity():
    # Empty hit list: no data source is needed downstream, but the
    # vector-store ``search`` call (with ``min_similarity``) still happens.
    vs = _RecordingVectorStore([])

    class _FakeDataSource:
        def get_text_for_id(self, text_ids, surrounding_chunks=0, text_scores=None):
            return [_make_postgres_doc("docA", 0.9, page=1) for _ in text_ids]

    engine = SearchEngineCore(vector_store=vs, data_source=_FakeDataSource())
    engine.hybrid_search(
        query_text="pytanie",
        collection_id=1,
        filter_doc_names=[],
        max_results=10,
        rerank=False,
        hybrid_search=False,
        min_similarity=0.42,
    )
    assert vs.last_min_similarity == 0.42


# ---------------------------------------------------------------------------
# normalize_relevance — mapping raw scores onto [0, 1]
# ---------------------------------------------------------------------------


def test_normalize_relevance_cosine_is_linear_on_unit_interval():
    assert normalize_relevance(-1.0, SCORE_SCALE_COSINE) == 0.0
    assert normalize_relevance(0.0, SCORE_SCALE_COSINE) == 0.5
    assert normalize_relevance(1.0, SCORE_SCALE_COSINE) == 1.0
    assert normalize_relevance(0.8, SCORE_SCALE_COSINE) == 0.9


def test_normalize_relevance_clamps_out_of_range_values():
    assert normalize_relevance(3.5, SCORE_SCALE_COSINE) == 1.0
    assert normalize_relevance(-3.5, SCORE_SCALE_COSINE) == 0.0
    assert normalize_relevance(1.4, SCORE_SCALE_RAW) == 1.0
    assert normalize_relevance(-0.4, SCORE_SCALE_RAW) == 0.0


def test_normalize_relevance_reranker_squashes_logits():
    assert normalize_relevance(0.0, SCORE_SCALE_RERANKER) == 0.5
    assert normalize_relevance(4.0, SCORE_SCALE_RERANKER) > 0.5
    assert normalize_relevance(-4.0, SCORE_SCALE_RERANKER) < 0.5
    # monotone and overflow-safe
    assert normalize_relevance(-1000.0, SCORE_SCALE_RERANKER) == 0.0
    assert normalize_relevance(1000.0, SCORE_SCALE_RERANKER) == 1.0


def test_normalize_relevance_rrf_accounts_for_the_fused_lists():
    top = 1.0 / 61.0
    # first place in a single list
    assert normalize_relevance(top, SCORE_SCALE_RRF) == 0.5
    # first place in both lists is the maximum
    assert normalize_relevance(2.0 * top, SCORE_SCALE_RRF) == 1.0
    # anything above the theoretical maximum is clamped
    assert normalize_relevance(4.0 * top, SCORE_SCALE_RRF) == 1.0


def test_normalize_relevance_ignores_garbage():
    assert normalize_relevance(None, SCORE_SCALE_COSINE) == 0.0
    assert normalize_relevance("junk", SCORE_SCALE_COSINE) == 0.0
    assert normalize_relevance(float("nan"), SCORE_SCALE_COSINE) == 0.0
    assert normalize_relevance(float("inf"), SCORE_SCALE_COSINE) == 1.0
    assert normalize_relevance(float("-inf"), SCORE_SCALE_COSINE) == 0.0


def test_normalize_relevance_rejects_unknown_scale():
    try:
        normalize_relevance(0.5, "euclidean")
    except ValueError as exc:
        assert "euclidean" in str(exc)
    else:
        raise AssertionError("unknown score scale must be rejected")


def test_retrieval_score_scale_follows_the_search_configuration():
    assert retrieval_score_scale() == SCORE_SCALE_COSINE
    assert retrieval_score_scale(hybrid_search=True) == SCORE_SCALE_RRF
    # reranking overwrites every retained score, so it wins
    assert retrieval_score_scale(rerank=True) == SCORE_SCALE_RERANKER
    assert (
        retrieval_score_scale(rerank=True, hybrid_search=True)
        == SCORE_SCALE_RERANKER
    )


# ---------------------------------------------------------------------------
# prepare_documents_stats — normalized, comparable document weights
# ---------------------------------------------------------------------------


def test_stats_empty_input():
    assert prepare_documents_stats([]) == {}


def test_stats_score_normalized_is_bounded():
    docs = [
        _make_postgres_doc("neg", -0.7, page=1),
        _make_postgres_doc("pos", 0.7, page=1),
    ]
    stats = prepare_documents_stats(docs, score_scale=SCORE_SCALE_COSINE)
    for res in stats.values():
        assert 0.0 <= res["score_normalized"] <= 1.0
    # the raw mean keeps the original metric for the API consumers
    assert stats["neg"]["score"] == -0.7
    assert math.isclose(stats["neg"]["score_normalized"], 0.15, rel_tol=1e-9)


def test_stats_shares_sum_to_one():
    docs = [
        _make_postgres_doc("a", 0.91, page=1),
        _make_postgres_doc("a", 0.88, page=3),
        _make_postgres_doc("b", 0.42, page=1),
        _make_postgres_doc("c", -0.3, page=2),
    ]
    stats = prepare_documents_stats(docs)
    total = sum(res["score_weighted_scaled"] for res in stats.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_stats_shares_sum_to_one_for_non_positive_scores():
    """A regression test: with every log-weight negative the old scaling
    returned 1.0 per document, so the mass was three times too big."""
    docs = [_make_postgres_doc(n, 0.7, page=1) for n in ("a", "b", "c")]
    stats = prepare_documents_stats(docs)
    total = sum(res["score_weighted_scaled"] for res in stats.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)
    for res in stats.values():
        assert math.isclose(res["score_weighted_scaled"], 1.0 / 3.0, rel_tol=1e-9)


def test_stats_identical_documents_are_weighted_equally():
    docs = []
    for name in ("a", "b", "c"):
        docs.extend(
            [
                _make_postgres_doc(name, 0.66, page=1),
                _make_postgres_doc(name, 0.66, page=2),
            ]
        )
    stats = prepare_documents_stats(docs)
    shares = {doc: res["score_weighted_scaled"] for doc, res in stats.items()}
    assert math.isclose(shares["a"], shares["b"], rel_tol=1e-9)
    assert math.isclose(shares["b"], shares["c"], rel_tol=1e-9)


def test_stats_single_document_takes_the_whole_mass():
    stats = prepare_documents_stats([_make_postgres_doc("only", 0.9, page=1)])
    assert stats["only"]["score_weighted_scaled"] == 1.0
    assert stats["only"]["weight_normalized"] == 1.0


def test_stats_negative_reranker_logits_keep_their_ordering():
    """A regression test: negative logits used to be clamped to the smooth
    factor, making a -1 and a -8 hit indistinguishable."""
    docs = [
        _make_postgres_doc("better", -1.0, page=1),
        _make_postgres_doc("worse", -8.0, page=1),
    ]
    stats = prepare_documents_stats(docs, score_scale=SCORE_SCALE_RERANKER)
    assert stats["better"]["score_weighted"] > stats["worse"]["score_weighted"]
    assert stats["better"]["score_weighted_scaled"] > 0.9
    assert stats["worse"]["score_weighted_scaled"] > 0.0
    total = sum(res["score_weighted_scaled"] for res in stats.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_stats_weight_normalized_is_relative_to_the_best_document():
    docs = [
        _make_postgres_doc("best", 0.95, page=1),
        _make_postgres_doc("mid", 0.6, page=1),
        _make_postgres_doc("weak", 0.3, page=1),
    ]
    stats = prepare_documents_stats(docs)
    assert stats["best"]["weight_normalized"] == 1.0
    assert (
        stats["best"]["weight_normalized"]
        > stats["mid"]["weight_normalized"]
        > stats["weak"]["weight_normalized"]
        > 0.0
    )
    assert stats["mid"]["weight_normalized"] <= 1.0


def test_stats_weights_are_never_negative():
    docs = [
        _make_postgres_doc("a", -0.99, page=1),
        _make_postgres_doc("b", -1.0, page=1),
    ]
    stats = prepare_documents_stats(docs)
    for res in stats.values():
        assert res["score_weighted"] >= 0.0
        assert res["score_weighted_scaled"] >= 0.0
        assert math.isfinite(res["weight_normalized"])


def test_stats_without_any_relevance_falls_back_to_uniform_shares():
    docs = [
        _make_postgres_doc("a", -1.0, page=1),
        _make_postgres_doc("b", -1.0, page=1),
    ]
    stats = prepare_documents_stats(docs)
    for res in stats.values():
        assert res["score_weighted"] == 0.0
        assert math.isclose(res["score_weighted_scaled"], 0.5, rel_tol=1e-9)
        assert res["weight_normalized"] == 0.0


def test_stats_rrf_scale_is_rescaled_before_weighting():
    top = 1.0 / 61.0
    docs = [
        _make_postgres_doc("top", top, page=1),
        _make_postgres_doc("tail", top / 10.0, page=1),
    ]
    stats = prepare_documents_stats(docs, score_scale=SCORE_SCALE_RRF)
    assert stats["top"]["score_normalized"] == 0.5
    assert math.isclose(stats["tail"]["score_normalized"], 0.05, rel_tol=1e-9)
    assert stats["top"]["weight_normalized"] == 1.0


def test_stats_accumulated_docs_follow_the_normalized_shares():
    docs = []
    for page in range(1, 6):
        docs.append(_make_postgres_doc("dominant", 0.95, page=page))
    docs.append(_make_postgres_doc("other", 0.6, page=1))
    stats = prepare_documents_stats(docs)
    results = {"stats": stats}
    assert get_accumulated_docs_by_rank_perc(results, 10) == ["dominant"]
    selected = get_accumulated_docs_by_rank_perc(results, 100)
    assert selected[0] == "dominant"
    assert set(selected) == {"dominant", "other"}


# ---------------------------------------------------------------------------
# Score propagation through SearchEngineCore and the data adapter
# ---------------------------------------------------------------------------


def _engine_hits_and_pages():
    """Ten hits spread over two documents of five hits / five pages each.

    ``SearchEngineCore`` prunes the stats to documents with at least five
    hits on at least three pages, so both documents have to clear that
    threshold for the shares of the returned stats to sum up to one.
    """
    hits, pages = [], {}
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
    for index, score in enumerate(scores):
        text_id = 101 + index
        page = (index % 5) + 1
        hits.append({"score": score, "metadata": {"external_text_id": str(text_id)}})
        pages[text_id] = page
    return hits, pages


class _CapturingDataSource:
    """Data source that mirrors the scores handed over by the engine."""

    def __init__(self, pages, fts_results=None):
        self.pages = pages
        self.fts_results = fts_results or []
        self.scores = None
        self.texts = None

    def fts_search(self, query_str, collection_id, max_results, language=None):
        return list(self.fts_results)

    def get_text_for_id(
        self, text_ids, surrounding_chunks=0, text_scores=None, text_similarity=None
    ):
        self.scores = list(text_scores or [])
        self.similarity = list(text_similarity or []) if text_similarity else None
        self.texts = [
            _make_postgres_doc(
                "doc101" if text_id <= 105 else "doc102",
                score,
                page=self.pages.get(text_id, 1),
            )
            for text_id, score in zip(text_ids, self.scores)
        ]
        return self.texts


def test_search_engine_core_passes_vector_scores_to_the_data_source():
    hits, pages = _engine_hits_and_pages()
    ds = _CapturingDataSource(pages)
    engine = SearchEngineCore(
        vector_store=_RecordingVectorStore(hits), data_source=ds
    )
    outcome = engine.hybrid_search(
        query_text="pytanie",
        collection_id=1,
        filter_doc_names=[],
        max_results=10,
        rerank=False,
        hybrid_search=False,
    )
    # the cosine scores of the vector search reach the data source
    assert ds.scores == [0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
    stats = outcome["stats"]
    assert math.isclose(
        stats["doc101"]["score"], (0.95 + 0.9 + 0.85 + 0.8 + 0.75) / 5, rel_tol=1e-9
    )
    assert stats["doc101"]["weight_normalized"] == 1.0
    assert 0.0 < stats["doc101"]["score_normalized"] < 1.0


def test_search_engine_core_stats_use_the_rrf_scale_for_hybrid_results():
    hits, pages = _engine_hits_and_pages()
    top = 1.0 / 61.0
    fts_results = [(101 + index, 9.0 - index) for index in range(10)]
    ds = _CapturingDataSource(pages, fts_results=fts_results)
    engine = SearchEngineCore(
        vector_store=_RecordingVectorStore(hits), data_source=ds
    )
    outcome = engine.hybrid_search(
        query_text="pytanie",
        collection_id=1,
        filter_doc_names=[],
        max_results=10,
        rerank=False,
        hybrid_search=True,
        rrf_k=60,
    )
    # the fused scores, not the cosine ones, are handed over
    assert math.isclose(ds.scores[0], 2.0 * top, rel_tol=1e-9)
    # the ranking is decided by the fusion, so no cosine is reported
    assert ds.similarity is None
    assert outcome["score_scale"] == "rrf"
    assert outcome["stats"]["doc101"]["similarity"] is None
    chunk = outcome["detailed_results"][0]["result"]["text"]
    assert chunk["similarity"] is None
    stats = outcome["stats"]
    # the fused scores are normalized onto [0, 1] before being weighted
    assert 0.0 < stats["doc101"]["score_normalized"] <= 1.0
    assert stats["doc101"]["score_normalized"] > stats["doc102"]["score_normalized"]
    assert stats["doc101"]["weight_normalized"] == 1.0
    # the shares of the *displayed* (filtered) documents sum up to 1.0
    total = sum(res["score_weighted_scaled"] for res in stats.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)
