"""
test_similarity.py
------------------

Tests for the COSINE metric alignment and the cosine similarity cutoff
(``min_similarity``) introduced in the search engine.

These tests are intentionally DB‑free:

* pure helpers from ``engine.engine_core.search_engine`` are tested
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
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
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
            "data",
            "system",
            "engine",
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

from data.controllers.constants import DEFAULT_MIN_SIMILARITY  # noqa: E402
from engine.controllers.database.milvus import (  # noqa: E402
    INDEX_QUERY_PARAMS,
    MilvusHandler,
)
from engine.controllers.search.semantic import (  # noqa: E402
    DBSemanticSearchController,
)
from engine.engine_core.search_engine import (  # noqa: E402
    SearchEngineCore,
    prepare_documents_stats,
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

    def __init__(self, distances):
        self._distances = distances
        self.last_kwargs = None

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


def _make_handler(distances):
    handler = MilvusHandler(
        jsonl_config_path="configs/milvus_config.json",
        collection_name="test_collection",
        create_db_if_not_exists=False,
        load_embedder=False,
        load_collection_and_schema=False,
    )
    handler._milvus_client = _FakeMilvusClient(distances)
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
        def get_text_for_id(self, text_ids, surrounding_chunks=0):
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
