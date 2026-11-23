"""
test_rrf_fusion.py
------------------

The weighted rank fusion of the hybrid search (plan 04):

* both legs of a fusion contribute the same candidate pool, so a lexical
  match is not discarded before the fusion happens,
* the weights of the legs are a property of the search (the request or the
  profile of the collection decides), the defaults keep the historical
  behaviour,
* the cosine similarity of a chunk survives the fusion, and
* the document filter of the vector leg reaches the full-text leg.
"""

import math
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
        MAIN_LOGGER=__import__("logging").getLogger("test_rrf_fusion"),
    )
    django.setup()

from sse_api.engine.core.search_engine import (  # noqa: E402
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_LEXICAL_WEIGHT,
    rrf_merge,
)

# ---------------------------------------------------------------------------
# weighted rrf_merge
# ---------------------------------------------------------------------------


def _milvus_hit(text_id, score):
    return {"score": score, "metadata": {"external_text_id": str(text_id)}}


def test_rrf_merge_default_weights_split_the_rank_by_the_defaults():
    ids, scores = rrf_merge(
        [_milvus_hit(1, 0.9), _milvus_hit(2, 0.8)],
        [(1, 9.0), (2, 8.0)],
        k=60,
    )
    # a hit ranked first in both legs carries the whole pair of reciprocal
    # ranks, weighted by the defaults
    assert ids == [1, 2]
    assert math.isclose(
        scores[0],
        DEFAULT_DENSE_WEIGHT / 61 + DEFAULT_LEXICAL_WEIGHT / 61,
        rel_tol=1e-9,
    )
    assert math.isclose(
        scores[1],
        DEFAULT_DENSE_WEIGHT / 62 + DEFAULT_LEXICAL_WEIGHT / 62,
        rel_tol=1e-9,
    )


def test_rrf_merge_weights_decide_the_order():
    # id 1 is first in the vector leg only, id 2 is first in the text leg
    # only; the leg with the bigger weight decides the winner
    dense_wins, _ = rrf_merge(
        [_milvus_hit(1, 0.9), _milvus_hit(2, 0.1)],
        [(2, 9.0), (1, 8.0)],
        k=60,
        dense_weight=0.9,
        lexical_weight=0.1,
    )
    lexical_wins, _ = rrf_merge(
        [_milvus_hit(1, 0.9), _milvus_hit(2, 0.1)],
        [(2, 9.0), (1, 8.0)],
        k=60,
        dense_weight=0.1,
        lexical_weight=0.9,
    )
    assert dense_wins[0] == 1
    assert lexical_wins[0] == 2


def test_rrf_merge_only_the_ratio_of_the_weights_matters():
    first, first_scores = rrf_merge(
        [_milvus_hit(1, 0.5)], [(2, 9.0)], k=60, dense_weight=0.6, lexical_weight=0.4
    )
    second, second_scores = rrf_merge(
        [_milvus_hit(1, 0.5)], [(2, 9.0)], k=60, dense_weight=6.0, lexical_weight=4.0
    )
    assert first == second
    assert all(
        math.isclose(a, b, rel_tol=1e-9) for a, b in zip(first_scores, second_scores)
    )


def test_rrf_merge_a_zero_weight_drops_the_leg():
    ids, scores = rrf_merge(
        [_milvus_hit(1, 0.9), _milvus_hit(2, 0.8)],
        [(2, 9.0)],
        k=60,
        dense_weight=1.0,
        lexical_weight=0.0,
    )
    # the text leg contributes nothing: id 2 loses its only reciprocal rank
    assert ids == [1, 2]
    assert math.isclose(scores[0], 1.0 / 61, rel_tol=1e-9)
    assert math.isclose(scores[1], 1.0 / 62, rel_tol=1e-9)


def test_rrf_merge_refuses_a_fusion_without_any_weight():
    import pytest

    with pytest.raises(ValueError):
        rrf_merge([], [], k=60, dense_weight=0.0, lexical_weight=0.0)
    with pytest.raises(ValueError):
        rrf_merge([], [], k=60, dense_weight=-0.5, lexical_weight=0.5)


# ---------------------------------------------------------------------------
# query instruction of the embedder
# ---------------------------------------------------------------------------


def test_query_instruction_prefixes_only_the_query(monkeypatch):
    from sse_api.engine.controllers.database import milvus as milvus_module

    handler = milvus_module.MilvusHandler(
        jsonl_config_path=os.path.join(
            _APP_DIR, "configs", "milvus_config.example.json"
        ),
        collection_name="test_collection",
        create_db_if_not_exists=False,
        load_embedder=False,
        load_collection_and_schema=False,
        embedder_model_path="/models/google/embeddinggemma-2",
    )
    embedded_texts = []
    handler.prepare_embeddings = lambda texts: embedded_texts.extend(texts) or [
        [0.0] * 4 for _ in texts
    ]

    from unittest.mock import MagicMock

    handler._milvus_client = MagicMock()
    handler._milvus_client.search.return_value = [[]]
    handler._prepare_milvus_client = MagicMock()
    handler._MilvusHandler__prepare_filtering_options = MagicMock(return_value="")

    from sse_api.engine.core.config.models_config import EmbeddingModelsConfig

    monkeypatch.setattr(
        EmbeddingModelsConfig,
        "get_embedder_query_instruction",
        staticmethod(lambda name: "task: search result | query: "),
    )
    from sse_api.engine.core.models import backend as backend_module

    monkeypatch.setattr(
        backend_module,
        "resolve_model_name",
        lambda kind, path: "google/embeddinggemma-2",
    )

    handler.search(search_text="pytanie", max_results=5)

    assert embedded_texts == ["task: search result | query: pytanie"]


def test_query_instruction_is_empty_for_an_unconfigured_model():
    from sse_api.engine.controllers.database.milvus import MilvusHandler

    handler = MilvusHandler(
        jsonl_config_path=os.path.join(
            _APP_DIR, "configs", "milvus_config.example.json"
        ),
        collection_name="test_collection",
        create_db_if_not_exists=False,
        load_embedder=False,
        load_collection_and_schema=False,
        embedder_model_path="/models/plain-sentence-transformer",
    )
    # a model which is not in the catalog asks for no instruction
    assert handler.query_text_for_embedding("pytanie") == "pytanie"
