import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sse_api.engine.controllers.search.semantic", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.search.query", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.search.relational", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.database.relational_db", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.database.semantic_db", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.models_logic.embedders_rerankers", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.system_logic.system", MagicMock())

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.tests.settings_django")
import django
django.setup()

from sse_api.engine.controllers.database.milvus import (
    INDEX_QUERY_PARAMS,
    MilvusHandler,
)

DEFAULT_INDEX_TYPE = MilvusHandler.DEFAULT_INDEX_TYPE
DB_FIELD_EMBEDDING = MilvusHandler.DB_FIELD_EMBEDDING


def test_default_index_params_structure():
    assert MilvusHandler.DEFAULT_INDEX_PARAMS["field_name"] == DB_FIELD_EMBEDDING
    assert (
        MilvusHandler.DEFAULT_INDEX_PARAMS["index_type"]
        == INDEX_QUERY_PARAMS[DEFAULT_INDEX_TYPE]["INDEX_PARAMS"]["index_type"]
    )
    assert (
        MilvusHandler.DEFAULT_INDEX_PARAMS["metric_type"]
        == INDEX_QUERY_PARAMS[DEFAULT_INDEX_TYPE]["INDEX_PARAMS"]["metric_type"]
    )


def test_prepare_collection_index_params_default():
    handler = object.__new__(MilvusHandler)
    handler.index_name = DEFAULT_INDEX_TYPE
    handler.DB_FIELD_EMBEDDING = DB_FIELD_EMBEDDING
    handler._collection_index_params = None

    handler._MilvusHandler__prepare_collection_index_params(None)

    assert handler._collection_index_params["field_name"] == DB_FIELD_EMBEDDING
    assert handler._collection_index_params["index_type"] == "IVF_FLAT"
    assert handler._collection_index_params["metric_type"] == "COSINE"


def test_prepare_collection_index_params_custom_with_field_name():
    handler = object.__new__(MilvusHandler)
    handler.index_name = "HNSW"
    handler.DB_FIELD_EMBEDDING = DB_FIELD_EMBEDDING
    handler._collection_index_params = None

    custom_params = {
        "field_name": "custom_vec",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    handler._MilvusHandler__prepare_collection_index_params(custom_params)

    assert handler._collection_index_params["field_name"] == "custom_vec"
    assert handler._collection_index_params["index_type"] == "HNSW"


def test_prepare_collection_index_params_custom_without_field_name():
    handler = object.__new__(MilvusHandler)
    handler.index_name = "HNSW"
    handler.DB_FIELD_EMBEDDING = DB_FIELD_EMBEDDING
    handler._collection_index_params = None

    custom_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    handler._MilvusHandler__prepare_collection_index_params(custom_params)

    assert handler._collection_index_params["field_name"] == DB_FIELD_EMBEDDING
    assert handler._collection_index_params["index_type"] == "HNSW"


def test_add_index_to_collection_invokes_milvus_correctly():
    handler = object.__new__(MilvusHandler)
    handler._collection_name = "test_collection"
    handler.DB_FIELD_EMBEDDING = DB_FIELD_EMBEDDING
    handler.index_name = DEFAULT_INDEX_TYPE
    handler._collection_index_params = {
        "field_name": "embedding",
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1536},
        "index_name": "emb_idx",
    }

    mock_client = MagicMock()
    mock_index_params = MagicMock()
    mock_client.prepare_index_params.return_value = mock_index_params
    handler._milvus_client = mock_client

    handler._MilvusHandler__add_index_to_collection()

    mock_client.prepare_index_params.assert_called_once_with()
    mock_index_params.add_index.assert_called_once_with(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1536},
        index_name="emb_idx",
    )
    mock_client.create_index.assert_called_once_with(
        collection_name="test_collection",
        index_params=mock_index_params,
    )


def test_search_uses_metric_type_of_existing_index():
    handler = object.__new__(MilvusHandler)
    handler._collection_name = "test_collection"
    handler.index_name = DEFAULT_INDEX_TYPE
    handler._search_params = None
    handler._milvus_client = MagicMock()
    handler._milvus_client.list_indexes.return_value = ["emb_idx"]
    handler._milvus_client.describe_index.return_value = {
        "metric_type": "IP"
    }
    handler._milvus_client.search.return_value = [[]]
    handler._MilvusHandler__prepare_milvus_client = MagicMock()
    handler._MilvusHandler__prepare_filtering_options = MagicMock(
        return_value=""
    )
    handler.prepare_embeddings = MagicMock(return_value=[[0.1, 0.2]])

    handler.search(search_text="test query")

    handler._milvus_client.describe_index.assert_called_once_with(
        collection_name="test_collection",
        index_name="emb_idx",
    )
    search_params = handler._milvus_client.search.call_args.kwargs[
        "search_params"
    ]
    assert search_params["metric_type"] == "IP"
