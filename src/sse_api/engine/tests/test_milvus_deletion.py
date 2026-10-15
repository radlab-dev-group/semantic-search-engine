"""
Unit tests of the deletion part of MilvusHandler.

The vector database is mocked; the tests pin how the handler translates
"delete texts/documents by their external ids" into a delete by primary
key, how the primary keys are resolved in batches, and how the collection
itself is dropped.
"""

import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sse_api.engine.controllers.search.semantic", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.search.query", MagicMock())
sys.modules.setdefault("sse_api.engine.controllers.search.relational", MagicMock())
sys.modules.setdefault(
    "sse_api.engine.controllers.database.relational_db", MagicMock()
)
sys.modules.setdefault(
    "sse_api.engine.controllers.database.semantic_db", MagicMock()
)
sys.modules.setdefault(
    "sse_api.engine.controllers.models_logic.embedders_rerankers", MagicMock()
)
sys.modules.setdefault("sse_api.engine.controllers.system_logic.system", MagicMock())

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.tests.settings_django")
import django

django.setup()

from sse_api.engine.controllers.database.milvus import MilvusHandler


def make_handler(query_results=None, existing_collections=None):
    handler = object.__new__(MilvusHandler)
    handler._collection_name = "test_collection"
    client = MagicMock()
    handler._milvus_client = client
    # the mocked client stands for an already established connection
    handler._prepare_milvus_client = lambda load_collection=False: None
    collections = (
        ["test_collection"] if existing_collections is None else existing_collections
    )
    handler._list_collections = lambda: list(collections)
    if query_results is not None:
        client.query.side_effect = query_results
    return handler


def test_dropping_a_collection_removes_it_from_milvus():
    handler = make_handler()

    dropped = handler.drop_collection()

    assert dropped is True
    handler._milvus_client.drop_collection.assert_called_once_with("test_collection")


def test_dropping_a_missing_collection_removes_nothing():
    handler = make_handler(existing_collections=[])

    dropped = handler.drop_collection()

    assert dropped is False
    handler._milvus_client.drop_collection.assert_not_called()


def test_the_matching_primary_keys_are_queried_and_removed_in_batches():
    handler = make_handler(query_results=[[{"pk": 10}, {"pk": 11}], []])

    deleted = handler._delete_entries('metadata["x"] in ["1"]')

    assert deleted == 2

    query_call, second_query_call = handler._milvus_client.query.call_args_list
    for call in (query_call, second_query_call):
        assert call.kwargs["collection_name"] == "test_collection"
        assert call.kwargs["filter"] == 'metadata["x"] in ["1"]'
        assert call.kwargs["output_fields"] == ["pk"]
        assert call.kwargs["limit"] == MilvusHandler.MILVUS_DELETE_QUERY_LIMIT

    (delete_call,) = handler._milvus_client.delete.call_args_list
    assert delete_call.kwargs["collection_name"] == "test_collection"
    assert delete_call.kwargs["filter"] == "pk in [10, 11]"


def test_primary_keys_are_queried_until_the_round_returns_nothing_new():
    handler = make_handler(query_results=[[{"pk": 10}], [{"pk": 10}]])

    deleted = handler._delete_entries('metadata["x"] in ["1"]')

    assert deleted == 1
    assert handler._milvus_client.query.call_count == 2
    assert handler._milvus_client.delete.call_count == 1


def test_deleting_texts_by_their_external_ids():
    handler = make_handler()
    handler._delete_entries = MagicMock(return_value=2)

    deleted = handler.delete_texts_by_external_ids(["1", 2])

    assert deleted == 2
    # ids are compared as strings, the order is kept, duplicates are removed
    handler._delete_entries.assert_called_once_with(
        'metadata["external_text_id"] in ["1", "2"]'
    )


def test_deleting_documents_by_their_external_ids():
    handler = make_handler()
    handler._delete_entries = MagicMock(return_value=0)

    deleted = handler.delete_documents_by_external_ids([5])

    assert deleted == 0
    handler._delete_entries.assert_called_once_with(
        'metadata["external_document_id"] in ["5"]'
    )


def test_deleting_without_any_ids_removes_nothing():
    handler = make_handler()

    deleted = handler.delete_texts_by_external_ids([])

    assert deleted == 0
    handler._milvus_client.query.assert_not_called()
    handler._milvus_client.delete.assert_not_called()


def test_deleting_from_a_missing_collection_removes_nothing():
    handler = make_handler(existing_collections=[])

    deleted = handler.delete_texts_by_external_ids(["1"])

    assert deleted == 0
    handler._milvus_client.query.assert_not_called()
    handler._milvus_client.delete.assert_not_called()
