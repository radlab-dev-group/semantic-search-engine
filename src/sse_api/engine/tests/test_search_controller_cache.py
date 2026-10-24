"""
The search path must not re-load the embedder / reranker / tokenizer for
every request.  ``controller_for_collection`` returns one cached controller
per ``(collection, index, embedder, reranker, config)`` tuple, exactly the
way the indexing pipeline caches its controllers per worker process.

The factory is patched in every test: what is under test is the caching
policy, not the model loading itself.
"""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from sse_api.engine.controllers.search import query as query_module
from sse_api.engine.controllers.search.query import (
    _SEARCH_CONTROLLERS,
    clear_search_controller_cache,
    controller_for_collection,
)


def _collection(pk, name="docs"):
    collection = MagicMock()
    collection.pk = pk
    collection.name = name
    collection.embedder_index_type = "HNSW"
    collection.model_embedder = "radlab/emb"
    collection.model_reranker = "radlab/rer"
    return collection


class SearchControllerCacheTests(SimpleTestCase):
    def setUp(self):
        clear_search_controller_cache()
        self._factory = patch.object(
            query_module.DBSemanticSearchController,
            "prepare_controller_for_collection",
        )
        self.mocked_factory = self._factory.start()

    def tearDown(self):
        self._factory.stop()
        clear_search_controller_cache()

    def test_second_call_returns_the_same_controller(self):
        controller_for_collection(_collection(7))
        second = controller_for_collection(_collection(7))
        self.assertEqual(self.mocked_factory.call_count, 1)
        self.assertIs(second, self.mocked_factory.return_value)

    def test_different_collection_gets_a_different_controller(self):
        controller_for_collection(_collection(1, "a"))
        controller_for_collection(_collection(2, "b"))
        self.assertEqual(self.mocked_factory.call_count, 2)
        self.assertEqual(len(_SEARCH_CONTROLLERS), 2)

    def test_controller_is_built_with_the_collection_model(self):
        collection = _collection(3)
        collection.embedder_index_type = "IVF_FLAT"
        controller_for_collection(collection, jsonl_config_path="/cfg.jsonl")
        kwargs = self.mocked_factory.call_args.kwargs
        self.assertIs(kwargs["collection"], collection)
        self.assertEqual(kwargs["sse_engin_config_path"], "/cfg.jsonl")

    def test_config_path_defaults_to_the_milvus_config(self):
        controller_for_collection(_collection(4))
        config_path = self.mocked_factory.call_args.kwargs["sse_engin_config_path"]
        self.assertTrue(config_path.endswith("milvus_config.json"))

    def test_clearing_the_cache_forgets_every_controller(self):
        controller_for_collection(_collection(1))
        self.assertEqual(len(_SEARCH_CONTROLLERS), 1)
        clear_search_controller_cache()
        self.assertEqual(len(_SEARCH_CONTROLLERS), 0)
