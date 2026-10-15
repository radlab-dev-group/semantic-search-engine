"""
Model configuration: ``max_model_length`` and the model listings.

``max_model_length`` is the largest number of tokens a model takes in: the
input text of an embedder, the ``query + text`` pair of a reranker.  A value
which is missing or nonsensical means "unknown" and must never impose a limit.
"""

from unittest.mock import patch

from django.test import SimpleTestCase

from sse_api.engine.api import ListEmbeddersModels, ListRerankersModels
from sse_api.engine.controllers.models_logic import (
    embedders_rerankers as legacy_module,
)
from sse_api.engine.core.config.models_config import (
    ALL_AVAILABLE_EMBEDDERS_MODELS,
    ALL_AVAILABLE_RERANKERS_MODELS,
    EmbeddingModelsConfig,
    _max_model_length,
)


class MaxModelLengthReadingTests(SimpleTestCase):
    def test_declared_length(self):
        self.assertEqual(_max_model_length({"max_model_length": 512}), 512)

    def test_number_as_text_is_accepted(self):
        self.assertEqual(_max_model_length({"max_model_length": "2048"}), 2048)

    def test_unknown_when_missing_or_not_a_length(self):
        for config in (
            {},
            {"max_model_length": None},
            {"max_model_length": True},
            {"max_model_length": 0},
            {"max_model_length": -8},
            {"max_model_length": "wide"},
        ):
            with self.subTest(config=config):
                self.assertIsNone(_max_model_length(config))


class ConfiguredModelsTests(SimpleTestCase):
    """The accessors report what ``configs/*.json`` declares."""

    def test_every_embedder_reports_its_own_limit(self):
        EmbeddingModelsConfig()
        for name, config in ALL_AVAILABLE_EMBEDDERS_MODELS.items():
            self.assertEqual(
                EmbeddingModelsConfig.get_embedder_max_model_length(name),
                _max_model_length(config),
                name,
            )

    def test_every_reranker_reports_its_own_limit(self):
        EmbeddingModelsConfig()
        for name, config in ALL_AVAILABLE_RERANKERS_MODELS.items():
            self.assertEqual(
                EmbeddingModelsConfig.get_reranker_max_model_length(name),
                _max_model_length(config),
                name,
            )

    def test_embedders_info_describes_the_models(self):
        EmbeddingModelsConfig()
        self.assertEqual(
            EmbeddingModelsConfig.embedders_info(),
            [
                {
                    "name": name,
                    "vector_size": config.get("vector_size"),
                    "max_model_length": _max_model_length(config),
                    "device": config.get("device"),
                }
                for name, config in ALL_AVAILABLE_EMBEDDERS_MODELS.items()
            ],
        )

    def test_rerankers_info_describes_the_models(self):
        EmbeddingModelsConfig()
        self.assertEqual(
            [entry["name"] for entry in EmbeddingModelsConfig.rerankers_info()],
            EmbeddingModelsConfig.rerankers(),
        )
        for entry in EmbeddingModelsConfig.rerankers_info():
            self.assertNotIn("vector_size", entry)


class ModelsEndpointsTests(SimpleTestCase):
    def setUp(self):
        self.embedders = ["test/embedder"]
        self.rerankers = ["test/reranker"]
        self.embedders_info = [
            {
                "name": "test/embedder",
                "vector_size": 8,
                "max_model_length": 2048,
                "device": "cpu",
            }
        ]
        self.rerankers_info = [
            {
                "name": "test/reranker",
                "max_model_length": 512,
                "device": "cpu",
            }
        ]

    def get_body(self, view):
        from rest_framework.test import APIRequestFactory

        request = APIRequestFactory().get("/")
        return view.as_view()(request).data["body"]

    def test_embedders_endpoint_reports_names_and_details(self):
        with (
            patch(
                "sse_api.engine.api.EmbeddingModelsConfig.embedders",
                return_value=self.embedders,
            ),
            patch(
                "sse_api.engine.api.EmbeddingModelsConfig.embedders_info",
                return_value=self.embedders_info,
            ),
        ):
            body = self.get_body(ListEmbeddersModels)
        self.assertEqual(body["models"], self.embedders)
        self.assertEqual(body["models_info"], self.embedders_info)

    def test_rerankers_endpoint_reports_names_and_details(self):
        with (
            patch(
                "sse_api.engine.api.EmbeddingModelsConfig.rerankers",
                return_value=self.rerankers,
            ),
            patch(
                "sse_api.engine.api.EmbeddingModelsConfig.rerankers_info",
                return_value=self.rerankers_info,
            ),
        ):
            body = self.get_body(ListRerankersModels)
        self.assertEqual(body["models"], self.rerankers)
        self.assertEqual(body["models_info"], self.rerankers_info)


class SingleRegistryTests(SimpleTestCase):
    """The old import path is a shim of the one registry, not a second copy."""

    def test_shim_exposes_the_same_class(self):
        self.assertIs(legacy_module.EmbeddingModelsConfig, EmbeddingModelsConfig)

    def test_shim_exposes_the_same_registries(self):
        self.assertIs(
            legacy_module.ALL_AVAILABLE_EMBEDDERS_MODELS,
            ALL_AVAILABLE_EMBEDDERS_MODELS,
        )
        self.assertIs(
            legacy_module.ALL_AVAILABLE_RERANKERS_MODELS,
            ALL_AVAILABLE_RERANKERS_MODELS,
        )
