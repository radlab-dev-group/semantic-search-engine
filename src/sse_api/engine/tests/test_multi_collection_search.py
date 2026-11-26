from unittest.mock import MagicMock, patch
from django.test import TestCase

from sse_api.engine.api import SearchWithOptions
from sse_api.tests.input_support import EndpointInputMixin


class MultiCollectionSearchTests(EndpointInputMixin, TestCase):
    def test_multi_collection_search_global_limit(self):
        col_a = MagicMock()
        col_a.name = "col_a"
        col_a.display_name = "Kolekcja A"

        col_b = MagicMock()
        col_b.name = "col_b"
        col_b.display_name = "Kolekcja B"

        def mock_get_collection(collection_name, created_by):
            if collection_name == "col_a":
                return col_a
            elif collection_name == "col_b":
                return col_b
            return None

        def mock_search_with_options(**kwargs):
            collection = kwargs.get("collection")
            if collection.name == "col_a":
                return (
                    {
                        "stats": {
                            "doc_a1.pdf": {
                                "score_weighted": 0.8,
                                "score_weighted_scaled": 1.0,
                                "hits": 2,
                            }
                        },
                        "detailed_results": {
                            "0": {
                                "score": 0.85,
                                "document_name": "doc_a1.pdf",
                                "text_str": "Text A1",
                            },
                            "1": {
                                "score": 0.45,
                                "document_name": "doc_a1.pdf",
                                "text_str": "Text A2",
                            },
                        },
                    },
                    [],
                    [],
                )
            else:
                return (
                    {
                        "stats": {
                            "doc_b1.pdf": {
                                "score_weighted": 0.9,
                                "score_weighted_scaled": 1.0,
                                "hits": 1,
                            }
                        },
                        "detailed_results": {
                            "0": {
                                "score": 0.95,
                                "document_name": "doc_b1.pdf",
                                "text_str": "Text B1",
                            }
                        },
                    },
                    [],
                    [],
                )

        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                side_effect=mock_get_collection,
            ),
            patch(
                "sse_api.engine.api.DBSemanticSearchController.prepare_controller_for_collection"
            ) as mock_prep,
            patch("sse_api.engine.models.UserQuery.objects.create") as mock_uq_create,
            patch(
                "sse_api.engine.models.UserQueryResponse.objects.create"
            ) as mock_uqr_create,
        ):
            mock_sem_ctrl = MagicMock()
            mock_sem_ctrl.search_with_options.side_effect = mock_search_with_options
            mock_prep.return_value = mock_sem_ctrl

            mock_uq = MagicMock()
            mock_uq_create.return_value = mock_uq

            mock_uqr = MagicMock()
            mock_uqr.pk = 777
            mock_uqr_create.return_value = mock_uqr

            payload = {
                "collection_name": "col_a",
                "collection_names": '["col_a", "col_b"]',
                "query_str": "Jakie sa procedury?",
                "options": {
                    "max_results": 2,
                    "multi_collection_mode": "global_limit",
                },
            }
            response = self.invoke(SearchWithOptions, payload)
            self.assertTrue(response.data["status"])
            body = response.data["body"]
            self.assertEqual(body["query_response_id"], 777)

            results = body["results"]
            detailed = results["detailed_results"]
            # With max_results=2 in global_limit, only the top 2 out of 3 should be returned
            self.assertEqual(len(detailed), 2)
            # First item should be score 0.95 from col_b
            self.assertEqual(detailed["0"]["score"], 0.95)
            self.assertEqual(detailed["0"]["collection_name"], "col_b")
            self.assertEqual(
                detailed["0"]["collection_display_name"], "Kolekcja B"
            )
            self.assertEqual(
                detailed["0"]["original_document_name"], "doc_b1.pdf"
            )
            self.assertEqual(
                detailed["0"]["document_name"], "[Kolekcja B] doc_b1.pdf"
            )

            # Second item should be score 0.85 from col_a
            self.assertEqual(detailed["1"]["score"], 0.85)
            self.assertEqual(detailed["1"]["collection_name"], "col_a")
            self.assertEqual(
                detailed["1"]["collection_display_name"], "Kolekcja A"
            )
            self.assertEqual(
                detailed["1"]["original_document_name"], "doc_a1.pdf"
            )
            self.assertEqual(
                detailed["1"]["document_name"], "[Kolekcja A] doc_a1.pdf"
            )

            # Verify stats keys and scaling
            stats = results["stats"]
            self.assertIn("[Kolekcja B] doc_b1.pdf", stats)
            self.assertIn("[Kolekcja A] doc_a1.pdf", stats)
            # sum of score_weighted_scaled should be 1.0
            total_scaled = sum(
                s["score_weighted_scaled"] for s in stats.values()
            )
            self.assertAlmostEqual(total_scaled, 1.0, places=4)

    def test_multi_collection_search_per_collection(self):
        col_a = MagicMock()
        col_a.name = "col_a"
        col_a.display_name = "Kolekcja A"

        col_b = MagicMock()
        col_b.name = "col_b"
        col_b.display_name = "Kolekcja B"

        def mock_get_collection(collection_name, created_by):
            if collection_name == "col_a":
                return col_a
            elif collection_name == "col_b":
                return col_b
            return None

        def mock_search_with_options(**kwargs):
            collection = kwargs.get("collection")
            if collection.name == "col_a":
                return (
                    {
                        "stats": {
                            "doc_a1.pdf": {
                                "score_weighted": 0.5,
                                "score_weighted_scaled": 1.0,
                            }
                        },
                        "detailed_results": {
                            "0": {
                                "score": 0.6,
                                "document_name": "doc_a1.pdf",
                                "text_str": "A1",
                            },
                            "1": {
                                "score": 0.4,
                                "document_name": "doc_a1.pdf",
                                "text_str": "A2",
                            },
                        },
                    },
                    [],
                    [],
                )
            else:
                return (
                    {
                        "stats": {
                            "doc_b1.pdf": {
                                "score_weighted": 0.5,
                                "score_weighted_scaled": 1.0,
                            }
                        },
                        "detailed_results": {
                            "0": {
                                "score": 0.7,
                                "document_name": "doc_b1.pdf",
                                "text_str": "B1",
                            },
                        },
                    },
                    [],
                    [],
                )

        with (
            patch(
                "sse_api.engine.api.RelationalDBController.get_collection",
                side_effect=mock_get_collection,
            ),
            patch(
                "sse_api.engine.api.DBSemanticSearchController.prepare_controller_for_collection"
            ) as mock_prep,
            patch("sse_api.engine.models.UserQuery.objects.create") as mock_uq_create,
            patch(
                "sse_api.engine.models.UserQueryResponse.objects.create"
            ) as mock_uqr_create,
        ):
            mock_sem_ctrl = MagicMock()
            mock_sem_ctrl.search_with_options.side_effect = mock_search_with_options
            mock_prep.return_value = mock_sem_ctrl

            mock_uq = MagicMock()
            mock_uq_create.return_value = mock_uq

            mock_uqr = MagicMock()
            mock_uqr.pk = 888
            mock_uqr_create.return_value = mock_uqr

            payload = {
                "collection_name": "col_a",
                "collection_names": '["col_a", "col_b"]',
                "query_str": "Wszystkie wyniki",
                "options": {
                    "max_results": 2,
                    "multi_collection_mode": "per_collection",
                },
            }
            response = self.invoke(SearchWithOptions, payload)
            self.assertTrue(response.data["status"])
            body = response.data["body"]
            results = body["results"]
            detailed = results["detailed_results"]
            # per_collection pools ALL results (2 from col_a + 1 from col_b = 3)
            self.assertEqual(len(detailed), 3)

    def test_multi_collection_access_denied(self):
        col_a = MagicMock()
        col_a.name = "col_a"

        def mock_get_collection(collection_name, created_by):
            if collection_name == "col_a":
                return col_a
            return None  # col_b denied

        with patch(
            "sse_api.engine.api.RelationalDBController.get_collection",
            side_effect=mock_get_collection,
        ):
            payload = {
                "collection_name": "col_a",
                "collection_names": '["col_a", "col_forbidden"]',
                "query_str": "Pytanie",
                "options": {},
            }
            response = self.invoke(SearchWithOptions, payload)
            self.assertFalse(response.data["status"])
            self.assertIn("errors", response.data)
