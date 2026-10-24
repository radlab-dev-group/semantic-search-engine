"""
Hybrid search must not pay the latency of the vector search and the full
text search one after the other: the two retrievals read different
databases, so they run in parallel and the RRF merge waits for both.

The test below makes each side sleep and asserts the total time is close to
one side, not the sum of both.  A serial implementation fails it by a wide
margin.
"""

import threading
import time

from django.test import SimpleTestCase
from unittest.mock import MagicMock

from sse_api.engine.controllers.search.semantic import (
    DBSemanticSearchController,
)

SLEEP_SECONDS = 0.3


class HybridConcurrentRetrievalTests(SimpleTestCase):
    def test_vector_and_fulltext_run_in_parallel(self):
        controller = object.__new__(DBSemanticSearchController)
        controller._logger = MagicMock()
        controller._vector_store = None
        controller._milvus_handler = None
        template_controller = MagicMock()
        template_controller.prepare_templates_for_user.return_value = []
        controller._template_controller = template_controller
        controller._rrf_merge = lambda milvus_results, postgres_results, k, max_results: (
            [1, 2],
            [0.5, 0.4],
        )

        def slow_vector(*args, **kwargs):
            time.sleep(SLEEP_SECONDS)
            return [{"metadata": {"external_text_id": 1}, "score": 0.9}]

        def slow_fulltext(*args, **kwargs):
            time.sleep(SLEEP_SECONDS)
            return [(2, 0.8), (3, 0.7)]

        textual_controller = MagicMock()
        textual_controller.search_texts.side_effect = slow_fulltext
        textual_controller.get_texts.return_value = [
            {
                "result": {
                    "text": {
                        "text_str": "chunk one",
                        "score": 0.5,
                        "document_name": "doc1",
                        "page_number": 1,
                        "relative_path": "/doc1.txt",
                    }
                },
            }
        ]

        with (
            __import__(
                "unittest.mock", fromlist=["patch"]
            ).patch.object(
                DBSemanticSearchController, "search", side_effect=slow_vector
            ),
            __import__(
                "unittest.mock", fromlist=["patch"]
            ).patch(
                "sse_api.engine.controllers.search.semantic"
                ".DBTextSearchController"
            ) as mocked_textual,
        ):
            mocked_textual.return_value = textual_controller
            started = time.monotonic()
            DBSemanticSearchController.search_with_options(
                controller,
                question_str="query",
                search_params={
                    "max_results": 10,
                    "hybrid_search": True,
                    "rerank_results": False,
                },
                collection=MagicMock(),
                organisation_user=None,
                user_query=None,
            )
            elapsed = time.monotonic() - started

        # In parallel the two sides overlap; serially the total is >= 2 x the
        # sleep.  Allow some slack for thread start-up, but not for a full
        # serial run.
        self.assertLess(
            elapsed,
            2 * SLEEP_SECONDS,
            "hybrid retrieval appears to run serially (%.3fs)" % elapsed,
        )
