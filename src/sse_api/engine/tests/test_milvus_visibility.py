"""
Deactivating vectors in Milvus (``MilvusHandler.set_active_by_external_ids``).

No Milvus server is needed: the handler gets a fake ``MilvusClient`` which
records the queries and the upserts, so the produced filter expression and the
written ``is_active`` flag are observable.
"""

import json
import os
import re
import tempfile
import unittest

from sse_api.engine.controllers.database.milvus import MilvusHandler


class FakeMilvusClient:
    """Records what the handler asks the vector database to do."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = []
        self.upserts = []
        self.deleted_collections = []

    def list_collections(self):
        return ["collection"]

    def query(self, collection_name, filter, output_fields, limit):  # noqa: A002
        self.queries.append({"filter": filter, "limit": limit})
        # Emulate Milvus: match the documents listed in the filter.
        wanted = set(re.findall(r'"(\d+)"', filter))
        return [
            dict(row)
            for row in self.rows
            if str(row["metadata"]["external_document_id"]) in wanted
        ]

    def upsert(self, collection_name, data):
        self.upserts.append(data)

    def close(self):
        return None


class SetActiveByExternalIdsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"pk": 1, "text": "a", "metadata": {"external_document_id": "10"}},
            {"pk": 2, "text": "b", "metadata": {"external_document_id": "10"}},
            {"pk": 3, "text": "c", "metadata": {"external_document_id": "20"}},
        ]
        self.client = FakeMilvusClient(self.rows)

        config = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"milvus_db_connection": self._connection()}, config)
        config.close()
        self.addCleanup(os.unlink, config.name)

        self.handler = MilvusHandler(
            jsonl_config_path=config.name,
            collection_name="collection",
            create_db_if_not_exists=False,
            load_embedder=False,
            load_collection_and_schema=False,
        )
        self.handler._milvus_client = self.client

    @staticmethod
    def _connection():
        return {
            "uri": "http://localhost",
            "host": "localhost",
            "port": 19530,
            "db_name": "sse",
            "user": "user",
            "password": "password",
        }

    def test_the_matched_entries_are_deactivated(self):
        updated = self.handler.set_active_by_external_ids([10], False)

        self.assertEqual(updated, 2)
        self.assertEqual(len(self.client.upserts), 1)
        for row in self.client.upserts[0]:
            self.assertFalse(row["is_active"])
        # The primary keys are kept, so the entries are updated in place.
        self.assertEqual(sorted(row["pk"] for row in self.client.upserts[0]), [1, 2])

    def test_the_entries_are_activated_again(self):
        updated = self.handler.set_active_by_external_ids([10, 20], True)

        self.assertEqual(updated, 3)
        self.assertTrue(all(row["is_active"] for row in self.client.upserts[0]))

    def test_the_filter_selects_the_documents_by_their_external_id(self):
        self.handler.set_active_by_external_ids([10, 20], False)

        self.assertEqual(
            self.client.queries[0]["filter"],
            'metadata["external_document_id"] in ["10", "20"]',
        )

    def test_the_base_is_active_filter_is_not_used(self):
        # Otherwise the entries which are being activated again could never
        # be found (they are currently is_active == false).
        self.handler.set_active_by_external_ids([10], True)

        self.assertNotIn("is_active == true", self.client.queries[0]["filter"])

    def test_no_ids_means_no_request_at_all(self):
        self.assertEqual(self.handler.set_active_by_external_ids([], False), 0)
        self.assertEqual(self.client.queries, [])
        self.assertEqual(self.client.upserts, [])

    def test_the_ids_are_deduplicated(self):
        self.handler.set_active_by_external_ids([10, "10", 10], False)

        self.assertEqual(
            self.client.queries[0]["filter"],
            'metadata["external_document_id"] in ["10"]',
        )


if __name__ == "__main__":
    unittest.main()
