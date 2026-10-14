import datetime
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.system.models import Organisation, OrganisationUser
from sse_api.data.models import CollectionOfDocuments
from sse_api.engine.models import (
    UserQuery,
    UserQueryResponse,
    UserQueryResponseAnswer,
)
from sse_api.engine.api import ListOfSearchHistory, GetSearchHistoryById
from sse_api.engine.core.errors import RESPONSE_ACCESS_DENIED, ENGINE_ERRORS
from sse_api.core.errors_list import NO_REQUIRED_PARAMS, BUILT_IN_GENERAL_ERRORS
from sse_api.core.errors_constants import ECODE


class SearchHistoryApiTests(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.org = Organisation.objects.create(name="Test Org")

        self.user1 = User.objects.create_user(username="user1", password="password")
        self.org_user1 = OrganisationUser.objects.create(
            auth_user=self.user1, organisation=self.org
        )

        self.user2 = User.objects.create_user(username="user2", password="password")
        self.org_user2 = OrganisationUser.objects.create(
            auth_user=self.user2, organisation=self.org
        )

        self.collection1 = CollectionOfDocuments.objects.create(
            name="col1", created_by=self.org_user1
        )
        self.collection2 = CollectionOfDocuments.objects.create(
            name="col2", created_by=self.org_user1
        )

        # Create search query and response for user1 in collection1
        self.query1 = UserQuery.objects.create(
            organisation_user=self.org_user1,
            collection=self.collection1,
            query_str_prompt="what is semantic search?",
            query_options={"max_results": 10},
        )
        self.response1 = UserQueryResponse.objects.create(
            user_query=self.query1,
            general_stats_json={"time": 0.05, "total": 1},
            detailed_results_json={
                "0": {
                    "document_name": "intro.pdf",
                    "score": 0.95,
                    "text_str": "Semantic search uses embeddings.",
                }
            },
            structured_results=[{"fact": "search"}],
        )
        self.answer1 = UserQueryResponseAnswer.objects.create(
            user_response=self.response1,
            is_generative=True,
            answer_options={"generative_model": "test-model"},
            query_instruction_prompt="Explain simply",
            generated_answer="Semantic search finds meaning.",
            generated_answer_translated="Wyszukiwanie semantyczne znajduje znaczenie.",
            rate_value=5,
            rate_nax_value=5,
            rate_comment="helpful",
            generation_time=datetime.timedelta(seconds=1.2),
        )

        # Create second search query and response for user1 in collection2 (no answer)
        self.query2 = UserQuery.objects.create(
            organisation_user=self.org_user1,
            collection=self.collection2,
            query_str_prompt="what is milvus?",
            query_options={"max_results": 5},
        )
        self.response2 = UserQueryResponse.objects.create(
            user_query=self.query2,
            general_stats_json={"time": 0.02, "total": 1},
            detailed_results_json={
                "0": {
                    "document_name": "milvus.pdf",
                    "score": 0.88,
                    "text_str": "Milvus is a vector database.",
                }
            },
            structured_results=[],
        )

        # Create search query and response for user2
        self.query_u2 = UserQuery.objects.create(
            organisation_user=self.org_user2,
            collection=self.collection1,
            query_str_prompt="secret user2 query",
            query_options={},
        )
        self.response_u2 = UserQueryResponse.objects.create(
            user_query=self.query_u2,
            general_stats_json={},
            detailed_results_json={"0": {"text_str": "secret"}},
            structured_results=[],
        )

    def test_list_search_history_all_user_entries(self):
        request = self.factory.get("/api/search_history")
        force_authenticate(request, user=self.user1)
        response = ListOfSearchHistory.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        history = response.data["body"]["history"]
        self.assertEqual(len(history), 2)

        # Ordered from newest to oldest
        first = history[0]
        self.assertEqual(first["query_response_id"], self.response2.pk)
        self.assertEqual(first["query"], "what is milvus?")
        self.assertEqual(first["collection_name"], "col2")
        self.assertEqual(first["answers"], [])
        self.assertIsNone(first["generated_answer"])

        second = history[1]
        self.assertEqual(second["query_response_id"], self.response1.pk)
        self.assertEqual(second["query"], "what is semantic search?")
        self.assertEqual(second["collection_name"], "col1")
        self.assertEqual(
            second["detailed_results"]["0"]["document_name"], "intro.pdf"
        )
        self.assertEqual(second["results"]["detailed_results"]["0"]["score"], 0.95)
        self.assertEqual(len(second["answers"]), 1)
        self.assertEqual(
            second["generated_answer"], "Semantic search finds meaning."
        )
        self.assertEqual(
            second["generated_answer_translated"],
            "Wyszukiwanie semantyczne znajduje znaczenie.",
        )
        self.assertEqual(second["answer"], "Semantic search finds meaning.")
        self.assertEqual(second["answers"][0]["rate_value"], 5)
        self.assertEqual(second["answers"][0]["rate_value_max"], 5)
        self.assertEqual(second["answers"][0]["rate_comment"], "helpful")
        self.assertEqual(second["answers"][0]["generation_time"], 1.2)

    def test_list_search_history_filtered_by_collection(self):
        request = self.factory.get("/api/search_history", {"collection_name": "col1"})
        force_authenticate(request, user=self.user1)
        response = ListOfSearchHistory.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        history = response.data["body"]["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["query_response_id"], self.response1.pk)
        self.assertEqual(history[0]["collection_name"], "col1")

    def test_list_search_history_with_query_response_id(self):
        request = self.factory.get(
            "/api/search_history", {"query_response_id": self.response1.pk}
        )
        force_authenticate(request, user=self.user1)
        response = ListOfSearchHistory.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        history = response.data["body"]["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["query_response_id"], self.response1.pk)

    def test_get_search_history_by_id(self):
        request = self.factory.get(
            "/api/get_search_history", {"query_response_id": self.response1.pk}
        )
        force_authenticate(request, user=self.user1)
        response = GetSearchHistoryById.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        entry = response.data["body"]
        self.assertEqual(entry["query_response_id"], self.response1.pk)
        self.assertEqual(entry["query_id"], self.query1.pk)
        self.assertEqual(entry["query"], "what is semantic search?")
        self.assertEqual(entry["collection_name"], "col1")
        self.assertEqual(
            entry["detailed_results"]["0"]["text_str"],
            "Semantic search uses embeddings.",
        )
        self.assertEqual(
            entry["generated_answer"], "Semantic search finds meaning."
        )
        self.assertEqual(
            entry["generated_answer_translated"],
            "Wyszukiwanie semantyczne znajduje znaczenie.",
        )

    def test_get_search_history_requires_query_response_id(self):
        request = self.factory.get("/api/get_search_history")
        force_authenticate(request, user=self.user1)
        response = GetSearchHistoryById.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0][ECODE],
            BUILT_IN_GENERAL_ERRORS[NO_REQUIRED_PARAMS][ECODE],
        )

    def test_security_idor_isolation_between_users(self):
        # User 1 cannot access User 2's search history via get_search_history
        request = self.factory.get(
            "/api/get_search_history", {"query_response_id": self.response_u2.pk}
        )
        force_authenticate(request, user=self.user1)
        response = GetSearchHistoryById.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0][ECODE],
            ENGINE_ERRORS[RESPONSE_ACCESS_DENIED][ECODE],
        )

        # User 2 list history does not contain User 1's entries
        request2 = self.factory.get("/api/search_history")
        force_authenticate(request2, user=self.user2)
        response2 = ListOfSearchHistory.as_view()(request2)

        self.assertEqual(response2.status_code, 200)
        self.assertTrue(response2.data["status"])
        history2 = response2.data["body"]["history"]
        self.assertEqual(len(history2), 1)
        self.assertEqual(history2[0]["query_response_id"], self.response_u2.pk)
        self.assertEqual(history2[0]["query"], "secret user2 query")
