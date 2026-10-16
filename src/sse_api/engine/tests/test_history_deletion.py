"""
Deletion of the search history: the controller and the ``delete_search_history``
endpoint.

Tests use a real (in memory) relational database: both the happy path and the
IDOR isolation of the endpoint are observable on the rows which disappear.
"""

import datetime

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.chat.models import (
    Chat,
    Message,
    MessageState,
    RAGMessageState,
    ContentSupervisorState,
)
from sse_api.system.models import Organisation, OrganisationUser
from sse_api.data.models import CollectionOfDocuments
from sse_api.engine.api import DeleteSearchHistory
from sse_api.engine.controllers.history.deletion import (
    HistoryDeletionProtected,
    SearchHistoryDeletionController,
)
from sse_api.engine.models import (
    UserQuery,
    UserQueryResponse,
    UserQueryResponseAnswer,
)
from sse_api.core.errors_list import BUILT_IN_GENERAL_ERRORS, INVALID_PARAMS
from sse_api.core.errors_constants import ECODE
from sse_api.engine.core.errors import (
    QUERY_NOT_FOUND,
    ENGINE_ERRORS,
)


class FixtureTestCase(TestCase):
    """
    Two users; the first owns two queries (one with an answer and a chat
    RAG state), the second owns one query which must stay untouched.
    """

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

        self.collection = CollectionOfDocuments.objects.create(
            name="collection", created_by=self.org_user1
        )

        self.query1 = self.add_query(self.org_user1, "query one")
        self.response1 = self.add_response(self.query1)
        self.answer1 = self.add_answer(self.response1)
        self.query2 = self.add_query(self.org_user1, "query two")
        self.response2 = self.add_response(self.query2)
        self.query_u2 = self.add_query(self.org_user2, "secret query")
        self.response_u2 = self.add_response(self.query_u2)

    def add_query(self, organisation_user, prompt):
        return UserQuery.objects.create(
            organisation_user=organisation_user,
            collection=self.collection,
            query_str_prompt=prompt,
            query_options={"max_results": 10},
        )

    def add_response(self, user_query):
        return UserQueryResponse.objects.create(
            user_query=user_query,
            general_stats_json={"total": 1},
            detailed_results_json={"0": {"text_str": "a fragment"}},
            structured_results=[],
        )

    def add_answer(self, user_response):
        return UserQueryResponseAnswer.objects.create(
            user_response=user_response,
            is_generative=True,
            answer_options={"generative_model": "model"},
            generated_answer="an answer",
            generation_time=datetime.timedelta(seconds=1.0),
        )

    def add_rag_message(self, user_query, user_response):
        rag_state = RAGMessageState.objects.create(
            contains_query=True,
            contains_instruction=False,
            extracted_query="extracted",
            sse_query=user_query,
            sse_response=user_response,
        )
        content_state = ContentSupervisorState.objects.create(
            state_type="moderation"
        )
        message_state = MessageState.objects.create(
            rag_message_state=rag_state,
            content_supervisor_state=content_state,
        )
        chat = Chat.objects.create(
            organisation_user=self.org_user1, collection=self.collection
        )
        return Message.objects.create(
            chat=chat, role="assistant", text="msg", number=1, state=message_state
        )


class SearchHistoryDeletionControllerTests(FixtureTestCase):
    def setUp(self):
        super().setUp()
        self.controller = SearchHistoryDeletionController()
        self.rag_message = self.add_rag_message(self.query1, self.response1)

    def test_queries_are_removed_with_their_responses_and_answers(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_ids=[self.query1.pk, self.query2.pk],
        )

        self.assertEqual(
            result.deleted,
            {
                "answers": 1,
                "responses": 2,
                "queries": 2,
                "rag_states": 1,
            },
        )
        self.assertEqual(
            result.not_found,
            {"query_ids": [], "query_response_ids": []},
        )
        self.assertFalse(
            UserQuery.objects.filter(organisation_user=self.org_user1).exists()
        )
        self.assertFalse(
            UserQueryResponse.objects.filter(
                user_query__organisation_user=self.org_user1
            ).exists()
        )
        self.assertFalse(UserQueryResponseAnswer.objects.exists())
        self.assertFalse(RAGMessageState.objects.exists())
        # The state of the message is kept, only the RAG reference is gone.
        self.assertEqual(MessageState.objects.count(), 1)
        self.assertIsNone(MessageState.objects.first().rag_message_state)

    def test_the_chat_message_keeps_its_stateless_life(self):
        # The message stays, its state was the only link to the history.
        self.controller.delete_queries(
            organisation_user=self.org_user1, query_ids=[self.query1.pk]
        )

        message = Message.objects.get(pk=self.rag_message.pk)
        self.assertIsNotNone(message.state)
        self.assertIsNone(message.state.rag_message_state)
        self.assertEqual(Message.objects.count(), 1)

    def test_the_queries_of_other_users_are_not_removed(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1, query_ids=[self.query_u2.pk]
        )

        self.assertEqual(result.deleted["queries"], 0)
        self.assertEqual(
            result.not_found,
            {"query_ids": [self.query_u2.pk], "query_response_ids": []},
        )
        self.assertTrue(UserQuery.objects.filter(pk=self.query_u2.pk).exists())
        self.assertTrue(
            UserQueryResponse.objects.filter(pk=self.response_u2.pk).exists()
        )

    def test_the_ids_which_do_not_exist_are_reported(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_ids=[self.query1.pk, 999999],
        )

        self.assertEqual(result.deleted["queries"], 1)
        self.assertEqual(
            result.not_found,
            {"query_ids": [999999], "query_response_ids": []},
        )

    def test_responses_are_removed_by_their_ids(self):
        # The history API serves query_response_id, so deleting by it has to
        # work exactly like deleting by query_id.
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_response_ids=[self.response1.pk],
        )

        self.assertEqual(result.deleted["queries"], 1)
        self.assertEqual(result.deleted["responses"], 1)
        self.assertEqual(result.deleted["answers"], 1)
        self.assertEqual(
            result.not_found,
            {"query_ids": [], "query_response_ids": []},
        )
        self.assertFalse(UserQuery.objects.filter(pk=self.query1.pk).exists())
        self.assertTrue(UserQuery.objects.filter(pk=self.query2.pk).exists())

    def test_a_single_response_id_reaches_its_query(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_response_ids=self.response2.pk,
        )

        self.assertEqual(result.deleted["queries"], 1)
        self.assertFalse(UserQuery.objects.filter(pk=self.query2.pk).exists())

    def test_both_identifiers_delete_the_union_only_once(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_ids=[self.query1.pk],
            query_response_ids=[self.response1.pk, self.response2.pk],
        )

        # query1 is addressed twice but has to be counted once.
        self.assertEqual(result.deleted["queries"], 2)
        self.assertEqual(result.deleted["responses"], 2)

    def test_responses_of_other_users_are_not_removed(self):
        result = self.controller.delete_queries(
            organisation_user=self.org_user1,
            query_response_ids=[self.response_u2.pk],
        )

        self.assertEqual(result.deleted["queries"], 0)
        self.assertEqual(
            result.not_found,
            {"query_ids": [], "query_response_ids": [self.response_u2.pk]},
        )
        self.assertTrue(
            UserQueryResponse.objects.filter(pk=self.response_u2.pk).exists()
        )

    def test_an_answer_quoted_by_a_state_of_another_query_is_protected(self):
        # The RAG state belongs to query 2 but quotes the answer of query 1
        # through sse_answer (PROTECT), so the deletion has to fail entirely.
        RAGMessageState.objects.create(
            contains_query=True,
            contains_instruction=False,
            sse_query=self.query2,
            sse_answer=self.answer1,
        )

        with self.assertRaises(HistoryDeletionProtected):
            self.controller.delete_queries(
                organisation_user=self.org_user1, query_ids=[self.query1.pk]
            )

        self.assertTrue(UserQuery.objects.filter(pk=self.query1.pk).exists())
        self.assertTrue(
            UserQueryResponseAnswer.objects.filter(pk=self.answer1.pk).exists()
        )

    def test_a_failure_midway_leaves_the_history_untouched(self):
        from unittest import mock
        from django.db import transaction

        # If the transaction blows up, nothing may have changed: the deletion
        # of the history is atomic.
        with mock.patch.object(
            transaction, "atomic", side_effect=RuntimeError("the database is gone")
        ):
            with self.assertRaises(RuntimeError):
                self.controller.delete_queries(
                    organisation_user=self.org_user1, query_ids=[self.query1.pk]
                )

        self.assertTrue(UserQuery.objects.filter(pk=self.query1.pk).exists())
        self.assertTrue(
            UserQueryResponse.objects.filter(pk=self.response1.pk).exists()
        )
        self.assertTrue(
            UserQueryResponseAnswer.objects.filter(pk=self.answer1.pk).exists()
        )
        self.assertTrue(RAGMessageState.objects.exists())
        self.assertTrue(MessageState.objects.exists())


class DeleteSearchHistoryApiTests(FixtureTestCase):
    def setUp(self):
        super().setUp()
        self.rag_message = self.add_rag_message(self.query1, self.response1)

    def call(self, user, payload=None, method="post"):
        request = getattr(self.factory, method)(
            "/api/delete_search_history", payload or {}
        )
        force_authenticate(request, user=user)
        return DeleteSearchHistory.as_view()(request)

    def test_queries_are_deleted_by_their_ids(self):
        response = self.call(
            self.user1, {"query_ids[]": [self.query1.pk, self.query2.pk]}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        body = response.data["body"]
        self.assertEqual(body["deleted"]["queries"], 2)
        self.assertEqual(body["deleted"]["responses"], 2)
        self.assertEqual(body["deleted"]["answers"], 1)
        self.assertEqual(body["not_found"]["query_ids"], [])
        self.assertEqual(body["not_found"]["query_response_ids"], [])
        self.assertFalse(
            UserQuery.objects.filter(organisation_user=self.org_user1).exists()
        )

    def test_the_method_of_the_request_may_be_delete(self):
        request = self.factory.delete(
            "/api/delete_search_history",
            {"query_ids[]": [self.query1.pk]},
            format="json",
        )
        force_authenticate(request, user=self.user1)
        response = DeleteSearchHistory.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        self.assertFalse(UserQuery.objects.filter(pk=self.query1.pk).exists())

    def test_the_history_of_other_users_is_protected(self):
        response = self.call(self.user1, {"query_ids[]": [self.query_u2.pk]})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0][ECODE],
            ENGINE_ERRORS[QUERY_NOT_FOUND][ECODE],
        )
        self.assertTrue(UserQuery.objects.filter(pk=self.query_u2.pk).exists())

    def test_responses_are_deleted_by_their_ids(self):
        response = self.call(
            self.user1, {"query_response_ids[]": [self.response1.pk]}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        self.assertEqual(response.data["body"]["deleted"]["queries"], 1)
        self.assertFalse(UserQuery.objects.filter(pk=self.query1.pk).exists())
        self.assertTrue(UserQuery.objects.filter(pk=self.query2.pk).exists())

    def test_a_single_query_response_id_is_accepted(self):
        # The shape which the clients of get_search_history already hold.
        response = self.call(
            self.user1, {"query_response_id": int(self.response2.pk)}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        self.assertEqual(response.data["body"]["deleted"]["queries"], 1)
        self.assertFalse(UserQuery.objects.filter(pk=self.query2.pk).exists())

    def test_a_single_query_id_is_accepted(self):
        response = self.call(self.user1, {"query_id": int(self.query1.pk)})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["status"])
        self.assertFalse(UserQuery.objects.filter(pk=self.query1.pk).exists())

    def test_no_identifiers_at_all_are_rejected(self):
        response = self.call(self.user1, {})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertEqual(
            response.data["errors"][0][ECODE],
            BUILT_IN_GENERAL_ERRORS[INVALID_PARAMS][ECODE],
        )
        self.assertTrue(UserQuery.objects.filter(pk=self.query1.pk).exists())
