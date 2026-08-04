from django.test import TestCase
from django.contrib.auth.models import User
from system.models import Organisation, OrganisationUser
from data.models import CollectionOfDocuments
from engine.models import UserQuery, UserQueryResponse, UserQueryResponseAnswer
from engine.controllers.search.query import SearchQueryController
from engine.controllers.models_logic.generative import GenerativeModelController

class SecurityIDORTestCase(TestCase):
    def setUp(self):
        self.org = Organisation.objects.create(name="TestOrg")
        
        self.user1 = User.objects.create_user(username="user1")
        self.org_user1 = OrganisationUser.objects.create(auth_user=self.user1, organisation=self.org)
        
        self.user2 = User.objects.create_user(username="user2")
        self.org_user2 = OrganisationUser.objects.create(auth_user=self.user2, organisation=self.org)
        
        self.collection1 = CollectionOfDocuments.objects.create(
            name="Coll1", created_by=self.org_user1
        )
        
        self.query1 = UserQuery.objects.create(
            organisation_user=self.org_user1,
            collection=self.collection1,
            query_str_prompt="What is test?",
            query_options={}
        )
        
        self.response1 = UserQueryResponse.objects.create(
            user_query=self.query1,
            general_stats_json={},
            detailed_results_json={},
            structured_results={}
        )
        
        self.answer1 = UserQueryResponseAnswer.objects.create(
            user_response=self.response1,
            is_generative=True,
            answer_options={}
        )

    def test_get_user_response_by_id_owner(self):
        # User 1 should be able to access their own response
        resp = SearchQueryController.get_user_response_by_id(self.response1.pk, self.org_user1)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.pk, self.response1.pk)

    def test_get_user_response_by_id_non_owner(self):
        # User 2 should NOT be able to access User 1's response
        resp = SearchQueryController.get_user_response_by_id(self.response1.pk, self.org_user2)
        self.assertIsNone(resp)

    def test_get_user_query_response_answer_owner(self):
        # User 1 should be able to access their own answer
        ans = GenerativeModelController.get_user_query_response_answer(self.answer1.pk, self.org_user1)
        self.assertIsNotNone(ans)
        self.assertEqual(ans.pk, self.answer1.pk)

    def test_get_user_query_response_answer_non_owner(self):
        # User 2 should NOT be able to access User 1's answer
        ans = GenerativeModelController.get_user_query_response_answer(self.answer1.pk, self.org_user2)
        self.assertIsNone(ans)
