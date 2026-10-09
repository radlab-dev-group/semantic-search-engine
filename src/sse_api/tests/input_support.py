import json

from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.system.models import Organisation, OrganisationUser

INDEXING_OPTIONS = {
    "prepare_proper_pages": True,
    "merge_document_pages": False,
    "clear_text": True,
    "use_text_denoiser": False,
    "check_text_lang": False,
    "max_tokens_in_chunk": 128,
    "number_of_overlap_tokens": 0,
}
GENERATION_OPTIONS = {
    "generative_model": "test-model",
    "percentage_rank_mass": 40,
    "use_doc_names_in_response": False,
    "translate_answer": False,
}


class EndpointInputMixin:
    def setUp(self):
        organisation = Organisation.objects.create(name="Input organisation")
        self.user = User.objects.create_user(username="input-user")
        self.profile = OrganisationUser.objects.create(
            auth_user=self.user,
            organisation=organisation,
        )
        self.factory = APIRequestFactory()

    def invoke(self, view, payload=None, method="post", url="/", body=None):
        if body is not None:
            request = self.factory.generic(
                "GET", url, json.dumps(body), content_type="application/json"
            )
        elif method == "get":
            request = self.factory.get(url, payload or {})
        else:
            request = self.factory.post(url, payload, format="json")
        force_authenticate(request, user=self.user)
        return view.as_view()(request)

    def assert_denial(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertTrue(response.data["errors"])
