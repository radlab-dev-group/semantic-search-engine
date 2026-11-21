"""
test_collection_options.py
--------------------------

The per-collection search and RAG profile (plan 03).

A collection stores its own hyperparameters of the search and the language
model which answers about it, in ``metadata_json`` under
``default_search_options`` and ``default_generation_options``.  A request
which does not name a value gets the one of the collection; the serializer
and ``POST set_collection_options`` expose the profile, and the validators
keep the sampling parameters inside their ranges.
"""

from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.core.validation import (
    collection_generation_options,
    collection_search_options,
    generation_options,
)
from sse_api.data.api import ListCollections, SetCollectionOptions
from sse_api.data.core.constants import (
    GENERATION_OPTIONS_METADATA_KEY,
    GLOBAL_DEFAULT_GENERATION_OPTIONS,
    GLOBAL_DEFAULT_SEARCH_OPTIONS,
    SEARCH_OPTIONS_METADATA_KEY,
)
from sse_api.data.models import CollectionOfDocuments
from sse_api.system.models import Organisation, OrganisationUser


class CollectionProfileModelTests(TestCase):
    """The profile accessors of ``CollectionOfDocuments``."""

    def setUp(self):
        self.organisation = Organisation.objects.create(name="profile org")
        self.user = User.objects.create_user(username="profile-user")
        self.organisation_user = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )

    def make_collection(self, name="profiled", metadata=None):
        return CollectionOfDocuments.objects.create(
            name=name,
            display_name=name.title(),
            created_by=self.organisation_user,
            metadata_json=metadata,
        )

    def test_a_fresh_collection_has_no_profile(self):
        collection = self.make_collection(metadata=None)
        self.assertEqual(collection.default_search_options(), {})
        self.assertEqual(collection.default_generation_options(), {})
        self.assertEqual(collection.default_generative_model, "")

    def test_defaults_complete_what_the_collection_declares(self):
        collection = self.make_collection(
            metadata={
                SEARCH_OPTIONS_METADATA_KEY: {"max_results": 15},
                GENERATION_OPTIONS_METADATA_KEY: {"generative_model": "gpt-x"},
            }
        )
        search = collection.search_options_with_defaults()
        self.assertEqual(search["max_results"], 15)
        self.assertEqual(
            search["hybrid_search"], GLOBAL_DEFAULT_SEARCH_OPTIONS["hybrid_search"]
        )
        generation = collection.generation_options_with_defaults()
        self.assertEqual(generation["generative_model"], "gpt-x")
        self.assertEqual(
            generation["percentage_rank_mass"],
            GLOBAL_DEFAULT_GENERATION_OPTIONS["percentage_rank_mass"],
        )
        self.assertEqual(collection.default_generative_model, "gpt-x")

    def test_setters_keep_the_other_metadata(self):
        collection = self.make_collection(
            metadata={"chunking": {"max_tokens_in_chunk": 10}}
        )
        collection.set_default_search_options({"max_results": 5})
        collection.set_default_generation_options({"temperature": 0.2})
        self.assertEqual(
            collection.metadata_json["chunking"], {"max_tokens_in_chunk": 10}
        )
        self.assertEqual(collection.default_search_options(), {"max_results": 5})
        self.assertEqual(
            collection.default_generation_options(), {"temperature": 0.2}
        )

    def test_broken_metadata_is_not_a_profile(self):
        for metadata in (
            {SEARCH_OPTIONS_METADATA_KEY: "many"},
            {GENERATION_OPTIONS_METADATA_KEY: [1, 2]},
            "not a dict",
        ):
            with self.subTest(metadata=metadata):
                collection = CollectionOfDocuments(metadata_json=metadata)
                self.assertEqual(collection.default_search_options(), {})
                self.assertEqual(collection.default_generation_options(), {})

    def test_request_values_win_over_the_profile(self):
        collection = self.make_collection(
            metadata={
                SEARCH_OPTIONS_METADATA_KEY: {"max_results": 15, "rrf_k": 40},
                GENERATION_OPTIONS_METADATA_KEY: {
                    "generative_model": "gpt-x",
                    "temperature": 0.1,
                },
            }
        )
        search = collection.apply_search_defaults({"max_results": 7, "other": None})
        self.assertEqual(search["max_results"], 7)
        self.assertEqual(search["rrf_k"], 40)

        generation = collection.apply_generation_defaults({"temperature": 0.9})
        self.assertEqual(generation["temperature"], 0.9)
        self.assertEqual(generation["generative_model"], "gpt-x")

    def test_generation_defaults_are_not_injected_into_a_request(self):
        """A collection without a profile leaves the request as it is."""
        collection = self.make_collection()
        self.assertEqual(
            collection.apply_generation_defaults({"top_k": 3}), {"top_k": 3}
        )
        self.assertEqual(collection.apply_generation_defaults(None), {})


class CollectionOptionsValidatorTests(TestCase):
    def test_search_profile_accepts_the_fusion_weights(self):
        options = collection_search_options(
            {
                "max_results": 15,
                "hybrid_search": False,
                "dense_weight": "0.7",
                "lexical_weight": 0.3,
                "min_similarity": 0.42,
                "max_chunks_per_document": 3,
            }
        )
        self.assertEqual(options["max_results"], 15)
        self.assertFalse(options["hybrid_search"])
        self.assertEqual(options["dense_weight"], 0.7)
        self.assertEqual(options["lexical_weight"], 0.3)
        self.assertEqual(options["min_similarity"], 0.42)
        self.assertEqual(options["max_chunks_per_document"], 3)

    def test_search_profile_rejects_a_weight_outside_the_range(self):
        for key, value in (
            ("dense_weight", 1.5),
            ("lexical_weight", -0.1),
            ("min_similarity", 2),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    collection_search_options({key: value})

    def test_generation_profile_validates_the_sampling_parameters(self):
        options = collection_generation_options(
            {
                "generative_model": "google/gemma-4-12b-it",
                "temperature": 0.2,
                "repetition_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "typical_p": 0.95,
                "max_new_tokens": 2048,
            }
        )
        self.assertEqual(options["temperature"], 0.2)
        self.assertEqual(options["repetition_penalty"], 1.1)
        self.assertEqual(options["top_p"], 0.9)
        self.assertEqual(options["top_k"], 40)
        self.assertEqual(options["typical_p"], 0.95)
        self.assertEqual(options["max_new_tokens"], 2048)

    def test_generation_profile_normalizes_the_aliases(self):
        options = collection_generation_options(
            {
                "rep_penalty": 1.05,
                "topp": 0.8,
                "topk": 20,
                "typicalp": 0.9,
                "max_tokens": 512,
            }
        )
        self.assertEqual(options["repetition_penalty"], 1.05)
        self.assertEqual(options["top_p"], 0.8)
        self.assertEqual(options["top_k"], 20)
        self.assertEqual(options["typical_p"], 0.9)
        self.assertEqual(options["max_new_tokens"], 512)
        for alias in ("rep_penalty", "topp", "topk", "typicalp", "max_tokens"):
            self.assertNotIn(alias, options)

    def test_generation_profile_rejects_values_outside_the_range(self):
        for options in (
            {"temperature": -1.0},
            {"temperature": 3.5},
            {"top_p": 1.5},
            {"top_p": 0.0},
            {"top_k": -1},
            {"top_k": 1.5},
            {"typical_p": 0.0},
            {"max_new_tokens": 0},
            {"repetition_penalty": 0.0},
            {"percentage_rank_mass": 101},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    collection_generation_options(options)

    def test_query_generation_options_accept_the_sampling_parameters(self):
        options = generation_options(
            {
                "generative_model": "gpt-x",
                "percentage_rank_mass": 40,
                "rep_penalty": 1.15,
                "max_tokens": 640,
            }
        )
        self.assertEqual(options["repetition_penalty"], 1.15)
        self.assertEqual(options["max_new_tokens"], 640)


class SetCollectionOptionsEndpointTests(TestCase):
    """``POST set_collection_options`` and the profile in ``GET collections``."""

    def setUp(self):
        self.factory = APIRequestFactory()
        self.organisation = Organisation.objects.create(name="endpoint org")
        self.user = User.objects.create_user(username="endpoint-user")
        self.organisation_user = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = CollectionOfDocuments.objects.create(
            name="options-collection",
            display_name="Options",
            created_by=self.organisation_user,
            metadata_json=None,
        )

    def post(self, payload):
        request = self.factory.post("/", payload, format="json")
        force_authenticate(request, user=self.user)
        response = SetCollectionOptions.as_view()(request)
        response.render()
        return response

    def get_collections(self):
        request = self.factory.get("/")
        force_authenticate(request, user=self.user)
        with (
            patch("sse_api.data.api.SemanticDBController") as semantic,
            patch(
                "sse_api.engine.controllers.database.relational_db."
                "RelationalDBController.get_user_collections",
                return_value=CollectionOfDocuments.objects.all(),
            ),
            patch(
                "sse_api.engine.controllers.database.relational_db."
                "RelationalDBController.get_user_organisation_collections",
                return_value=CollectionOfDocuments.objects.none(),
            ),
        ):
            semantic.return_value.collections.return_value = [self.collection.name]
            response = ListCollections.as_view()(request)
            response.render()
        return response

    def test_saving_a_full_profile(self):
        response = self.post(
            {
                "collection_name": self.collection.name,
                "search_options": {"max_results": 15, "hybrid_search": False},
                "generation_options": {
                    "generative_model": "google/gemma-4-12b-it",
                    "temperature": 0.2,
                    "repetition_penalty": 1.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "typical_p": 0.95,
                    "max_new_tokens": 2048,
                },
            }
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.assertTrue(response.data["status"], response.data)
        body = response.data["body"]
        self.assertEqual(body["default_generative_model"], "google/gemma-4-12b-it")
        self.assertEqual(body["default_search_options"]["max_results"], 15)
        self.assertFalse(body["default_search_options"]["hybrid_search"])
        self.assertEqual(body["default_generation_options"]["temperature"], 0.2)

        self.collection.refresh_from_db()
        saved = self.collection.metadata_json[GENERATION_OPTIONS_METADATA_KEY]
        self.assertEqual(saved["max_new_tokens"], 2048)

    def test_a_second_call_updates_only_what_it_sends(self):
        self.post(
            {
                "collection_name": self.collection.name,
                "search_options": {"max_results": 15},
                "generation_options": {"generative_model": "model-one"},
            }
        )
        self.post(
            {
                "collection_name": self.collection.name,
                "generation_options": {"temperature": 0.3},
            }
        )
        self.collection.refresh_from_db()
        search = self.collection.default_search_options()
        generation = self.collection.default_generation_options()
        self.assertEqual(search["max_results"], 15)
        self.assertEqual(generation["generative_model"], "model-one")
        self.assertEqual(generation["temperature"], 0.3)

    def test_generative_model_alone_is_enough(self):
        response = self.post(
            {
                "collection_name": self.collection.name,
                "generative_model": "model-two",
            }
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.default_generative_model, "model-two")

    def test_invalid_options_are_refused(self):
        response = self.post(
            {
                "collection_name": self.collection.name,
                "generation_options": {"temperature": 9.0},
            }
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.assertFalse(response.data.get("status"), response.data)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.default_generation_options(), {})

    def test_another_users_collection_is_not_reachable(self):
        other_user = User.objects.create_user(username="someone-else")
        other_organisation_user = OrganisationUser.objects.create(
            auth_user=other_user, organisation=self.organisation
        )
        request = self.factory.post(
            "/",
            {"collection_name": self.collection.name, "generative_model": "m"},
            format="json",
        )
        request.user = other_user
        response = SetCollectionOptions.as_view()(request)
        response.render()
        # The collection belongs to self.user: a request from another user
        # must not see it at all.
        self.assertNotEqual(response.status_code, 200)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.default_generative_model, "")

    def test_collections_endpoint_returns_the_profiles(self):
        self.post(
            {
                "collection_name": self.collection.name,
                "search_options": {"rrf_k": 30},
                "generation_options": {"generative_model": "model-three"},
            }
        )
        response = self.get_collections()
        self.assertEqual(response.status_code, 200, response.content)
        collections = response.data["body"]["collections"]
        entry = next(c for c in collections if c["name"] == self.collection.name)
        self.assertEqual(entry["default_search_options"]["rrf_k"], 30)
        self.assertEqual(
            entry["default_generation_options"]["generative_model"], "model-three"
        )
        # The answer carries a complete set, not only what was saved.
        self.assertIn("dense_weight", entry["default_search_options"])
        self.assertIn("temperature", entry["default_generation_options"])
