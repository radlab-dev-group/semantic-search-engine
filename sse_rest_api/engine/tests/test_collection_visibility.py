from django.contrib.auth.models import User
from django.db.models import QuerySet
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIRequestFactory, force_authenticate
from unittest.mock import patch
import json

from data.models import CollectionOfDocuments
from engine.controllers.database.relational_db import RelationalDBController
from system.models import Organisation, OrganisationUser


class EndpointVisibilityMixin:
    indexing_options = json.dumps({
        "prepare_proper_pages": True, "merge_document_pages": False,
        "clear_text": True, "use_text_denoiser": False, "check_text_lang": False,
        "max_tokens_in_chunk": 128, "number_of_overlap_tokens": 0,
    })

    def request_view(self, view, user, payload=None):
        factory = APIRequestFactory()
        if payload is None:
            request = factory.get("/")
        else:
            request = factory.post("/", payload, format="multipart")
        force_authenticate(request, user=user.auth_user if hasattr(user, "auth_user") else user)
        return view.as_view()(request)

    def test_filtering_categories_only_from_visible_searchable_documents(self):
        from data.api import ListFilteringOptions
        from data.models import Document

        for collection in (self.private, self.shared, self.foreign, self.member_owned):
            Document.objects.create(
                name=collection.name, collection=collection, category=collection.name,
                added_by=collection.created_by, path="/test", relative_path="test",
                document_hash=collection.name,
            )
        Document.objects.create(
            name="disabled", collection=self.shared, category="disabled",
            added_by=self.owner, use_in_search=False, path="/disabled",
            relative_path="disabled", document_hash="disabled",
        )
        for user, categories in ((self.owner, ["Private", "Shared"]),
                                 (self.member, ["Shared", "Member owned", "Foreign"]),
                                 (self.nonmember, []), (self.outsider, ["Foreign"])):
            with self.subTest(user=user.pk):
                response = self.request_view(ListFilteringOptions, user)
                self.assertTrue(response.data["status"])
                self.assertCountEqual(response.data["body"]["categories"], categories)

    def test_list_endpoint_filters_visibility_and_deduplicates(self):
        from data.api import ListCollections

        names = list(CollectionOfDocuments.objects.values_list("name", flat=True))
        with patch("data.api.SemanticDBController") as semantic:
            semantic.return_value.collections.return_value = names
            for user in (self.owner, self.member, self.nonmember, self.outsider):
                with self.subTest(user=user.pk):
                    response = self.request_view(ListCollections, user)
                    self.assertTrue(response.data["status"])
                    self.assertCountEqual(
                        [row["id"] for row in response.data["body"]["collections"]],
                        self.visible_ids(user),
                    )

    def test_search_rejects_nonmember_before_query_execution(self):
        from engine.api import SearchWithOptions

        with patch("engine.api.SearchQueryController.new_query") as query:
            response = self.request_view(SearchWithOptions, self.nonmember, {
                "collection_name": self.shared.name, "query_str": "question", "options": "{}",
            })
            self.assertFalse(response.data["status"])
            query.assert_not_called()

    def test_search_passes_visible_collection_and_rejects_revocation(self):
        from engine.api import SearchWithOptions

        with patch("engine.api.SearchQueryController.new_query", return_value={}) as query:
            payload = {"collection_name": self.shared.name, "query_str": "question", "options": "{}"}
            response = self.request_view(SearchWithOptions, self.member, payload)
            self.assertTrue(response.data["status"])
            self.assertEqual(query.call_args.kwargs["collection"], self.shared)
            self.member.user_groups.clear()
            query.reset_mock()
            response = self.request_view(SearchWithOptions, self.member, payload)
            self.assertFalse(response.data["status"])
            query.assert_not_called()

    def test_upload_denied_before_external_indexing(self):
        from data.api import UploadAndIndexFilesToCollection

        with patch("data.api.UploadDocumentsController") as upload:
            response = self.request_view(UploadAndIndexFilesToCollection, self.nonmember, {
                "collection_name": self.shared.name, "indexing_options": self.indexing_options,
                "files[]": SimpleUploadedFile("test.txt", b"test"),
            })
            self.assertFalse(response.data["status"])
            upload.return_value.store_and_index_files_rel_db_post_request.assert_not_called()

    def test_upload_passes_visible_collection_to_indexer(self):
        from data.api import UploadAndIndexFilesToCollection
        from data.models import UploadedDocuments

        uploaded = UploadedDocuments.objects.create(
            dir_hash="visibility-upload", dir_path="/test", uploaded_by=self.member,
        )
        with patch("data.api.UploadDocumentsController") as upload:
            upload.return_value.store_and_index_files_rel_db_post_request.return_value = uploaded
            response = self.request_view(UploadAndIndexFilesToCollection, self.member, {
                "collection_name": self.shared.name, "indexing_options": self.indexing_options,
                "files[]": SimpleUploadedFile("test.txt", b"test"),
            })
            self.assertTrue(response.data["status"])
            self.assertEqual(
                upload.return_value.store_and_index_files_rel_db_post_request.call_args.kwargs["collection"],
                self.shared,
            )

    def test_rag_chat_creation_uses_visibility_and_revocation(self):
        with patch("engine.controllers.models_logic.generative.GenerativeModelConfig.load"):
            from chat.api import NewChat
        from chat.models import Chat

        payload = {"collection_name": self.shared.name}
        response = self.request_view(NewChat, self.member, payload)
        self.assertTrue(response.data["status"])
        self.assertEqual(Chat.objects.get().collection, self.shared)
        self.member.user_groups.clear()
        response = self.request_view(NewChat, self.member, payload)
        self.assertFalse(response.data["status"])
        self.assertEqual(Chat.objects.count(), 1)

    def test_missing_profile_returns_permission_denied(self):
        from data.api import ListFilteringOptions

        user = User.objects.create_user(username="no-profile")
        response = self.request_view(ListFilteringOptions, user)
        self.assertEqual(response.status_code, 403)


class CollectionVisibilityTestCase(EndpointVisibilityMixin, TestCase):
    def setUp(self):
        self.organisation = Organisation.objects.create(name="Visibility organisation")
        self.other_organisation = Organisation.objects.create(name="Other organisation")
        self.owner = self.make_user("owner", self.organisation)
        self.member = self.make_user("member", self.organisation)
        self.nonmember = self.make_user("nonmember", self.organisation)
        self.outsider = self.make_user("outsider", self.other_organisation)
        self.private = CollectionOfDocuments.objects.create(
            name="Private", created_by=self.owner
        )
        self.shared = CollectionOfDocuments.objects.create(
            name="Shared", created_by=self.owner
        )
        self.foreign = CollectionOfDocuments.objects.create(
            name="Foreign", created_by=self.outsider
        )
        self.member_owned = CollectionOfDocuments.objects.create(
            name="Member owned", created_by=self.member
        )
        self.group = self.make_group("Readers")
        self.second_group = self.make_group("Second readers")
        self.member.user_groups.add(self.group, self.second_group)
        self.shared.visible_to_groups.add(self.group, self.second_group)
        # Local group grants apply regardless of the collection owner's organisation.
        self.foreign.visible_to_groups.add(self.group)
        # Membership in a group outside the user's organisation must not grant access.
        self.outsider.user_groups.add(self.group)

    @staticmethod
    def make_user(username, organisation):
        return OrganisationUser.objects.create(
            auth_user=User.objects.create_user(username=username),
            organisation=organisation,
        )

    def make_group(self, name):
        group_model = OrganisationUser._meta.get_field("user_groups").related_model
        values = {"group_name": name}
        if any(field.name == "organisation" for field in group_model._meta.fields):
            values["organisation"] = self.organisation
        return group_model.objects.create(**values)

    def visible_ids(self, user, **kwargs):
        queryset = RelationalDBController.get_visible_collections(user, **kwargs)
        self.assertIsInstance(queryset, QuerySet)
        return list(queryset.values_list("pk", flat=True))

    def test_owner_sees_private_and_shared_collections_without_membership(self):
        self.assertCountEqual(
            self.visible_ids(self.owner), [self.private.pk, self.shared.pk]
        )

    def test_group_member_sees_shared_and_owned_collections_once(self):
        self.assertCountEqual(
            self.visible_ids(self.member),
            [self.shared.pk, self.member_owned.pk, self.foreign.pk],
        )

    def test_same_organisation_nonmember_sees_no_collections(self):
        self.assertEqual(self.visible_ids(self.nonmember), [])

    def test_cross_organisation_group_membership_does_not_grant_access(self):
        self.assertEqual(self.visible_ids(self.outsider), [self.foreign.pk])

    def test_foreign_group_grant_does_not_expose_private_collection(self):
        group_model = OrganisationUser._meta.get_field("user_groups").related_model
        foreign_group = group_model.objects.create(
            group_name="Foreign readers", organisation=self.other_organisation,
        )
        self.member.user_groups.add(foreign_group)
        self.private.visible_to_groups.add(foreign_group)
        self.assertNotIn(self.private.pk, self.visible_ids(self.member))
        self.assertIsNone(
            RelationalDBController.get_collection(self.member, self.private.name)
        )

    def test_unconstrained_lookup_still_requires_ownership_or_group_membership(self):
        # The plan preserves owner-only access when organisation checking is disabled.
        self.assertCountEqual(
            self.visible_ids(self.member, check_in_organisation=False),
            [self.member_owned.pk],
        )
        self.assertIsNone(RelationalDBController.get_collection(
            self.member, self.shared.name, check_in_organisation=False,
        ))
        self.assertEqual(RelationalDBController.get_collection(
            self.member, self.member_owned.name, check_in_organisation=False,
        ), self.member_owned)
        self.assertEqual(
            self.visible_ids(self.nonmember, check_in_organisation=False), []
        )

    def test_owner_and_multiple_group_matches_are_distinct(self):
        self.owner.user_groups.add(self.group, self.second_group)
        self.assertCountEqual(
            self.visible_ids(self.owner), [self.private.pk, self.shared.pk, self.foreign.pk]
        )

    def test_membership_revocation_takes_effect_on_next_lookup(self):
        self.assertIn(self.shared.pk, self.visible_ids(self.member))
        self.member.user_groups.remove(self.group, self.second_group)
        self.assertEqual(self.visible_ids(self.member), [self.member_owned.pk])

    def test_collection_grant_revocation_takes_effect_on_next_lookup(self):
        self.assertIn(self.shared.pk, self.visible_ids(self.member))
        self.shared.visible_to_groups.clear()
        self.assertCountEqual(self.visible_ids(self.member), [self.member_owned.pk, self.foreign.pk])
        self.assertIsNone(RelationalDBController.get_collection(self.member, self.shared.name))

    def test_collection_listing_uses_the_same_visibility_rules(self):
        for user in (self.owner, self.member, self.nonmember, self.outsider):
            with self.subTest(user=user.pk):
                collections = RelationalDBController.get_user_organisation_collections(
                    user, list(CollectionOfDocuments.objects.values_list("name", flat=True))
                )
                self.assertCountEqual(
                    [collection.pk for collection in collections],
                    self.visible_ids(user),
                )

    def test_collection_lookup_matches_visibility_for_every_user(self):
        for user in (self.owner, self.member, self.nonmember, self.outsider):
            visible_ids = self.visible_ids(user)
            for collection in (
                self.private, self.shared, self.foreign, self.member_owned
            ):
                with self.subTest(user=user.pk, collection=collection.pk):
                    result = RelationalDBController.get_collection(user, collection.name)
                    if collection.pk in visible_ids:
                        self.assertIsNotNone(result)
                        self.assertEqual(result.pk, collection.pk)
                    else:
                        self.assertIsNone(result)

    def test_collection_lookup_denies_revoked_membership(self):
        self.assertIsNotNone(
            RelationalDBController.get_collection(self.member, self.shared.name)
        )
        self.member.user_groups.clear()
        self.assertIsNone(
            RelationalDBController.get_collection(self.member, self.shared.name)
        )