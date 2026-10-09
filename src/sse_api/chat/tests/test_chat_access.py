import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

with patch(
    "sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"
):
    from sse_api.chat.api import (
        AddUserMessageToChatWithSystemResponse,
        GetSavedChatByHash,
        SetChatStateAsSaved,
    )
from sse_api.chat.core.errors import (
    ALL_ERRORS_CHATS,
    CHAT_ID_NOT_FOUND,
    USER_DENIED_TO_CHAT,
    CANNOT_ADD_MESSAGE_CHAT_RO,
    COLLECTION_NOT_FOUND,
)
from sse_api.chat.models import Chat, Message
from sse_api.data.models import CollectionOfDocuments
from sse_api.core.errors import MSG, MSG_EN, ECODE
from sse_api.system.models import Organisation, OrganisationUser


class ChatAccessTests(TestCase):
    def setUp(self):
        self.organisation = Organisation.objects.create(name="Chat tests")
        self.owner = self.make_user("owner")
        self.other = self.make_user("other")
        self.collection = CollectionOfDocuments.objects.create(
            name="Own",
            created_by=self.owner,
        )
        self.shared = CollectionOfDocuments.objects.create(
            name="Shared",
            created_by=self.other,
        )
        group_model = OrganisationUser._meta.get_field("user_groups").related_model
        self.group = group_model.objects.create(
            group_name="Readers",
            organisation=self.organisation,
        )
        self.owner.user_groups.add(self.group)
        self.shared.visible_to_groups.add(self.group)
        self.chat = Chat.objects.create(
            organisation_user=self.owner,
            collection=self.collection,
            hash="own-chat",
            options={},
        )
        self.foreign = Chat.objects.create(
            organisation_user=self.other,
            collection=self.shared,
            hash="foreign-chat",
            is_saved=True,
            read_only=True,
        )
        self.secret = Message.objects.create(
            chat=self.foreign,
            role="user",
            text="secret",
            number=1,
        )

    def make_user(self, username):
        return OrganisationUser.objects.create(
            auth_user=User.objects.create_user(username=username),
            organisation=self.organisation,
        )

    def request_view(self, view, payload, method="post"):
        factory = APIRequestFactory()
        if method == "get" and any(value is None for value in payload.values()):
            request = factory.generic(
                "GET", "/", json.dumps(payload), content_type="application/json"
            )
        else:
            request = (
                factory.get("/", payload)
                if method == "get"
                else factory.post("/", payload, format="json")
            )
        force_authenticate(request, user=self.owner.auth_user)
        return view.as_view()(request)

    def assert_error(self, response, name):
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["status"])
        self.assertNotIn("body", response.data)
        error = response.data["errors"][0]
        self.assertEqual(error[ECODE], ALL_ERRORS_CHATS[name][ECODE])
        self.assertEqual(error[MSG], ALL_ERRORS_CHATS[name][MSG][MSG_EN])

    def test_save_owned_chat_and_make_editable_again(self):
        for read_only in (True, "false"):
            response = self.request_view(
                SetChatStateAsSaved,
                {
                    "chat_id": self.chat.pk,
                    "read_only": read_only,
                },
            )
            self.assertTrue(response.data["status"])
            self.assertEqual(response.data["body"], {"chat_hash": self.chat.hash})
            self.chat.refresh_from_db()
            self.assertTrue(self.chat.is_saved)
            self.assertEqual(self.chat.read_only, read_only is True)

    def test_save_missing_or_foreign_chat_does_not_mutate(self):
        for chat_id, error in (
            (999999, CHAT_ID_NOT_FOUND),
            (self.foreign.pk, USER_DENIED_TO_CHAT),
        ):
            with self.subTest(chat_id=chat_id):
                response = self.request_view(
                    SetChatStateAsSaved,
                    {
                        "chat_id": chat_id,
                        "read_only": False,
                    },
                )
                self.assert_error(response, error)
                self.foreign.refresh_from_db()
                self.assertTrue(self.foreign.read_only)
                self.assertTrue(self.foreign.is_saved)

    def test_saved_hash_owned_history_and_unknown_or_unsaved_empty(self):
        self.chat.is_saved = True
        self.chat.read_only = True
        self.chat.save()
        message = Message.objects.create(
            chat=self.chat,
            role="user",
            text="question",
            number=1,
        )
        response = self.request_view(
            GetSavedChatByHash,
            {
                "chat_hash": self.chat.hash,
            },
            "get",
        )
        self.assertTrue(response.data["status"])
        self.assertEqual(response.data["body"]["chat_id"], self.chat.pk)
        self.assertTrue(response.data["body"]["is_read_only"])
        self.assertEqual(response.data["body"]["chat_history"][0]["id"], message.pk)
        self.chat.is_saved = False
        self.chat.save()
        for chat_hash in ("unknown", self.chat.hash):
            response = self.request_view(
                GetSavedChatByHash,
                {
                    "chat_hash": chat_hash,
                },
                "get",
            )
            self.assertEqual(
                response.data,
                {
                    "status": True,
                    "body": {
                        "chat_id": None,
                        "is_read_only": None,
                        "chat_history": [],
                    },
                },
            )

    def test_foreign_saved_hash_denied_without_history_read(self):
        with patch("sse_api.chat.api.ChatController.get_chat_messages") as history:
            response = self.request_view(
                GetSavedChatByHash,
                {
                    "chat_hash": self.foreign.hash,
                },
                "get",
            )
            self.assert_error(response, USER_DENIED_TO_CHAT)
            history.assert_not_called()

    def send(self, **overrides):
        payload = {
            "chat_id": self.chat.pk,
            "user_message": "question",
            "options": {},
        }
        payload.update(overrides)
        return self.request_view(AddUserMessageToChatWithSystemResponse, payload)

    def assert_message_denied(self, error, **payload):
        before = list(Message.objects.values())
        with patch(
            "sse_api.chat.api.ChatController.generate_assistant_message_cs_rag"
        ) as generate:
            response = self.send(**payload)
            self.assert_error(response, error)
            generate.assert_not_called()
        self.assertEqual(list(Message.objects.values()), before)

    def test_message_missing_foreign_and_read_only(self):
        self.assert_message_denied(CHAT_ID_NOT_FOUND, chat_id=999999)
        self.assert_message_denied(USER_DENIED_TO_CHAT, chat_id=self.foreign.pk)
        self.chat.read_only = True
        self.chat.save()
        self.assert_message_denied(CANNOT_ADD_MESSAGE_CHAT_RO)

    def test_message_explicit_collection_precedes_saved_collection(self):
        self.assert_successful_message(self.shared, collection_name=self.shared.name)
        self.chat.refresh_from_db()
        self.assertEqual(self.chat.collection, self.collection)

    def test_message_inherits_collection(self):
        self.assert_successful_message(self.collection)
        self.assert_successful_message(
            self.collection, collection_name=None, options="{}"
        )

    def assert_successful_message(self, collection, **payload):
        with patch(
            "sse_api.chat.api.ChatController.generate_assistant_message_cs_rag",
            return_value=("answer", [], None, 0.1),
        ) as generate:
            response = self.send(**payload)
            self.assertTrue(response.data["status"])
            self.assertEqual(generate.call_args.kwargs["collection"], collection)
            message = generate.call_args.kwargs["last_user_message"]
            self.assertTrue(
                Message.objects.filter(pk=message.pk, chat=self.chat).exists()
            )
            self.assertEqual(message.text, "question")

    def test_message_revoked_membership_or_grant_denied_explicit_and_inherited(self):
        self.chat.collection = self.shared
        self.chat.save()
        for revoke in (
            self.owner.user_groups.clear,
            self.shared.visible_to_groups.clear,
        ):
            self.owner.user_groups.add(self.group)
            self.shared.visible_to_groups.add(self.group)
            revoke()
            self.assert_message_denied(COLLECTION_NOT_FOUND)
            self.assert_message_denied(
                COLLECTION_NOT_FOUND, collection_name=self.shared.name
            )
            self.assert_successful_message(
                self.collection, collection_name=self.collection.name
            )

    def test_message_absent_collection_or_invalid_explicit_does_not_fallback(self):
        self.assert_message_denied(COLLECTION_NOT_FOUND, collection_name="missing")
        self.assert_message_denied(COLLECTION_NOT_FOUND, collection_name=" ")
        self.owner.user_groups.clear()
        self.assert_message_denied(
            COLLECTION_NOT_FOUND, collection_name=self.shared.name
        )
        self.chat.collection = None
        self.chat.save()
        self.assert_message_denied(COLLECTION_NOT_FOUND)
        self.assert_successful_message(
            self.collection, collection_name=self.collection.name
        )

    def test_invalid_input_has_no_writes_or_generation(self):
        invalid = (
            {"chat_id": None},
            {"chat_id": []},
            {"user_message": None},
            {"user_message": " "},
            {"options": []},
            {"options": "[1]"},
            {"collection_name": 123},
            {"search_options": []},
            {"system_prompt": {}},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                before = list(Message.objects.values())
                with patch(
                    "sse_api.chat.api.ChatController.generate_assistant_message_cs_rag"
                ) as generate:
                    response = self.send(**payload)
                    self.assertFalse(response.data["status"])
                    generate.assert_not_called()
                self.assertEqual(list(Message.objects.values()), before)
        for view, payload, method in (
            (
                SetChatStateAsSaved,
                {"chat_id": self.chat.pk, "read_only": []},
                "post",
            ),
            (SetChatStateAsSaved, {"chat_id": None, "read_only": True}, "post"),
            (GetSavedChatByHash, {"chat_hash": None}, "get"),
            (GetSavedChatByHash, {"chat_hash": " "}, "get"),
        ):
            response = self.request_view(view, payload, method)
            self.assertFalse(response.data["status"])
            self.chat.refresh_from_db()
            self.assertFalse(self.chat.is_saved)
