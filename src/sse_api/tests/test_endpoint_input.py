from unittest.mock import patch

from django.test import TestCase

with patch("sse_api.authorization.utils.config.RdlAuthConfig.load_config"):
    from sse_api.authorization.api.authorization import CreateRdlAuthToken, RefreshToken

from sse_api.core.errors_constants import MSG
from sse_api.tests.input_support import EndpointInputMixin


class SharedEndpointInputTests(EndpointInputMixin, TestCase):
    def test_authorization_bad_input_before_provider(self):
        with patch.object(CreateRdlAuthToken.token_handler, "get_token_with_options") as provider:
            for key, value in (("code", None), ("code", []), ("state", True), ("session_state", {})):
                with self.subTest(key=key, value=value):
                    self.assert_denial(self.invoke(CreateRdlAuthToken, {"code": "code", "state": "state", key: value}))
            provider.assert_not_called()
        with patch.object(RefreshToken.token_handler, "verify_and_decode_token") as verify:
            for token in (None, [], False, 0, " "):
                self.assert_denial(self.invoke(RefreshToken, {"refresh_token": token}))
            verify.assert_not_called()

    def test_non_object_body_before_optional_endpoint_side_effects(self):
        from sse_api.authorization.api.authorization import GenerateLoginUrl
        with patch.object(GenerateLoginUrl, "state_handler") as handler:
            for payload in ([], "text", False, 3):
                self.assert_denial(self.invoke(GenerateLoginUrl, payload))
            self.assertEqual(handler.mock_calls, [])

    def test_language_invalid_types_and_url_precedence(self):
        from sse_api.data.api import ListCategoriesFromCollection
        with patch("sse_api.data.api.RelationalDBController.get_collection", return_value=None):
            response = self.invoke(ListCategoriesFromCollection, url="/?collection_name=url&lang=pl",
                                   body={"collection_name": "body", "lang": "en"})
            self.assert_denial(response)
            self.assertEqual(response.data["errors"][0][MSG], "Nie odnaleziono kolekcji!")
            for lang in (None, [], {}, False):
                self.assert_denial(self.invoke(CreateRdlAuthToken, {"code": "c", "state": "s", "lang": lang}))

    def test_chat_invalid_input_has_no_controller_side_effects(self):
        with patch("sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"):
            from sse_api.chat.api import (
                NewChat,
                AddUserMessageToChatWithSystemResponse,
                SetChatStateAsSaved,
            )
        for view, payload, invalid in (
            (NewChat, {}, (("options", []), ("options", None), ("collection_name", False), ("search_options", "[]"))),
            (AddUserMessageToChatWithSystemResponse, {"chat_id": 1, "user_message": "hi", "options": {}},
             (("chat_id", "bad"), ("chat_id", False), ("user_message", {}), ("options", "[]"),
              ("system_prompt", 2), ("search_options", None))),
            (SetChatStateAsSaved, {"chat_id": 1, "read_only": False},
             (("chat_id", []), ("read_only", "bad"))),
        ):
            with patch.object(view, "chat_controller") as controller:
                for key, value in invalid:
                    with self.subTest(view=view.__name__, key=key):
                        self.assert_denial(self.invoke(view, {**payload, key: value}))
                self.assertEqual(controller.mock_calls, [])

    def test_chat_get_hash_and_false_state_contract(self):
        with patch("sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"):
            from sse_api.chat.api import GetSavedChatByHash, SetChatStateAsSaved
        with patch.object(GetSavedChatByHash, "chat_controller") as controller:
            controller.get_chat_by_chat_hash.return_value = None
            response = self.invoke(GetSavedChatByHash, url="/?chat_hash=url", body={"chat_hash": "body"})
            self.assertTrue(response.data["status"])
            self.assertEqual(controller.get_chat_by_chat_hash.call_args.kwargs["chat_hash"], "url")
        with patch.object(SetChatStateAsSaved, "chat_controller") as controller:
            controller.get_chat_by_id.return_value.organisation_user = self.profile
            controller.set_chat_as_saved.return_value = "hash"
            response = self.invoke(SetChatStateAsSaved, {"chat_id": "1", "read_only": "false"})
            self.assertTrue(response.data["status"])
            self.assertIs(controller.set_chat_as_saved.call_args.kwargs["read_only"], False)

    def test_new_chat_object_options_are_normalized_before_storage(self):
        with patch("sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"):
            from sse_api.chat.api import NewChat
        from sse_api.chat.models import Chat
        for options in ({}, "{}"):
            with self.subTest(options=options):
                response = self.invoke(NewChat, {"options": options, "search_options": "{}"})
                self.assertTrue(response.data["status"])
        self.assertEqual(Chat.objects.count(), 2)
        for chat in Chat.objects.all():
            self.assertEqual(chat.options, {})