import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

with patch(
    "sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"
):
    from sse_api.chat.api import (
        AddUserMessageToChatWithSystemResponse,
        DeleteChat,
        GetSavedChatByHash,
        ListOfUserChats,
        SetChatStateAsSaved,
    )
from sse_api.chat.core.errors import (
    ALL_ERRORS_CHATS,
    CHAT_ID_NOT_FOUND,
    USER_DENIED_TO_CHAT,
    CANNOT_ADD_MESSAGE_CHAT_RO,
    COLLECTION_NOT_FOUND,
)
from sse_api.chat.models import (
    Chat,
    ContentSupervisorState,
    Message,
    MessageState,
    RAGMessageState,
)
from sse_api.data.models import CollectionOfDocuments
from sse_api.core.errors import MSG, MSG_EN, ECODE
from sse_api.system.models import Organisation, OrganisationUser
from sse_api.chat.controllers.message import MessageLogicController
from sse_api.chat.serializer import RAGStateSerializer
from sse_api.engine.models import (
    UserQuery,
    UserQueryResponse,
    UserQueryResponseAnswer,
)


class QueryRewriteTests(SimpleTestCase):
    def setUp(self):
        self.controller = object.__new__(MessageLogicController)
        self.history = [
            SimpleNamespace(role="user", text="What are the obligations?"),
            SimpleNamespace(role="assistant", text="Providers must keep records."),
        ]

    def test_rewrite_sends_questions_and_answers_using_configured_model(self):
        with patch.object(
            self.controller, "gen_model_controller", create=True
        ) as gen:
            gen.gen_model_controller.rewrite_query.return_value = "Records for SMEs?"
            result = self.controller._rewrite_query_with_llm(
                "And SMEs?",
                self.history,
                {"query_rewrite_model": "rewrite-model", "lang": "en"},
            )
            gen.gen_model_controller.rewrite_query.assert_called_once_with(
                query="And SMEs?",
                history=[
                    {"user": self.history[0].text, "assistant": self.history[1].text}
                ],
                model_name="rewrite-model",
                lang="en",
            )
        self.assertEqual(result, "Records for SMEs?")

    def test_rewrite_failure_returns_original_question(self):
        with patch.object(
            self.controller, "gen_model_controller", create=True
        ) as gen:
            gen.gen_model_controller.rewrite_query.side_effect = RuntimeError(
                "timeout"
            )
            result = self.controller._rewrite_query_with_llm(
                "And SMEs?", self.history, {"generative_model": "answer-model"}
            )
        self.assertEqual(result, "And SMEs?")

    def test_rewrite_uses_environment_model(self):
        with (
            patch.dict(
                "os.environ", {"LLM_ROUTER_QUERY_REWRITE_MODEL": "env-model"}
            ),
            patch.object(
                self.controller, "gen_model_controller", create=True
            ) as gen,
        ):
            gen.gen_model_controller.rewrite_query.return_value = "SMEs?"
            self.controller._rewrite_query_with_llm(
                "And SMEs?", self.history, {"generative_model": "answer-model"}
            )
            self.assertEqual(
                gen.gen_model_controller.rewrite_query.call_args.kwargs[
                    "model_name"
                ],
                "env-model",
            )

    def test_first_question_needs_no_router_call(self):
        self.assertEqual(
            self.controller._rewrite_query_with_llm("Hello", [], {}), "Hello"
        )

    def test_rag_state_serializes_saved_citations_and_search_query(self):
        citations = [{"n": 2, "text_id": 12, "document_id": 3, "page_number": 4}]
        state = RAGMessageState(
            extracted_query="Standalone question",
            sse_answer=UserQueryResponseAnswer(
                is_generative=True,
                answer_options={},
                generative_qa_result_json={"citations": citations},
            ),
        )
        data = RAGStateSerializer(state).data
        self.assertEqual(data["rewritten_query"], "Standalone question")
        self.assertEqual(data["citations"], citations)

    def test_template_instructions_do_not_replace_user_prompt(self):
        message = SimpleNamespace(text="question", options={}, search_options={})
        response = SimpleNamespace(user_query=object())
        with (
            patch(
                "sse_api.chat.controllers.message.SearchQueryController"
            ) as search,
            patch("sse_api.chat.controllers.message.RAGMessageState.objects.create"),
            patch.object(
                self.controller, "gen_model_controller", create=True
            ) as gen,
        ):
            search.new_query.return_value = {
                "query_response_id": 1,
                "template_prompts": ["Template instructions"],
            }
            search.get_user_response_by_id.return_value = response
            self.controller._prepare_rag_supervisor_state(
                message=message,
                history=[],
                collection=object(),
                organisation_user=object(),
                sse_engin_config_path="unused",
                user_msg_is_rag_question=True,
                system_prompt="Use a table and citations",
            )
            self.assertEqual(
                gen.generative_answer_for_response.call_args.kwargs["system_prompt"],
                "Template instructions\n\nUse a table and citations",
            )

    def test_history_transformation_skips_search_and_keeps_context(self):
        context = {
            "history": [{"user": "Obowiązki?", "assistant": "Ewidencja [4]."}],
            "sources": [{"n": 4, "text_id": 9}],
            "registry": [{"n": 4, "text_id": 9}],
            "question": "Zrób z tego tabelę",
            "history_only": False,
        }
        previous = SimpleNamespace(sse_response=object(), sse_query=object())
        message = SimpleNamespace(
            text=context["question"], options={}, search_options={}
        )
        with (
            patch.object(
                self.controller,
                "_prepare_rag_memory",
                return_value=(context, previous),
            ),
            patch.object(
                self.controller, "gen_model_controller", create=True
            ) as gen,
            patch(
                "sse_api.chat.controllers.message.SearchQueryController"
            ) as search,
            patch(
                "sse_api.chat.controllers.message.RAGMessageState.objects.create"
            ) as state,
        ):
            self.controller._prepare_rag_supervisor_state(
                message, [], object(), object(), "unused"
            )
            search.new_query.assert_not_called()
            args = gen.generative_answer_for_response.call_args.kwargs
            self.assertTrue(args["rag_context"]["history_only"])
            self.assertEqual(args["rag_context"]["sources"][0]["n"], 4)
            self.assertIs(args["user_response"], previous.sse_response)
            self.assertEqual(state.call_args.kwargs["extracted_query"], "")

    def test_disabled_rewrite_never_concatenates_old_topics(self):
        self.assertEqual(
            self.controller._prepare_sse_question_based_on_history(
                SimpleNamespace(text="New topic"),
                self.history,
                False,
                {"use_intelligent_query_rewrite": False},
            ),
            "New topic",
        )


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
                        "collection_name": None,
                        "chat_history": [],
                    },
                },
            )

    def test_list_chats_returns_collection_name_and_message_state(self):
        query = UserQuery.objects.create(
            organisation_user=self.owner,
            collection=self.collection,
            query_str_prompt="question",
            query_options={},
        )
        q_resp = UserQueryResponse.objects.create(
            user_query=query,
            general_stats_json={},
            detailed_results_json={"1": {"document_name": "doc-a.pdf", "text_str": "chunk"}},
        )
        rag_state = RAGMessageState.objects.create(
            contains_query=True,
            contains_instruction=False,
            extracted_query="question",
            sse_query=query,
            sse_response=q_resp,
        )
        msg_state = MessageState.objects.create(rag_message_state=rag_state)
        Message.objects.create(
            chat=self.chat,
            role="user",
            text="question",
            number=1,
            state=msg_state,
        )
        response = self.request_view(ListOfUserChats, {}, "get")
        self.assertTrue(response.data["status"])
        history = response.data["body"]["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["chat_id"], self.chat.pk)
        self.assertEqual(history[0]["collection_name"], self.collection.name)
        msg_data = history[0]["chat_history"][0]
        self.assertIsNotNone(msg_data["state"])
        self.assertEqual(
            msg_data["state"]["rag_message_state"]["sse_response"]["detailed_results_json"],
            {"1": {"document_name": "doc-a.pdf", "text_str": "chunk"}},
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
            (DeleteChat, {}, "post"),
            (DeleteChat, {"chat_id": "bad"}, "post"),
            (DeleteChat, {"chat_ids[]": []}, "post"),
        ):
            response = self.request_view(view, payload, method)
            self.assertFalse(response.data["status"])
            self.chat.refresh_from_db()
            self.assertFalse(self.chat.is_saved)

    def test_delete_owned_chat_removes_messages_and_states_keeping_queries(self):
        query = UserQuery.objects.create(
            organisation_user=self.owner,
            collection=self.collection,
            query_str_prompt="question",
            query_options={},
        )
        q_resp = UserQueryResponse.objects.create(
            user_query=query,
            general_stats_json={},
            detailed_results_json={},
        )
        rag_state = RAGMessageState.objects.create(
            contains_query=True,
            contains_instruction=False,
            extracted_query="question",
            sse_query=query,
            sse_response=q_resp,
        )
        cs_state = ContentSupervisorState.objects.create(
            state_type="moderation",
        )
        msg_state = MessageState.objects.create(
            rag_message_state=rag_state,
            content_supervisor_state=cs_state,
        )
        user_msg = Message.objects.create(
            chat=self.chat,
            role="user",
            text="question",
            number=1,
            state=msg_state,
        )
        asst_msg = Message.objects.create(
            chat=self.chat,
            role="assistant",
            text="answer",
            number=2,
            prev_message=user_msg,
        )
        user_msg.next_message = asst_msg
        user_msg.save()

        response = self.request_view(
            DeleteChat,
            {"chat_id": self.chat.pk},
        )
        self.assertTrue(response.data["status"])
        body = response.data["body"]
        self.assertEqual(body["deleted"]["chats"], 1)
        self.assertEqual(body["deleted"]["messages"], 2)
        self.assertEqual(body["deleted"]["message_states"], 1)
        self.assertEqual(body["deleted"]["rag_states"], 1)
        self.assertEqual(body["deleted"]["content_supervisor_states"], 1)
        self.assertEqual(body["not_found"]["chat_ids"], [])
        self.assertFalse(Chat.objects.filter(pk=self.chat.pk).exists())
        self.assertFalse(Message.objects.filter(chat_id=self.chat.pk).exists())
        self.assertFalse(MessageState.objects.filter(pk=msg_state.pk).exists())
        self.assertFalse(RAGMessageState.objects.filter(pk=rag_state.pk).exists())
        self.assertFalse(ContentSupervisorState.objects.filter(pk=cs_state.pk).exists())
        # Underlying search query and foreign chat stay untouched
        self.assertTrue(UserQuery.objects.filter(pk=query.pk).exists())
        self.assertTrue(Chat.objects.filter(pk=self.foreign.pk).exists())

    def test_delete_missing_or_foreign_chat_is_denied(self):
        for chat_id in (999999, self.foreign.pk):
            with self.subTest(chat_id=chat_id):
                response = self.request_view(
                    DeleteChat,
                    {"chat_id": chat_id},
                )
                self.assert_error(response, CHAT_ID_NOT_FOUND)
        self.assertTrue(Chat.objects.filter(pk=self.foreign.pk).exists())
        self.assertTrue(Message.objects.filter(pk=self.secret.pk).exists())
