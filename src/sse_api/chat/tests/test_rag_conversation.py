from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase

with patch(
    "sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"
):
    from sse_api.chat.api import AddUserMessageToChatWithSystemResponse
    from sse_api.chat.controllers.chat import ChatController

from llm_router_lib.data_models.response import GenerativeAnswerResponse
from sse_api.chat.models import Chat, Message
from sse_api.data.controllers.chunk_offsets import chunk_offsets_for_texts
from sse_api.data.models import DocumentPageText
from sse_api.data.tests.test_preview import PreviewFixture
from sse_api.engine.models import UserQuery, UserQueryResponse
from sse_api.tests.input_support import EndpointInputMixin


class RagConversationTests(EndpointInputMixin, PreviewFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.make_fixture()
        self.chat = Chat.objects.create(
            organisation_user=self.profile, collection=self.collection
        )
        self.chunks = list(
            DocumentPageText.objects.filter(
                page=self.first, text_chunk_type="clear_texts"
            ).order_by("text_number")
        )
        self.payloads = []
        self.router = Mock()
        self.router.rewrite_query.return_value = SimpleNamespace(
            rewritten_query="AI Act dla MŚP"
        )
        self.router.generative_answer_with_citations.side_effect = self.generate

    def new_controller(self):
        with patch(
            "sse_api.engine.controllers.models_logic.generative.GenerativeModelConfig.load"
        ):
            return ChatController(add_to_db=True)

    def search(self, **kwargs):
        chunk = self.chunks[min(UserQuery.objects.count(), 1)]
        start, end = chunk_offsets_for_texts([chunk])[chunk.pk]
        query = UserQuery.objects.create(
            organisation_user=self.profile,
            collection=self.collection,
            query_str_prompt=kwargs["query_str"],
            query_options={},
        )
        response = UserQueryResponse.objects.create(
            user_query=query,
            general_stats_json={},
            detailed_results_json={
                "0": {
                    "document_name": self.document.name,
                    "document_id": self.document.pk,
                    "text_id": chunk.pk,
                    "page_number": 1,
                    "text_str": chunk.text_str,
                    "char_start": start,
                    "char_end": end,
                }
            },
        )
        return {"query_response_id": response.pk}

    def generate(self, *, payload):
        data = payload.model_dump()
        self.payloads.append(data)
        if "tabelę" in data["question_str"]:
            text = "| Obowiązek |\n|---|\n| Ewidencja [1] |\n| Dokumentacja [2] |"
        elif data["historical_messages"]:
            text = "Dla MŚP: ewidencja [1], dokumentacja [2]."
        else:
            text = "Ewidencja [1]."
        return GenerativeAnswerResponse(
            response=text,
            generation_time=0.1,
            citations=data["enumerated_documents"],
        )

    def send(self, text):
        return self.invoke(
            AddUserMessageToChatWithSystemResponse,
            {
                "chat_id": self.chat.pk,
                "user_message": text,
                "options": {
                    "use_rag_supervisor": True,
                    "generative_model": "test-model",
                    "percentage_rank_mass": 100,
                    "use_doc_names_in_response": False,
                },
            },
        ).data

    def test_three_turns_resume_and_transform_without_search(self):
        with (
            patch(
                "sse_api.chat.controllers.message.SearchQueryController.new_query",
                side_effect=self.search,
            ) as search,
            patch(
                "sse_api.engine.controllers.models_logic.generative.get_accumulated_docs_by_rank_perc",
                return_value=[self.document.name],
            ),
            patch(
                "sse_api.engine.controllers.models_logic.generative.GenerativeModelControllerApi._create_router_client",
                return_value=self.router,
            ),
        ):
            with patch.object(
                AddUserMessageToChatWithSystemResponse,
                "chat_controller",
                self.new_controller(),
            ):
                self.assertTrue(self.send("Obowiązki dostawców AI Act?")["status"])
                self.assertTrue(
                    self.send("A które z nich dotyczą małych firm?")["status"]
                )
            self.chat.refresh_from_db()
            with patch.object(
                AddUserMessageToChatWithSystemResponse,
                "chat_controller",
                self.new_controller(),
            ):
                response = self.send("Zrób z tego tabelę")
            self.assertTrue(response["status"])
            body = response["body"]
            self.assertEqual(body["answer_source"], "history")
            self.assertEqual(body["rewritten_query"], "")
            self.assertEqual(search.call_count, 2)
            self.assertEqual(self.router.rewrite_query.call_count, 1)
            self.assertEqual([item["n"] for item in body["citations"]], [1, 2])
            self.assertIn(
                "MŚP", self.payloads[-1]["historical_messages"][-1]["assistant"]
            )
            self.assertIn("|", body["generated_assistant_message"])
            self.assertEqual(Message.objects.filter(chat=self.chat).count(), 6)
            self.assertEqual(
                body["last_state"][0]["rag_message_state"]["answer_source"],
                "history",
            )

    def test_disabled_or_changed_sources_are_not_reused(self):
        with (
            patch(
                "sse_api.chat.controllers.message.SearchQueryController.new_query",
                side_effect=self.search,
            ),
            patch(
                "sse_api.engine.controllers.models_logic.generative.get_accumulated_docs_by_rank_perc",
                return_value=[self.document.name],
            ),
            patch(
                "sse_api.engine.controllers.models_logic.generative.GenerativeModelControllerApi._create_router_client",
                return_value=self.router,
            ),
            patch.object(
                AddUserMessageToChatWithSystemResponse,
                "chat_controller",
                self.new_controller(),
            ),
        ):
            self.assertTrue(self.send("Obowiązki?")["status"])
        controller = self.new_controller()
        history = controller.get_chat_messages(self.chat)
        message = SimpleNamespace(text="skróć", options={}, search_options={})
        for change in ("disabled", "changed"):
            with self.subTest(change=change):
                if change == "disabled":
                    self.document.use_in_search = False
                    self.document.save()
                else:
                    self.document.use_in_search = True
                    self.document.save()
                    DocumentPageText.objects.filter(pk=self.chunks[0].pk).update(
                        text_str="Changed"
                    )
                context, _ = controller._prepare_rag_memory(
                    message, history, self.collection
                )
                self.assertEqual(context["sources"], [])
                self.assertEqual(context["history"], [])

    def test_unknown_citation_rolls_back_the_whole_turn(self):
        self.router.generative_answer_with_citations.side_effect = (
            lambda **kwargs: GenerativeAnswerResponse(
                response="Unsupported [999].", generation_time=0.1, citations=[]
            )
        )
        with (
            patch(
                "sse_api.chat.controllers.message.SearchQueryController.new_query",
                side_effect=self.search,
            ),
            patch(
                "sse_api.engine.controllers.models_logic.generative.get_accumulated_docs_by_rank_perc",
                return_value=[self.document.name],
            ),
            patch(
                "sse_api.engine.controllers.models_logic.generative.GenerativeModelControllerApi._create_router_client",
                return_value=self.router,
            ),
            patch.object(
                AddUserMessageToChatWithSystemResponse,
                "chat_controller",
                self.new_controller(),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unknown citation"):
                self.send("Obowiązki?")
        self.assertFalse(Message.objects.filter(chat=self.chat).exists())
        self.assertEqual(UserQuery.objects.count(), 0)
