import logging
import os
import random
import datetime

from typing import List, Dict

from sse_api.chat.models import (
    Chat,
    Message,
    MessageState,
    ContentSupervisorState,
    RAGMessageState,
)
from sse_api.data.models import (
    OrganisationUser,
    CollectionOfDocuments,
    DocumentPageText,
)
from sse_api.data.controllers.chunk_offsets import chunk_offsets_for_texts
from sse_api.chat.controllers.memory import (
    build_memory,
    is_history_transformation,
    scoped_history,
)

from sse_api.content_supervisor.supervisor import ContentSupervisor
from sse_api.content_supervisor.processors.regex_processors import URLRegexProcessor

from sse_api.engine.controllers.search.query import SearchQueryController
from sse_api.engine.controllers.models_logic.generative import (
    GenerativeModelController,
)


class MessageLogicController:
    USER_ROLE = "user"
    SYSTEM_ROLE = "system"
    ASSISTANT_ROLE = "assistant"
    AVAILABLE_SYSTEM_ROLES = [SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE]
    REMOVE_LAST_QUESTION_CHAR = [".", ",", "?", "!", ";", "/", "-"]
    USER_QUERY_PROMPT_HISTORY = (
        "\n\nTrzymając kontekst wcześniejszych pytań, odpowiedz na pytanie: "
    )

    def __init__(self, add_to_db: bool = True):
        self._add_to_db = add_to_db
        self.cs_controller = ContentSupervisor(options={"use_executor": True})
        self.gen_model_controller = GenerativeModelController(store_to_db=True)

    def add_user_message(
        self, chat: Chat, message: str, options: dict, search_options: dict
    ) -> (List[Message], Message):
        history = self.get_chat_messages(chat=chat)
        user_message = Message.objects.create(
            chat=chat,
            role=self.USER_ROLE,
            text=message,
            number=len(history) + 1,
            options=options,
            search_options=search_options,
        )
        if len(history):
            last_hist_msg = history[-1]
            last_hist_msg.next_message = user_message
            user_message.prev_message = last_hist_msg
            if self._add_to_db:
                last_hist_msg.save()
                user_message.save()
        return history, user_message

    def generate_assistant_message_cs_rag(
        self,
        chat: Chat,
        collection: CollectionOfDocuments | None,
        last_user_message: Message,
        history: List[Message],
        options: dict,
        organisation_user: OrganisationUser,
        sse_engin_config_path: str,
        last_questions_to_query: int = -1,
        system_prompt: str or None = None,
    ) -> (str, List[Message], MessageState, float | None):
        gen_assistant_msg_str, message_state = self._prepare_message_state(
            user_message=last_user_message,
            history=history,
            collection=collection,
            organisation_user=organisation_user,
            options=options,
            sse_engin_config_path=sse_engin_config_path,
            last_questions_to_query=last_questions_to_query,
            system_prompt=system_prompt,
        )

        if message_state is not None:
            last_user_message.state = message_state
            if self._add_to_db:
                last_user_message.save()

        generation_time = 0
        gen_assistant_msg_str_translated = None
        if gen_assistant_msg_str is None or not len(gen_assistant_msg_str.strip()):
            history_turns = self._convert_to_history_turns(history)
            (
                gen_assistant_msg_str,
                gen_assistant_msg_str_translated,
                generation_time,
            ) = self.gen_model_controller.model_response_cs_rag(
                query_options=options,
                last_user_message=last_user_message.text,
                history=history_turns,
                message_state=message_state,
            )
            if gen_assistant_msg_str is None:
                raise Exception("Problem with generating message for assistant!")
            elif type(gen_assistant_msg_str) in [dict]:
                raise Exception(gen_assistant_msg_str)
            elif not len(gen_assistant_msg_str.strip()):
                raise Exception("Problem with generating message for assistant!")

        assistant_msg = Message.objects.create(
            chat=chat,
            role=self.ASSISTANT_ROLE,
            text=gen_assistant_msg_str,
            text_translated=gen_assistant_msg_str_translated,
            number=len(history) + 2,
            state=message_state,
            generation_time=datetime.timedelta(seconds=generation_time),
            prev_message=last_user_message,
        )

        last_user_message.next_message = assistant_msg
        if self._add_to_db:
            last_user_message.save()

        return (
            gen_assistant_msg_str,
            history + [last_user_message, assistant_msg],
            message_state,
            generation_time,
        )

    @staticmethod
    def _extract_question_from_message(message: Message) -> str | None:
        question_str = None
        last_user_prompt = message.text
        if "pytanie: " in last_user_prompt:
            question_str = last_user_prompt.split("pytanie:")[1].strip()
        elif "Pytanie: " in last_user_prompt:
            question_str = last_user_prompt.split("Pytanie:")[1].strip()
        elif "pytanie:" in last_user_prompt.lower():
            question_str = last_user_prompt.lower().split("pytanie:")[1].strip()
        return question_str

    @staticmethod
    def _extract_instruction_from_message(message: Message) -> str | None:
        if message.options is None or not len(message.options):
            return ""
        return message.options.get("query_instruction", "")

    def _prepare_rag_supervisor_state(
        self,
        message: Message,
        history: List[Message],
        collection: CollectionOfDocuments,
        organisation_user: OrganisationUser,
        sse_engin_config_path: str,
        user_msg_is_rag_question: bool = False,
        ignore_question_lang_detect: bool = True,
        last_questions_to_query: int = -1,
        system_prompt: str or None = None,
    ) -> RAGMessageState | None:
        if collection is None:
            return None

        instruction_str = self._extract_instruction_from_message(message)
        options = dict(message.options or {})
        options.setdefault("system_prompt_with_enumerated_documents", True)
        rag_context, previous_rag = self._prepare_rag_memory(
            message, history, collection
        )
        history_only = bool(
            rag_context["history"]
            and rag_context["sources"]
            and previous_rag is not None
            and options["system_prompt_with_enumerated_documents"]
            and (
                options.get("history_only") is True
                or is_history_transformation(message.text)
            )
        )
        rag_context["history_only"] = history_only

        if history_only:
            answer = self.gen_model_controller.generative_answer_for_response(
                user_response=previous_rag.sse_response,
                query_instruction=instruction_str,
                query_options=options,
                system_prompt=system_prompt,
                rag_context=rag_context,
            )
            return RAGMessageState.objects.create(
                contains_query=False,
                contains_instruction=True,
                extracted_query="",
                extracted_instruction=message.text,
                sse_query=previous_rag.sse_query,
                sse_response=previous_rag.sse_response,
                sse_answer=answer,
                system_prompt=system_prompt,
            )

        if user_msg_is_rag_question or options.get(
            "use_intelligent_query_rewrite", True
        ):
            question_str = self._prepare_sse_question_based_on_history(
                user_message=message,
                history=history,
                only_first_user_history_message=False,
                options=options,
                last_questions=last_questions_to_query,
                instruction_str=instruction_str,
                historical_turns=rag_context["history"],
            )
        else:
            question_str = self._extract_question_from_message(message)
            if question_str is None or not len(question_str.strip()):
                return None

        # Add new query to DB
        query_db_results_dict = SearchQueryController.new_query(
            query_str=question_str,
            search_options_dict=message.search_options,
            collection=collection,
            organisation_user=organisation_user,
            sse_engin_config_path=sse_engin_config_path,
            ignore_question_lang_detect=ignore_question_lang_detect,
        )

        user_response = SearchQueryController.get_user_response_by_id(
            query_response_id=query_db_results_dict["query_response_id"],
            organisation_user=organisation_user,
        )

        if "template_prompts" in query_db_results_dict:
            prompts = query_db_results_dict["template_prompts"]
            prompts = [p for p in prompts if p is not None and len(p.strip())]
            if len(prompts):
                random.shuffle(prompts)
                system_prompt = "\n\n".join(
                    prompt for prompt in (prompts[0], system_prompt) if prompt
                )

        # Prepare generative answer
        query_response = self.gen_model_controller.generative_answer_for_response(
            user_response=user_response,
            query_instruction=instruction_str,
            query_options=options,
            system_prompt=system_prompt,
            rag_context=rag_context,
        )

        rag_state = RAGMessageState.objects.create(
            contains_query=True,
            contains_instruction=False,
            extracted_query=question_str,
            extracted_instruction=instruction_str,
            sse_query=user_response.user_query,
            sse_response=user_response,
            sse_answer=query_response,
            system_prompt=system_prompt,
        )

        return rag_state

    @staticmethod
    def _prepare_rag_memory(message, history, collection):
        _, registry, _ = build_memory(history, max_turns=0)
        selected_history, previous_rag = scoped_history(
            history, getattr(collection, "pk", None), message.search_options
        )

        options = message.options or {}
        max_turns = options.get("rag_history_max_turns", 8)
        max_chars = options.get("rag_history_max_chars", 24000)
        max_turns = min(max(max_turns, 0), 50) if type(max_turns) is int else 8
        max_chars = (
            min(max(max_chars, 0), 100000) if type(max_chars) is int else 24000
        )
        turns, registry, sources = build_memory(
            selected_history, max_turns, max_chars, initial_registry=registry
        )
        if sources:
            chunks = list(
                DocumentPageText.objects.filter(
                    id__in=[source.get("text_id") for source in sources],
                    page__document__collection=collection,
                    page__document__use_in_search=True,
                ).select_related("page__document")
            )
            current = {chunk.pk: chunk for chunk in chunks}
            valid = [
                source
                for source in sources
                if source.get("text_id") in current
                and current[source["text_id"]].text_str == source.get("text_str")
            ]
            if len(valid) != len(sources):
                turns = []
                sources = []
            else:
                offsets = chunk_offsets_for_texts(chunks)
                sources = [
                    dict(
                        source,
                        document_id=current[source["text_id"]].page.document_id,
                        document_name=current[source["text_id"]].page.document.name,
                        page_number=current[source["text_id"]].page.page_number,
                        char_start=offsets[source["text_id"]][0],
                        char_end=offsets[source["text_id"]][1],
                    )
                    for source in valid
                ]
        return {
            "history": turns,
            "registry": registry,
            "sources": sources,
            "question": message.text,
            "history_only": False,
        }, previous_rag

    def _prepare_sse_question_based_on_history(
        self,
        user_message: Message,
        history: List[Message],
        only_first_user_history_message: bool,
        options: dict,
        last_questions: int = -1,
        instruction_str: str or None = None,
        historical_turns: list | None = None,
    ) -> str:
        if options.get("use_intelligent_query_rewrite", True):
            return self._rewrite_query_with_llm(
                query=user_message.text,
                history=history,
                options=options,
                historical_turns=historical_turns,
            )

        return user_message.text

    def _rewrite_query_with_llm(
        self,
        query: str,
        history: List[Message],
        options: dict,
        historical_turns: list | None = None,
    ) -> str:
        """
        Rewrite user query using LLM to make it a standalone question.
        """
        if not history:
            return query

        history_turns = []
        turn = None
        for msg in history[-10:]:
            if msg.role == self.USER_ROLE:
                turn = {self.USER_ROLE: msg.text}
            elif msg.role == self.ASSISTANT_ROLE and turn is not None:
                turn[self.ASSISTANT_ROLE] = msg.text
                history_turns.append(turn)
                turn = None
        if historical_turns is not None:
            history_turns = historical_turns
        if not history_turns:
            return query

        model_name = (
            options.get("query_rewrite_model")
            or os.getenv("LLM_ROUTER_QUERY_REWRITE_MODEL", "").strip()
            or options.get("generative_model")
        )
        if not model_name or not history_turns:
            return query
        try:
            rewritten_query = (
                self.gen_model_controller.gen_model_controller.rewrite_query(
                    query=query,
                    history=history_turns,
                    model_name=model_name,
                    lang=options.get("lang", "pl"),
                )
            )
            if isinstance(rewritten_query, str) and rewritten_query.strip():
                return rewritten_query.strip()
        except Exception as e:
            logging.error(f"Error during query rewriting: {e}")

        return query

    def _prepare_content_supervisor_state(
        self, user_message: Message
    ) -> ContentSupervisorState | None:
        supervised_outputs = self.cs_controller.check_text(
            text_str=user_message.text
        )
        if supervised_outputs is None or not len(supervised_outputs):
            return

        state_name = ""
        grabbed_www_contents = None
        for cs_output in supervised_outputs:
            if len(state_name):
                state_name += "|"
            state_name += cs_output.content_type
            if cs_output.content_type == URLRegexProcessor.SELF_PROCESSOR_NAME:
                grabbed_www_contents = cs_output.content_body
        if grabbed_www_contents is not None and len(grabbed_www_contents):
            cs_state = ContentSupervisorState.objects.create(
                state_type=state_name, www_content=grabbed_www_contents
            )
            return cs_state
        return None

    def _prepare_message_state(
        self,
        user_message: Message,
        history: List[Message],
        collection: CollectionOfDocuments,
        organisation_user: OrganisationUser,
        options: Dict,
        sse_engin_config_path,
        last_questions_to_query: int = -1,
        system_prompt: str or None = None,
    ) -> (str | None, MessageState | None):
        rag_state = None
        if options.get("use_rag_supervisor", False):
            rag_state = self._prepare_rag_supervisor_state(
                message=user_message,
                history=history,
                collection=collection,
                organisation_user=organisation_user,
                sse_engin_config_path=sse_engin_config_path,
                user_msg_is_rag_question=options.get(
                    "user_msg_is_rag_question", False
                ),
                ignore_question_lang_detect=True,
                last_questions_to_query=last_questions_to_query,
                system_prompt=system_prompt,
            )

        cs_state = None
        if options.get("use_content_supervisor", False):
            cs_state = self._prepare_content_supervisor_state(
                user_message=user_message
            )

        message_state = None
        if cs_state is not None or rag_state is not None:
            message_state = MessageState.objects.create(
                rag_message_state=rag_state, content_supervisor_state=cs_state
            )

        response_str = ""
        if rag_state is not None and rag_state.sse_answer is not None:
            if rag_state.sse_answer.generated_answer_translated is not None:
                if len(rag_state.sse_answer.generated_answer_translated.strip()):
                    response_str = rag_state.sse_answer.generated_answer_translated
            if response_str is None or not len(response_str.strip()):
                response_str = rag_state.sse_answer.generated_answer
        return response_str, message_state

    def _convert_to_history_turns(
        self, history: List[Message]
    ) -> List[Dict[str, str]]:
        turn = None
        chat_turns = []
        for msg in history:
            if msg.role == self.USER_ROLE:
                user_txt_msg = self.gen_model_controller.handle_message_state(
                    last_user_message=msg.text, message_state=msg.state
                )
                turn = {self.USER_ROLE: user_txt_msg, self.ASSISTANT_ROLE: None}
            elif msg.role == self.ASSISTANT_ROLE:
                turn[self.ASSISTANT_ROLE] = msg.text
                chat_turns.append(turn)
                turn = None
        if turn is not None:
            raise Exception("Last turn is not complete?")
        return chat_turns

    @staticmethod
    def get_chat_messages(chat: Chat) -> List[Message]:
        return list(
            Message.objects.filter(chat=chat)
            .select_related(
                "state__rag_message_state__sse_answer",
                "state__rag_message_state__sse_query",
                "state__rag_message_state__sse_response",
            )
            .order_by("number")
        )
