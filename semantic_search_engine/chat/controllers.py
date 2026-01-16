import string
import random
import logging
import datetime
from typing import List, Dict

from django.db.models import QuerySet

from content_supervisor.supervisor import ContentSupervisor
from content_supervisor.processors.regex_processors import URLRegexProcessor

from chat.models import (
    Chat,
    Message,
    MessageState,
    ContentSupervisorState,
    RAGMessageState,
)
from data.models import OrganisationUser, CollectionOfDocuments

# from engine.models import UserQueryResponseAnswer
from engine.controllers.search.relational import SearchQueryController
from engine.controllers.models_logic.generative import GenerativeModelController


class ChatController:
    USER_ROLE = "user"
    SYSTEM_ROLE = "system"
    ASSISTANT_ROLE = "assistant"
    AVAILABLE_SYSTEM_ROLES = [SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE]
    REMOVE_LAST_QUESTION_CHAR = [".", ",", "?", "!", ";", "/", "-"]
    USER_QUERY_PROMPT_HISTORY = (
        "\n\nTrzymając kontekst wcześniejszych pytań, odpowiedz na pytanie: "
    )

    def __init__(self, add_to_db: bool = True, new_chat_hash_length: int = 128):
        self._add_to_db = add_to_db
        self.cs_controller = ContentSupervisor(options={"use_executor": True})
        self.gen_model_controller = GenerativeModelController(store_to_db=True)

        self._new_chat_hash_length = new_chat_hash_length

    def new_chat(
        self,
        organisation_user: OrganisationUser,
        collection: CollectionOfDocuments | None,
        options: dict | None,
        search_options: dict | None = None,
    ) -> Chat:
        """
        Create a new chat.
        :param organisation_user: Chat user
        :param collection: Collection of documents to conversation
        :param options: Dictionary with chat options.
        :param search_options: Dictionary with search options.
        :return: Created chat object
        """
        chat_hash = self._generate_chat_hash()
        logging.info(f"Created new chat with hash: {chat_hash}")

        new_chat = Chat(
            organisation_user=organisation_user,
            collection=collection,
            options=options,
            search_options=search_options,
            hash=chat_hash,
        )
        if self._add_to_db:
            new_chat.save()
        return new_chat

    def set_chat_as_saved(self, chat: Chat, read_only: bool) -> str:
        """
        Return hash of saved chat
        :param chat:
        :param read_only:
        :return:
        """
        chat.is_saved = True
        chat.read_only = read_only
        if self._add_to_db:
            chat.save()
        return chat.hash

    @staticmethod
    def get_list_of_user_chats(user: OrganisationUser) -> QuerySet[Chat]:
        return Chat.objects.filter(organisation_user=user)

    @staticmethod
    def get_chat_by_chat_hash(
        chat_hash: str, only_saved: bool = True
    ) -> Chat | None:
        try:
            return Chat.objects.get(is_saved=only_saved, hash=chat_hash)
        except Chat.DoesNotExist:
            return None

    def _generate_chat_hash(self) -> str:
        chat_hash = "".join(
            random.SystemRandom().choice(
                string.ascii_lowercase + string.ascii_uppercase + string.digits
            )
            for _ in range(self._new_chat_hash_length)
        )
        return chat_hash

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
            state=None,
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
        """
        If question state exists into message:
          - prepare new db query
          - run query search into relational/semantic db
          - generate answer based on the query result

        :param message:
        :param collection:
        :param organisation_user:
        :return:
        """
        if collection is None:
            return None

        instruction_str = self._extract_instruction_from_message(message)

        if user_msg_is_rag_question:
            question_str = self._prepare_sse_question_based_on_history(
                user_message=message,
                history=history,
                only_first_user_history_message=False,
                last_questions=last_questions_to_query,
                instruction_str=instruction_str,
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
            query_response_id=query_db_results_dict["query_response_id"]
        )
        # if user_msg_is_rag_question:
        #     if len(history):
        #         user_response.user_query.query_str_prompt += (
        #             self.USER_QUERY_PROMPT_HISTORY + message.text
        #         )
        #
        #     question_str = user_response.user_query.query_str_prompt

        if "template_prompts" in query_db_results_dict:
            prompts = query_db_results_dict["template_prompts"]
            prompts = [p for p in prompts if p is not None and len(p.strip())]
            if len(prompts):
                random.shuffle(prompts)
                system_prompt = prompts[0]

        # Prepare generative answer
        query_response = self.gen_model_controller.generative_answer_for_response(
            user_response=user_response,
            query_instruction=instruction_str,
            query_options=message.options,
            system_prompt=system_prompt,
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

    def _prepare_sse_question_based_on_history(
        self,
        user_message: Message,
        history: List[Message],
        only_first_user_history_message: bool,
        last_questions: int = -1,
        instruction_str: str or None = None,
    ) -> str:
        if last_questions == 0:
            history = []
        elif last_questions > 0:
            # *2 because history contains user and assisntant messages
            history = history[-last_questions * 2 :]

        user_question = ""
        for msg in history:
            if msg.role == self.USER_ROLE:
                if msg.text in user_question:
                    continue

                user_question += msg.text + "\n"
                if only_first_user_history_message:
                    # user_question = user_question.strip()
                    # for rc in self.REMOVE_LAST_QUESTION_CHAR:
                    #     user_question = user_question.rstrip(rc)
                    # if len(user_question):
                    #     user_question += " oraz "
                    break

        if not len(user_question.strip()):
            return user_message.text

        user_question = "Użytkownik wcześniej pytał o:\n" + user_question + "\n\n"
        user_question += "Odpowiedz teraz na pytanie: " + user_message.text

        # If query-instruction is given, then add it to user prompt
        if instruction_str is not None:
            instruction_str = instruction_str.strip()
            if len(instruction_str):
                user_question += (
                    "\n\n" + "Udzielając odpowiedzi zastosuj się do zasad:"
                )
                user_question += "\n" + instruction_str

        return user_question.strip()

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
        """
        Check if message contains any state. Any question, instruction etc.
        There classifier for "importance" should be used!
        :param user_message: User message to check state
        :return: Pair of generated response (or None) and message state (or None)
        """
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
        return [m for m in Message.objects.filter(chat=chat).order_by("number")]

    @staticmethod
    def get_chat_by_id(chat_id) -> Chat | None:
        try:
            return Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return None
