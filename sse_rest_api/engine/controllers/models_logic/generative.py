import json
import os
import logging
import datetime

from typing import Dict, List

from radlab_data.text.utils import TextUtils

from llm_router_lib.client import LLMRouterClient

from chat.models import MessageState
from engine.models import UserQueryResponse, UserQueryResponseAnswer
from engine.controllers.search.semantic import DBSemanticSearchController


class GenerativeModelConfig:
    """
    Konfigurator modeli generatywnych.

    Odczytuje konfigurację z pliku ``generative-models.json`` w następującym
    formacie:

    {
        "api_hosts": {
            "<model_name>": "<host_url>",
            ...
        },
        "active_api_models": [
            "<model_name>",
            ...
        ]
    }

    Po wczytaniu pliku dostępne są:
    * ``active_local_models_hosts`` – mapowanie nazwy modelu → URL hosta.
    """

    # Nazwy kluczy w pliku JSON
    JSON_API_HOSTS = "api_hosts"
    JSON_ACTIVE_API_MODELS = "active_api_models"

    def __init__(self, config_path: str | None = "configs/generative-models.json"):
        """
        Inicjalizacja konfiguracji.

        Parameters
        ----------
        config_path : str | None
            Ścieżka do pliku JSON z konfiguracją. ``None`` oznacza,
            że konfiguracja nie zostanie wczytana.
        """
        self._config_path = config_path
        self._models_config_json: dict | None = None

        # Mappings exposed via właściwości
        self._active_local_models_hosts: dict = {}

        if self._config_path is not None:
            self.load()

    # ------------------------------------------------------------------
    # Publiczne właściwości
    # ------------------------------------------------------------------
    @property
    def active_local_models_hosts(self) -> dict:
        """
        Zwraca mapowanie aktywnych modeli → adresów hostów.
        """
        return self._active_local_models_hosts

    # ------------------------------------------------------------------
    # Ładowanie i przetwarzanie pliku konfiguracyjnego
    # ------------------------------------------------------------------
    def load(self, config_path: str | None = None) -> None:
        """
        Wczytuje (lub przeładowuje) konfigurację z pliku JSON.

        Parameters
        ----------
        config_path : str | None
            Opcjonalna alternatywna ścieżka do pliku. Jeśli podana,
            zostanie użyta zamiast ścieżki podanej przy konstrukcji.
        """
        if config_path is not None:
            self._config_path = config_path

        with open(self._config_path, "rt", encoding="utf-8") as f:
            self._models_config_json = json.load(f)

        self._process_config_file()

    def _process_config_file(self) -> None:
        """
        Buduje wewnętrzne słowniki ``_active_local_models_hosts``
        na podstawie wczytanej konfiguracji.
        """
        self._active_local_models_hosts.clear()

        all_api_hosts: dict = self._models_config_json[self.JSON_API_HOSTS]
        active_models: list = self._models_config_json[self.JSON_ACTIVE_API_MODELS]

        for model_name in active_models:
            host_url = all_api_hosts.get(model_name)
            if host_url:
                if host_url.endswith("/"):
                    host_url = host_url[:-1]
                self._active_local_models_hosts[model_name] = host_url
            else:
                logging.warning(
                    f"Model '{model_name}' is active but no definition in "
                    f"'{self.JSON_API_HOSTS}'."
                )


class GenerativeModelControllerApi:
    """
    API client for custom local generative models.
    """

    class LocalModelAPI:
        """
        Helper that builds URLs and request templates for a specific local model.
        """

        @staticmethod
        def get_request_data_template(generation_options: dict | None) -> dict:
            """
            Create a baseline request payload for the local API.

            Parameters
            ----------
            generation_options : dict | None
                Optional generation parameters that will overwrite defaults.

            Returns
            -------
            dict
                JSON‑serialisable request body.
            """
            request_data = {
                "question_str": "",
                "question_prompt": "",
                "texts": {},
                "model_name": "",
                "proper_input": True,
                "post_proc_output": False,
                "top_k": 50,
                "top_p": 0.99,
                "temperature": 0.7,
                "typical_p": 1,
                "repetition_penalty": 1.2,
            }
            if generation_options is not None:
                for go, val in generation_options.items():
                    request_data[go] = val

            return request_data

    def __init__(self, deepl_api_key: str):
        """
        Initialise the façade with a DeepL API key.

        Parameters
        ----------
        deepl_api_key : str
            Authentication token for DeepL translation service.
        """
        self.deepl_api_key = deepl_api_key
        self.models_config = GenerativeModelConfig(
            config_path="configs/generative-models.json"
        )

    def generative_answer_local_api_model(
        self,
        question_str: str,
        search_results: dict,
        qa_gen_model: str,
        question_prompt: str = "",
        which_docs: list | None = None,
        use_doc_names_in_response: bool = False,
        generation_options: dict | None = None,
        system_prompt: str or None = None,
    ) -> (str | None, float | None):
        """
        Request a generative answer from a locally hosted model.

        Parameters
        ----------
        question_str : str
            The user’s question.
        search_results : dict
            Search results to be transformed into context.
        qa_gen_model : str
            Model name as defined in the configuration.
        question_prompt : str
            Optional prompt that will be appended to the question.
        which_docs : list | None
            Sub‑set of documents to include.
        use_doc_names_in_response : bool
            If ``True``, document names are added to the generated answer.
        generation_options : dict | None
            Model‑specific generation parameters.
        system_prompt : str | None
            Optional system prompt that overrides the default.

        Returns
        -------
        tuple
            ``(generated_answer, generation_time)`` where
            ``generated_answer`` is a string or ``None`` on failure,
            and ``generation_time`` is the elapsed time in seconds.
        """

        request_data = self.LocalModelAPI.get_request_data_template(
            generation_options
        )
        request_data["question_str"] = question_str
        request_data["question_prompt"] = question_prompt
        request_data["texts"] = (
            DBSemanticSearchController.convert_search_results_to_doc2answer(
                search_results=search_results,
                which_docs=which_docs,
                use_doc_names_in_response=use_doc_names_in_response,
            )
        )
        request_data["model_name"] = qa_gen_model

        if not len(request_data["texts"]):
            return None, None

        if system_prompt is not None and len(system_prompt.strip()):
            request_data["system_prompt"] = system_prompt

        _r_client = LLMRouterClient(
            api=self.models_config.active_local_models_hosts[qa_gen_model],
            timeout=120,
        )
        generated_answer = _r_client.generative_answer(payload=request_data)

        generation_time = generated_answer["generation_time"]
        if "response" not in generated_answer:
            logging.error(generated_answer)
            return generated_answer, generation_time

        generated_answer = generated_answer["response"]
        if isinstance(generated_answer, dict):
            logging.error(generated_answer)
            return generated_answer, generation_time

        return generated_answer, generation_time

    def conversation_with_local_model(
        self,
        history: List[Dict[str, str]],
        last_user_message: str,
        options: Dict[str, str],
        model_name_path: str,
    ) -> (str | None, float | None):
        """
        Conduct a multi‑turn conversation with a local generative model.

        Parameters
        ----------
        history : List[Dict[str, str]]
            List of prior ``{\"role\": ..., \"content\": ...}`` messages.
        last_user_message : str
            The most recent user utterance.
        options : Dict[str, str]
            Generation options that will be merged into the request payload.
        model_name_path : str
            Name of the model to query.

        Returns
        -------
        tuple
            ``(assistant_message, generation_time)`` where ``assistant_message``
            is the model’s reply (or ``None`` on error) and ``generation_time`` is
            the elapsed time in seconds.
        """
        request_data = self.LocalModelAPI.get_request_data_template(options)

        request_data["user_last_statement"] = last_user_message
        request_data["historical_messages"] = history
        request_data["model_name"] = model_name_path

        _r_client = LLMRouterClient(
            api=self.models_config.active_local_models_hosts[model_name_path],
            timeout=120,
        )
        chat_assistant_response = _r_client.conversation_with_model(
            payload=request_data
        )

        if "response" not in chat_assistant_response:
            logging.error(chat_assistant_response)
            return chat_assistant_response, None

        generation_time = chat_assistant_response["generation_time"]
        assistant_message = chat_assistant_response["response"].strip()
        return assistant_message, generation_time


class GenerativeModelController:
    """
    High‑level orchestrator for generative QA workflows.
    """

    available_generation_options = [
        "top_k",
        "top_p",
        "temperature",
        "typical_p",
        "repetition_penalty",
        "max_new_tokens",
    ]

    def __init__(self, store_to_db: bool = True):
        """
        Initialise sub‑controllers and environment variables.

        Parameters
        ----------
        store_to_db : bool
            If ``True`` the generated answers are persisted via Django ORM.
        """
        self.store_to_db = store_to_db
        self.deepl_api_key = os.environ.get("DEEPL_AUTH_KEY", None)

        self.gen_model_controller = GenerativeModelControllerApi(
            deepl_api_key=self.deepl_api_key
        )

    @staticmethod
    def get_user_query_response_answer(
        user_query_response_id: int,
    ) -> UserQueryResponseAnswer | None:
        """
        Retrieve a stored ``UserQueryResponseAnswer`` by its primary key.

        Parameters
        ----------
        user_query_response_id : int
            Database identifier.

        Returns
        -------
        UserQueryResponseAnswer | None
            The answer instance, or ``None`` if it does not exist.
        """
        try:
            return UserQueryResponseAnswer.objects.get(id=user_query_response_id)
        except UserQueryResponseAnswer.DoesNotExist:
            return None

    def _prepare_generation_options(self, query_options: dict) -> Dict[str, str]:
        """
        Filter user‑provided options to those supported by the model.

        Parameters
        ----------
        query_options : dict
            Dictionary supplied by the client.

        Returns
        -------
        dict
            Sub‑dictionary containing only recognised generation parameters.
        """
        g_opt = {}
        for opt in self.available_generation_options:
            if opt in query_options:
                g_opt[opt] = query_options[opt]
        return g_opt

    def model_response_cs_rag(
        self,
        query_options: dict,
        last_user_message: str,
        history: List[Dict[str, str]],
        message_state: MessageState | None,
    ) -> (str | None, str | None, str | None):
        """
        Generate a conversational answer using a local model.

        Parameters
        ----------
        query_options : dict
            Options controlling generation, model choice, translation, etc.
        last_user_message : str
            The most recent user utterance.
        history : List[Dict[str, str]]
            Prior dialogue turns.
        message_state : MessageState | None
            Optional state that can modify the incoming user message.

        Returns
        -------
        tuple
            ``(answer, translated_answer, generation_time)`` where any element
            may be ``None`` on failure.
        """
        generation_options = self._prepare_generation_options(query_options)

        last_user_message = self.handle_message_state(
            last_user_message=last_user_message, message_state=message_state
        )

        ai_message_str, generation_time = (
            self.gen_model_controller.conversation_with_local_model(
                history=history,
                last_user_message=last_user_message,
                options=generation_options,
                model_name_path=query_options["generative_model"],
            )
        )
        if ai_message_str is None:
            logging.error(
                "Problem while generating the assistant message "
                "[model_response_cs_rag!"
            )
            return None, None, None

        ai_message_str_translated = None
        if query_options.get("translate_answer", False):
            if self.deepl_api_key:
                ai_message_str_translated = TextUtils.translate_text_deepl(
                    text_str=ai_message_str,
                    target_lang=query_options["answer_language"],
                    auth_key=self.deepl_api_key,
                )
            else:
                logging.error("DEEPL_AUTH_KEY is not defined!")
        return ai_message_str, ai_message_str_translated, generation_time

    def handle_message_state(
        self, last_user_message: str, message_state: MessageState | None
    ) -> str:
        """
        Apply optional ``MessageState`` transformations to the user message.

        Parameters
        ----------
        last_user_message : str
            Original message.
        message_state : MessageState | None
            Optional state object.

        Returns
        -------
        str
            Possibly modified message.
        """
        if message_state is None:
            return last_user_message
        last_user_message = self._handle_message_state_content_supervisor(
            last_user_message=last_user_message, message_state=message_state
        )
        return last_user_message

    @staticmethod
    def _handle_message_state_content_supervisor(
        last_user_message: str, message_state: MessageState | None
    ) -> str:
        """
        Append ``www_content`` from a ``ContentSupervisorState`` to the message.

        Parameters
        ----------
        last_user_message : str
            Current message text.
        message_state : MessageState | None
            May contain a ``content_supervisor_state`` with ``www_content``.

        Returns
        -------
        str
            Message with any additional content appended.
        """
        if message_state is None:
            return last_user_message
        cs_state = message_state.content_supervisor_state
        if cs_state is None:
            return last_user_message

        if cs_state.www_content and len(cs_state.www_content):
            add_content = ""
            for content_str in cs_state.www_content.values():
                add_content += content_str + "\n"
            last_user_message += f"\n\n{add_content.strip()}"

        return last_user_message

    def generative_answer_for_response(
        self,
        user_response: UserQueryResponse,
        query_instruction: str,
        query_options: dict,
        system_prompt: str or None = None,
    ) -> UserQueryResponseAnswer | None:
        """
        Generate a (possibly translated) answer for a stored ``UserQueryResponse``.

        Parameters
        ----------
        user_response : UserQueryResponse
            The persisted user query object.
        query_instruction : str
            Additional instruction that guides the generation.
        query_options : dict
            Dictionary with model choice, generation settings, translation flags, etc.
        system_prompt : str | None
            Optional system prompt for the underlying model.

        Returns
        -------
        UserQueryResponseAnswer | None
            The newly created answer record, or ``None`` on failure.
        """
        generated_answer = []
        query_response_answer = UserQueryResponseAnswer.objects.create(
            user_response=user_response,
            is_generative=True,
            answer_options=query_options,
            query_instruction_prompt=query_instruction,
            generated_answer=json.dumps(generated_answer),
        )

        generation_options = self._prepare_generation_options(query_options)

        generated_answer, generation_time = (
            self.generative_answer_for_response_from_api(
                user_response=user_response,
                generative_model=query_options["generative_model"],
                query_instruction=query_instruction,
                percentage_rank_mass=query_options["percentage_rank_mass"],
                use_doc_names_in_response=query_options["use_doc_names_in_response"],
                generation_options=generation_options,
                system_prompt=system_prompt,
            )
        )

        if generated_answer is None:
            return None

        query_response_answer.generation_time = datetime.timedelta(
            seconds=generation_time
        )
        query_response_answer.generated_answer = generated_answer
        if self.store_to_db:
            query_response_answer.save()

        if query_options.get("translate_answer", False):
            if self.deepl_api_key:
                target_lang = query_options["answer_language"]
                generated_answer_translated = TextUtils.translate_text_deepl(
                    text_str=generated_answer,
                    target_lang=target_lang,
                    auth_key=self.deepl_api_key,
                )
                query_response_answer.generated_answer_translated = (
                    generated_answer_translated
                )
                if self.store_to_db:
                    query_response_answer.save()
            else:
                logging.error("DEEPL_AUTH_KEY is not defined!")

        return query_response_answer

    def generative_answer_for_response_from_api(
        self,
        user_response: UserQueryResponse,
        generative_model: str,
        query_instruction: str,
        percentage_rank_mass: int,
        use_doc_names_in_response: bool = False,
        generation_options: dict | None = None,
        dont_response_when_no_documents: bool = True,
        system_prompt: str | None = None,
    ) -> (str | None, float | None):
        """
        Generate an answer using a locally hosted model via HTTP API.

        Parameters
        ----------
        user_response : UserQueryResponse
            The persisted query object.
        generative_model : str
            Name of the local model.
        query_instruction : str
            Prompt that guides generation.
        percentage_rank_mass : int
            Percent of top‑ranked documents to include.
        use_doc_names_in_response : bool
            Whether to prepend document names to each generated fragment.
        generation_options : dict | None
            Model‑specific generation parameters.
        dont_response_when_no_documents : bool
            If ``True`` and no documents are selected, a short explanatory
            string is returned instead of calling the model.
        system_prompt : str | None
            Optional system prompt.

        Returns
        -------
        tuple
            ``(generated_answer, generation_time)`` where ``generated_answer`` is a
            string (or ``None`` on error) and ``generation_time`` is the elapsed
            time in seconds.
        """
        which_docs = DBSemanticSearchController.get_accumulated_docs_by_rank_perc(
            results={"stats": user_response.general_stats_json},
            perc_rank_gen_qa=percentage_rank_mass,
        )
        logging.info(f"Number of documents to generate response: {len(which_docs)}")
        logging.info(f"generative model to generate answer: {generative_model}")

        if not len(which_docs) and dont_response_when_no_documents:
            return "Brak treści spełniających parametry wyszukiwania", 0.0

        generative_answer_str, generation_time = (
            self.gen_model_controller.generative_answer_local_api_model(
                question_str=user_response.user_query.query_str_prompt,
                search_results=user_response.detailed_results_json,
                qa_gen_model=generative_model,
                question_prompt=query_instruction,
                which_docs=which_docs,
                use_doc_names_in_response=use_doc_names_in_response,
                generation_options=generation_options,
                system_prompt=system_prompt,
            )
        )

        return generative_answer_str, generation_time
