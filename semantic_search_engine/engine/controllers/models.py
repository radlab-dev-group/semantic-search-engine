"""
Models module
--------------

Provides a set of controllers and utilities for working with generative
language models, extractive question‑answering, and configuration handling.
The module includes:

* ``GenerativeModelConfig`` – loads and parses a JSON configuration that
  defines active OpenAI and locally‑hosted models.
* ``ExtractiveQAController`` – runs a HuggingFace “question‑answering” pipeline
  on retrieved document passages.
* ``OpenAIGenerativeController`` – builds prompts and calls the OpenAI chat API.
* ``GenerativeModelControllerApi`` – thin wrapper around a custom local generative
  model HTTP API.
* ``GenerativeModelController`` – high‑level façade that coordinates the above
  components, optionally translating answers via DeepL and persisting results.
"""

import datetime
import json
import logging
import os
from typing import Dict, List

import requests
from openai import OpenAI
from transformers import pipeline

from chat.models import MessageState
from engine.controllers.search import DBSemanticSearchController
from engine.models import UserQueryResponse, UserQueryResponseAnswer
from radlab_data.text.utils import TextUtils

ALL_AVAILABLE_GENAI_MODELS_NAMES = []


class GenerativeModelConfig:
    """
    Controller for generative model configurations
    """

    JSON_API_HOSTS = "api_hosts"
    JSON_API_EP_LIST = "ep"
    JSON_API_HOSTS_LIST = "hosts"
    JSON_ACTIVE_API_MODELS = "active_api_models"
    JSON_API_HOSTS_OPENAI = "openai_models_api"
    JSON_API_HOSTS_LOCAL_MODELS = "local_models_api"

    def __init__(self, config_path: str | None = "configs/generative-models.json"):
        """
        Initialise the configuration loader.

        Parameters
        ----------
        config_path : str | None
            Path to the JSON file containing generative‑model configuration.
            If ``None`` the configuration is not loaded.
        """
        self._config_path = config_path

        self._models_config_json = None
        self._active_local_models_hosts = {}
        self._local_models_endpoints = {}
        self._active_openai_hosts = {}

        if self._config_path is not None:
            self.load()

    @property
    def active_openai_hosts(self) -> dict:
        """
        Returns the active OpenAI models.

        Returns
        -------
        dict
            Mapping where the key is the model name and the value is the
            OpenAI model identifier.
        """
        return self._active_openai_hosts

    @property
    def active_local_models_hosts(self) -> dict:
        """
        Returns the active locally‑provided models.

        Returns
        -------
        dict
            Mapping where the key is the model name and the value is the model host URL.
        """
        return self._active_local_models_hosts

    @property
    def local_models_endpoints(self) -> dict:
        """
        Returns the local models API endpoints.

        Returns
        -------
        dict
            Mapping of endpoint names to their URLs.
        """
        return self._local_models_endpoints

    def load(self, config_path: str | None = None) -> None:
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        config_path : str | None
            Optional alternative path. If supplied, it overrides the instance’s
            stored path.
        """
        if config_path is not None:
            self._config_path = config_path

        with open(self._config_path, "rt") as models_in:
            self._models_config_json = json.load(models_in)

        self._process_config_file()

    def _process_config_file(self) -> None:
        """
        Populate internal dictionaries with active OpenAI and local models.
        """
        self._active_local_models_hosts.clear()
        self._active_openai_hosts.clear()

        all_api_hosts = self._models_config_json[self.JSON_API_HOSTS]
        active_api_models = self._models_config_json[self.JSON_ACTIVE_API_MODELS]
        local_api_models = all_api_hosts[self.JSON_API_HOSTS_LOCAL_MODELS][
            self.JSON_API_HOSTS_LIST
        ]
        openai_api_models = all_api_hosts[self.JSON_API_HOSTS_OPENAI][
            self.JSON_API_HOSTS_LIST
        ]

        self._local_models_endpoints = all_api_hosts[
            self.JSON_API_HOSTS_LOCAL_MODELS
        ][self.JSON_API_EP_LIST]

        for active_model in active_api_models:
            if active_model not in ALL_AVAILABLE_GENAI_MODELS_NAMES:
                ALL_AVAILABLE_GENAI_MODELS_NAMES.append(active_model)

            if active_model in local_api_models:
                self._active_local_models_hosts[active_model] = local_api_models[
                    active_model
                ]
            elif active_model in openai_api_models:
                self._active_openai_hosts[active_model] = openai_api_models[
                    active_model
                ]


class ExtractiveQAController:
    """
    Controller that runs extractive question‑answering using a HuggingFace pipeline.
    """

    model_path = "radlab/polish-qa-v2"

    def __init__(
        self, model_path: str | None, qa_pipeline=None, device: str = "cpu"
    ):
        """
        Initialise the controller.

        Parameters
        ----------
        model_path : str | None
            Path or identifier of the model to load if ``qa_pipeline`` is not
            provided.
        qa_pipeline : object | None
            An already‑initialised ``pipeline`` object. If supplied, ``model_path``
            is ignored.
        device : str
            Device to run the model on (e.g., ``"cpu"`` or ``"cuda"``).
        """
        assert model_path is not None or qa_pipeline is not None

        self.device = device
        if qa_pipeline is not None:
            self.question_answerer = qa_pipeline
        else:
            self.question_answerer = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load a HuggingFace “question‑answering” pipeline.

        Parameters
        ----------
        model_path : str
            Model identifier or path.

        Returns
        -------
        pipeline
            A ready‑to‑use question‑answering pipeline.
        """
        return pipeline("question-answering", model=model_path, device=self.device)

    def run_extractive_qa(self, question_str: str, search_results: dict):
        """
        Perform extractive QA on a set of retrieved passages.

        Parameters
        ----------
        question_str : str
            The user’s question.
        search_results : dict
            Dictionary produced by the semantic‑search controller. Must contain a
            ``"results"`` list where each entry holds a ``"result"`` dict with
            ``"text"`` sub‑fields.

        Returns
        -------
        dict
            Mapping of document name to a dict containing page number, text
            number, the extracted answer and its confidence score.
        """
        document_answers = {}
        for answer in search_results["results"]:
            answer_str = answer["result"]["text"]["text_str"]
            document_name = answer["result"]["text"]["document_name"]
            page_number = answer["result"]["text"]["page_number"]
            text_number = answer["result"]["text"]["text_number"]

            q_res = self.question_answerer(question=question_str, context=answer_str)
            document_answers[document_name] = {
                "page_number": page_number,
                "text_number": text_number,
                "answer": q_res["answer"],
                "score": q_res["score"],
            }
        return document_answers


class OpenAIGenerativeController:
    """
    Wrapper around the OpenAI chat API for generative summarisation.
    """

    models_config = GenerativeModelConfig(
        config_path="configs/generative-models.json"
    )

    def __init__(self, openai_api_key: str) -> None:
        """
        Initialise the OpenAI client.

        Parameters
        ----------
        openai_api_key : str
            API key for authenticating with the OpenAI service.
        """
        self.openai_api_key = openai_api_key
        self.client = None
        if openai_api_key is not None and len(openai_api_key.strip()) > 5:
            self.client = OpenAI(api_key=self.openai_api_key)

    def generative_summarization_openai(
        self,
        question_str,
        query_instruction,
        search_results,
        qa_gen_model,
        which_docs,
        generation_options,
        system_prompt: str or None = None,
    ):
        """
        Generate a summary using an OpenAI model.

        Parameters
        ----------
        question_str : str
            The original user question.
        query_instruction : str
            Additional instruction for the model.
        search_results : dict
            Search results to be turned into context.
        qa_gen_model : str
            Name of the active OpenAI model to use.
        which_docs : list
            List of document identifiers to include in the prompt.
        generation_options : dict
            Generation parameters (currently unused).
        system_prompt : str | None
            Optional system‑level prompt that overrides the default.

        Returns
        -------
        list[str]
            List of generated answer strings (one per ``choice``).
        """
        doc2answers = (
            DBSemanticSearchController.convert_search_results_to_doc2answer(
                search_results=search_results, which_docs=which_docs
            )
        )

        input_prompt = self.prepare_input_message_for_openai(
            question_str, query_instruction, doc2answers, system_prompt=system_prompt
        )

        response = self.client.chat.completions.create(
            model=self.models_config.active_openai_hosts[qa_gen_model],
            messages=input_prompt,
        )

        answers = []
        for choice in response.choices:
            answers.append(choice.message.content)

        return answers

    def prepare_input_message_for_openai(
        self,
        question_str: str,
        query_instruction: str,
        doc2answers: dict,
        system_prompt: str or None = None,
    ) -> list:
        """
        Build the list of messages required by the OpenAI chat endpoint.

        Parameters
        ----------
        question_str : str
            The user’s question.
        query_instruction : str
            Optional extra instruction for the model.
        doc2answers : dict
            Mapping of document name to a list of extracted answer strings.
        system_prompt : str | None
            Custom system prompt; if ``None`` the default prompt is used.

        Returns
        -------
        list[dict]
            Ordered list of ``{\"role\": ..., \"content\": ...}`` messages.
        """
        system_prompt = self.__get_openai_system_prompt(system_prompt)

        messages = [{"role": "system", "content": system_prompt}]
        actual_role = "user"
        for doc_name, answers in doc2answers.items():
            for answer in answers:
                message = {
                    "role": actual_role,
                    "content": f"{doc_name}: {answer}",
                }
                messages.append(message)
                actual_role = "system" if actual_role == "user" else "user"

        question_message = {
            "role": actual_role,
            "content": f"Pytanie: {question_str}",
        }
        if query_instruction is not None and len(query_instruction.strip()):
            question_message["content"] += "\n" + query_instruction.strip()
        messages.append(question_message)
        return messages

    @staticmethod
    def __get_openai_system_prompt(system_prompt):
        """
        Return a system prompt for the OpenAI model.

        Parameters
        ----------
        system_prompt : str | None
            Custom prompt supplied by the caller.

        Returns
        -------
        str
            The system prompt to be sent to the API.
        """
        if system_prompt is not None and len(system_prompt.strip()):
            return system_prompt

        system_prompt = """
            You are QA assistance, to prepare your answer use only 
            knowledge from the given texts. Your response have to 
            be created with the same style as in original texts. 
            Return only response for question without your comment. 
            The texts to prepare answer for question will be given in format 
            like: doc_name: <context>. 
            doc_name is the name of document where <context> comes from. 
            Use only knowledge from <context>. And generating response inform 
            about document names from the response comes from. 
            The language of response have to be in polish. 
            The response length have to be about 1000k tokens. 
            The question for the context will be given at the end. 
            """

        return system_prompt


class GenerativeModelControllerApi:
    """
    API client for custom local generative models.
    """

    class LocalModelAPI:
        """
        Helper that builds URLs and request templates for a specific local model.
        """

        JSON_FIELD_EP_GENERATIVE_ANSWER = "generative_answer"
        JSON_FIELD_EP_CONVERSATION_WITH_MODEL = "conversation_with_model"

        def __init__(self, qa_gen_model: str, models_config: GenerativeModelConfig):
            """
            Initialise URLs for the chosen local model.

            Parameters
            ----------
            qa_gen_model : str
                Name of the model as defined in the configuration.
            models_config : GenerativeModelConfig
                Loaded configuration providing host and endpoint data.
            """
            api_host = models_config.active_local_models_hosts[qa_gen_model]
            if api_host.endswith("/"):
                api_host = api_host[:-1]

            ep_url_generative_answer = models_config.local_models_endpoints[
                self.JSON_FIELD_EP_GENERATIVE_ANSWER
            ]
            if ep_url_generative_answer.startswith("/"):
                ep_url_generative_answer = ep_url_generative_answer[1:]
            self.api_generative_answer_ep_url = (
                f"{api_host}/{ep_url_generative_answer}"
            )

            ep_url_conv_with_model = models_config.local_models_endpoints[
                self.JSON_FIELD_EP_CONVERSATION_WITH_MODEL
            ]
            if ep_url_conv_with_model.startswith("/"):
                ep_url_conv_with_model = ep_url_conv_with_model[1:]

            self.api_conversation_with_model_ep_url = (
                f"{api_host}/{ep_url_conv_with_model}"
            )

            self.api_header = {"Content-Type": "application/json; charset=utf-8"}

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

    models_config = GenerativeModelConfig(
        config_path="configs/generative-models.json"
    )

    def __init__(self, deepl_api_key: str):
        """
        Initialise the façade with a DeepL API key.

        Parameters
        ----------
        deepl_api_key : str
            Authentication token for DeepL translation service.
        """
        self.deepl_api_key = deepl_api_key

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
        local_model_api = self.LocalModelAPI(qa_gen_model, self.models_config)

        request_data = local_model_api.get_request_data_template(generation_options)
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

        generated_answer = requests.post(
            local_model_api.api_generative_answer_ep_url,
            headers=local_model_api.api_header,
            json=request_data,
        )
        if not generated_answer.ok:
            logging.error(generated_answer)
            return None, None

        generated_answer = generated_answer.json()
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
        local_model_api = self.LocalModelAPI(model_name_path, self.models_config)
        request_data = local_model_api.get_request_data_template(options)
        request_data["user_last_statement"] = last_user_message
        request_data["historical_messages"] = history
        request_data["model_name"] = model_name_path
        chat_assistant_response = requests.post(
            local_model_api.api_conversation_with_model_ep_url,
            headers=local_model_api.api_header,
            json=request_data,
        )

        if not chat_assistant_response.ok:
            logging.error(chat_assistant_response)
            return chat_assistant_response, None

        chat_assistant_response = chat_assistant_response.json()
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
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", None)

        self.gen_model_controller = GenerativeModelControllerApi(
            deepl_api_key=self.deepl_api_key
        )
        self.openai_controller = OpenAIGenerativeController(
            openai_api_key=self.openai_api_key
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

        if "openai" in query_options["generative_model"].lower():
            raise Exception(
                "Conversational search with OpenAI is not implemented yet"
            )

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
        if "openai" in query_options["generative_model"].lower():
            generated_answer, generation_time = (
                self.generative_answer_for_response_openai(
                    user_response=user_response,
                    generative_model=query_options["generative_model"],
                    query_instruction=query_instruction,
                    percentage_rank_mass=query_options["percentage_rank_mass"],
                    generation_options=generation_options,
                    system_prompt=system_prompt,
                )
            )
        else:
            generated_answer, generation_time = (
                self.generative_answer_for_response_from_api(
                    user_response=user_response,
                    generative_model=query_options["generative_model"],
                    query_instruction=query_instruction,
                    percentage_rank_mass=query_options["percentage_rank_mass"],
                    use_doc_names_in_response=query_options[
                        "use_doc_names_in_response"
                    ],
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

    def generative_answer_for_response_openai(
        self,
        user_response: UserQueryResponse,
        generative_model: str,
        query_instruction: str,
        percentage_rank_mass: int,
        generation_options: dict = None,
        system_prompt: str or None = None,
    ) -> (str | None, float | None):
        """
        Generate an answer using an OpenAI model.

        Parameters
        ----------
        user_response : UserQueryResponse
            The persisted query with search results.
        generative_model : str
            Name of the OpenAI model to use.
        query_instruction : str
            Additional instruction for the model.
        percentage_rank_mass : int
            Percent of top‑ranked documents to include.
        generation_options : dict | None
            Optional generation settings (currently unused).
        system_prompt : str | None
            Optional system prompt that overrides the default.

        Returns
        -------
        tuple
            ``(generated_answer, generation_time)`` where ``generated_answer`` is a
            list of strings (or ``None`` on error) and ``generation_time`` is the
            elapsed time in seconds.
        """
        if self.openai_api_key is None or len(self.openai_api_key) < 10:
            logging.error("OPENAI_API_KEY is not defined!")
            return None

        which_docs = DBSemanticSearchController.get_accumulated_docs_by_rank_perc(
            results={"stats": user_response.general_stats_json},
            perc_rank_gen_qa=percentage_rank_mass,
        )
        logging.info(f"Number of documents to generate response: {len(which_docs)}")

        start = datetime.datetime.now()
        generated_answer = self.openai_controller.generative_summarization_openai(
            question_str=user_response.user_query.query_str_prompt,
            query_instruction=query_instruction,
            search_results=user_response.detailed_results_json,
            qa_gen_model=generative_model,
            which_docs=which_docs,
            generation_options=generation_options,
            system_prompt=system_prompt,
        )
        generation_time = datetime.datetime.now() - start
        return generated_answer, generation_time.total_seconds()

    def generative_answer_for_response_from_api(
        self,
        user_response: UserQueryResponse,
        generative_model: str,
        query_instruction: str,
        percentage_rank_mass: int,
        use_doc_names_in_response: bool = False,
        generation_options: dict | None = None,
        dont_response_when_no_documents: bool = True,
        system_prompt: str or None = None,
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
